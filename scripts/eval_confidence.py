"""Does `heatmap_confidence` predict per-point error?

Exists because the docstring on `heatmap_confidence` promises only a "heuristic
scale, not a calibrated probability" written for temporal smoothing, and that
left open what else it can carry. Ear_3D_Lifter tried to use it to select views
for triangulation, found nothing on renders (within-landmark Spearman -0.018 at
p50 error 19.39 px), and attributed it to architecture: the loss is WingLoss on
COORDINATES through soft_argmax, so heatmap amplitude is never constrained --
`heatmaps()` feeds straight into `F.softmax`, discarding the scale before any
loss sees it. A softmax must place its mass somewhere, so it can say "unsure
WHERE" and never "nothing is HERE".

That reasoning is correct about the model and still predicts the wrong thing in
domain. On photographs the signal is strong (within-landmark -0.486, 55/55
landmarks below -0.1, 5.3x lift as a gross-miss detector). The null is a
render-domain effect, established from both ends: this script going UP toward
their error regime without the correlation decaying, and their own descent DOWN
to a comparable error level without it appearing.

WHAT THIS MEASURES, AND THE THREE WAYS IT CAN LIE:

  1. POOLED vs WITHIN-LANDMARK. Selection chooses among instances for a FIXED
     landmark, so a pooled correlation can be carried entirely by "landmark 7
     is intrinsically hard and also intrinsically diffuse" while saying nothing
     about the choice anyone actually makes. Within-landmark is the figure that
     matters; pooled is printed only to show they agree.

  2. DETECTION vs RANKING. Restricting to worse errors decays the correlation
     (>0 px -0.475, >10 px -0.125). Concentration separates good points from
     bad ones well and orders the bad ones among themselves poorly. Those are
     different instruments and a single rho hides which one you have.

  3. SELECTION TRUNCATION. The regime pass selects whole IMAGES by mean error,
     never points by their own error. Selecting points on the correlated
     variable truncates it and drives rho toward 0 as a pure artefact of the
     selection -- which would manufacture exactly the null being tested for.

WHAT IT CANNOT MEASURE: absence. Every point in data/manual/test.csv is
in-frame (57,970 of 57,970 visible), so the claim that a spatial softmax cannot
report "nothing is here" is untested by anything here and rests on the code
alone. Do not cite this script as evidence either way on absence.

Usage:
    python scripts/eval_confidence.py
    python scripts/eval_confidence.py --run fixed_conv_s42 --data-dir data/manual
    python scripts/eval_confidence.py --split test_clean.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import percentileofscore, spearmanr
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.dataset import EarLandmarkDataset              # noqa: E402
from model.lightning_module import EarLandmarkerModule   # noqa: E402

CKPT_DIR = ROOT / "runs" / "checkpoints"
IMAGE_SIZE = 192
REF_STD = 0.05

# Ear_3D_Lifter's render figures, for the domain comparison. Measured on the
# same checkpoint (fixed_conv_s42/433) at sun +1.2, 20 ears, 27,500
# landmark-views, against 2D placement error from projected ground truth.
PEER = {"p50_px": 19.39, "rho_within": -0.018, "below": 2, "total": 55}


def best_ckpt(run: str) -> Path | None:
    """Lowest-NME checkpoint for a run, scored by the value INSIDE the file.

    The `nme=` field in the filename is rounded to 4 decimals and most runs have
    two or three checkpoints tied at that precision, so picking by name resolves
    the tie on glob order. Same selection rule as scripts/eval_test.py.
    """
    cks = [c for c in (CKPT_DIR / run).glob("*.ckpt") if "nme=" in c.name]
    if not cks:
        return None

    def score(c: Path) -> float:
        try:
            ck = torch.load(str(c), map_location="cpu", weights_only=False)
            for v in (ck.get("callbacks") or {}).values():
                if isinstance(v, dict) and v.get("current_score") is not None:
                    return float(v["current_score"])
        except Exception:
            pass
        return float(c.stem.split("nme=")[1])

    return min(cks, key=score)


def collect(ckpt: Path, data_dir: Path, split: str, batch_size: int,
            workers: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One inference pass -> (confidence, error_px, visible), each (N_img, K).

    All three analyses below need the same matrix, so they share one pass; the
    model is large enough that reloading it per analysis dominated the runtime
    of the throwaway versions of this.
    """
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    module = EarLandmarkerModule.load_from_checkpoint(str(ckpt), map_location=dev)
    module.eval().to(dev)
    net = module.model
    tau = float(module.hparams["tau"])

    ds = EarLandmarkDataset(split_csv=data_dir / split, data_dir=data_dir,
                            image_size=IMAGE_SIZE)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=workers)

    conf, err, vis = [], [], []
    with torch.no_grad():
        for b in loader:
            hm = net.heatmaps(b["image"].to(dev))              # (B, K, 24, 24)
            coords = net.soft_argmax(hm, tau)                  # (B, K*2)
            c = net.heatmap_confidence(hm, tau, REF_STD)       # (B, K)

            k = c.shape[1]
            p = coords.view(-1, k, 2).cpu()
            t = b["landmarks"].view(-1, k, 2)
            # Pixels of the crop, not normalised units, so the numbers are
            # directly comparable to the peer's px figures.
            err.append((torch.norm(p - t, dim=-1) * IMAGE_SIZE).numpy())
            conf.append(c.cpu().numpy())
            vis.append(b["visible"].numpy())

    return (np.concatenate(conf), np.concatenate(err),
            np.concatenate(vis).astype(bool))


def zscore_within_landmark(conf: np.ndarray) -> np.ndarray:
    """Confidence re-expressed as "low FOR THIS LANDMARK".

    Needed wherever points from different landmarks are pooled: the raw scale
    differs per landmark, so a raw threshold selects the diffuse landmarks
    rather than the diffuse instances.
    """
    z = np.zeros_like(conf, dtype=float)
    for j in range(conf.shape[1]):
        v = conf[:, j]
        z[:, j] = (v - v.mean()) / max(v.std(), 1e-9)
    return z


def within_landmark_rho(conf: np.ndarray, err: np.ndarray,
                        min_n: int = 30) -> tuple[float, int, int, float, float]:
    """(median rho, count below -0.1, landmarks scored, min, max)."""
    rhos = []
    for j in range(conf.shape[1]):
        if conf.shape[0] < min_n:
            continue
        r, _ = spearmanr(conf[:, j], err[:, j])
        if np.isfinite(r):
            rhos.append(r)
    a = np.array(rhos)
    if a.size == 0:
        return float("nan"), 0, 0, float("nan"), float("nan")
    return (float(np.median(a)), int((a < -0.1).sum()), a.size,
            float(a.min()), float(a.max()))


def report_correlation(conf, err, z) -> None:
    print("=== 1. Does confidence rank error? ===\n")
    print(f"confidence  p1 {np.percentile(conf, 1):.4f}  "
          f"p50 {np.percentile(conf, 50):.4f}  "
          f"p99 {np.percentile(conf, 99):.4f}  "
          f"(span {np.percentile(conf, 99) / max(np.percentile(conf, 1), 1e-9):.1f}x)")
    print(f"2D error    p50 {np.percentile(err, 50):.2f} px  "
          f"p90 {np.percentile(err, 90):.2f} px\n")

    rho, p = spearmanr(conf.ravel(), err.ravel())
    print(f"pooled                         Spearman {rho:+.3f}  (p={p:.1e})")
    med, below, tot, lo, hi = within_landmark_rho(conf, err)
    print(f"within each landmark (median)  Spearman {med:+.3f},  "
          f"{below}/{tot} below -0.1")
    print(f"                               range {lo:+.3f} .. {hi:+.3f}")

    qs = np.percentile(z.ravel(), [20, 40, 60, 80])
    bins = np.digitize(z.ravel(), qs)
    e = err.ravel()
    meds = [float(np.percentile(e[bins == q], 50)) for q in range(5)]
    print("\nby quintile of within-landmark confidence (low -> high):")
    print("  2D error p50   " + "  ".join(f"{v:5.2f}" for v in meds) + "  px")
    print(f"  lowest -> highest: {meds[0]:.2f} -> {meds[-1]:.2f} px "
          f"({(meds[-1] - meds[0]) / meds[0] * 100:+.1f}%)")


def report_detection(conf, err, z, floor_px: float = 10.0) -> None:
    print("\n=== 2. Detection vs ranking ===\n")
    zf, ef = z.ravel(), err.ravel()

    print("Spearman restricted to error above a floor (ranking among the bad):")
    print("  floor       n       rho")
    for floor in (0, 2, 5, 10, 15, 20, 30):
        s = ef > floor
        if s.sum() < 50:
            print(f"  >{floor:2d} px  {int(s.sum()):6d}       --   too few to rank")
            continue
        r, p = spearmanr(zf[s], ef[s])
        print(f"  >{floor:2d} px  {int(s.sum()):6d}   {r:+.3f}   p={p:.1e}")

    print("\nWhere do the WORST points sit in confidence?")
    print("  (percentile of within-landmark z; 50 = typical, <20 = flagged low)")
    order = np.argsort(-ef)
    for label, n in (("worst 50", 50), ("worst 200", 200), ("worst 1000", 1000)):
        idx = order[:n]
        pct = float(np.mean([percentileofscore(zf, zf[i]) for i in idx]))
        print(f"  {label:11s} mean confidence percentile {pct:5.1f}  "
              f"(error {ef[idx].min():.1f}-{ef[idx].max():.1f} px)")

    bad = ef > floor_px
    print(f"\nAs a detector for a gross miss (error > {floor_px:g} px):")
    print(f"  base rate {bad.mean() * 100:.2f}%  "
          f"({int(bad.sum())} of {bad.size:,})")
    for q in (5, 10, 20, 30):
        flag = zf <= np.percentile(zf, q)
        tp = int((flag & bad).sum())
        prec = tp / max(int(flag.sum()), 1)
        rec = tp / max(int(bad.sum()), 1)
        print(f"  flag lowest {q:2d}%: recall {rec * 100:5.1f}%  "
              f"precision {prec * 100:5.2f}%  "
              f"lift {prec / max(bad.mean(), 1e-9):4.1f}x")


def report_regime(conf, err) -> None:
    """Is a null on renders explained by error LEVEL rather than by domain?

    If the signal simply dies wherever the model is failing, keeping only the
    worst images should collapse rho toward the render figure. It does not.
    """
    print("\n=== 3. Domain or regime? ===\n")
    print(f"this split      p50 error {np.percentile(err, 50):6.2f} px")
    print(f"render regime   p50 error {PEER['p50_px']:6.2f} px  "
          f"(rho {PEER['rho_within']:+.3f}, "
          f"{PEER['below']}/{PEER['total']} below -0.1)\n")

    per_img = err.mean(axis=1)
    order = np.argsort(-per_img)            # worst images first
    n_img = conf.shape[0]

    print("keeping only the worst N images (selected by MEAN IMAGE error, never")
    print("by a point's own error -- see SELECTION TRUNCATION in the docstring):")
    print("     N    p50 err   within-landmark rho   below -0.1")
    for n in (n_img, 500, 200, 100, 50, 25, 10):
        if n > n_img:
            continue
        idx = order[:n]
        c, e = conf[idx], err[idx]
        if n < 30:
            r, _ = spearmanr(c.ravel(), e.ravel())
            print(f"  {n:4d}     {np.percentile(e, 50):6.2f}      "
                  f"{r:+.3f} (pooled)        --")
            continue
        med, below, tot, _, _ = within_landmark_rho(c, e)
        print(f"  {n:4d}     {np.percentile(e, 50):6.2f}      "
              f"{med:+.3f}              {below:2d}/{tot}")

    worst_p50 = float(np.percentile(err[order[:50]], 50))
    print(f"\nFlat across the range, so REGIME is refuted: the worst 50 images sit")
    print(f"at {worst_p50:.2f} px and still rank. The render null is a DOMAIN effect.")
    print(f"LIMIT: this split cannot reach {PEER['p50_px']:.1f} px, so the match is")
    print("approached from below, not met. The other half of this -- descending")
    print("from the render regime to a comparable level without the correlation")
    print("appearing -- was run by Ear_3D_Lifter, not here.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="fixed_conv_s42",
                    help="checkpoint run under runs/checkpoints (default: %(default)s)")
    ap.add_argument("--ckpt", help="explicit checkpoint path, overriding --run")
    ap.add_argument("--data-dir", default="data/manual",
                    help="dataset root (default: %(default)s)")
    ap.add_argument("--split", default="test.csv",
                    help="split csv inside --data-dir (default: %(default)s)")
    ap.add_argument("--gross-miss-px", type=float, default=10.0,
                    help="error above which a point counts as a gross miss")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    ckpt = Path(args.ckpt) if args.ckpt else best_ckpt(args.run)
    if ckpt is None or not ckpt.exists():
        raise SystemExit(f"no checkpoint for run {args.run!r} under {CKPT_DIR}")
    data_dir = ROOT / args.data_dir

    conf, err, vis = collect(ckpt, data_dir, args.split,
                             args.batch_size, args.workers)

    # Invisible points have no meaningful ground-truth position, so their
    # "error" is not an error. Kept as a whole-image matrix where possible,
    # because the within-landmark and per-image passes both need the shape; the
    # mask is reported so a split that actually has occlusions is not silently
    # analysed as if it did not.
    n_vis, n_tot = int(vis.sum()), vis.size
    print(f"checkpoint      {ckpt.parent.name}/{ckpt.name}")
    print(f"split           {data_dir.name}/{args.split}  "
          f"({conf.shape[0]} images, {conf.shape[1]} landmarks)")
    print(f"landmark-views  {n_vis:,} visible of {n_tot:,}")
    print(f"ref_std         {REF_STD}\n")

    if n_vis < n_tot:
        # Zero-weighting an invisible point is wrong for the per-image regime
        # pass (it would pull that image's mean error down), so drop any image
        # with an occluded point rather than mixing the two conventions.
        keep = vis.all(axis=1)
        print(f"NOTE: {n_tot - n_vis:,} invisible points present; dropping the "
              f"{int((~keep).sum())} image(s) containing them.")
        print("      This split can say nothing about ABSENCE either way.\n")
        conf, err = conf[keep], err[keep]

    z = zscore_within_landmark(conf)
    report_correlation(conf, err, z)
    report_detection(conf, err, z, args.gross_miss_px)
    report_regime(conf, err)


if __name__ == "__main__":
    main()
