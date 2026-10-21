"""Compare two checkpoints on one test split, PAIRED, with a per-point breakdown.

Exists because the two obvious ways to compare runs both fail here.

A single test NME cannot settle it: three seeds of one configuration give
0.02919 +/- 0.00031, so the resolution of this setup is about 1% relative and
anything smaller is a seed, not a result. Two independent means throw away the
fact that both models scored the *same images*, and this repo has already paid
for that lesson once -- in the flip review (RESULTS.md part 2), an unpaired version of the handedness
test returned |t| < 0.7 for every cut, because row-to-row difficulty swamps the
effect, while the paired version found +17.05% at p=2.7e-19. Pairing is not a
refinement here, it is the difference between seeing an effect and not.

And a pooled mean hides where the error lives. Ear_3D_Lifter's handoff is
explicit that the 2D model's tail is what costs them: the upper helix (points
0-8) is the largest cluster past 5 mm, and the lobe (15-19) is the unstable one.
A change that fixes the helix and does nothing else will be invisible in pooled
NME. So the breakdown is reported beside the mean, not instead of it, and the
helix and lobe groups are named because they are the ones with a consumer.

WHAT IS AND IS NOT CONTROLLED. This scores whatever two checkpoints you name on
whatever split you name; it does not know whether either model trained on it.
For the v2-vs-v3 comparison the split is data/v3_manual/test.csv, drawn by
scripts/ingest_manual_v3.py from ears absent from data/manual/train.csv, so
neither model has seen them. Point that at a split one model trained on and the
number will be meaningless and this script will not notice.

Usage:
    python scripts/compare_runs_paired.py fixed_conv_s42 v3_conv_s42 \
        --data-dir data/v3_manual
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ttest_rel, wilcoxon
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.dataset import EarLandmarkDataset              # noqa: E402
from model.lightning_module import EarLandmarkerModule   # noqa: E402

CKPT_DIR = ROOT / "runs" / "checkpoints"
IMAGE_SIZE = 192

# The 3D side addresses these indices as fixed anatomical addresses, so they are
# named here rather than left as slices for a reader to decode.
GROUPS = {
    "upper helix 0-8": range(0, 9),
    "rim rest 9-19": range(9, 20),
    "lobe 15-19": range(15, 20),
    "helix whole 0-34": range(0, 35),
    "concha 40-49": range(40, 50),
    "canal 39": range(39, 40),
    "everything else 35-54": list(range(35, 55)),
}


def best_ckpt(run: str) -> Path | None:
    """Lowest-NME checkpoint, by the score stored INSIDE the file.

    The nme= in the filename is rounded to 4 decimals and runs routinely have
    two or three checkpoints tied there, so selecting by name resolves the tie
    on glob order and silently picks a different checkpoint than the one that
    was tested. Same rule as scripts/eval_test.py.
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


def per_point_error(ckpt: Path, data_dir: Path, split: str,
                    batch: int, workers: int) -> tuple[np.ndarray, np.ndarray]:
    """-> (err (N, K) in normalised units, visible (N, K) bool).

    Error is left in normalised [0,1] crop units, matching how `nme` is defined
    everywhere else in this repo, so a number here is comparable to a number in
    RESULTS.md rather than to a pixel count that depends on image_size.
    """
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    module = EarLandmarkerModule.load_from_checkpoint(str(ckpt), map_location=dev)
    module.eval().to(dev)

    ds = EarLandmarkDataset(split_csv=data_dir / split, data_dir=data_dir,
                            image_size=IMAGE_SIZE)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers)

    errs, vis = [], []
    with torch.no_grad():
        for b in loader:
            pred = module.model(b["image"].to(dev)).cpu()
            k = b["landmarks"].shape[1] // 2
            p = pred.view(-1, k, 2)
            t = b["landmarks"].view(-1, k, 2)
            errs.append(torch.norm(p - t, dim=-1).numpy())
            vis.append(b["visible"].numpy().astype(bool))
    return np.concatenate(errs), np.concatenate(vis)


def summarise(name: str, err: np.ndarray, vis: np.ndarray) -> float:
    m = err[vis]
    print(f"  {name:16s} nme {m.mean():.5f}   median {np.median(m):.5f}   "
          f"p90 {np.percentile(m, 90):.5f}   n={m.size:,}")
    return float(m.mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("baseline", help="run name, e.g. fixed_conv_s42")
    ap.add_argument("candidate", help="run name, e.g. v3_conv_s42")
    ap.add_argument("--data-dir", default="data/v3_manual")
    ap.add_argument("--split", default="test.csv")
    ap.add_argument("--gross", type=float, default=0.05,
                    help="normalised error above which a point is a gross miss")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    data_dir = ROOT / args.data_dir
    runs = {}
    for role, name in (("baseline", args.baseline), ("candidate", args.candidate)):
        ck = best_ckpt(name)
        if ck is None:
            raise SystemExit(f"no checkpoint for run {name!r} under {CKPT_DIR}")
        runs[role] = (name, ck)

    print(f"split      {args.data_dir}/{args.split}")
    for role, (name, ck) in runs.items():
        print(f"{role:10s} {name}  ({ck.name})")
    print()

    eb, vb = per_point_error(runs["baseline"][1], data_dir, args.split,
                             args.batch_size, args.workers)
    ec, vc = per_point_error(runs["candidate"][1], data_dir, args.split,
                             args.batch_size, args.workers)
    assert eb.shape == ec.shape, "the two runs scored different numbers of rows"
    # Both models see the same loader, so visibility is identical; intersect
    # anyway rather than assume, since a mismatch would silently misalign pairs.
    vis = vb & vc
    if not np.array_equal(vb, vc):
        print("NOTE: visibility masks differed; using the intersection\n")

    print("pooled over all visible points")
    mb = summarise(runs["baseline"][0], eb, vis)
    mc = summarise(runs["candidate"][0], ec, vis)
    rel = (mc - mb) / mb * 100
    print(f"  {'delta':16s} {mc - mb:+.5f}   ({rel:+.2f}% relative)")
    print("  the seed floor on this setup is about 1% relative, so a gap")
    print("  smaller than that is not a result\n")

    # --- paired, per image ---------------------------------------------------
    # Per-image mean over that image's visible points, so each row contributes
    # once and the pairing is image-to-image.
    def per_image(e):
        s = np.where(vis, e, 0.0).sum(1)
        n = vis.sum(1)
        keep = n > 0
        return (s[keep] / n[keep]), keep

    pb, keep = per_image(eb)
    pc, _ = per_image(ec)
    d = pc - pb
    t, p_t = ttest_rel(pc, pb)
    try:
        _, p_w = wilcoxon(pc, pb)
    except ValueError:
        p_w = float("nan")
    better = int((d < 0).sum())
    print(f"paired by image, n={d.size}")
    print(f"  candidate better on {better} of {d.size} images "
          f"({better / d.size * 100:.1f}%)")
    print(f"  mean paired delta {d.mean():+.5f}   t={t:+.2f}  p={p_t:.2e}")
    print(f"  wilcoxon p={p_w:.2e}  (no normality assumption)\n")

    # --- where the change lives ---------------------------------------------
    print("by anatomical group (mean over visible points in the group)")
    print(f"  {'group':22s} {'baseline':>9s} {'candidate':>10s} {'delta':>9s} "
          f"{'rel':>8s}")
    for label, idx in GROUPS.items():
        idx = list(idx)
        m = vis[:, idx]
        if not m.any():
            continue
        a = eb[:, idx][m].mean()
        b = ec[:, idx][m].mean()
        print(f"  {label:22s} {a:9.5f} {b:10.5f} {b - a:+9.5f} "
              f"{(b - a) / a * 100:+7.2f}%")

    # --- the tail, which a mean hides ---------------------------------------
    g = args.gross
    bad_b = (eb > g) & vis
    bad_c = (ec > g) & vis
    nb, nc = int(bad_b.sum()), int(bad_c.sum())
    tot = int(vis.sum())
    print(f"\ngross misses, error > {g:g} normalised ({g * IMAGE_SIZE:.0f} px "
          f"at {IMAGE_SIZE})")
    print(f"  {runs['baseline'][0]:16s} {nb:6d}  ({nb / tot * 100:.3f}%)")
    print(f"  {runs['candidate'][0]:16s} {nc:6d}  ({nc / tot * 100:.3f}%)")
    # McNemar on discordant points: the shared points cancel, and only the
    # points where exactly one model failed carry information.
    b01 = int((~bad_b & bad_c & vis).sum())
    b10 = int((bad_b & ~bad_c & vis).sum())
    if b01 + b10:
        chi = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
        from scipy.stats import chi2
        print(f"  discordant: candidate-only {b01}, baseline-only {b10}, "
              f"McNemar p={1 - chi2.cdf(chi, 1):.2e}")


if __name__ == "__main__":
    main()
