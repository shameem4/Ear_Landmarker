"""Score landmark models on real full frames WITHOUT ground-truth landmarks.

WHY THIS EXISTS: the preprocessed test set is 55% collectionB, whose annotation
convention v6_persp65 was trained on. A model that reproduces that convention
scores well there whether or not it is anatomically right, and a model trained
elsewhere scores badly whether or not it is wrong. Comparing two such models on
that benchmark measures agreement with collectionB, not accuracy.

This removes ground truth from the comparison entirely. Both models run through
the identical two-stage pipeline on the same real photographs, and each
prediction is scored by how strongly the outer-helix contour lies on actual
image structure (Sobel gradient magnitude, per-image normalised so blur and
exposure cannot bias the comparison).

WHAT THIS CAN AND CANNOT SHOW: edge response rewards a contour that follows a
real intensity boundary. It cannot tell the helix rim from the jawline, so a
contour that overshoots onto another strong edge is not penalised as it should
be. Treat a small difference as inconclusive; it is evidence about contour
placement, not a replacement for annotated error.

Images come from BlazeEar's local full-scene data (see
scripts/build_local_testset.py). They carry ground-truth ear BOXES, used here
only to pick frames that really contain an ear.

Usage:
    python scripts/eval_gt_free.py --limit 100 v6_persp65 manual_occ_s42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_local_testset import find_sources, load_boxes   # noqa: E402
from eval_test import best_ckpt                            # noqa: E402
from inference import (                                    # noqa: E402
    BLAZEEAR_DIR, DETECTOR_WEIGHTS, EarLandmarkerPipeline,
)

OUTER_HELIX = (0, 20)


def resample(poly: np.ndarray, n: int = 300) -> np.ndarray:
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    if cum[-1] < 1e-9:
        return np.repeat(poly[:1], n, 0)
    t = np.linspace(0, cum[-1], n)
    return np.stack([np.interp(t, cum, poly[:, 0]), np.interp(t, cum, poly[:, 1])], 1)


def bilinear(gm: np.ndarray, pts: np.ndarray) -> np.ndarray:
    x = np.clip(pts[:, 0], 0, gm.shape[1] - 1.001)
    y = np.clip(pts[:, 1], 0, gm.shape[0] - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    return (gm[y0, x0] * (1 - fx) * (1 - fy) + gm[y0, x0 + 1] * fx * (1 - fy)
            + gm[y0 + 1, x0] * (1 - fx) * fy + gm[y0 + 1, x0 + 1] * fx * fy)


def edge_map(path: Path) -> np.ndarray:
    g = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    g = ndimage.gaussian_filter(g, 1.5)
    gm = np.hypot(ndimage.sobel(g, 0), ndimage.sobel(g, 1))
    return gm / (gm.mean() + 1e-9)


def collect_frames(limit: int, seed: int) -> list[Path]:
    """Full frames that ground truth says contain at least one ear."""
    frames: list[Path] = []
    for img_dir, lbl_dir in find_sources():
        for p in sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")):
            lbl = lbl_dir / (p.stem + ".txt")
            if lbl.exists():
                frames.append(p)
    rng = np.random.default_rng(seed)
    if len(frames) > limit:
        frames = [frames[i] for i in sorted(rng.choice(len(frames), limit, replace=False))]
    return frames


def main() -> None:
    p = argparse.ArgumentParser(description="Ground-truth-free landmark comparison")
    p.add_argument("runs", nargs="+")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None, help="npz of per-frame scores")
    args = p.parse_args()

    frames = collect_frames(args.limit, args.seed)
    print(f"{len(frames)} full frames with ground-truth ear boxes\n")

    pipes = {r: EarLandmarkerPipeline(BLAZEEAR_DIR / DETECTOR_WEIGHTS,
                                      best_ckpt(r), smooth=False) for r in args.runs}
    scores = {r: [] for r in args.runs}
    aspects = {r: [] for r in args.runs}
    frame_of: list[int] = []
    paired = 0
    for n, f in enumerate(frames):
        gm = edge_map(f)
        img = np.asarray(Image.open(f).convert("RGB"))
        dets = {r: pipes[r](img) for r in args.runs}
        # Only score frames where EVERY model found the same number of ears, so
        # the comparison is paired and detector disagreement cannot skew it.
        counts = {len(d) for d in dets.values()}
        if len(counts) != 1 or counts == {0}:
            continue
        paired += 1
        frame_of.extend([n] * len(dets[args.runs[0]]))
        for r in args.runs:
            for det in dets[r]:
                lm = np.asarray(det["landmarks"], dtype=float)
                scores[r].append(bilinear(gm, resample(lm[slice(*OUTER_HELIX)])).mean())
                L = max(np.ptp(lm[:, 0]), np.ptp(lm[:, 1]))
                Wd = min(np.ptp(lm[:, 0]), np.ptp(lm[:, 1]))
                aspects[r].append(L / max(Wd, 1e-9))
        if (n + 1) % 25 == 0:
            print(f"  {n+1}/{len(frames)} frames, {paired} paired")

    print(f"\n{paired} frames usable, {len(scores[args.runs[0]])} ears scored\n")
    print(f"{'model':20s} {'edge response':>14s} {'sd':>8s} {'aspect med':>11s}")
    for r in args.runs:
        s = np.array(scores[r])
        print(f"{r:20s} {s.mean():14.4f} {s.std():8.4f} {np.median(aspects[r]):11.2f}")

    if len(args.runs) == 2:
        a, b = (np.array(scores[r]) for r in args.runs)
        d = b - a
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"\n{args.runs[1]} - {args.runs[0]}: {d.mean():+.4f} +/- {se:.4f} (SE)"
              f"   t = {d.mean()/se:+.2f}   n = {len(d)} ears  [UNCLUSTERED]")
        # Two ears from one photo share subject, lighting and focus, so treating
        # them as independent overstates the evidence. Aggregate per frame first.
        fo = np.array(frame_of)
        per = np.array([d[fo == u].mean() for u in np.unique(fo)])
        sec = per.std(ddof=1) / np.sqrt(len(per))
        print(f"{'':>{len(args.runs[1]) + len(args.runs[0]) + 3}} {per.mean():+.4f} +/- {sec:.4f} (SE)"
              f"   t = {per.mean()/sec:+.2f}   n = {len(per)} frames [CLUSTERED]")
        print(f"{args.runs[1]} higher on {100*(d>0).mean():.1f}% of ears, "
              f"{100*(per>0).mean():.1f}% of frames")
    if args.out:
        np.savez(args.out, **{r: np.array(scores[r]) for r in args.runs})
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
