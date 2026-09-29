"""Resample every linestrip to uniform arc-length spacing.

Why: 72% of the model's squared error is tangential -- points sitting at the
wrong place *along* a contour. That is a label problem, not a model problem. The
source datasets space their points differently along the same anatomy:

    source        spacing CV (helix)   model tangential error
    audioear2d              0.137              2.74 px
    collectionB             0.250              4.96 px
    collectionA             0.286              7.44 px

The relationship is monotone: the more irregularly a source distributes its
points, the worse the model localises them along the contour. "Point 7" simply
does not mean the same thing across sources, so the model learns an average of
mutually inconsistent conventions and hedges.

Resampling each strip to uniform arc length gives every source one convention.
Endpoints are preserved exactly, since those are the only anatomically anchored
vertices; interior points are redistributed.

LIMITATION: this can only fix spacing *within* a strip. If two sources start or
end a strip at different anatomical places, resampling aligns the parameterisation
but not the anatomy, and the residual mismatch stays. Whether that residual is
small is an empirical question -- compare tangential error before and after.

NOTE: this changes what each landmark INDEX means. Any downstream code that
measures between specific indices must be re-derived. model/measure.py is
index-free precisely so it is unaffected.

Usage:
    python data/canonicalize.py              # writes landmarks_canonical.npy
    python data/canonicalize.py --report     # spacing stats only, writes nothing
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

PREP_DIR = Path(__file__).resolve().parent / "preprocessed"

# (name, start, end) -- end exclusive
LINESTRIPS = [("outer_helix", 0, 20), ("inner_helix", 20, 35),
              ("concha_border", 35, 50), ("superior_crus", 50, 55)]


def resample_uniform(points: np.ndarray, spline: bool = True) -> np.ndarray:
    """Resample a (k, 2) strip to k points spaced uniformly by arc length.

    Endpoints are preserved exactly.

    `spline=True` interpolates with a monotone cubic (PCHIP) through the original
    vertices before resampling. This matters more than it sounds: with only 15-20
    vertices on a curved strip, resampling along the piecewise-LINEAR path cuts
    corners, so the contour itself moves rather than just its vertices. Measured
    on the real data, linear resampling deformed the contour by 2.09 px mean /
    5.31 px p95 -- comparable to the model's entire 2.48 px normal error, and it
    shifted the contour-based measurements by 1.17 mm on a 70 mm ear. That would
    have traded one error for another.

    PCHIP rather than a natural cubic because it does not overshoot between
    vertices, which on a tight concha bowl would invent curvature that is not in
    the annotation.
    """
    pts = np.asarray(points, dtype=np.float64)
    k = len(pts)
    if k < 3:
        return pts.copy()

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total < 1e-9:                       # degenerate strip, leave it alone
        return pts.copy()

    # Duplicate/coincident vertices give zero-length segments, so the arc-length
    # parameter repeats and PCHIP rejects it. Drop the repeats for fitting; the
    # curve is unchanged by removing a vertex that sits on top of its neighbour.
    keep = np.concatenate([[True], np.diff(cum) > 1e-12])
    fit_cum, fit_pts = cum[keep], pts[keep]

    if spline and len(fit_pts) >= 4:
        from scipy.interpolate import PchipInterpolator

        fx = PchipInterpolator(fit_cum, fit_pts[:, 0])
        fy = PchipInterpolator(fit_cum, fit_pts[:, 1])
        # Arc length along the SPLINE, not the chords, so spacing is uniform on
        # the curve we actually interpolate.
        dense_t = np.linspace(0.0, total, 512)
        dense = np.stack([fx(dense_t), fy(dense_t)], axis=1)
        dseg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
        dcum = np.concatenate([[0.0], np.cumsum(dseg)])
        target_s = np.linspace(0.0, dcum[-1], k)
        t_at = np.interp(target_s, dcum, dense_t)
        out = np.stack([fx(t_at), fy(t_at)], axis=1)
    else:
        target = np.linspace(0.0, total, k)
        out = np.stack([
            np.interp(target, cum, pts[:, 0]),
            np.interp(target, cum, pts[:, 1]),
        ], axis=1)

    out[0], out[-1] = pts[0], pts[-1]      # exact endpoints
    return out


def canonicalize(landmarks: np.ndarray) -> np.ndarray:
    """Apply uniform arc-length resampling to every strip of (N, 55, 2)."""
    out = np.asarray(landmarks, dtype=np.float32).copy()
    for i in range(len(out)):
        for _, a, b in LINESTRIPS:
            out[i, a:b] = resample_uniform(out[i, a:b]).astype(np.float32)
    return out


def spacing_cv(landmarks: np.ndarray) -> dict[str, float]:
    """Mean coefficient of variation of segment lengths, per strip."""
    stats = {}
    for name, a, b in LINESTRIPS:
        seg = np.linalg.norm(np.diff(landmarks[:, a:b, :], axis=1), axis=-1)
        cv = seg.std(axis=1) / np.maximum(seg.mean(axis=1), 1e-9)
        stats[name] = float(cv.mean())
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="Canonicalize landmark arc-length spacing")
    p.add_argument("--report", action="store_true", help="Report only; write nothing")
    p.add_argument("--out", default="landmarks_canonical.npy")
    args = p.parse_args()

    lms = np.load(PREP_DIR / "landmarks.npy")
    with open(PREP_DIR / "manifest.csv", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))
    by_source = defaultdict(list)
    for r in manifest:
        by_source[r["source"]].append(int(r["idx"]))

    canon = canonicalize(lms)

    print(f"{'source':14s} {'strip':11s} {'CV before':>10s} {'CV after':>10s}")
    print("-" * 50)
    for src, idx in sorted(by_source.items()):
        before = spacing_cv(lms[idx])
        after = spacing_cv(canon[idx])
        for name, _, _ in LINESTRIPS:
            print(f"{src:14s} {name:11s} {before[name]:10.4f} {after[name]:10.4f}")

    # How far did points actually move?
    moved = np.linalg.norm(canon - lms, axis=-1)
    print(f"\nlandmark displacement from canonicalization (px at 192):")
    print(f"  mean {moved.mean()*192:.3f}  p50 {np.percentile(moved,50)*192:.3f}  "
          f"p95 {np.percentile(moved,95)*192:.3f}  max {moved.max()*192:.3f}")

    if args.report:
        print("\n--report given; nothing written.")
        return

    out_path = PREP_DIR / args.out
    np.save(out_path, canon.astype(np.float32))
    print(f"\nwrote {out_path} ({canon.shape})")
    print("Train against it with: python train.py --landmarks landmarks_canonical.npy")


if __name__ == "__main__":
    main()
