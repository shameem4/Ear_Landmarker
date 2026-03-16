"""Automated data quality checks for the unified ear landmark dataset.

Runs without manual inspection. Prints a text report and writes
flagged sample indices to data/preprocessed/flagged.json for optional removal.

Checks:
  1. Image integrity     -- can every image be opened and decoded?
  2. Landmark bounds     -- points at extremes (clipped) or outside [0,1]
  3. Landmark geometry   -- convex hull area, inter-point distances, degenerate shapes
  4. Linestrip smoothness-- consecutive points should be spatially close (detects misordering)
  5. Per-point stats     -- per-landmark mean/std across dataset; outlier detection (>3 sigma)
  6. Cross-source shape  -- mean shape similarity between sources (detects annotation convention mismatch)
  7. Resolution audit    -- image size distribution, extreme aspect ratios

Usage:
    python data/validate.py
    python data/validate.py --strict   # lower outlier threshold (2.5 sigma)
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

PREP_DIR = Path("C:/Users/shame/OneDrive/Desktop/ear_stuff/Ear Landmarker/data/preprocessed")
NUM_LANDMARKS = 55

# Linestrip grouping: shape 0 (20 pts), shape 1 (15 pts), shape 2 (15 pts), shape 3 (5 pts)
LINESTRIP_RANGES = [(0, 20), (20, 35), (35, 50), (50, 55)]


def load_data():
    landmarks = np.load(PREP_DIR / "landmarks.npy")
    with open(PREP_DIR / "manifest.csv", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))
    return landmarks, manifest


# ---------------------------------------------------------------------------
# Check 1: Image integrity
# ---------------------------------------------------------------------------
def check_images(manifest: list[dict]) -> list[int]:
    """Verify every image can be opened and decoded."""
    bad = []
    for row in manifest:
        idx = int(row["idx"])
        path = PREP_DIR / row["image_file"]
        try:
            img = Image.open(path)
            img.verify()
        except Exception:
            bad.append(idx)
    return bad


# ---------------------------------------------------------------------------
# Check 2: Landmark bounds
# ---------------------------------------------------------------------------
def check_bounds(landmarks: np.ndarray, eps: float = 0.005) -> dict:
    """Find samples with landmarks at extreme edges (likely clipped)."""
    clipped_at_zero = np.any(landmarks < eps, axis=(1, 2))
    clipped_at_one = np.any(landmarks > 1.0 - eps, axis=(1, 2))
    clipped = np.where(clipped_at_zero | clipped_at_one)[0].tolist()

    outside = np.where(np.any((landmarks < 0) | (landmarks > 1), axis=(1, 2)))[0].tolist()
    return {"clipped_edge": clipped, "outside_01": outside}


# ---------------------------------------------------------------------------
# Check 3: Landmark geometry
# ---------------------------------------------------------------------------
def _convex_hull_area(points: np.ndarray) -> float:
    """Shoelace formula on convex hull (simplified: use bounding box area as proxy)."""
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return float((maxs[0] - mins[0]) * (maxs[1] - mins[1]))


def check_geometry(landmarks: np.ndarray, min_area: float = 0.01) -> dict:
    """Flag degenerate shapes: tiny bounding box, collapsed points."""
    tiny = []
    collapsed = []
    for i in range(len(landmarks)):
        area = _convex_hull_area(landmarks[i])
        if area < min_area:
            tiny.append(i)
        # Check if many points overlap (std < threshold)
        if landmarks[i].std() < 0.02:
            collapsed.append(i)
    return {"tiny_bbox": tiny, "collapsed": collapsed}


# ---------------------------------------------------------------------------
# Check 4: Linestrip smoothness
# ---------------------------------------------------------------------------
def check_linestrip_smoothness(landmarks: np.ndarray, max_jump: float = 0.15) -> list[int]:
    """Flag samples where consecutive points within a linestrip jump too far."""
    flagged = []
    for i in range(len(landmarks)):
        bad = False
        for start, end in LINESTRIP_RANGES:
            strip = landmarks[i, start:end]
            diffs = np.linalg.norm(np.diff(strip, axis=0), axis=1)
            if np.any(diffs > max_jump):
                bad = True
                break
        if bad:
            flagged.append(i)
    return flagged


# ---------------------------------------------------------------------------
# Check 5: Per-point outlier detection
# ---------------------------------------------------------------------------
def check_outliers(landmarks: np.ndarray, sigma: float = 3.0) -> dict:
    """Flag samples where any landmark is >sigma std from the per-point mean."""
    mean = landmarks.mean(axis=0)  # (55, 2)
    std = landmarks.std(axis=0)    # (55, 2)
    std = np.maximum(std, 1e-6)

    z_scores = np.abs(landmarks - mean) / std  # (N, 55, 2)
    max_z = z_scores.max(axis=(1, 2))  # (N,)
    flagged = np.where(max_z > sigma)[0].tolist()

    return {
        "flagged": flagged,
        "per_point_mean": mean,
        "per_point_std": std,
        "worst_z": float(max_z.max()),
        "median_z": float(np.median(max_z)),
    }


# ---------------------------------------------------------------------------
# Check 6: Cross-source mean shape comparison
# ---------------------------------------------------------------------------
def check_cross_source(landmarks: np.ndarray, manifest: list[dict]) -> dict:
    """Compare mean landmark shape between sources via Procrustes-like distance."""
    by_source: dict[str, list[int]] = defaultdict(list)
    for row in manifest:
        by_source[row["source"]].append(int(row["idx"]))

    means = {}
    for src, indices in by_source.items():
        means[src] = landmarks[indices].mean(axis=0)  # (55, 2)

    # Pairwise L2 distance between mean shapes (normalized by sqrt(55))
    sources = sorted(means.keys())
    distances = {}
    for i, s1 in enumerate(sources):
        for s2 in sources[i + 1:]:
            d = float(np.linalg.norm(means[s1] - means[s2]) / np.sqrt(NUM_LANDMARKS))
            distances[f"{s1} <-> {s2}"] = d

    return {"source_means": {s: m.tolist() for s, m in means.items()}, "distances": distances}


# ---------------------------------------------------------------------------
# Check 7: Resolution audit
# ---------------------------------------------------------------------------
def check_resolutions(manifest: list[dict]) -> dict:
    widths = [int(r["width"]) for r in manifest]
    heights = [int(r["height"]) for r in manifest]
    aspects = [w / max(h, 1) for w, h in zip(widths, heights)]

    extreme_aspect = [int(manifest[i]["idx"]) for i, a in enumerate(aspects) if a > 2.0 or a < 0.5]
    tiny = [int(manifest[i]["idx"]) for i, (w, h) in enumerate(zip(widths, heights)) if w < 32 or h < 32]

    return {
        "width_range": (min(widths), max(widths)),
        "height_range": (min(heights), max(heights)),
        "aspect_ratio_range": (round(min(aspects), 2), round(max(aspects), 2)),
        "extreme_aspect": extreme_aspect,
        "tiny_images": tiny,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="Use 2.5-sigma outlier threshold")
    args = parser.parse_args()
    sigma = 2.5 if args.strict else 3.0

    print("Loading data...")
    landmarks, manifest = load_data()
    N = len(landmarks)
    print(f"  {N} samples, {NUM_LANDMARKS} landmarks each\n")

    all_flagged: dict[str, list[int]] = {}

    # 1. Image integrity
    print("[1/7] Image integrity...")
    bad_imgs = check_images(manifest)
    all_flagged["bad_image"] = bad_imgs
    print(f"  FAIL: {len(bad_imgs)}" if bad_imgs else "  PASS")

    # 2. Landmark bounds
    print("[2/7] Landmark bounds...")
    bounds = check_bounds(landmarks)
    all_flagged["outside_01"] = bounds["outside_01"]
    print(f"  Outside [0,1]: {len(bounds['outside_01'])}")
    print(f"  Near edge (clipped): {len(bounds['clipped_edge'])}")

    # 3. Geometry
    print("[3/7] Landmark geometry...")
    geom = check_geometry(landmarks)
    all_flagged["tiny_bbox"] = geom["tiny_bbox"]
    all_flagged["collapsed"] = geom["collapsed"]
    print(f"  Tiny bbox (<1% area): {len(geom['tiny_bbox'])}")
    print(f"  Collapsed points: {len(geom['collapsed'])}")

    # 4. Linestrip smoothness
    print("[4/7] Linestrip smoothness...")
    jumpy = check_linestrip_smoothness(landmarks)
    all_flagged["jumpy_linestrip"] = jumpy
    print(f"  Large jumps (>0.15): {len(jumpy)}")

    # 5. Per-point outliers
    print(f"[5/7] Per-point outliers (>{sigma} sigma)...")
    outliers = check_outliers(landmarks, sigma)
    all_flagged["statistical_outlier"] = outliers["flagged"]
    print(f"  Flagged: {len(outliers['flagged'])}")
    print(f"  Worst z-score: {outliers['worst_z']:.1f}")
    print(f"  Median max-z: {outliers['median_z']:.2f}")

    # 6. Cross-source consistency
    print("[6/7] Cross-source shape consistency...")
    xsource = check_cross_source(landmarks, manifest)
    print("  Mean shape L2 distances (lower = more consistent):")
    for pair, d in sorted(xsource["distances"].items(), key=lambda x: -x[1]):
        status = "WARN" if d > 0.08 else "OK"
        print(f"    {pair:40s} {d:.4f}  [{status}]")

    # 7. Resolution
    print("[7/7] Resolution audit...")
    res = check_resolutions(manifest)
    print(f"  Width range:  {res['width_range']}")
    print(f"  Height range: {res['height_range']}")
    print(f"  Aspect ratio: {res['aspect_ratio_range']}")
    print(f"  Extreme aspect: {len(res['extreme_aspect'])}")
    print(f"  Tiny (<32px): {len(res['tiny_images'])}")
    all_flagged["extreme_aspect"] = res["extreme_aspect"]
    all_flagged["tiny_image"] = res["tiny_images"]

    # Aggregate
    all_bad = set()
    for indices in all_flagged.values():
        all_bad.update(indices)

    print(f"\n{'=' * 50}")
    print(f"Total samples:  {N}")
    print(f"Total flagged:  {len(all_bad)} ({100 * len(all_bad) / N:.1f}%)")
    print(f"Clean samples:  {N - len(all_bad)}")
    print(f"{'=' * 50}")

    # Breakdown of flagged by source
    by_source: dict[str, int] = defaultdict(int)
    for idx in all_bad:
        by_source[manifest[idx]["source"]] += 1
    if by_source:
        print("Flagged by source:")
        for src, count in sorted(by_source.items(), key=lambda x: -x[1]):
            total_src = sum(1 for r in manifest if r["source"] == src)
            print(f"  {src:20s} {count:>4d} / {total_src} ({100 * count / total_src:.0f}%)")

    # Save flagged indices
    output = {
        "sigma": sigma,
        "total_flagged": len(all_bad),
        "flagged_indices": sorted(all_bad),
        "by_reason": {k: v for k, v in all_flagged.items() if v},
    }
    out_path = PREP_DIR / "flagged.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nFlagged indices saved to {out_path}")


if __name__ == "__main__":
    main()
