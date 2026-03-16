"""Create stratified train/val splits from the preprocessed manifest.

Stratifies by source so each split reflects the overall source distribution.
Excludes samples flagged as tiny (<32px) in flagged.json.

Usage:
    python data/split.py                  # 85/15 split, seed=42
    python data/split.py --val-ratio 0.2  # 80/20 split
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PREP_DIR = Path("C:/Users/shame/OneDrive/Desktop/ear_stuff/Ear Landmarker/data/preprocessed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/val splits")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load manifest
    with open(PREP_DIR / "manifest.csv", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))

    # Load flagged tiny images to exclude
    exclude = set()
    flagged_path = PREP_DIR / "flagged.json"
    if flagged_path.exists():
        with open(flagged_path, encoding="utf-8") as f:
            flagged = json.load(f)
        exclude.update(flagged.get("by_reason", {}).get("tiny_image", []))

    # Filter
    samples = [r for r in manifest if int(r["idx"]) not in exclude]
    excluded = len(manifest) - len(samples)

    # Stratify by source
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in samples:
        by_source[r["source"]].append(r)

    rng = np.random.default_rng(args.seed)
    train_rows, val_rows = [], []

    for source, rows in sorted(by_source.items()):
        indices = rng.permutation(len(rows))
        n_val = max(1, int(len(rows) * args.val_ratio))
        val_idx = set(indices[:n_val].tolist())
        for i, row in enumerate(rows):
            if i in val_idx:
                val_rows.append(row)
            else:
                train_rows.append(row)

    # Shuffle within splits
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    # Write splits
    fields = list(manifest[0].keys())
    for name, rows in [("train", train_rows), ("val", val_rows)]:
        path = PREP_DIR / f"{name}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    # Summary
    print(f"Excluded: {excluded} (tiny images)")
    print(f"Train:    {len(train_rows)}")
    print(f"Val:      {len(val_rows)}")
    print(f"Total:    {len(train_rows) + len(val_rows)}")
    print()

    # Per-source breakdown
    for split_name, rows in [("train", train_rows), ("val", val_rows)]:
        by_src = defaultdict(int)
        for r in rows:
            by_src[r["source"]] += 1
        print(f"{split_name}:")
        for src, n in sorted(by_src.items(), key=lambda x: -x[1]):
            print(f"  {src:20s} {n:>5d}")


if __name__ == "__main__":
    main()
