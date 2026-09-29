"""Create stratified train/val/test splits from the preprocessed manifest.

Stratifies by source so each split reflects the overall source distribution.
Excludes samples flagged as tiny (<32px) in flagged.json.

`test` is held out for final reporting only -- nothing (early stopping,
checkpoint selection, hyperparameter tuning) may select on it, otherwise the
reported number is a model-selection metric rather than a generalisation
estimate.

AudioEar3D is kept entirely in train: its 112 samples are 56 subjects x
(left, right), and left/right ears of one subject are near-mirrors. With
horizontal-flip augmentation on, splitting them randomly would put effectively
the same ear in both train and eval. The manifest records only "left.json" /
"right.json", so subjects cannot be grouped -- confining the source to train
removes the leak outright.

Usage:
    python data/split.py                     # 70/15/15 split, seed=42
    python data/split.py --val-ratio 0.2 --test-ratio 0.2
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PREP_DIR = Path(__file__).resolve().parent / "preprocessed"

# Sources that must not be split across train/eval (see module docstring).
TRAIN_ONLY_SOURCES = {"audioear3d"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/val/test splits")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.val_ratio + args.test_ratio >= 1.0:
        parser.error("val-ratio + test-ratio must be < 1.0")

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
    train_rows, val_rows, test_rows = [], [], []

    for source, rows in sorted(by_source.items()):
        if source in TRAIN_ONLY_SOURCES:
            train_rows.extend(rows)
            continue

        perm = rng.permutation(len(rows))
        n_val = max(1, int(len(rows) * args.val_ratio))
        n_test = max(1, int(len(rows) * args.test_ratio))
        val_idx = set(perm[:n_val].tolist())
        test_idx = set(perm[n_val:n_val + n_test].tolist())
        for i, row in enumerate(rows):
            if i in val_idx:
                val_rows.append(row)
            elif i in test_idx:
                test_rows.append(row)
            else:
                train_rows.append(row)

    # Shuffle within splits
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)

    # Sanity: no sample may appear in more than one split
    splits = {"train": train_rows, "val": val_rows, "test": test_rows}
    seen: dict[str, str] = {}
    for name, rows in splits.items():
        for r in rows:
            if r["idx"] in seen:
                raise RuntimeError(
                    f"idx {r['idx']} in both {seen[r['idx']]} and {name}"
                )
            seen[r["idx"]] = name
    assert len(seen) == len(samples), "split lost or duplicated samples"

    # Write splits
    fields = list(manifest[0].keys())
    for name, rows in splits.items():
        path = PREP_DIR / f"{name}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    # Summary
    print(f"Excluded: {excluded} (tiny images)")
    for name, rows in splits.items():
        print(f"{name.capitalize():9s} {len(rows):>5d}")
    print(f"{'Total':9s} {len(seen):>5d}")
    print()

    # Per-source breakdown
    for split_name, rows in splits.items():
        by_src: dict[str, int] = defaultdict(int)
        for r in rows:
            by_src[r["source"]] += 1
        print(f"{split_name}:")
        for src, n in sorted(by_src.items(), key=lambda x: -x[1]):
            note = "  (train-only source)" if src in TRAIN_ONLY_SOURCES else ""
            print(f"  {src:20s} {n:>5d}{note}")


if __name__ == "__main__":
    main()
