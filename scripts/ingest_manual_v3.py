"""Ingest the v3 cherry-picked manual annotations into the repo's data format.

`v3_manual_data/` arrives as 6,340 flat `ec_NNNNNN.png` + `ec_NNNNNN_annotation.txt`
pairs, where the annotation is 55 lines of `x y` in the PNG's own pixel space.
`EarLandmarkDataset` wants a different shape: a `landmarks.npy` of (N, 55, 2)
normalised to the crop and indexed by a manifest `idx`, plus split CSVs. This
builds that.

WHY THE SPLIT IS NOT RANDOM, which is the whole reason this script exists
rather than a one-liner. The v3 ids overlap the existing corpus:

    4,452 ids also in data/manual   (same ear, DIFFERENT crop and fresh labels)
    1,888 ids not in it at all
    6,083 old ids dropped from v3

and specifically **456 of the old test split's 1,054 ears are in v3**. So the
two obvious designs are both broken:

  - train on all of v3 and score on `data/manual/test.csv`: 43% of that test
    set is in training, under a different crop. The number goes up and means
    nothing.
  - draw a v3 test split at random: 3,561 v3 ids are in the OLD TRAIN split, so
    a random split hands the *baseline* ears it was trained on, and the leak
    favours the incumbent instead.

So the test split is drawn only from v3 ids **absent from `data/manual/train.csv`**
(2,779 candidates). Neither model has seen those: the baseline never trained on
them, and this script keeps them out of the v3 train split. Both models are
clean on the same yardstick.

The crops differ between the two corpora for the same ear -- sampled pairs ran
181 vs 140, 1316 vs 987, and 277 vs 324 the other way -- so the overlap is a
shared ear identity, not a shared image. Identity is what has to be held apart,
which is why the split keys on the `ec_NNNNNN` stem.

Usage:
    python scripts/ingest_manual_v3.py                 # build data/v3_manual/
    python scripts/ingest_manual_v3.py --test-size 900
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "v3_manual_data"
OUT = ROOT / "data" / "v3_manual"
OLD = ROOT / "data" / "manual"
N_POINTS = 55


def old_split_stems(name: str) -> set[str]:
    """The `ec_NNNNNN` stems in one of the existing split CSVs."""
    f = OLD / f"{name}.csv"
    if not f.exists():
        raise SystemExit(f"missing {f}; the split design depends on it")
    with open(f, encoding="utf-8") as fh:
        return {Path(r["image_file"]).stem for r in csv.DictReader(fh)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--test-size", type=int, default=1000,
                    help="held-out ears, drawn from ids absent from old train")
    ap.add_argument("--val-frac", type=float, default=0.11,
                    help="validation fraction of what remains (default ~1:8)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pngs = sorted(SRC.glob("ec_*.png"))
    if not pngs:
        raise SystemExit(f"no ec_*.png under {SRC}")

    # --- read every pair, normalise to its own crop --------------------------
    stems, rows, pts = [], [], []
    skipped: list[tuple[str, str]] = []
    for png in pngs:
        ann = png.with_name(f"{png.stem}_annotation.txt")
        if not ann.exists():
            skipped.append((png.stem, "no annotation"))
            continue
        a = np.loadtxt(ann, dtype=np.float64)
        if a.shape != (N_POINTS, 2):
            skipped.append((png.stem, f"shape {a.shape}"))
            continue
        w, h = Image.open(png).size
        if w <= 0 or h <= 0:
            skipped.append((png.stem, "bad image size"))
            continue
        # Normalised to the crop, and deliberately NOT clamped: the dataset
        # treats a coordinate at or outside 0/1 as out-of-crop and leaves it
        # unsupervised, so clamping would forge a label on the border.
        n = np.empty_like(a, dtype=np.float32)
        n[:, 0] = a[:, 0] / w
        n[:, 1] = a[:, 1] / h
        stems.append(png.stem)
        pts.append(n)
        rows.append({"image_file": f"images/{png.name}", "source": "manual_v3",
                     "original_path": f"v3_manual_data/{png.name}",
                     "width": w, "height": h})

    land = np.stack(pts)                                    # (N, 55, 2) float32
    n_total = len(stems)
    print(f"read {n_total} pairs, skipped {len(skipped)}")
    for s, why in skipped[:10]:
        print(f"  skipped {s}: {why}")

    outside = ((land <= 0.0) | (land >= 1.0)).any(axis=2)
    print(f"points outside the crop: {outside.sum():,} of {outside.size:,} "
          f"({outside.sum()/outside.size*100:.2f}%), affecting "
          f"{outside.any(axis=1).sum()} of {n_total} ears "
          f"-- left unsupervised, not clamped")

    # --- split on EAR IDENTITY, not at random --------------------------------
    old_train = old_split_stems("train")
    eligible = [s for s in stems if s not in old_train]      # baseline-clean
    contaminated = n_total - len(eligible)
    print(f"\ntest-eligible (absent from old train): {len(eligible)}")
    print(f"ineligible (in old train, so the baseline has seen them): "
          f"{contaminated}")
    if len(eligible) < args.test_size:
        raise SystemExit(f"only {len(eligible)} eligible for a "
                         f"{args.test_size}-ear test split")

    rng = random.Random(args.seed)
    test = set(rng.sample(sorted(eligible), args.test_size))
    rest = [s for s in stems if s not in test]
    rng.shuffle(rest)
    n_val = int(round(len(rest) * args.val_frac))
    val, train = set(rest[:n_val]), set(rest[n_val:])

    # --- write ---------------------------------------------------------------
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    idx_of = {s: i for i, s in enumerate(stems)}
    np.save(OUT / "landmarks.npy", land)

    fields = ["idx", "image_file", "source", "original_path", "width", "height"]

    def write(name: str, keep: set[str] | None) -> int:
        n = 0
        with open(OUT / f"{name}.csv", "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=fields)
            wr.writeheader()
            for s, r in zip(stems, rows):
                if keep is None or s in keep:
                    wr.writerow({"idx": idx_of[s], **r})
                    n += 1
        return n

    counts = {k: write(k, v) for k, v in
              (("manifest", None), ("train", train), ("val", val), ("test", test))}

    # Images are symlinked, not copied: 750 MB, and the source is the record.
    #
    # Rebuilt rather than topped up, because the source set SHRINKS. The v3
    # corpus went 6,340 -> 6,099 when filtering tightened, and a link left
    # behind for a dropped ear dangles. Worse, `Path.exists()` FOLLOWS the
    # link, so a dangling one reads as absent and `symlink_to` then raises
    # FileExistsError -- the second run of this script would have crashed
    # rather than updated. `lexists` asks about the link itself.
    want = set(stems)
    pruned = 0
    for dst in (OUT / "images").iterdir():
        if dst.stem not in want:
            dst.unlink()
            pruned += 1
    linked = 0
    for s in stems:
        dst = OUT / "images" / f"{s}.png"
        src = SRC / f"{s}.png"
        if dst.is_symlink() and dst.readlink() == src:
            continue                      # already correct
        if os.path.lexists(dst):
            dst.unlink()                  # stale or dangling
        dst.symlink_to(src)
        linked += 1

    print(f"\nwrote {OUT}")
    for k in ("manifest", "train", "val", "test"):
        print(f"  {k+'.csv':14s} {counts[k]:5d}")
    print(f"  landmarks.npy  {land.shape}")
    print(f"  images/        {linked} symlinks created "
          f"({len(stems)} total)")

    # --- assert the thing the whole design rests on --------------------------
    assert not (test & old_train), "test split overlaps old train -- leak"
    assert not (test & train) and not (test & val), "split overlap"
    print("\nverified: test split shares no ear with data/manual/train.csv, "
          "so the baseline is clean on it")


if __name__ == "__main__":
    main()
