"""Ingest the commissioned manual annotation set as a standalone data source.

10,535 square ear crops from the wild, each with a 55-point `_annotation.txt`
(one "x y" pixel pair per line, same linestrip convention as the rest of the
pipeline: 0-19 outer helix, 20-34 inner helix, 35-49 concha border, 50-54
superior crus).

WHY THIS MATTERS: the shipped model's training data is non-commercial
(iBUG-derived labels, FFHQ-derived images). Here both sides are clean -- the
annotations are commissioned work-for-hire, the images were vetted for
permissive licensing -- so a model trained on this alone has no inherited
restriction. See NOTICE.

DESIGNED FOR FREE REVERSION, same contract as scripts/ingest_synthetic.py.
Nothing here touches data/preprocessed: images, landmarks, manifest and splits
are written to data/manual/. To revert, delete that directory.

TWO THINGS THIS SCRIPT IS CAREFUL ABOUT:
  1. Framing. These crops put the ear at 0.707 of the frame. The existing
     training data averages 0.777 but with a WIDE spread (sd 0.097, range 0.50)
     because its sources frame differently -- audioear2d 0.658, collectionB
     0.853. Cropping every image to a constant 0.777 (which this script did
     originally, copying ingest_synthetic.py) matches the mean and destroys the
     variance: models trained that way see exactly one scale, and 99% of the
     existing test set then lies outside their training range. Each image now
     draws its own target occupancy from the existing data's distribution.
     Targets below the native 0.707 need more context than the source image
     holds, so those crops are grey-128 padded -- the same padding
     inference.py's square_roi_crop applies when a ROI runs off frame.
  2. Split leakage. Wild-scraped collections repeat the same ear at different
     crops or scales. Splitting per-sample would put near-duplicates on both
     sides of the test boundary and inflate the score. Samples are grouped by
     perceptual hash first, and whole groups are assigned to a split.

Usage:
    python scripts/ingest_manual.py
    python scripts/ingest_manual.py --limit 2000 --phash-threshold 8
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dct

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "new_manual_data"
OUT = ROOT / "data" / "manual"
IMAGES = OUT / "images"

# Occupancy of the ear in the existing TRAINING crops (n=4,142): the mean is what
# inference.py's ROI refinement targets, the sd is what the model must tolerate.
TARGET_OCCUPANCY = 0.777
OCCUPANCY_SD = 0.097
OCCUPANCY_CLIP = (0.55, 0.95)   # beyond the existing data's 5-95% either way
NUM_LANDMARKS = 55
PHASH_SIZE = 32          # DCT input side; low 8x8 block is kept
# Hamming distance at or below which two images group. Union-find is transitive,
# so a threshold that is too loose chains unrelated images into one giant group:
# measured on this set, 8 produced a 735-member group in which only 0.2% of
# pairs were actually within 8 bits. 4 keeps the largest group at 6 and involves
# 2.3% of samples. Over-merging only costs split balance; under-merging leaks a
# near-duplicate across the test boundary, so err loose rather than tight.
PHASH_THRESHOLD = 4


def phash(img: Image.Image) -> np.uint64:
    """64-bit DCT perceptual hash."""
    g = np.asarray(img.convert("L").resize((PHASH_SIZE, PHASH_SIZE),
                                           Image.BILINEAR), dtype=np.float64)
    d = dct(dct(g, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8]
    flat = d.flatten()
    # Drop the DC term from the median so a uniform brightness shift cannot
    # flip every bit at once.
    bits = flat > np.median(flat[1:])
    return np.uint64(int("".join("1" if b else "0" for b in bits), 2))


def group_by_phash(hashes: np.ndarray, threshold: int) -> np.ndarray:
    """Union-find grouping of near-duplicate hashes. Returns a group id per item.

    O(n^2) in bit-count comparisons, vectorised one row at a time: 10.5k items
    is ~55M 64-bit popcounts, a few seconds.
    """
    n = len(hashes)
    parent = np.arange(n)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    # popcount via a 16-bit lookup table applied to the four halfwords.
    lut = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8)

    def hamming_row(i: int) -> np.ndarray:
        x = np.bitwise_xor(hashes[i], hashes)
        return (lut[(x & 0xFFFF).astype(np.uint16)]
                + lut[((x >> np.uint64(16)) & 0xFFFF).astype(np.uint16)]
                + lut[((x >> np.uint64(32)) & 0xFFFF).astype(np.uint16)]
                + lut[((x >> np.uint64(48)) & 0xFFFF).astype(np.uint16)])

    for i in range(n):
        close = np.flatnonzero(hamming_row(i)[i + 1:] <= threshold) + i + 1
        ri = find(i)
        for j in close:
            rj = find(int(j))
            if rj != ri:
                parent[rj] = ri
                ri = find(i)
    return np.array([find(i) for i in range(n)])


def crop_to_occupancy(img: Image.Image, kp: np.ndarray, occ: float):
    """Square crop centred on the ear so it fills `occ` of the frame.

    Returns (crop, landmarks normalised to the crop) or None if it cannot fit.
    When the window runs past the source image -- which it does for any target
    below the native 0.707 -- PIL's crop grey-128 pads it rather than shrinking
    the window, so the requested occupancy is delivered exactly.
    """
    x0, x1 = kp[:, 0].min(), kp[:, 0].max()
    y0, y1 = kp[:, 1].min(), kp[:, 1].max()
    ear = max(x1 - x0, y1 - y0)
    side = ear / occ
    if side < 32:
        return None
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    li, ti, si = int(round(cx - side / 2)), int(round(cy - side / 2)), int(round(side))
    # Pad explicitly: Image.crop fills out-of-bounds with 0 (black), and the
    # pipeline's out-of-frame colour is grey 128 everywhere else.
    canvas = Image.new("RGB", (si, si), (128, 128, 128))
    sx0, sy0 = max(0, li), max(0, ti)
    sx1, sy1 = min(img.width, li + si), min(img.height, ti + si)
    if sx1 <= sx0 or sy1 <= sy0:
        return None
    canvas.paste(img.crop((sx0, sy0, sx1, sy1)), (sx0 - li, sy0 - ti))
    lm = np.stack([(kp[:, 0] - li) / si, (kp[:, 1] - ti) / si], axis=1)
    return canvas, lm


def read_annotation(path: Path) -> np.ndarray | None:
    """Parse an `_annotation.txt` into (55, 2) float pixel coords, or None."""
    try:
        kp = np.loadtxt(path, dtype=np.float64)
    except (ValueError, OSError):
        return None
    if kp.ndim != 2 or kp.shape != (NUM_LANDMARKS, 2) or not np.isfinite(kp).all():
        return None
    return kp


def main() -> None:
    p = argparse.ArgumentParser(description="Ingest commissioned manual annotations")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--occupancy", type=float, default=TARGET_OCCUPANCY,
                   help="Mean target occupancy")
    p.add_argument("--occupancy-sd", type=float, default=OCCUPANCY_SD,
                   help="Per-image spread of target occupancy. 0 reproduces the "
                        "original constant-scale behaviour, which cost 99%% "
                        "coverage of the existing test set's scale range.")
    p.add_argument("--phash-threshold", type=int, default=PHASH_THRESHOLD,
                   help="Hamming distance for near-duplicate grouping (0 = exact only)")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--quality", type=int, default=95)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not SRC.is_dir():
        sys.exit(f"source not found: {SRC}")
    pairs = []
    for ann in sorted(SRC.glob("*_annotation.txt")):
        img = ann.with_name(ann.name.replace("_annotation.txt", ".png"))
        if img.exists():
            pairs.append((img, ann))
    print(f"found {len(pairs)} image/annotation pairs")
    if args.limit:
        rng = np.random.default_rng(args.seed)
        keep = sorted(rng.choice(len(pairs), min(args.limit, len(pairs)), replace=False))
        pairs = [pairs[i] for i in keep]
        print(f"limited to {len(pairs)}")

    IMAGES.mkdir(parents=True, exist_ok=True)
    rng_occ = np.random.default_rng(args.seed)

    # ---- pass 1: read, validate, exact-dedup, crop, hash -------------------
    kept, hashes = [], []
    seen_md5: set[str] = set()
    bad_ann = dup = uncroppable = 0
    for n, (img_path, ann_path) in enumerate(pairs):
        kp = read_annotation(ann_path)
        if kp is None:
            bad_ann += 1
            continue
        raw = img_path.read_bytes()
        h = hashlib.md5(raw).hexdigest()
        if h in seen_md5:
            dup += 1
            continue
        seen_md5.add(h)
        img = Image.open(img_path).convert("RGB")
        target = args.occupancy
        if args.occupancy_sd > 0:
            target = float(np.clip(rng_occ.normal(args.occupancy, args.occupancy_sd),
                                   *OCCUPANCY_CLIP))
        got = crop_to_occupancy(img, kp, target)
        if got is None:
            uncroppable += 1
            continue
        # Separate CANONICAL crop, at the fixed mean occupancy, used only for
        # near-duplicate hashing. The source images are themselves variably
        # framed, so hashing them misses duplicates; hashing the training crop
        # misses them too now that its scale is randomised. Normalising scale
        # first is what makes two crops of one ear compare equal.
        canon = crop_to_occupancy(img, kp, args.occupancy)
        crop, lm = got
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=args.quality)
        jpg = buf.getvalue()
        kept.append((img_path.name, jpg, crop.size, lm.astype(np.float32)))
        hash_src = canon[0] if canon is not None else crop
        hashes.append(phash(hash_src))
        if (n + 1) % 1000 == 0:
            print(f"  read {n+1}/{len(pairs)}  kept {len(kept)}")
    if not kept:
        sys.exit("nothing ingested")
    print(f"\nkept {len(kept)}  (dropped: {bad_ann} malformed, {dup} exact dup, "
          f"{uncroppable} uncroppable)")

    # ---- pass 2: group near-duplicates, assign splits by group -------------
    hashes = np.array(hashes, dtype=np.uint64)
    print(f"grouping by pHash (threshold {args.phash_threshold})...")
    groups = group_by_phash(hashes, args.phash_threshold)
    uniq = np.unique(groups)
    sizes = np.bincount(np.searchsorted(uniq, groups))
    print(f"  {len(uniq)} groups for {len(kept)} samples "
          f"({(sizes > 1).sum()} groups hold >1, largest {sizes.max()})")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(uniq))
    n_val = int(round(len(kept) * args.val_frac))
    n_test = int(round(len(kept) * args.test_frac))
    split_of_group: dict[int, str] = {}
    cv = ct = 0
    for gi in order:
        g = int(uniq[gi])
        sz = int(sizes[gi])
        if ct < n_test:
            split_of_group[g] = "test"; ct += sz
        elif cv < n_val:
            split_of_group[g] = "val"; cv += sz
        else:
            split_of_group[g] = "train"

    # ---- write -------------------------------------------------------------
    rows, lms, splits = [], [], []
    for i, (name, jpg, size, lm) in enumerate(kept):
        out_name = Path(name).with_suffix(".jpg").name
        (IMAGES / out_name).write_bytes(jpg)
        rows.append({"idx": i, "image_file": f"images/{out_name}",
                     "source": "manual", "original_path": f"new_manual_data/{name}",
                     "width": size[0], "height": size[1]})
        lms.append(lm)
        splits.append(split_of_group[int(groups[i])])

    arr = np.stack(lms)
    np.save(OUT / "landmarks.npy", arr)
    fields = list(rows[0].keys())
    for fname, want in (("manifest.csv", None), ("train.csv", "train"),
                        ("val.csv", "val"), ("test.csv", "test")):
        sel = rows if want is None else [r for r, s in zip(rows, splits) if s == want]
        with open(OUT / fname, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(sel)
        print(f"  {fname:14s} {len(sel):>6d}")

    span = np.maximum(arr[:, :, 0].max(1) - arr[:, :, 0].min(1),
                      arr[:, :, 1].max(1) - arr[:, :, 1].min(1))
    inb = ((arr >= 0) & (arr <= 1)).all(axis=(1, 2))
    print(f"\nlandmarks.npy {arr.shape}")
    print(f"occupancy achieved: mean {span.mean():.3f} sd {span.std():.3f} "
          f"p5 {np.percentile(span,5):.3f} p95 {np.percentile(span,95):.3f}  "
          f"(existing training data: mean 0.778 sd 0.097)")
    print(f"samples fully inside the crop: {inb.mean()*100:.1f}%")
    print(f"\nwrote {OUT}")
    print(f"train with:  python train.py --data-dir data/manual --run-name manual_only")


if __name__ == "__main__":
    main()
