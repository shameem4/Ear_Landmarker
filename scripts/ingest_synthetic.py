"""Ingest the AudioEar synthetic renders as an OPTIONAL extra training source.

10,002 samples with exact 55-point labels (sythc_data-002.zip / sythetic_anno.npz),
where kp_3d[:, :, :2] are direct 2D coordinates in the 480x480 render and all
550,110 keypoints land inside the frame.

DESIGNED FOR FREE REVERSION. Nothing here touches data/preprocessed:
  - images, landmarks and manifest are written to data/synthetic/
  - the real train/val/test splits are untouched
  - training only uses it when --synthetic-ratio > 0 is passed
  - synthetic NEVER enters val or test, so the held-out metrics stay purely real
To revert: omit the flag, or delete data/synthetic/. That is the whole procedure.

WHY SYNTHETIC COULD REGRESS THINGS, and what is done about it:
  - domain gap: renders are not photographs. Mitigated by using
    render_background.jpg (ear composited onto hair/head) rather than the
    isolated render, and by keeping the real held-out sets as the only judge.
  - dominance: 10,002 synthetic against 4,142 real training samples would be 71%
    synthetic, and audioear2d (already synthetic) is a further 34% of the real
    side. --synthetic-ratio subsamples so synthetic does not outweigh real.
  - framing mismatch: synthetic ears fill 0.637 of their frame against 0.777 in
    the real training data. Each render is therefore cropped to put the ear at
    the real distribution's occupancy, so the model sees one consistent framing.

Usage:
    python scripts/ingest_synthetic.py                 # all 10,002
    python scripts/ingest_synthetic.py --limit 3000
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT.parent / "sythc_data-002.zip"
OUT = ROOT / "data" / "synthetic"
IMAGES = OUT / "images"

# Occupancy of the ear in the real training crops, measured over all 5,870
# samples. Synthetic renders are cropped to match so framing is consistent.
TARGET_OCCUPANCY = 0.777
NUM_LANDMARKS = 55


def crop_to_occupancy(img: Image.Image, kp: np.ndarray, occ: float):
    """Square crop centred on the ear so it fills `occ` of the frame.

    Returns (cropped_image, landmarks_normalised_to_crop) or None if the crop
    would fall outside the render.
    """
    w, h = img.size
    x0, x1 = kp[:, 0].min(), kp[:, 0].max()
    y0, y1 = kp[:, 1].min(), kp[:, 1].max()
    ear = max(x1 - x0, y1 - y0)
    side = ear / occ
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left, top = cx - side / 2, cy - side / 2

    # Shift the window back inside the frame rather than shrinking it, so the
    # occupancy stays on target; only give up if it cannot fit at all.
    left = float(np.clip(left, 0, max(0, w - side)))
    top = float(np.clip(top, 0, max(0, h - side)))
    if side > w or side > h:
        return None

    li, ti, si = int(round(left)), int(round(top)), int(round(side))
    si = min(si, w - li, h - ti)
    if si < 32:
        return None

    crop = img.crop((li, ti, li + si, ti + si))
    lm = np.stack([(kp[:, 0] - li) / si, (kp[:, 1] - ti) / si], axis=1)
    return crop, lm


def main() -> None:
    p = argparse.ArgumentParser(description="Ingest synthetic ear renders")
    p.add_argument("--limit", type=int, default=None,
                   help="Max samples to ingest (default: all)")
    p.add_argument("--occupancy", type=float, default=TARGET_OCCUPANCY)
    p.add_argument("--quality", type=int, default=92)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not ZIP.exists():
        sys.exit(f"archive not found: {ZIP}")

    IMAGES.mkdir(parents=True, exist_ok=True)
    zf = zipfile.ZipFile(ZIP)

    with zf.open("sythc_data/sythetic_anno.npz") as fh:
        anno = np.load(io.BytesIO(fh.read()), allow_pickle=True)
    parts = []
    for split in ("train", "test"):
        d = anno[split].item()
        if len(d["image_index"]):
            parts.append((list(d["image_index"]), np.asarray(d["kp_3d"])))
    ids = [i for p_ids, _ in parts for i in p_ids]
    kps = np.concatenate([k for _, k in parts], 0)
    print(f"annotation: {len(ids)} samples, {kps.shape[1]} keypoints")

    order = np.arange(len(ids))
    if args.limit and args.limit < len(ids):
        order = np.random.default_rng(args.seed).choice(
            len(ids), args.limit, replace=False)
        order.sort()
    print(f"ingesting {len(order)} samples at occupancy {args.occupancy}\n")

    rows, lms = [], []
    skipped_missing = skipped_crop = 0
    for n, i in enumerate(order):
        sid = ids[i]
        member = f"sythc_data/{sid}/render_background.jpg"
        try:
            with zf.open(member) as fh:
                img = Image.open(io.BytesIO(fh.read())).convert("RGB")
        except KeyError:
            skipped_missing += 1
            continue
        got = crop_to_occupancy(img, kps[i][:, :2].astype(np.float64), args.occupancy)
        if got is None:
            skipped_crop += 1
            continue
        crop, lm = got
        name = f"synth_{sid}.jpg"
        crop.save(IMAGES / name, quality=args.quality)
        rows.append({"idx": len(rows), "image_file": f"images/{name}",
                     "source": "sythc", "original_path": member,
                     "width": crop.width, "height": crop.height})
        lms.append(lm.astype(np.float32))
        if (n + 1) % 500 == 0:
            print(f"  {n+1}/{len(order)}  kept {len(rows)}")

    if not rows:
        sys.exit("nothing ingested")

    arr = np.stack(lms)
    np.save(OUT / "landmarks.npy", arr)
    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wri.writeheader()
        wri.writerows(rows)
    # train.csv here mirrors the real split format; synthetic is train-only by
    # construction, so no val/test file is written and none should be.
    with open(OUT / "train.csv", "w", newline="", encoding="utf-8") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wri.writeheader()
        wri.writerows(rows)

    span = np.maximum(arr[:, :, 0].max(1) - arr[:, :, 0].min(1),
                      arr[:, :, 1].max(1) - arr[:, :, 1].min(1))
    inb = ((arr >= 0) & (arr <= 1)).all(axis=(1, 2))
    print(f"\ningested {len(rows)}  (skipped: {skipped_missing} missing, "
          f"{skipped_crop} uncroppable)")
    print(f"occupancy achieved: mean {span.mean():.3f} "
          f"p5 {np.percentile(span,5):.3f} p95 {np.percentile(span,95):.3f}  "
          f"(target {args.occupancy}, real data 0.777)")
    print(f"samples with all landmarks inside the crop: {inb.mean()*100:.1f}%")
    print(f"\nwrote {OUT}")
    print("train with:  python train.py --synthetic-ratio 1.0   (0 = off, the default)")


if __name__ == "__main__":
    main()
