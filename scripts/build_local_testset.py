"""Build a real full-frame ear test set from the BlazeEar training data already
on disk, and use its ground-truth boxes to settle ROI_EXPAND.

Why this replaced downloading from the internet: BlazeEar's own data directory
holds 7,157 full-scene images with ear bounding boxes (Open Images derived,
"Ear Detection from Full Face image"), plus ~4,300 more across five other ear
datasets. That is real, diverse, already local, and it carries ground truth --
all things the Wikimedia download was not. The first download attempt yielded
14 detections from 56 images of which only four were usable modern ears; the
rest were marble busts and false positives on painted faces.

What this gives that the preprocessed test set cannot:
  - full frames, so crop framing can actually be varied
  - ground-truth ear BOXES, so the detector can be scored
  - thousands of subjects, poses, distances and lighting conditions

THE ROI_EXPAND CALCULATION
The landmarker was trained with the ear filling 0.777 of its crop (measured over
5,870 training samples). At inference the crop side is
    roi_side = max(detector_box_w, detector_box_h) * ROI_EXPAND
so the true ear ends up occupying
    occupancy = true_ear_extent / roi_side
Rearranged, the expansion that puts the ear at its training occupancy is
    ROI_EXPAND = (true_ear_extent / detector_box_extent) / 0.777
The only unknown is the ratio of true ear extent to detector box extent, which
this script measures against ground truth instead of guessing it.

CAVEAT: the annotations are incomplete -- in multi-person scenes only some ears
are boxed. So an unmatched detection is NOT necessarily a false positive, and no
false-positive rate is reported here.

Usage:
    python scripts/build_local_testset.py --limit 600
    python scripts/build_local_testset.py --limit 600 --save-crops
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BLAZE = Path(os.environ.get("BLAZEEAR_DIR", ROOT.parent / "BlazeEar"))
YOLO = BLAZE / "data" / "yolo11_ears"
OUT = ROOT / "data" / "local_testset"

# Training-set occupancy of the landmarker, measured over all 5,870 samples.
TRAIN_OCCUPANCY = 0.777


def find_sources() -> list[tuple[Path, Path]]:
    """(images_dir, labels_dir) pairs across every bundled ear dataset."""
    pairs = []
    for split in ("train", "val"):
        img_root, lab_root = YOLO / "images" / split, YOLO / "labels" / split
        if not img_root.is_dir():
            continue
        for ds in sorted(img_root.iterdir()):
            if not ds.is_dir():
                continue
            lab_ds = lab_root / ds.name
            for sub in ("train", "valid", "test", ""):
                i, l = (ds / sub), (lab_ds / sub)
                if i.is_dir() and l.is_dir() and any(i.glob("*.jpg")):
                    pairs.append((i, l))
    return pairs


def load_boxes(label: Path, w: int, h: int) -> np.ndarray:
    """YOLO label file -> (N,4) xmin,ymin,xmax,ymax in pixels."""
    if not label.exists():
        return np.zeros((0, 4))
    out = []
    for line in label.read_text().split("\n"):
        p = line.split()
        if len(p) != 5:
            continue
        _, cx, cy, bw, bh = (float(v) for v in p)
        out.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                    (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.asarray(out) if out else np.zeros((0, 4))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description="Local full-frame ear test set")
    p.add_argument("--limit", type=int, default=600)
    p.add_argument("--save-crops", action="store_true")
    p.add_argument("--match-iou", type=float, default=0.3)
    args = p.parse_args()

    sys.path.insert(0, str(BLAZE))
    from blazeear import BlazeEar                      # type: ignore
    from utils.anchor_utils import anchor_options      # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BlazeEar()
    ck = torch.load(str(BLAZE / "runs/checkpoints/BlazeEar_best.pth"),
                    map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("model_state_dict", ck), strict=True)
    model.to(device).eval()
    model.generate_anchors(anchor_options)

    sources = find_sources()
    print(f"found {len(sources)} image/label directories")
    files = []
    for img_dir, lab_dir in sources:
        for f in sorted(img_dir.glob("*.jpg")):
            files.append((f, lab_dir / (f.stem + ".txt")))
    rng = np.random.default_rng(0)
    if len(files) > args.limit:
        files = [files[i] for i in rng.choice(len(files), args.limit, replace=False)]
    print(f"evaluating {len(files)} images\n")

    if args.save_crops:
        (OUT / "ear_crops").mkdir(parents=True, exist_ok=True)

    ratios, ious, confs = [], [], []
    n_img = n_gt = n_det = n_matched = n_multi_per_gt = 0
    records = []

    for path, lab in files:
        try:
            rgb = np.asarray(Image.open(path).convert("RGB"))
        except Exception:
            continue
        h, w = rgb.shape[:2]
        gt = load_boxes(lab, w, h)
        with torch.no_grad():
            dets = model.process(rgb)
        if isinstance(dets, torch.Tensor):
            dets = dets.cpu().numpy()
        dets = np.atleast_2d(np.asarray(dets)) if np.size(dets) else np.zeros((0, 17))

        n_img += 1
        n_gt += len(gt)
        n_det += len(dets)

        # BlazeEar returns (N,17): box 0-3 as ymin,xmin,ymax,xmax; conf last.
        dboxes = [(d[1], d[0], d[3], d[2]) for d in dets] if len(dets) else []
        dconf = [float(d[-1]) for d in dets] if len(dets) else []

        for g in gt:
            best, best_iou = None, args.match_iou
            n_over = 0
            for db in dboxes:
                v = iou(np.asarray(db), g)
                if v >= args.match_iou:
                    n_over += 1
                if v > best_iou:
                    best, best_iou = db, v
            if n_over > 1:
                n_multi_per_gt += 1
            if best is None:
                continue
            n_matched += 1
            gt_extent = max(g[2] - g[0], g[3] - g[1])
            det_extent = max(best[2] - best[0], best[3] - best[1])
            if det_extent > 1:
                ratios.append(gt_extent / det_extent)
            ious.append(best_iou)

        confs.extend(dconf)
        if args.save_crops and len(dboxes):
            for k, db in enumerate(dboxes):
                side = max(db[2] - db[0], db[3] - db[1]) * 1.3
                cx, cy = (db[0] + db[2]) / 2, (db[1] + db[3]) / 2
                x1, y1 = max(0, int(cx - side / 2)), max(0, int(cy - side / 2))
                x2, y2 = min(w, int(cx + side / 2)), min(h, int(cy + side / 2))
                if x2 - x1 > 24 and y2 - y1 > 24:
                    Image.fromarray(rgb[y1:y2, x1:x2]).save(
                        OUT / "ear_crops" / f"{path.stem}_e{k}.png")
        records.append({"file": str(path), "n_gt": len(gt), "n_det": len(dets)})

    ratios = np.asarray(ratios)
    print(f"images              {n_img}")
    print(f"ground-truth ears   {n_gt}")
    print(f"detections          {n_det}")
    print(f"matched (IoU>={args.match_iou})  {n_matched}  "
          f"({n_matched / max(n_gt,1)*100:.1f}% recall of annotated ears)")
    print(f"GT ears with >1 overlapping detection: {n_multi_per_gt} "
          f"({n_multi_per_gt / max(n_gt,1)*100:.1f}%)   <- duplicate rate")
    if len(confs):
        c = np.asarray(confs)
        print(f"detection confidence: mean {c.mean():.3f} "
              f"p5 {np.percentile(c,5):.3f} p95 {np.percentile(c,95):.3f}")

    if len(ratios) < 20:
        print("\nnot enough matches to estimate ROI_EXPAND")
        return

    print(f"\ntrue-ear-extent / detector-box-extent   (n={len(ratios)})")
    print(f"  mean {ratios.mean():.3f}  median {np.median(ratios):.3f}  "
          f"p25 {np.percentile(ratios,25):.3f}  p75 {np.percentile(ratios,75):.3f}")

    print(f"\nROI_EXPAND that lands the true ear at the landmarker's training")
    print(f"occupancy of {TRAIN_OCCUPANCY}:")
    for label, r in (("median", np.median(ratios)), ("mean", ratios.mean()),
                     ("p75 (cover larger ears)", np.percentile(ratios, 75))):
        print(f"  using {label:24s} ratio {r:.3f}  ->  ROI_EXPAND {r / TRAIN_OCCUPANCY:.2f}")
    print(f"\ncurrent ROI_EXPAND is 1.30, which implies the ear occupies "
          f"{np.median(ratios)/1.30:.3f} of the crop")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps({
        "images": n_img, "gt_ears": n_gt, "detections": n_det,
        "matched": n_matched, "duplicate_gt": n_multi_per_gt,
        "ratio_median": float(np.median(ratios)),
        "ratio_mean": float(ratios.mean()),
        "implied_roi_expand_median": float(np.median(ratios) / TRAIN_OCCUPANCY),
    }, indent=1))
    print(f"\nsummary -> {OUT/'summary.json'}")


if __name__ == "__main__":
    main()
