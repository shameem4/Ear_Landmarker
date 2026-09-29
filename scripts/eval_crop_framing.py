"""Find the crop framing the landmarker actually performs best at, and test
whether that optimum is stable across conditions.

Motivation: on four real capture frames the ear filled 0.886 of the ROI crop at
the current ROI_EXPAND=1.3, against a training distribution of 0.777. That
suggested the crop is too tight -- but it was measured from PREDICTED landmarks
on one person at one distance, which is both circular and a sample of four.

This measures the same thing against GROUND TRUTH on the held-out test set, and
stratifies it, so we can tell an artifact of one view from a real property.

Method: for each test image we know the true ear extent from its landmarks. We
build a square crop centred on the ear whose side is ear_size / f, so the ear
occupies exactly fraction `f` of the crop, run the model, map back, and measure
error against ground truth. Sweeping f gives the model's preferred framing.

Only crops that fit ENTIRELY inside the source image are used -- no synthetic
padding, since gray or reflected borders are not what a real ROI contains and
would bias the low-f end. The count of usable samples is reported per f, because
it falls as f shrinks.

Conditions tested, to answer "is one value enough for everything":
  - overall
  - by simulated yaw (0 / 25 / 45 deg), since foreshortening changes ear aspect
  - by source dataset
  - by ear aspect ratio (tall vs square ears)

Usage:
    python scripts/eval_crop_framing.py
    python scripts/eval_crop_framing.py --run v6_persp65 --limit 400
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_jitter import IMG_SIZE, best_checkpoint, load_checkpoint  # noqa: E402
from model.measure import LINESTRIPS, measure_ear, resample_polyline  # noqa: E402

DATA = ROOT / "data" / "preprocessed"


def yaw_warp(deg: float, focal: float = 2.0):
    """Homography simulating out-of-plane turn, in normalized coords."""
    th = np.radians(deg)

    def fwd(p: np.ndarray) -> np.ndarray:
        x = (p[:, 0] - 0.5) * 2
        y = (p[:, 1] - 0.5) * 2
        d = 1.0 + (x * np.sin(th)) / focal
        return np.stack([(x * np.cos(th)) / d / 2 + 0.5, y / d / 2 + 0.5], 1)

    return fwd


def crop_at_occupancy(img: np.ndarray, gt: np.ndarray, f: float):
    """Square crop where the ear occupies fraction `f`, or None if it would
    need padding.

    Returns (crop_uint8, gt_in_crop_normalized).
    """
    h, w = img.shape[:2]
    gx, gy = gt[:, 0] * w, gt[:, 1] * h
    cx, cy = (gx.min() + gx.max()) / 2, (gy.min() + gy.max()) / 2
    ear = max(gx.max() - gx.min(), gy.max() - gy.min())
    side = ear / f
    x0, y0 = cx - side / 2, cy - side / 2
    if x0 < 0 or y0 < 0 or x0 + side > w or y0 + side > h:
        return None                      # would need synthetic padding
    x0i, y0i, si = int(round(x0)), int(round(y0)), int(round(side))
    if si < 24 or x0i + si > w or y0i + si > h:
        return None
    crop = img[y0i:y0i + si, x0i:x0i + si]
    gt_crop = np.stack([(gx - x0i) / si, (gy - y0i) / si], 1)
    return crop, gt_crop


def contour_error(pred: np.ndarray, gt: np.ndarray) -> float:
    """Index-free point-to-contour error, in px at 192."""
    errs = []
    for _, (a, b) in LINESTRIPS.items():
        dense = resample_polyline(gt[a:b], 120)
        d = np.linalg.norm(pred[a:b][:, None, :] - dense[None, :, :], axis=-1)
        errs.append(d.min(axis=1).mean())
    return float(np.mean(errs)) * IMG_SIZE


def measurement_error(pred: np.ndarray, gt: np.ndarray) -> float:
    mg, mp = measure_ear(gt), measure_ear(pred)
    rel = [abs(mp[k] - mg[k]) / mg[k] for k in mg if mg[k] > 1e-6]
    return float(np.mean(rel)) * 100


@torch.no_grad()
def run_batch(model, crops, device):
    t = torch.stack([
        torch.from_numpy(
            cv2.resize(c, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            .astype(np.float32) / 255.0
        ).permute(2, 0, 1) for c in crops
    ])
    out = model(((t.to(device)) - 0.5) / 0.5)
    return out.reshape(len(crops), 55, 2).cpu().numpy().astype(np.float64)


def main() -> None:
    p = argparse.ArgumentParser(description="Validate ROI crop framing")
    p.add_argument("--run", default="v6_persp65")
    p.add_argument("--limit", type=int, default=400)
    p.add_argument("--yaws", type=float, nargs="*", default=[0.0, 25.0, 45.0])
    p.add_argument("--occupancies", type=float, nargs="*",
                   default=[0.62, 0.70, 0.78, 0.86, 0.94])
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, arch = load_checkpoint(best_checkpoint(ROOT / "runs" / "checkpoints" / args.run))
    model.to(device)

    rows = list(csv.DictReader(open(DATA / "test.csv", encoding="utf-8")))[:args.limit]
    lms = np.load(DATA / "landmarks.npy", mmap_mode="r")

    occupancies = args.occupancies
    print(f"run={args.run} arch={arch}  test images={len(rows)}  device={device}")
    print("ear occupies fraction f of the crop; only crops needing NO padding are used\n")

    # (yaw, f) -> lists; plus stratification keys
    agg = defaultdict(lambda: {"contour": [], "meas": [], "n": 0})
    strat = defaultdict(lambda: defaultdict(lambda: {"contour": [], "n": 0}))

    for yaw in args.yaws:
        fwd = yaw_warp(yaw) if yaw else None
        images, gts, metas = [], [], []
        for r in rows:
            im = np.asarray(Image.open(DATA / r["image_file"]).convert("RGB"))
            gt = np.asarray(lms[int(r["idx"])], dtype=np.float64)
            if fwd is not None:
                h, w = im.shape[:2]
                src = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
                dst = fwd(src.astype(np.float64).copy())
                m = cv2.getPerspectiveTransform(
                    (src * [w, h]).astype(np.float32), (dst * [w, h]).astype(np.float32))
                im = cv2.warpPerspective(im, m, (w, h), flags=cv2.INTER_LINEAR,
                                         borderValue=(128, 128, 128))
                gt = fwd(gt)
            images.append(im)
            gts.append(gt)
            gx, gy = gt[:, 0], gt[:, 1]
            aspect = (gx.max() - gx.min()) / max(gy.max() - gy.min(), 1e-6)
            metas.append((r["source"], aspect))

        # PAIRED comparison. A sample only enters if it supports EVERY occupancy
        # in the sweep. Without this the low-f rows contain only images with
        # spare margin -- overwhelmingly the synthetic audioear2d set, whose ears
        # fill 0.66 of frame -- while the high-f rows also include the tight
        # collectionB crops. That makes lower f look better purely because it is
        # scored on an easier subset.
        usable = [
            i for i in range(len(images))
            if all(crop_at_occupancy(images[i], gts[i], f) is not None
                   for f in occupancies)
        ]
        if yaw == args.yaws[0]:
            src_mix = defaultdict(int)
            for i in usable:
                src_mix[metas[i][0]] += 1
            print(f"paired sample: {len(usable)}/{len(images)} images support all "
                  f"{len(occupancies)} crop levels  {dict(src_mix)}\n")

        for f in occupancies:
            crops, keep_gt, keep_meta = [], [], []
            for i in usable:
                got = crop_at_occupancy(images[i], gts[i], f)
                if got is None:
                    continue
                crops.append(got[0])
                keep_gt.append(got[1])
                keep_meta.append(metas[i])
            if not crops:
                continue
            preds = []
            for i in range(0, len(crops), 64):
                preds.append(run_batch(model, crops[i:i + 64], device))
            preds = np.concatenate(preds, 0)

            for pr, gt, (src_name, aspect) in zip(preds, keep_gt, keep_meta):
                ce = contour_error(pr, gt)
                agg[(yaw, f)]["contour"].append(ce)
                agg[(yaw, f)]["meas"].append(measurement_error(pr, gt))
                agg[(yaw, f)]["n"] += 1
                if yaw == 0.0:
                    strat[("source", src_name)][f]["contour"].append(ce)
                    strat[("source", src_name)][f]["n"] += 1
                    bucket = "tall(<0.8)" if aspect < 0.8 else "square(>=0.8)"
                    strat[("aspect", bucket)][f]["contour"].append(ce)
                    strat[("aspect", bucket)][f]["n"] += 1

    def best_f(d):
        vals = [(f, np.mean(v["contour"])) for (f, v) in d.items() if v["n"] >= 10]
        return min(vals, key=lambda t: t[1]) if vals else (None, None)

    print("CONTOUR ERROR px@192 (and usable n) by occupancy f\n")
    hdr = f"{'yaw':>5s} " + " ".join(f"{f:>13.2f}" for f in occupancies) + "   best f"
    print(hdr); print("-" * len(hdr))
    for yaw in args.yaws:
        line = f"{yaw:4.0f}° "
        d = {f: agg[(yaw, f)] for f in occupancies if agg[(yaw, f)]["n"]}
        for f in occupancies:
            v = agg[(yaw, f)]
            line += f" {np.mean(v['contour']):7.3f}({v['n']:4d})" if v["n"] else f" {'--':>13s}"
        bf, bv = best_f(d)
        line += f"   {bf if bf else '--'}"
        print(line)

    print("\nMEASUREMENT ERROR % by occupancy f\n")
    print(hdr); print("-" * len(hdr))
    for yaw in args.yaws:
        line = f"{yaw:4.0f}° "
        for f in occupancies:
            v = agg[(yaw, f)]
            line += f" {np.mean(v['meas']):7.2f}({v['n']:4d})" if v["n"] else f" {'--':>13s}"
        print(line)

    print("\nSTRATIFIED (yaw 0 only) -- best f per subgroup, contour error")
    print(f"  {'group':22s} {'best f':>7s} {'err':>8s}   per-f errors")
    for (kind, name), d in sorted(strat.items()):
        bf, bv = best_f(d)
        per = " ".join(
            f"{f}:{np.mean(d[f]["contour"]):.2f}" if d[f]["n"] >= 10 else f"{f}:--"
            for f in occupancies if f in d)
        label = f"{kind}={name}"
        print(f"  {label:22s} {str(bf):>7s} {bv if bv else float('nan'):8.3f}   {per}")

    print("\nNOTE: low f needs a wider crop than some source images contain, so n")
    print("falls at the left of each row. Compare only where n is adequate.")


if __name__ == "__main__":
    main()
