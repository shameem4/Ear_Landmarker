"""Full image vs ear crop, with the project's own detector doing the finding.

The first attempt at this (qa_sam3_fullface.py) is recorded in RESULTS.md as
NOT RUN: it located ears with a Gemini box, and two of seven crops contained
no ear while the landmarker ran off the cheek on archive photographs. Both
faults were in the finding, not in the question.

This version uses the shipped pipeline end to end -- BlazeEar two-stage
detector, then EarLandmarker -- so an ear that cannot be found is simply not
scored, instead of producing a crop of someone's jaw. Detector confidence is
the filter, replacing my eyeballing of which photos "look usable".

Each detected ear is then segmented twice, full frame and cropped, and both
are scored against the SAME landmarks mapped into each frame. The landmarks
are model output, so the ABSOLUTE level is meaningless here; only the
difference between the two arms is, because the points cancel.

    python scripts/qa_sam3_fullface2.py IMAGE_DIR runs/fullface2
"""
import pathlib
import pickle
import sys

import numpy as np
import torch
from matplotlib.path import Path as MPath
from PIL import Image
from scipy import ndimage
from scipy.spatial import ConvexHull
from transformers import Sam3Model, Sam3Processor

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from inference import (BLAZEEAR_DIR, DETECTOR_WEIGHTS, EarLandmarkerPipeline,
                       find_best_checkpoint, square_roi_crop)

IMGDIR = pathlib.Path(sys.argv[1])
OUT = pathlib.Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
ROI_EXPAND = 1.3

dev = "cuda" if torch.cuda.is_available() else "cpu"
proc = Sam3Processor.from_pretrained("facebook/sam3")
sam = Sam3Model.from_pretrained("facebook/sam3", dtype=torch.float16).to(dev).eval()
pipe = EarLandmarkerPipeline(BLAZEEAR_DIR / DETECTOR_WEIGHTS,
                             find_best_checkpoint(), device=dev, smooth=False)


def sam_mask(img_rgb, xy):
    h, w = img_rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    try:
        hull = xy[ConvexHull(xy).vertices]
    except Exception:
        return None
    inhull = MPath(hull).contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(h, w)
    inp = proc(images=Image.fromarray(img_rgb), text="ear",
               return_tensors="pt").to(dev)
    with torch.inference_mode():
        o = sam(**inp)
    try:
        r = proc.post_process_instance_segmentation(
            o, threshold=0.3, mask_threshold=0.5, target_sizes=[(h, w)])[0]
        masks = np.asarray(r["masks"].cpu())
    except Exception:
        return None
    best, mk = -1.0, None
    for m_ in masks:
        m_ = np.asarray(m_) > 0
        if m_.ndim != 2 or not m_.any():
            continue
        v = (m_ & inhull).sum() / max((m_ | inhull).sum(), 1)
        if v > best:
            best, mk = v, m_
    if mk is None:
        return None
    lab, n = ndimage.label(mk)
    sz = ndimage.sum(mk, lab, range(1, n + 1))
    return lab == (1 + np.argmax(sz))


def score(mk, xy, h, w):
    sd = (ndimage.distance_transform_edt(~mk)
          - ndimage.distance_transform_edt(mk))
    eh = float(xy[:, 1].max() - xy[:, 1].min())
    px = np.clip(np.round(xy[:, 0]).astype(int), 0, w - 1)
    py = np.clip(np.round(xy[:, 1]).astype(int), 0, h - 1)
    return (sd[py, px] / eh * 100.0)[:20]


res = {"full image": [], "ear crop": []}
keep = {}
for p in sorted(IMGDIR.glob("*.jpg")):
    full = np.asarray(Image.open(p).convert("RGB"))
    H, W = full.shape[:2]
    try:
        dets = pipe(full)
    except Exception as e:
        print(f"  {p.stem}: pipeline {type(e).__name__} {e}"); continue
    if not dets:
        print(f"  {p.stem}: no ear detected"); continue
    d = max(dets, key=lambda r: r["confidence"])
    lm_full = np.asarray(d["landmarks"], float)
    if not np.isfinite(lm_full).all():
        continue
    ex = lm_full[:20, 0].max() - lm_full[:20, 0].min()
    ey = lm_full[:20, 1].max() - lm_full[:20, 1].min()
    if ey < 24:
        print(f"  {p.stem}: ear too small ({ey:.0f}px)"); continue

    cx, cy = lm_full[:20, 0].mean(), lm_full[:20, 1].mean()
    roi = square_roi_crop(full, cx, cy, max(ex, ey) * ROI_EXPAND)
    if roi is None:
        print(f"  {p.stem}: ROI unusable"); continue
    crop, ox, oy = roi
    lm_crop = lm_full - np.array([ox, oy])

    a = sam_mask(full, lm_full)
    b = sam_mask(crop, lm_crop)
    if a is None or b is None:
        print(f"  {p.stem}: no mask"); continue
    sa = score(a, lm_full, H, W)
    sb = score(b, lm_crop, crop.shape[0], crop.shape[1])
    res["full image"].append(sa.tolist())
    res["ear crop"].append(sb.tolist())
    keep[p.stem] = dict(full=full, crop=crop, lm_full=lm_full, lm_crop=lm_crop,
                        mk_full=a, mk_crop=b, conf=d["confidence"])
    print(f"  {p.stem}: conf {d['confidence']:.2f}  ear {ey:.0f}px  "
          f"19-12 full {sa[19]-sa[12]:+6.2f}  crop {sb[19]-sb[12]:+6.2f}")

pickle.dump(keep, open(OUT / "arms.pkl", "wb"))
print(f"\n{'arm':<14}{'n':>5}{'pt0':>8}{'pt12':>8}{'pt19':>8}{'19-12':>8}{'lobe':>8}")
for a, v in res.items():
    if not v:
        print(f"{a:<14}{0:>5}"); continue
    P = np.array(v)
    p0, p12, p19 = (np.median(P[:, j]) for j in (0, 12, 19))
    print(f"{a:<14}{len(v):>5}{p0:>8.2f}{p12:>8.2f}{p19:>8.2f}"
          f"{p19-p12:>8.2f}{np.median(P[:, 14:20]):>8.2f}")
if res["full image"]:
    A = np.array(res["full image"]); B = np.array(res["ear crop"])
    d = (A[:, 19] - A[:, 12]) - (B[:, 19] - B[:, 12])
    print(f"\npaired full-minus-crop on 19-12: median {np.median(d):+.2f}"
          f"  (negative = the full image helps)")
print("RENDER arms.pkl before believing any of this.")
