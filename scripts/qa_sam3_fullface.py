"""Does SAM 3 behave differently on a whole face than on an ear crop?

Every 2D QA number in this project comes from a tight crop, and this
corpus has no full-face view to check that against (see RESULTS.md). So
this brings in a handful of public-domain faces from Wikimedia Commons
and runs the same ear through SAM 3 twice: once in the full photo, once
cropped the way the corpus is cropped.

Pipeline, deliberately reusing what is already validated:
  Gemini box (median IoU 0.909 against annotated extent) locates the ear
  -> square ROI crop, as inference.py does
  -> the shipped landmarker predicts the 55 points
  -> SAM 3 on the FULL image and on the CROP, scored against the SAME
     points mapped into each frame.

The points are MODEL OUTPUT, not ground truth, and the landmarker was
itself trained on crops -- so its own lobe placement may be biased. That
bias cancels here: both arms are scored against the same points, so any
difference in the numbers belongs to the mask, which is the whole
question. The absolute level means nothing; only full-vs-crop does.

    python scripts/qa_sam3_fullface.py IMAGE_DIR BOXES.json runs/fullface
"""
import json
import pathlib
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
from inference import LandmarkPredictor, find_best_checkpoint, square_roi_crop

IMGDIR = pathlib.Path(sys.argv[1])
BOXES = json.load(open(sys.argv[2]))
OUT = pathlib.Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
ROI_EXPAND = 1.3

dev = "cuda" if torch.cuda.is_available() else "cpu"
proc = Sam3Processor.from_pretrained("facebook/sam3")
sam = Sam3Model.from_pretrained("facebook/sam3", dtype=torch.float16).to(dev).eval()
ck = find_best_checkpoint()
print(f"landmarker: {ck.name}")
lmk = LandmarkPredictor(ck, device=dev)


def score(im_rgb, xy):
    """Signed distance from each point to the SAM mask edge, % of ear height."""
    h, w = im_rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    try:
        hull = xy[ConvexHull(xy).vertices]
    except Exception:
        return None
    inhull = MPath(hull).contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(h, w)
    inp = proc(images=Image.fromarray(im_rgb), text="ear",
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
    mk = lab == (1 + np.argmax(sz))
    sd = (ndimage.distance_transform_edt(~mk)
          - ndimage.distance_transform_edt(mk))
    eh = float(xy[:, 1].max() - xy[:, 1].min())
    px = np.clip(np.round(xy[:, 0]).astype(int), 0, w - 1)
    py = np.clip(np.round(xy[:, 1]).astype(int), 0, h - 1)
    return (sd[py, px] / eh * 100.0)[:20], mk


res = {"full image": [], "ear crop": []}
keep = {}
for k in sorted(BOXES):
    p = IMGDIR / f"{k}.jpg"
    if not p.exists():
        continue
    full = np.asarray(Image.open(p).convert("RGB"))
    H, W = full.shape[:2]
    x0, y0, x1, y1 = BOXES[k]
    cx, cy = (x0 + x1) / 2 * W, (y0 + y1) / 2 * H
    side = max((x1 - x0) * W, (y1 - y0) * H) * ROI_EXPAND
    roi = square_roi_crop(full, cx, cy, side)
    if roi is None:
        print(f"  {k}: ROI unusable"); continue
    crop, ox, oy = roi                 # native resolution, grey-128 padded
    lm_crop = lmk.predict(crop)        # px in the crop
    lm_full = lm_crop + np.array([ox, oy])

    a = score(full, lm_full)
    b = score(crop, lm_crop)
    if a is None or b is None:
        print(f"  {k}: no mask"); continue
    res["full image"].append(a[0].tolist())
    res["ear crop"].append(b[0].tolist())
    keep[k] = dict(crop=crop, lm_crop=lm_crop, lm_full=lm_full,
                   mk_full=a[1], mk_crop=b[1])
    d = a[0][19] - a[0][12], b[0][19] - b[0][12]
    print(f"  {k}: 19-12 full {d[0]:+6.2f}   crop {d[1]:+6.2f}")

json.dump(res, open(OUT / "fullface.json", "w"))
print(f"\n{'arm':<14}{'n':>5}{'pt0':>8}{'pt12':>8}{'pt19':>8}{'19-12':>8}{'lobe':>8}")
for a, v in res.items():
    if not v:
        print(f"{a:<14}{0:>5}"); continue
    P = np.array(v)
    p0, p12, p19 = (np.median(P[:, j]) for j in (0, 12, 19))
    print(f"{a:<14}{len(v):>5}{p0:>8.2f}{p12:>8.2f}{p19:>8.2f}"
          f"{p19-p12:>8.2f}{np.median(P[:, 14:20]):>8.2f}")

import pickle
pickle.dump({k: {kk: vv for kk, vv in v.items()} for k, v in keep.items()},
            open(OUT / "arms.pkl", "wb"))
print(f"\nwrote {OUT/'arms.pkl'} for rendering -- LOOK at it before believing"
      " any of the above.")
