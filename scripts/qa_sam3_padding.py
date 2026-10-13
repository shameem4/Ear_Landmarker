"""Is SAM 3's lobe cutoff caused by the ear abutting the crop edge?

Prompting and VLM boxes both failed to move the mask boundary. A cheaper
hypothesis remains: the crops are tight (the ear spans 77% of the frame at
the median, 90% at p90), so the lobe often runs into the frame edge, and a
segmenter has no reason to carry a region past the border of its input.

If that is the cause, padding the crop should cure it, and which KIND of
padding works tells us why:

  grey       clean separation, no invented texture -- but a hard edge
  replicate  stretches the border pixels, so the lobe appears to continue
  reflect    mirrors real texture back in, the most plausible continuation

Landmarks are mapped into the padded frame, so the metric is unchanged:
signed distance to the mask edge as % of ear height, positive = outside.
Point 19 (lobe) minus point 12 (mid-helix) is the number that matters.

    python scripts/qa_sam3_padding.py runs/sam3_pad ROWS.txt
"""
import csv
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
OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
pick = [int(x) for x in pathlib.Path(sys.argv[2]).read_text().split()]

# (name, fraction of the crop added on EVERY side, numpy pad mode)
ARMS = [("none", 0.0, None),
        ("grey 25%", 0.25, "grey"),
        ("replicate 25%", 0.25, "edge"),
        ("reflect 25%", 0.25, "reflect"),
        ("grey 50%", 0.50, "grey"),
        ("reflect 50%", 0.50, "reflect")]

dev = "cuda" if torch.cuda.is_available() else "cpu"
proc = Sam3Processor.from_pretrained("facebook/sam3")
model = Sam3Model.from_pretrained("facebook/sam3", dtype=torch.float16).to(dev).eval()
L = np.load(ROOT / "data/manual/landmarks.npy")
rows = list(csv.DictReader(open(ROOT / "data/manual/manifest.csv")))


def pad(a, p, mode):
    if p == 0:
        return a, 0, 0
    ph, pw = round(a.shape[0] * p), round(a.shape[1] * p)
    if mode == "grey":
        b = np.pad(a, ((ph, ph), (pw, pw), (0, 0)), constant_values=128)
    else:
        b = np.pad(a, ((ph, ph), (pw, pw), (0, 0)), mode=mode)
    return b, pw, ph


res = {a[0]: [] for a in ARMS}
for c, i in enumerate(pick):
    im0 = np.asarray(Image.open(ROOT / "data/manual" / rows[i]["image_file"])
                     .convert("RGB"))
    h0, w0 = im0.shape[:2]
    xy0 = L[i] * np.array([w0, h0])
    if not np.isfinite(xy0).all():
        continue
    eh = float(xy0[:, 1].max() - xy0[:, 1].min())
    if eh < 24:
        continue
    for name, p, mode in ARMS:
        arr, dx, dy = pad(im0, p, mode)
        im = Image.fromarray(arr)
        h, w = arr.shape[:2]
        xy = xy0 + np.array([dx, dy])
        yy, xx = np.mgrid[0:h, 0:w]
        try:
            hull = xy[ConvexHull(xy).vertices]
        except Exception:
            continue
        inhull = MPath(hull).contains_points(
            np.c_[xx.ravel(), yy.ravel()]).reshape(h, w)
        try:
            inp = proc(images=im, text="ear", return_tensors="pt").to(dev)
            with torch.inference_mode():
                o = model(**inp)
            r = proc.post_process_instance_segmentation(
                o, threshold=0.3, mask_threshold=0.5, target_sizes=[(h, w)])[0]
            masks = np.asarray(r["masks"].cpu())
        except Exception:
            continue
        best, mk = -1.0, None
        for m_ in masks:
            m_ = np.asarray(m_) > 0
            if m_.ndim != 2 or not m_.any():
                continue
            v = (m_ & inhull).sum() / max((m_ | inhull).sum(), 1)
            if v > best:
                best, mk = v, m_
        if mk is None:
            continue
        lab, n = ndimage.label(mk)
        if n == 0:
            continue
        sz = ndimage.sum(mk, lab, range(1, n + 1))
        if sz.max() / max(sz.sum(), 1.0) < 0.80:
            continue
        mk = lab == (1 + np.argmax(sz))
        sd = (ndimage.distance_transform_edt(~mk)
              - ndimage.distance_transform_edt(mk))
        px = np.clip(np.round(xy[:, 0]).astype(int), 0, w - 1)
        py = np.clip(np.round(xy[:, 1]).astype(int), 0, h - 1)
        res[name].append((sd[py, px] / eh * 100.0)[:20].tolist())
    if (c + 1) % 25 == 0:
        print(f"  {c+1}/{len(pick)}", flush=True)

json.dump(res, open(OUT / "padding.json", "w"))
print(f"\n{'padding':<16}{'n':>5}{'pt0':>8}{'pt12':>8}{'pt19':>8}{'19-12':>8}{'lobe':>8}")
for name, _, _ in ARMS:
    v = res[name]
    if not v:
        print(f"{name:<16}{0:>5}"); continue
    P = np.array(v)
    p0, p12, p19 = (np.median(P[:, j]) for j in (0, 12, 19))
    print(f"{name:<16}{len(v):>5}{p0:>8.2f}{p12:>8.2f}{p19:>8.2f}"
          f"{p19-p12:>8.2f}{np.median(P[:, 14:20]):>8.2f}")
print("\n(signed % of ear height, >0 = annotation OUTSIDE the mask;"
      " '19-12' near 0 = the lobe is in.)")
