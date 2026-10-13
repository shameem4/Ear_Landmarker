"""Does SAM 3 CARVE INTO the ear, or just stop short of its rim?

A first attempt at this used mask-area/convex-hull-area and looked for
annotated points inside the hull deficit. Both were the wrong instrument
and returned a false negative:

  - an ear is not convex. The intertragic notch alone puts a perfectly
    good mask at ~0.93 solidity, which is what BOTH the disputed and the
    control sets scored. The metric had no resolution.
  - the 20 outer points sit ON the rim, and SAM under-reaches, so they are
    outside the mask already. They can never land "inside a bite". The
    bites are in the ear's INTERIOR, where that test had no points.

Rendering the masks showed obvious carving -- a hair strand driven as a
deep V into the middle of the ear -- that those numbers denied.

So measure the thing directly. The deficit is the annotated ear region
SAM did not mark. Every mask has a thin rind of it from under-reach; a
BITE reaches inward. Distance from the annotation's own rim separates
them, in units of ear height:

  rind      deficit within 10% of ear height of the annotated rim
  intrusion deficit deeper than that -- SAM excluding interior ear

    python scripts/qa_sam3_intrusion.py runs/intrusion ROWS.txt TAG
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
pick = [int(x) for x in pathlib.Path(sys.argv[2]).read_text().replace(",", " ").split()]
TAG = sys.argv[3] if len(sys.argv) > 3 else "rows"
DEEP = 0.10                      # intrusion threshold, fraction of ear height

dev = "cuda" if torch.cuda.is_available() else "cpu"
proc = Sam3Processor.from_pretrained("facebook/sam3")
sam = Sam3Model.from_pretrained("facebook/sam3", dtype=torch.float16).to(dev).eval()
L = np.load(ROOT / "data/manual/landmarks.npy")
rows = list(csv.DictReader(open(ROOT / "data/manual/manifest.csv")))

res = []
for c, i in enumerate(pick):
    g = L[i]
    if not np.isfinite(g).all():
        continue
    try:
        im = Image.open(ROOT / "data/manual" / rows[i]["image_file"]).convert("RGB")
    except Exception:
        continue
    w, h = im.size
    xy = g * np.array([w, h])
    eh = float(xy[:20, 1].max() - xy[:20, 1].min())
    if eh < 32:
        continue
    yy, xx = np.mgrid[0:h, 0:w]
    try:
        poly = xy[:20]                         # the annotated rim, in order
        inann = MPath(poly).contains_points(
            np.c_[xx.ravel(), yy.ravel()]).reshape(h, w)
    except Exception:
        continue
    if inann.sum() < 100:
        continue
    inp = proc(images=im, text="ear", return_tensors="pt").to(dev)
    with torch.inference_mode():
        o = sam(**inp)
    mk, best = None, -1.0
    try:
        r = proc.post_process_instance_segmentation(
            o, threshold=0.3, mask_threshold=0.5, target_sizes=[(h, w)])[0]
        for m_ in np.asarray(r["masks"].cpu()):
            m_ = np.asarray(m_) > 0
            if m_.ndim != 2 or not m_.any():
                continue
            v = (m_ & inann).sum() / max((m_ | inann).sum(), 1)
            if v > best:
                best, mk = v, m_
    except Exception:
        pass
    if mk is None:
        continue
    lab, n = ndimage.label(mk)
    if n == 0:
        continue
    mk = lab == (1 + np.argmax(ndimage.sum(mk, lab, range(1, n + 1))))

    # depth inside the ANNOTATED ear, so "deep" means far from its rim
    depth = ndimage.distance_transform_edt(inann) / eh
    deficit = inann & ~mk
    deep = deficit & (depth > DEEP)
    res.append(dict(idx=i,
                    cover=float((mk & inann).sum() / inann.sum()),
                    deficit_frac=float(deficit.sum() / inann.sum()),
                    intrusion_frac=float(deep.sum() / inann.sum()),
                    max_depth=float(depth[deficit].max()) if deficit.any() else 0.0,
                    deep_blobs=int(ndimage.label(
                        ndimage.binary_opening(deep, np.ones((3, 3))))[1])))
    if (c + 1) % 40 == 0:
        print(f"  {c+1}/{len(pick)}", flush=True)

json.dump(res, open(OUT / f"intrusion_{TAG}.json", "w"))
A = {k: np.array([r[k] for r in res]) for k in res[0]}
print(f"\n{TAG}: {len(res)} rows   (deep = >{DEEP:.0%} of ear height inside the rim)")
print(f"  coverage of annotated ear          {100*np.median(A['cover']):.0f}%")
print(f"  deficit, all of it                 {100*np.median(A['deficit_frac']):.1f}% of ear")
print(f"  of which DEEP (intrusion)          {100*np.median(A['intrusion_frac']):.1f}% of ear")
print(f"  deepest point of the deficit       {100*np.median(A['max_depth']):.1f}% of ear height")
print(f"  rows with a deep blob              {100*(A['deep_blobs']>0).mean():.0f}%")
print(f"  rows where intrusion > 5% of ear   {100*(A['intrusion_frac']>0.05).mean():.0f}%")
