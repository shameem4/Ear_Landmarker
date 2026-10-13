"""Is SAM 3 EXCLUDING hair, or just falling short at the rim?

Two different failures look alike in a coverage number and are not alike
at all:

  under-reach   the mask stops short of the ear. Its boundary is still
                ear-shaped; it is simply too small.
  hair carving  the mask takes a concave BITE out of an otherwise
                ear-shaped blob, following a strand across the ear. The
                shape is sacrificed to exclude a non-ear texture.

The second is the more interesting one, because the bite is an OCCLUSION
MAP that SAM has already computed. If annotated landmarks fall inside the
bite, SAM is telling us where the ear is hidden, for free.

Decomposed per row against the mask's own convex hull:

  solidity          mask area / hull area. An ear is a convex-ish blob, so
                    a low value means the mask is bitten into.
  pts in deficit    annotated points inside (hull - mask): carved out
  pts outside hull  annotated points beyond the hull entirely: under-reach

Deficit components are filtered by area, since a one-pixel rind along the
whole boundary is quantisation, not a bite.

    python scripts/qa_sam3_concavity.py runs/concavity ROWS.txt
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
from skimage.morphology import convex_hull_image
from transformers import Sam3Model, Sam3Processor

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
pick = [int(x) for x in pathlib.Path(sys.argv[2]).read_text().split()]
TAG = sys.argv[3] if len(sys.argv) > 3 else "rows"

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
        hull_a = xy[:20][ConvexHull(xy[:20]).vertices]
    except Exception:
        continue
    inann = MPath(hull_a).contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(h, w)
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

    hull_m = convex_hull_image(mk)
    deficit = hull_m & ~mk
    # a thin rind round the whole boundary is quantisation; a BITE is a
    # compact blob. Keep components of at least 1% of the mask's area.
    dl, dn = ndimage.label(deficit)
    keep = np.zeros_like(deficit)
    amin = 0.01 * mk.sum()
    for s in range(1, dn + 1):
        comp = dl == s
        if comp.sum() >= amin:
            keep |= comp
    px = np.clip(np.round(xy[:20, 0]).astype(int), 0, w - 1)
    py = np.clip(np.round(xy[:20, 1]).astype(int), 0, h - 1)
    in_def = keep[py, px]
    out_hull = ~hull_m[py, px]
    res.append(dict(idx=i,
                    solidity=float(mk.sum() / max(hull_m.sum(), 1)),
                    cover=float((mk & inann).sum() / max(inann.sum(), 1)),
                    bites=int(sum(1 for s in range(1, dn + 1)
                                  if (dl == s).sum() >= amin)),
                    pts_in_deficit=float(in_def.mean()),
                    pts_outside_hull=float(out_hull.mean())))
    if (c + 1) % 40 == 0:
        print(f"  {c+1}/{len(pick)}", flush=True)

json.dump(res, open(OUT / f"concavity_{TAG}.json", "w"))
A = {k: np.array([r[k] for r in res]) for k in res[0]}
print(f"\n{TAG}: {len(res)} rows")
print(f"  solidity (mask/hull)            {np.median(A['solidity']):.3f}")
print(f"  coverage of annotated ear       {100*np.median(A['cover']):.0f}%")
print(f"  rows with >=1 bite (>=1% area)  {100*(A['bites']>0).mean():.0f}%")
print(f"  outer points INSIDE a bite      {100*np.median(A['pts_in_deficit']):.0f}%"
      f"   mean {100*A['pts_in_deficit'].mean():.1f}%")
print(f"  outer points OUTSIDE the hull   {100*np.median(A['pts_outside_hull']):.0f}%"
      f"   mean {100*A['pts_outside_hull'].mean():.1f}%")
print("\n(inside a bite = SAM carved out ear the annotation includes;"
      "\n outside the hull = SAM simply did not reach that far.)")
