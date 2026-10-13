"""Where SAM 3 stops at the lobe -- is there an edge there, or nothing?

This decides which repairs are even possible. If the mask boundary near
the lobe sits on a real intensity edge, then GrabCut, snakes, matting and
every other edge-driven method will stop in the same place, and only a
shape prior can carry the contour further. If it sits on nothing, the
boundary is arbitrary and growth methods have something to bite on.

Measured on CLEAN rows only (the flagged ones are excluded) so the
annotation can be trusted as the target.

For each row: take SAM 3's mask, walk its boundary, and compare the image
gradient at the part of the boundary nearest the annotated lobe against
the gradient along the rest of the boundary. Also report how far the lobe
points sit outside, so the two can be read together.

    python scripts/qa_sam3_cutoff_diag.py runs/cutoff ROWS.txt
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
from skimage.filters import sobel
from transformers import Sam3Model, Sam3Processor

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
pick = [int(x) for x in pathlib.Path(sys.argv[2]).read_text().split()]

dev = "cuda" if torch.cuda.is_available() else "cpu"
proc = Sam3Processor.from_pretrained("facebook/sam3")
sam = Sam3Model.from_pretrained("facebook/sam3", dtype=torch.float16).to(dev).eval()
L = np.load(ROOT / "data/manual/landmarks.npy")
rows = list(csv.DictReader(open(ROOT / "data/manual/manifest.csv")))

res = []
for c, i in enumerate(pick):
    im = Image.open(ROOT / "data/manual" / rows[i]["image_file"]).convert("RGB")
    w, h = im.size
    xy = L[i][:20] * np.array([w, h])
    if not np.isfinite(xy).all():
        continue
    eh = float(xy[:, 1].max() - xy[:, 1].min())
    if eh < 32:
        continue
    g = np.asarray(im.convert("L"), float) / 255.0
    grad = sobel(ndimage.gaussian_filter(g, 1.0))

    yy, xx = np.mgrid[0:h, 0:w]
    try:
        hull = xy[ConvexHull(xy).vertices]
    except Exception:
        continue
    inhull = MPath(hull).contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(h, w)
    inp = proc(images=im, text="ear", return_tensors="pt").to(dev)
    with torch.inference_mode():
        o = sam(**inp)
    try:
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
    mk = lab == (1 + np.argmax(sz))

    bnd = mk ^ ndimage.binary_erosion(mk)
    by, bx = np.nonzero(bnd)
    if len(by) < 40:
        continue
    lobe = xy[14:20]
    d_to_lobe = np.min(np.hypot(bx[:, None] - lobe[None, :, 0],
                                by[:, None] - lobe[None, :, 1]), axis=1)
    near = d_to_lobe < 0.15 * eh          # boundary near the annotated lobe
    if near.sum() < 5:
        continue
    gb = grad[by, bx]
    sd = (ndimage.distance_transform_edt(~mk)
          - ndimage.distance_transform_edt(mk))
    px = np.clip(np.round(xy[:, 0]).astype(int), 0, w - 1)
    py = np.clip(np.round(xy[:, 1]).astype(int), 0, h - 1)
    dist = sd[py, px] / eh * 100
    res.append(dict(idx=i,
                    g_lobe=float(np.median(gb[near])),
                    g_rest=float(np.median(gb[~near])),
                    g_img=float(np.median(grad[inhull])),
                    lobe_out=float(np.median(dist[14:20])),
                    mid_out=float(dist[12])))
    if (c + 1) % 25 == 0:
        print(f"  {c+1}/{len(pick)}", flush=True)

json.dump(res, open(OUT / "cutoff.json", "w"))
A = {k: np.array([r[k] for r in res]) for k in res[0]}
print(f"\n{len(res)} rows\n")
print(f"gradient on the mask boundary NEAR the lobe   {np.median(A['g_lobe']):.4f}")
print(f"gradient on the REST of the mask boundary     {np.median(A['g_rest']):.4f}")
print(f"gradient everywhere inside the ear            {np.median(A['g_img']):.4f}")
ratio = np.median(A['g_lobe'] / np.maximum(A['g_rest'], 1e-6))
print(f"\nper-row ratio lobe/rest: median {ratio:.2f}")
print(f"  rows where the lobe boundary is WEAKER than the rest: "
      f"{(A['g_lobe'] < A['g_rest']).mean()*100:.0f}%")
print(f"\nlobe points outside the mask {np.median(A['lobe_out']):+.2f}%,"
      f" mid-chain {np.median(A['mid_out']):+.2f}%")
print("\n(ratio well below 1 = the mask stops where there is NO edge, so the"
      "\n boundary is arbitrary and growth methods have room. Near or above 1"
      "\n = it stops at a real edge, and edge-driven repairs will stop too.)")
