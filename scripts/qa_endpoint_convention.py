"""Where do the GOOD rows put the outer chain's endpoints?

Points 0 and 19 are the two ends of the open `outer_helix` linestrip, and
both terminate at the ear/head junction where no crisp image boundary
exists. The full-corpus screen shows that is exactly where the annotations
fail -- 4.83% of rows misplace point 0 and 6.01% point 19, against 0.02% at
point 12 in the middle of the chain.

Rather than invent a termination rule, this measures the one the clean
majority already follows, so the fix can be "match what 94.6% of the corpus
does" instead of a new convention nobody's annotations obey.

Everything is expressed in a frame derived from the SAM 3 MASK, never from
the landmarks, so a row with bad landmarks is still measured honestly:

    origin  mask centroid
    u       along the mask's long axis (PCA), + towards the helix end
    v       across it
    scale   the mask's extent along the long axis

Writes one record per row: the (u, v) of every outer point plus its signed
distance to the mask boundary.

    python scripts/qa_endpoint_convention.py runs/endpoint_conv ROWS.txt
"""
import csv
import json
import pathlib
import sys

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.spatial import ConvexHull
from matplotlib.path import Path as MPath
from transformers import Sam3Model, Sam3Processor

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from scheme import GROUPS, GROUP_NAMES          # noqa: E402

OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
ROWS = sys.argv[2]
OUTER = list(range(0, 20))          # the whole outer_helix strip, in order

dev = "cuda" if torch.cuda.is_available() else "cpu"
proc = Sam3Processor.from_pretrained("facebook/sam3")
model = Sam3Model.from_pretrained("facebook/sam3", dtype=torch.float16).to(dev).eval()
L = np.load(ROOT / "data/manual/landmarks.npy")
rows = list(csv.DictReader(open(ROOT / "data/manual/manifest.csv")))
pick = [int(x) for x in pathlib.Path(ROWS).read_text().split()]
print(f"measuring {len(pick)} rows")

res = []
for c, i in enumerate(pick):
    try:
        im = Image.open(ROOT / "data/manual" / rows[i]["image_file"]).convert("RGB")
    except Exception:
        continue
    w, h = im.size
    xy = L[i] * np.array([w, h])
    if not np.isfinite(xy).all():
        continue
    inp = proc(images=im, text="ear", return_tensors="pt").to(dev)
    with torch.inference_mode():
        o = model(**inp)
    try:
        r = proc.post_process_instance_segmentation(
            o, threshold=0.3, mask_threshold=0.5, target_sizes=[(h, w)])[0]
        masks = np.asarray(r["masks"].cpu())
    except Exception:
        continue
    if len(masks) == 0:
        continue
    yy, xx = np.mgrid[0:h, 0:w]
    try:
        hull = xy[ConvexHull(xy).vertices]
    except Exception:
        continue
    inhull = MPath(hull).contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(h, w)
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
    sizes = ndimage.sum(mk, lab, range(1, n + 1))
    if sizes.max() / max(sizes.sum(), 1.0) < 0.80:
        continue
    mk = lab == (1 + np.argmax(sizes))
    # ---- frame from the MASK, not the landmarks ----
    ys, xs = np.nonzero(mk)
    P = np.c_[xs, ys].astype(np.float64)
    cen = P.mean(0)
    _, _, Vt = np.linalg.svd(P - cen, full_matrices=False)
    e1 = Vt[0]                                   # long axis
    e2 = np.array([-e1[1], e1[0]])
    proj = (P - cen) @ e1
    scale = float(proj.max() - proj.min())
    if scale < 8:
        continue
    # orient +u towards the end nearer landmark 0, so left/right ears agree
    u0 = float((xy[0] - cen) @ e1)
    if u0 < 0:
        e1, u0 = -e1, -u0
        e2 = np.array([-e1[1], e1[0]])
        proj = (P - cen) @ e1
    sd = (ndimage.distance_transform_edt(~mk)
          - ndimage.distance_transform_edt(mk))
    px = np.clip(np.round(xy[:, 0]).astype(int), 0, w - 1)
    py = np.clip(np.round(xy[:, 1]).astype(int), 0, h - 1)
    rec = dict(idx=i, scale=scale)
    for j in OUTER:
        d = xy[j] - cen
        rec[f"u{j}"] = float(d @ e1) / scale
        rec[f"v{j}"] = float(d @ e2) / scale
        rec[f"d{j}"] = float(sd[py[j], px[j]]) / scale
    res.append(rec)
    if (c + 1) % 200 == 0:
        print(f"  {c+1}/{len(pick)}", flush=True)

json.dump(res, open(OUT / "endpoints.json", "w"))
print(f"\nwrote {len(res)} records -> {OUT/'endpoints.json'}")
