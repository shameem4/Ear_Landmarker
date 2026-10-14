"""Ear silhouette from SAM 3, prompted with the CONCEPT "ear".

Same output schema as the SAM 2.1, Florence-2 and U-Net versions, so
qa_find_bad_contours.py reads any of them unchanged.

SAM 3's reason for existing here is concept prompting: it is told the word
"ear" and finds it, where SAM 2.1 must be handed a box and a point and can
still take the hair. Among the returned instances we keep the one
overlapping the landmark hull best, which is coarse localisation of the same
kind the box gave SAM 2.1 -- no shape information, so the comparison is
fair.

LICENCE. SAM 3 is NOT Apache like SAM 2.1; it is Meta's custom SAM License
with gated weights, and the user accepted those terms for this account.
Reading the text: there is NO Llama-style restriction on using outputs to
train or distil, the grant is royalty-free with no commercial-use bar, and
the live restrictions are acceptable-use (military/ITAR/weapons) plus
no-reverse-engineering. The project's use is narrower still -- this model
only SCREENS which annotations are unsound, and never ships.

    python scripts/qa_silhouette_sam3.py /tmp/sam3_screen 1500 /tmp/idx.txt
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

OUT = pathlib.Path(sys.argv[1])
N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
IDX_FILE = sys.argv[3] if len(sys.argv) > 3 else None
OUT.mkdir(parents=True, exist_ok=True)

G = {n: list(range(lo, hi)) for n, (lo, hi) in zip(GROUP_NAMES, GROUPS)}
OUTER = G["ascending_helix"] + G["descending_helix"] + G["helix"] + G["ear_lobe"]
INNER = (G["ascending_inner_helix"] + G["descending_inner_helix"]
         + G["inner_helix"] + G["tragus"] + G["canal"] + G["concha"])

dev = "cuda" if torch.cuda.is_available() else "cpu"
proc = Sam3Processor.from_pretrained("facebook/sam3")
model = Sam3Model.from_pretrained("facebook/sam3", dtype=torch.float16).to(dev).eval()

L = np.load(ROOT / "data/manual/landmarks.npy")
rows = list(csv.DictReader(open(ROOT / "data/manual/manifest.csv")))
if IDX_FILE:
    pick = [int(x) for x in pathlib.Path(IDX_FILE).read_text().split()][:N]
    print(f"auditing {len(pick)} listed rows from {IDX_FILE}")
else:
    pick = np.random.default_rng(0).choice(len(rows), min(N, len(rows)), replace=False)

res, dropped = [], []
for c, i in enumerate(pick):
    i = int(i)
    try:
        im = Image.open(ROOT / "data/manual" / rows[i]["image_file"]).convert("RGB")
    except Exception:
        continue
    w, h = im.size
    xy = L[i] * np.array([w, h])
    if not np.isfinite(xy).all():
        continue
    eh = float(xy[:, 1].max() - xy[:, 1].min())
    if eh < 8:
        continue
    inp = proc(images=im, text="ear", return_tensors="pt").to(dev)
    with torch.inference_mode():
        o = model(**inp)
    try:
        r = proc.post_process_instance_segmentation(
            o, threshold=0.3, mask_threshold=0.5, target_sizes=[(h, w)])[0]
    except Exception:
        dropped.append(i); continue
    masks = np.asarray(r["masks"].cpu()) if hasattr(r["masks"], "cpu") else np.asarray(r["masks"])
    if masks.size == 0 or len(masks) == 0:
        dropped.append(i); continue
    yy, xx = np.mgrid[0:h, 0:w]
    try:
        hull = xy[ConvexHull(xy).vertices]
    except Exception:
        continue
    inhull = MPath(hull).contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(h, w)
    # keep the instance that best overlaps the landmark hull: coarse
    # localisation only, the same thing the box gave SAM 2.1
    best, mk = -1.0, None
    for m_ in masks:
        m_ = np.asarray(m_) > 0
        if m_.ndim != 2 or not m_.any():
            continue
        iou = (m_ & inhull).sum() / max((m_ | inhull).sum(), 1)
        if iou > best:
            best, mk = iou, m_
    if mk is None:
        dropped.append(i); continue
    lab, n = ndimage.label(mk)
    if n == 0:
        dropped.append(i); continue
    sizes = ndimage.sum(mk, lab, range(1, n + 1))
    if sizes.max() / max(sizes.sum(), 1.0) < 0.80:
        dropped.append(i); continue
    mk = lab == (1 + np.argmax(sizes))
    frac = float(mk.mean())
    if frac < 0.02 or frac > 0.95:
        dropped.append(i); continue
    covered = float((mk & inhull).sum() / max(inhull.sum(), 1))
    bloat = float(mk.sum()) / max(float(inhull.sum()), 1.0)
    if covered < 0.60 or bloat > 2.0:
        dropped.append(i); continue
    sd = ndimage.distance_transform_edt(~mk) - ndimage.distance_transform_edt(mk)
    px = np.clip(np.round(xy[:, 0]).astype(int), 0, w - 1)
    py = np.clip(np.round(xy[:, 1]).astype(int), 0, h - 1)
    dd = sd[py, px] / eh * 100.0
    res.append(dict(idx=i, iou=float(best), frac=frac, covered=covered,
                    bloat=bloat,
                    outer_abs=float(np.abs(dd[OUTER]).mean()),
                    inner_abs=float(np.abs(dd[INNER]).mean()),
                    outer_in=float((dd[OUTER] < 0).mean()),
                    per=dd.tolist()))
    if (c + 1) % 200 == 0:
        print(f"  {c+1}/{len(pick)}", flush=True)

json.dump(res, open(OUT / "sam2_silhouette.json", "w"))
print(f"\nSAM 3 failures dropped by the gates: {len(dropped)} "
      f"({len(dropped)/max(len(dropped)+len(res),1)*100:.1f}%)")
if res:
    D = np.array([r["per"] for r in res])
    o_ = np.abs(D[:, OUTER]).mean(); n_ = np.abs(D[:, INNER]).mean()
    print(f"n={len(res)} images   mask area {np.mean([r['frac'] for r in res])*100:.1f}% of crop")
    print(f"OUTER |dist| {o_:.2f}%   INNER |dist| {n_:.2f}%   "
          f"interior/outer ratio {n_/max(o_,1e-9):.2f}x")
print("wrote", OUT / "sam2_silhouette.json")
