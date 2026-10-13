"""Does a VLM box fix SAM 3's lobe shortfall?

Text alone does not: see qa_sam3_prompt_sweep.py, where naming the lobe
moves the point-19 excess by less than the run-to-run spread and
instruction-shaped prompts lose most of the masks. SAM 3 also takes BOXES,
so this feeds it the box from qa_ear_box_vlm.py and re-scores.

Scored exactly as the sweep: signed distance from each annotated point to
the mask edge, as % of ear height, positive = outside. The number that
matters is point 19 (lobe) minus point 12 (mid-helix, already good).

    python scripts/qa_sam3_boxed.py runs/vlmbox/boxes_gemini.json runs/sam3_boxed
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
BOXES = json.load(open(sys.argv[1]))
OUT = pathlib.Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)

dev = "cuda" if torch.cuda.is_available() else "cpu"
proc = Sam3Processor.from_pretrained("facebook/sam3")
model = Sam3Model.from_pretrained("facebook/sam3", dtype=torch.float16).to(dev).eval()
L = np.load(ROOT / "data/manual/landmarks.npy")
rows = list(csv.DictReader(open(ROOT / "data/manual/manifest.csv")))


def pick_mask(o, h, w, inhull):
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
    if n == 0:
        return None
    sizes = ndimage.sum(mk, lab, range(1, n + 1))
    if sizes.max() / max(sizes.sum(), 1.0) < 0.80:
        return None
    return lab == (1 + np.argmax(sizes))


arms = {"text only": [], "box only": [], "text + box": []}
for c, k in enumerate(BOXES):
    i = int(k)
    im = Image.open(ROOT / "data/manual" / rows[i]["image_file"]).convert("RGB")
    w, h = im.size
    xy = L[i] * np.array([w, h])
    if not np.isfinite(xy).all():
        continue
    eh = float(xy[:, 1].max() - xy[:, 1].min())
    if eh < 24:
        continue
    yy, xx = np.mgrid[0:h, 0:w]
    try:
        hull = xy[ConvexHull(xy).vertices]
    except Exception:
        continue
    inhull = MPath(hull).contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(h, w)
    px = np.clip(np.round(xy[:, 0]).astype(int), 0, w - 1)
    py = np.clip(np.round(xy[:, 1]).astype(int), 0, h - 1)
    x0, y0, x1, y1 = BOXES[k]
    bx = [[[x0 * w, y0 * h, x1 * w, y1 * h]]]

    for arm, kw in (("text only", dict(text="ear")),
                    ("box only", dict(input_boxes=bx)),
                    ("text + box", dict(text="ear", input_boxes=bx))):
        try:
            inp = proc(images=im, return_tensors="pt", **kw).to(dev)
            # the box tensors come out float32; the model is half, and a
            # box-prompt Linear refuses the mix even though the image path
            # is cast for us.
            for key, val in list(inp.items()):
                if torch.is_tensor(val) and val.is_floating_point():
                    inp[key] = val.half()
            with torch.inference_mode():
                o = model(**inp)
        except Exception as e:
            if c == 0:
                print(f"  {arm}: {type(e).__name__} {e}")
            continue
        mk = pick_mask(o, h, w, inhull)
        if mk is None:
            continue
        sd = (ndimage.distance_transform_edt(~mk)
              - ndimage.distance_transform_edt(mk))
        arms[arm].append((ndimage.map_coordinates(
            sd, [py, px], order=1) / eh * 100.0)[:20].tolist())
    if (c + 1) % 20 == 0:
        print(f"  {c+1}/{len(BOXES)}", flush=True)

json.dump(arms, open(OUT / "boxed.json", "w"))
print(f"\n{'arm':<14}{'n':>5}{'pt0':>8}{'pt12':>8}{'pt19':>8}{'19-12':>8}{'lobe':>8}")
for a, v in arms.items():
    if not v:
        print(f"{a:<14}{0:>5}"); continue
    P = np.array(v)
    p0, p12, p19 = (np.median(P[:, j]) for j in (0, 12, 19))
    print(f"{a:<14}{len(v):>5}{p0:>8.2f}{p12:>8.2f}{p19:>8.2f}"
          f"{p19-p12:>8.2f}{np.median(P[:, 14:20]):>8.2f}")
print("\n(signed % of ear height, >0 = annotation OUTSIDE the mask;"
      " '19-12' near 0 = the lobe is in.)")
