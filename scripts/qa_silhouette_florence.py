"""Ear silhouette from Florence-2 -- TRIED AND REJECTED, kept as the record.

VERDICT: do not use this. Florence-2 returns coarse, low-vertex VLM
polygons. Rendered against SAM 2.1 on the same six ears it gave an EMPTY
mask on one, a bare vertical line on three, nothing usable on another, and
a single crude angular polygon spanning hair and ear on the last, while SAM
traced the ear properly on five of six. On a 40-row sample it was dropped by
the gates on 60% of images against SAM's 33%, and separated interior from
outer landmarks at 2.30x against SAM's 3.02x.

It is the right tool for "where is the object" and the wrong one for "where
exactly is the rim", which is the only thing this check needs. Kept so the
MIT-licence appeal does not cost someone the same afternoon twice.

The original motivation follows.

SAM 2.1 is the bottleneck in the contour audit, not the method: a third of
images fail its gates and only ~43% of the rest give a mask tight enough to
judge a contour against. Florence-2 is MIT for code AND weights, ungated,
and natively supported in transformers, so it costs nothing in licence
terms -- unlike SAM 3, whose weights are gated behind a manual accept.

It is prompted with <REGION_TO_SEGMENTATION> and the padded landmark
bounding box, which is the same coarse localisation SAM gets, so neither
model is handed any shape information.

Writes sam2_silhouette.json in the SAME schema as qa_silhouette_sam.py, so
qa_find_bad_contours.py reads either without changes. The point is not to
replace SAM but to AGREE with it: a row where two independent segmenters
draw the same rim is a row whose mask can be trusted, which is a far better
gate than any single model's own confidence -- SAM's IoU score already
failed at that job, reading 0.39 on a near-perfect mask.

    python scripts/qa_silhouette_florence.py /tmp/flo_screen 1500 /tmp/idx.txt
"""
import csv
import json
import pathlib
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage
from matplotlib.path import Path as MPath
from scipy.spatial import ConvexHull
from transformers import AutoProcessor, Florence2ForConditionalGeneration

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
# The NATIVE conversion, not microsoft/Florence-2-large. The original repo
# ships the remote-code layout, whose tokenizer has no image_token, so the
# native processor in transformers 5.x raises on it. This one is MIT,
# ungated, model_type=florence2 and carries NO auto_map, so nothing remote
# is executed to load it.
MID = "florence-community/Florence-2-large"
proc = AutoProcessor.from_pretrained(MID)
# Florence2 is a conditional-generation model in transformers 5.x, not a
# plain causal LM -- AutoModelForCausalLM refuses Florence2Config.
model = Florence2ForConditionalGeneration.from_pretrained(
    MID, dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()

L = np.load(ROOT / "data/manual/landmarks.npy")
rows = list(csv.DictReader(open(ROOT / "data/manual/manifest.csv")))
if IDX_FILE:
    pick = [int(x) for x in pathlib.Path(IDX_FILE).read_text().split()][:N]
    print(f"auditing {len(pick)} listed rows from {IDX_FILE}")
else:
    pick = np.random.default_rng(0).choice(len(rows), min(N, len(rows)),
                                           replace=False)

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
    lo, hi = xy.min(0), xy.max(0)
    eh = float(hi[1] - lo[1])
    if eh < 8:
        continue
    pad = 0.15 * (hi - lo)
    box = [max(0.0, lo[0] - pad[0]), max(0.0, lo[1] - pad[1]),
           min(w - 1.0, hi[0] + pad[0]), min(h - 1.0, hi[1] + pad[1])]
    task = "<REGION_TO_SEGMENTATION>"
    # Florence-2 takes the region as quantised <loc_> tokens, 0-999
    q = [int(round(box[0] / w * 999)), int(round(box[1] / h * 999)),
         int(round(box[2] / w * 999)), int(round(box[3] / h * 999))]
    prompt = task + f"<loc_{q[0]}><loc_{q[1]}><loc_{q[2]}><loc_{q[3]}>"
    inp = proc(text=prompt, images=im, return_tensors="pt").to(dev)
    if "pixel_values" in inp and dev == "cuda":
        inp["pixel_values"] = inp["pixel_values"].half()
    with torch.inference_mode():
        ids = model.generate(input_ids=inp["input_ids"],
                             pixel_values=inp["pixel_values"],
                             max_new_tokens=1024, num_beams=3, do_sample=False)
    txt = proc.batch_decode(ids, skip_special_tokens=False)[0]
    try:
        parsed = proc.post_process_generation(txt, task=task,
                                              image_size=(w, h))[task]
    except Exception:
        dropped.append(i); continue
    polys = parsed.get("polygons") or []
    mk = np.zeros((h, w), bool)
    if not polys:
        dropped.append(i); continue
    canvas = Image.new("1", (w, h), 0)
    d = ImageDraw.Draw(canvas)
    for inst in polys:
        for pg in inst:
            pts = np.asarray(pg, float).reshape(-1, 2)
            if len(pts) >= 3:
                d.polygon([tuple(p) for p in pts], fill=1)
    mk = np.asarray(canvas, bool)
    if not mk.any():
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
    yy, xx = np.mgrid[0:h, 0:w]
    try:
        hull = xy[ConvexHull(xy).vertices]
    except Exception:
        continue
    inhull = MPath(hull).contains_points(
        np.c_[xx.ravel(), yy.ravel()]).reshape(h, w)
    covered = float((mk & inhull).sum() / max(inhull.sum(), 1))
    bloat = float(mk.sum()) / max(float(inhull.sum()), 1.0)
    if covered < 0.60 or bloat > 2.0:
        dropped.append(i); continue
    din = ndimage.distance_transform_edt(mk)
    dout = ndimage.distance_transform_edt(~mk)
    sd = dout - din
    px = np.clip(np.round(xy[:, 0]).astype(int), 0, w - 1)
    py = np.clip(np.round(xy[:, 1]).astype(int), 0, h - 1)
    dd = sd[py, px] / eh * 100.0
    res.append(dict(idx=i, iou=float("nan"), frac=frac, covered=covered,
                    bloat=bloat,
                    outer_abs=float(np.abs(dd[OUTER]).mean()),
                    inner_abs=float(np.abs(dd[INNER]).mean()),
                    outer_in=float((dd[OUTER] < 0).mean()),
                    per=dd.tolist()))
    if (c + 1) % 50 == 0:
        print(f"  {c+1}/{len(pick)}", flush=True)

json.dump(res, open(OUT / "sam2_silhouette.json", "w"))
print(f"\nFlorence-2 failures dropped by the gates: {len(dropped)} "
      f"({len(dropped)/max(len(dropped)+len(res),1)*100:.1f}%)")
if res:
    D = np.array([r["per"] for r in res])
    o = np.abs(D[:, OUTER]).mean(); n_ = np.abs(D[:, INNER]).mean()
    print(f"n={len(res)} images   mask area {np.mean([r['frac'] for r in res])*100:.1f}% of crop")
    print(f"OUTER |dist| {o:.2f}%   INNER |dist| {n_:.2f}%   "
          f"interior/outer ratio {n_/max(o,1e-9):.2f}x")
print("wrote", OUT / "sam2_silhouette.json")
