"""Does a different text prompt stop SAM 3 dropping the earlobe?

The endpoint "defect" turned out to be the mask: SAM 3 prompted with "ear"
systematically under-segments, worst at the two ends of the outer chain, and
on many rows it cuts across ABOVE the lobe. That makes distance-to-rim
useless for judging points 0 and 19, because a correct annotation and a
short mask look identical.

SAM 3 takes a CONCEPT, so the fix may be to name the concept better. This
sweeps prompts and scores each by how far the annotated points sit outside
the mask, SIGNED. The number that matters is point 19 (lobe end) against
point 12 (mid-helix, where the mask is already good): if a prompt includes
the lobe, point 19's excess over point 12 collapses.

Lower is better everywhere; the target is point 19 looking like point 12.

    python scripts/qa_sam3_prompt_sweep.py runs/prompt_sweep ROWS.txt [PROMPTS.txt]

Note SAM 3 takes a CONCEPT, not an instruction: an imperative prompt is
embedded as a bag of nouns, so "extrapolate the lobe if occluded" is really
being tested as "lobe, occluded" and should be read that way.
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

PROMPTS = [
    "ear",
    "ear including the earlobe",
    "whole ear with earlobe",
    "human outer ear and earlobe",
    "auricle",
    "pinna",
]

OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
ROWS = sys.argv[2]
if len(sys.argv) > 3:  # one prompt per line, to sweep a second batch
    PROMPTS = [l for l in pathlib.Path(sys.argv[3]).read_text().splitlines() if l.strip()]
dev = "cuda" if torch.cuda.is_available() else "cpu"
proc = Sam3Processor.from_pretrained("facebook/sam3")
model = Sam3Model.from_pretrained("facebook/sam3", dtype=torch.float16).to(dev).eval()
L = np.load(ROOT / "data/manual/landmarks.npy")
rows = list(csv.DictReader(open(ROOT / "data/manual/manifest.csv")))
pick = [int(x) for x in pathlib.Path(ROWS).read_text().split()]
print(f"{len(pick)} rows x {len(PROMPTS)} prompts")

res = {p: [] for p in PROMPTS}
fails = {p: 0 for p in PROMPTS}
for c, i in enumerate(pick):
    try:
        im = Image.open(ROOT / "data/manual" / rows[i]["image_file"]).convert("RGB")
    except Exception:
        continue
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
    for prompt in PROMPTS:
        inp = proc(images=im, text=prompt, return_tensors="pt").to(dev)
        with torch.inference_mode():
            o = model(**inp)
        try:
            r = proc.post_process_instance_segmentation(
                o, threshold=0.3, mask_threshold=0.5, target_sizes=[(h, w)])[0]
            masks = np.asarray(r["masks"].cpu())
        except Exception:
            fails[prompt] += 1; continue
        if len(masks) == 0:
            fails[prompt] += 1; continue
        best, mk = -1.0, None
        for m_ in masks:
            m_ = np.asarray(m_) > 0
            if m_.ndim != 2 or not m_.any():
                continue
            v = (m_ & inhull).sum() / max((m_ | inhull).sum(), 1)
            if v > best:
                best, mk = v, m_
        if mk is None:
            fails[prompt] += 1; continue
        lab, n = ndimage.label(mk)
        if n == 0:
            fails[prompt] += 1; continue
        sizes = ndimage.sum(mk, lab, range(1, n + 1))
        if sizes.max() / max(sizes.sum(), 1.0) < 0.80:
            fails[prompt] += 1; continue
        mk = lab == (1 + np.argmax(sizes))
        sd = (ndimage.distance_transform_edt(~mk)
              - ndimage.distance_transform_edt(mk))
        d = sd[py, px] / eh * 100.0
        res[prompt].append(dict(idx=i, per=d[:20].tolist(),
                                area=float(mk.mean())))
    if (c + 1) % 50 == 0:
        print(f"  {c+1}/{len(pick)}", flush=True)

json.dump({k: v for k, v in res.items()}, open(OUT / "prompt_sweep.json", "w"))
print(f"\n{'prompt':<32}{'n':>5}{'fail':>6}{'pt0':>8}{'pt12':>8}{'pt19':>8}"
      f"{'19-12':>8}{'lobe':>8}")
for p in PROMPTS:
    v = res[p]
    if not v:
        print(f"  {p:<30}{0:>5}"); continue
    P = np.array([r["per"] for r in v])
    p0, p12, p19 = (np.median(P[:, j]) for j in (0, 12, 19))
    lobe = np.median(P[:, 14:20])
    print(f"{p:<32}{len(v):>5}{fails[p]:>6}{p0:>8.2f}{p12:>8.2f}{p19:>8.2f}"
          f"{p19-p12:>8.2f}{lobe:>8.2f}")
print("\n(signed % of ear height; >0 = the annotation sits OUTSIDE the mask.")
print(" '19-12' is the lobe end's excess over mid-helix -- near 0 means the")
print(" prompt includes the lobe and the endpoint signal becomes usable.)")
