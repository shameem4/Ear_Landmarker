"""Render VLM boxes over the crops, with the annotation drawn on top.

A box statistic cannot tell a correct ear box from a confident box around
the concha, and the whole point of this exercise is that the lobe gets
left out -- which is a thing you SEE. Green box = VLM, orange chain =
annotated outer contour, red dot = point 19 (lobe end), blue = point 0.

    python scripts/qa_vlmbox_show.py runs/vlmbox/boxes_ollama.json out.png
"""
import csv
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
box = json.load(open(sys.argv[1]))
out = sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 24
L = np.load(ROOT / "data/manual/landmarks.npy")
rows = list(csv.DictReader(open(ROOT / "data/manual/manifest.csv")))

keys = list(box)[:n]
c = 6
r = (len(keys) + c - 1) // c
fig, ax = plt.subplots(r, c, figsize=(2.3 * c, 2.3 * r))
for a in np.ravel(ax):
    a.axis("off")
for a, k in zip(np.ravel(ax), keys):
    i = int(k)
    im = Image.open(ROOT / "data/manual" / rows[i]["image_file"]).convert("RGB")
    w, h = im.size
    a.imshow(im)
    xy = L[i] * np.array([w, h])
    a.plot(xy[:20, 0], xy[:20, 1], "-", color="orange", lw=1.0)
    a.plot(*xy[0], "o", color="deepskyblue", ms=3)
    a.plot(*xy[19], "o", color="red", ms=3)
    x0, y0, x1, y1 = box[k]
    a.add_patch(Rectangle((x0 * w, y0 * h), (x1 - x0) * w, (y1 - y0) * h,
                          fill=False, ec="lime", lw=1.2))
    below = xy[19, 1] > y1 * h          # lobe end outside the box bottom
    a.set_title(f"{i}{'  LOBE OUT' if below else ''}", fontsize=7,
                color="red" if below else "black")
fig.tight_layout()
fig.savefig(out, dpi=115)
print(f"wrote {out}")

cut = sum(1 for k in box
          if L[int(k)][19, 1] > box[k][3])
print(f"point 19 below the box bottom in {cut}/{len(box)} rows")
