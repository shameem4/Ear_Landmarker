"""Text-prompt vs box-prompt masks, side by side, on the lobe rows.

The two arms score within noise of each other, which could mean the box
changed nothing OR that SAM 3 ignored it. Those look identical in a table
and completely different on screen.

    python scripts/qa_sam3_boxed_show.py runs/vlmbox/boxes_gemini.json out.png
"""
import csv
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from PIL import Image
from scipy import ndimage
from transformers import Sam3Model, Sam3Processor

ROOT = pathlib.Path(__file__).resolve().parent.parent
BOXES = json.load(open(sys.argv[1]))
OUTPNG = sys.argv[2]
ROWS = [int(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3
                         else "113,155,318,428,506,157".split(","))]
dev = "cuda" if torch.cuda.is_available() else "cpu"
proc = Sam3Processor.from_pretrained("facebook/sam3")
model = Sam3Model.from_pretrained("facebook/sam3", dtype=torch.float16).to(dev).eval()
L = np.load(ROOT / "data/manual/landmarks.npy")
rows = list(csv.DictReader(open(ROOT / "data/manual/manifest.csv")))


def run(im, h, w, **kw):
    inp = proc(images=im, return_tensors="pt", **kw).to(dev)
    for k, v in list(inp.items()):
        if torch.is_tensor(v) and v.is_floating_point():
            inp[k] = v.half()
    with torch.inference_mode():
        o = model(**inp)
    r = proc.post_process_instance_segmentation(
        o, threshold=0.3, mask_threshold=0.5, target_sizes=[(h, w)])[0]
    m = np.asarray(r["masks"].cpu())
    if len(m) == 0:
        return None
    m = [np.asarray(x) > 0 for x in m]
    big = max(m, key=lambda x: x.sum())
    lab, n = ndimage.label(big)
    if n > 1:
        sz = ndimage.sum(big, lab, range(1, n + 1))
        big = lab == (1 + np.argmax(sz))
    return big


ARMS = ["text only", "box only", "text + box"]
fig, ax = plt.subplots(len(ROWS), 3, figsize=(3.1 * 3, 3.1 * len(ROWS)))
ax = np.atleast_2d(ax)
for a in np.ravel(ax):
    a.axis("off")
for r_, i in enumerate(ROWS):
    im = Image.open(ROOT / "data/manual" / rows[i]["image_file"]).convert("RGB")
    w, h = im.size
    xy = L[i] * np.array([w, h])
    b = BOXES.get(str(i))
    bx = [[[b[0] * w, b[1] * h, b[2] * w, b[3] * h]]] if b else None
    for c_, arm in enumerate(ARMS):
        a = ax[r_, c_]
        a.imshow(im)
        kw = (dict(text="ear") if arm == "text only" else
              dict(input_boxes=bx) if arm == "box only" else
              dict(text="ear", input_boxes=bx))
        if bx is None and arm != "text only":
            a.set_title("no box", fontsize=8); continue
        mk = run(im, h, w, **kw)
        if mk is not None:
            a.contour(mk, [0.5], colors="lime", linewidths=1.4)
        if b:
            a.add_patch(Rectangle((b[0] * w, b[1] * h), (b[2] - b[0]) * w,
                                  (b[3] - b[1]) * h, fill=False, ec="cyan",
                                  lw=0.9, ls="--"))
        a.plot(xy[:20, 0], xy[:20, 1], "-", color="orange", lw=1.0)
        a.plot(*xy[19], "o", color="red", ms=4)
        a.set_title(f"{i}  {arm}", fontsize=8)
fig.tight_layout()
fig.savefig(OUTPNG, dpi=115)
print(f"wrote {OUTPNG}")
