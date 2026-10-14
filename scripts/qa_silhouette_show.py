"""Look at the best and worst silhouette agreements before believing the table.

A large landmark-to-boundary distance has two causes that the number cannot
separate: SAM found the wrong thing, or the landmarks are wrong. Only the
picture tells them apart, so the worst cases are rendered with the mask
outline and the outer landmarks on top.
"""
import sys, csv, json, pathlib, numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from transformers import Sam2Model, Sam2Processor
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from scheme import GROUPS, GROUP_NAMES
R = pathlib.Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/Ear_Landmarker")
# The silhouette JSON used to be written into the 3D repo's scratch area;
# it belongs with the 2D QA runs. Pass a directory to override.
W = R / "runs" / "qa"
g = {n: list(range(lo, hi)) for n, (lo, hi) in zip(GROUP_NAMES, GROUPS)}
OUTER = g["ascending_helix"]+g["descending_helix"]+g["helix"]+g["ear_lobe"]
res = json.load(open(W/"sam2_silhouette.json" if len(sys.argv)<2 else pathlib.Path(sys.argv[1])/"sam2_silhouette.json"))
res.sort(key=lambda r: r["outer_abs"])
show = res[:6] + res[-6:]
dev = "cuda"; mid = "facebook/sam2.1-hiera-base-plus"
proc = Sam2Processor.from_pretrained(mid)
model = Sam2Model.from_pretrained(mid, dtype=torch.float16).to(dev).eval()
L = np.load(R/"data/manual/landmarks.npy")
rows = list(csv.DictReader(open(R/"data/manual/manifest.csv")))
fig, axes = plt.subplots(2, 6, figsize=(16.5, 6.0), dpi=135)
for ax, r in zip(axes.ravel(), show):
    i = r["idx"]; rec = rows[i]
    im = Image.open(R/"data/manual"/rec["image_file"]).convert("RGB")
    w, h = im.size; xy = L[i]*np.array([w, h])
    lo, hi = xy.min(0), xy.max(0); pad = 0.15*(hi-lo)
    box = [[float(lo[0]-pad[0]), float(lo[1]-pad[1]),
            float(hi[0]+pad[0]), float(hi[1]+pad[1])]]
    inp = proc(images=im, input_boxes=[box], return_tensors="pt").to(dev)
    with torch.inference_mode():
        o = model(**inp, multimask_output=True)
    m = proc.post_process_masks(o.pred_masks.float().cpu(),
                                inp["original_sizes"])[0].numpy()
    iou = o.iou_scores.float().cpu().numpy().ravel()
    mk = m.reshape(-1, *m.shape[-2:])[int(np.argmax(iou))] > 0
    lab, n = ndimage.label(mk)
    if n > 1:
        mk = lab == (1+np.argmax(ndimage.sum(mk, lab, range(1, n+1))))
    ax.imshow(im); ax.contour(mk, levels=[.5], colors="#17d4ff", linewidths=1.4)
    ax.plot(xy[OUTER, 0], xy[OUTER, 1], ".", ms=3.6, c="#ff2d2d")
    ax.set_axis_off()
    ax.set_title(f"outer {r['outer_abs']:.1f}%  IoU {r['iou']:.2f}", fontsize=8)
axes[0, 0].set_ylabel("best"); axes[1, 0].set_ylabel("worst")
fig.suptitle("SAM 2.1 mask (cyan) vs outer landmarks (red) -- top: best 6, "
             "bottom: worst 6", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(W/"sam2_cases.png")
print("wrote", W/"sam2_cases.png")
