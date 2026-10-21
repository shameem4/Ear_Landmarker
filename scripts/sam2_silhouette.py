"""Do the outer-contour landmarks lie on the ear's silhouette?

The check the project has been missing: every internal number agreed with
every other and all of them were wrong, so this one does not involve the
landmark model's own geometry. SAM 2.1 gets a BOX only -- the landmark
bounding box, padded -- which localises the ear and nothing more. The mask
BOUNDARY is image evidence, and the landmarks are then measured against it.

Outer groups (0-19: ascending helix, descending helix, helix, lobe) should
sit ON the boundary. INTERIOR groups (inner helix, tragus, canal, concha)
should sit well inside it. The interior set is the control: if both land on
the boundary the metric is measuring nothing, and if the outer set is as
deep as the interior one the model is drawing the helix inside the ear.

Distances are in % of ear height, so crops of different sizes compare.
SAM 2.1 is Apache 2.0, code and checkpoint.
"""
import sys, csv, json, pathlib, numpy as np, torch
from PIL import Image
from scipy import ndimage
from transformers import Sam2Model, Sam2Processor
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from scheme import GROUPS, GROUP_NAMES  # was ear3d.scheme; identical, moved at the repo split
R = pathlib.Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/Ear_Landmarker")
OUT = pathlib.Path(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
g = {n: list(range(lo, hi)) for n, (lo, hi) in zip(GROUP_NAMES, GROUPS)}
OUTER = g["ascending_helix"] + g["descending_helix"] + g["helix"] + g["ear_lobe"]
INNER = (g["ascending_inner_helix"] + g["descending_inner_helix"]
         + g["inner_helix"] + g["tragus"] + g["canal"] + g["concha"])

dev = "cuda" if torch.cuda.is_available() else "cpu"
mid = "facebook/sam2.1-hiera-base-plus"
proc = Sam2Processor.from_pretrained(mid)
model = Sam2Model.from_pretrained(mid, dtype=torch.float16).to(dev).eval()
L = np.load(R/"data/manual/landmarks.npy")
rows = list(csv.DictReader(open(R/"data/manual/manifest.csv")))
rng = np.random.default_rng(0)
pick = rng.choice(len(rows), size=min(N, len(rows)), replace=False)

res, dropped = [], []
for c, i in enumerate(pick):
    r = rows[int(i)]
    try:
        im = Image.open(R/"data/manual"/r["image_file"]).convert("RGB")
    except Exception:
        continue
    w, h = im.size
    xy = L[int(i)] * np.array([w, h])
    if not np.isfinite(xy).all():
        continue
    lo, hi = xy.min(0), xy.max(0)
    eh = float(hi[1] - lo[1])
    if eh < 8:
        continue
    pad = 0.15*(hi - lo)
    box = [[float(lo[0]-pad[0]), float(lo[1]-pad[1]),
            float(hi[0]+pad[0]), float(hi[1]+pad[1])]]
    # A BOX ALONE LETS SAM SEGMENT THE HAIR. Looking at the worst cases, the
    # landmarks were right every time and the mask was wrong -- SAM had taken
    # the background, the hair or a fragment. One positive point inside the
    # ear fixes most of it. The point is the concha centroid, which is coarse
    # localisation like the box, not shape information.
    pt = [[[[float(xy[INNER, 0].mean()), float(xy[INNER, 1].mean())]]]]
    inp = proc(images=im, input_boxes=[box], input_points=pt,
               input_labels=[[[1]]], return_tensors="pt").to(dev)
    with torch.inference_mode():
        o = model(**inp, multimask_output=True)
    m = proc.post_process_masks(o.pred_masks.float().cpu(),
                                inp["original_sizes"])[0].numpy()
    iou = o.iou_scores.float().cpu().numpy().ravel()
    mk = m.reshape(-1, *m.shape[-2:])[int(np.argmax(iou))] > 0
    lab, n = ndimage.label(mk)
    if n == 0:
        continue
    if n > 1:                      # keep the largest blob only
        mk = lab == (1 + np.argmax(ndimage.sum(mk, lab, range(1, n+1))))
    frac = mk.mean()
    if frac < 0.02 or frac > 0.95:        # degenerate mask, not a usable test
        continue
    # MASK QUALITY GATE. The landmark hull is used ONLY to reject a mask that
    # segmented something else entirely -- a yes/no on gross overlap. It does
    # not touch the boundary distance, which stays image evidence. The IoU
    # score cannot do this job: it read 0.39 on a near-perfect mask and 0.72
    # on one that had taken the hair.
    from matplotlib.path import Path as _P
    yy, xx = np.mgrid[0:h, 0:w]
    try:
        from scipy.spatial import ConvexHull
        hull = xy[ConvexHull(xy).vertices]
    except Exception:
        continue
    inhull = _P(hull).contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(h, w)
    covered = float((mk & inhull).sum() / max(inhull.sum(), 1))
    if covered < 0.60:
        dropped.append(int(i)); continue
    # distance to the mask BOUNDARY, signed negative inside
    din = ndimage.distance_transform_edt(mk)
    dout = ndimage.distance_transform_edt(~mk)
    sd = dout - din                        # >0 outside, <0 inside
    px = np.clip(np.round(xy[:, 0]).astype(int), 0, w-1)
    py = np.clip(np.round(xy[:, 1]).astype(int), 0, h-1)
    d = sd[py, px] / eh * 100.0            # % of ear height
    res.append(dict(idx=int(i), iou=float(iou.max()), frac=float(frac),
                    covered=covered,
                    outer_abs=float(np.abs(d[OUTER]).mean()),
                    inner_abs=float(np.abs(d[INNER]).mean()),
                    outer_in=float((d[OUTER] < 0).mean()),
                    per=d.tolist()))
    if (c+1) % 50 == 0:
        print(f"  {c+1}/{len(pick)}", flush=True)

json.dump(res, open(OUT/"sam2_silhouette.json", "w"))
D = np.array([r["per"] for r in res])
print(f"\nSAM failures dropped by the hull gate: {len(dropped)} "
      f"({len(dropped)/max(len(dropped)+len(res),1)*100:.1f}%)")
print(f"\nn={len(res)} images   mean mask IoU {np.mean([r['iou'] for r in res]):.3f}"
      f"   mask area {np.mean([r['frac'] for r in res])*100:.1f}% of crop")
print(f"\n{'group':<24}{'|dist| to boundary':>20}{'median':>9}{'inside':>9}")
for nm, idx in (("OUTER (0-19)", OUTER), ("  ascending_helix", g["ascending_helix"]),
                ("  descending_helix", g["descending_helix"]),
                ("  helix", g["helix"]), ("  ear_lobe", g["ear_lobe"]),
                ("INNER control", INNER), ("  concha", g["concha"]),
                ("  tragus", g["tragus"]), ("  inner_helix", g["inner_helix"])):
    v = D[:, idx]
    print(f"{nm:<24}{np.abs(v).mean():>17.2f}%{np.median(np.abs(v)):>8.2f}%"
          f"{(v<0).mean()*100:>8.1f}%")
o, n_ = np.abs(D[:, OUTER]).mean(), np.abs(D[:, INNER]).mean()
print(f"\ninterior/outer ratio {n_/max(o,1e-9):.2f}x  "
      f"(near 1 means the metric separates nothing)")
print("wrote", OUT/"sam2_silhouette.json")
