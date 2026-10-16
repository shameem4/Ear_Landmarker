"""Does cropping to the ear first give SAM a finer silhouette?

A/B on the same images: full frame vs cropped to the padded landmark box
and re-encoded at 4x. SAM's encoder works at 1024x1024, so cropping raises
the ear's share of the frame -- and removes the hair and background that
caused every failure in the worst cases.

THE FRAME EDGE IS NOT A SILHOUETTE. The mask includes skin attached to the
ear, so on a crop it runs off the side; the first version rejected every
such image and threw away half the data. Worse, where it did NOT reject,
the cut edge counted as mask boundary, so a landmark near the frame read
as sitting on the silhouette when it was only sitting near the crop. Both
are fixed the same way: the distance transform is taken to boundary pixels
that are NOT on the frame edge, so a cut contributes nothing and the
measurement is to real silhouette only. Nothing is rejected for touching
the border -- only for a mask that is degenerate or segmented the wrong
object, which the hull gate catches.
"""
import sys, csv, pathlib, numpy as np, torch
from PIL import Image
from scipy import ndimage
from scipy.spatial import ConvexHull
from matplotlib.path import Path as MPath
from transformers import Sam2Model, Sam2Processor
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from ear3d.scheme import GROUPS, GROUP_NAMES
R = pathlib.Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/Ear_Landmarker")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
PAD = float(sys.argv[2]) if len(sys.argv) > 2 else 0.60
g = {n: list(range(lo, hi)) for n, (lo, hi) in zip(GROUP_NAMES, GROUPS)}
OUTER = g["ascending_helix"]+g["descending_helix"]+g["helix"]+g["ear_lobe"]
INNER = (g["ascending_inner_helix"]+g["descending_inner_helix"]
         + g["inner_helix"]+g["tragus"]+g["canal"]+g["concha"])
dev = "cuda"; mid = "facebook/sam2.1-hiera-base-plus"
proc = Sam2Processor.from_pretrained(mid)
model = Sam2Model.from_pretrained(mid, dtype=torch.float16).to(dev).eval()
L = np.load(R/"data/manual/landmarks.npy")
rows = list(csv.DictReader(open(R/"data/manual/manifest.csv")))
pick = np.random.default_rng(0).choice(len(rows), size=min(N, len(rows)), replace=False)


def segment(im, xy):
    w, h = im.size
    lo, hi = xy.min(0), xy.max(0); p = 0.15*(hi-lo)
    box = [[float(lo[0]-p[0]), float(lo[1]-p[1]),
            float(hi[0]+p[0]), float(hi[1]+p[1])]]
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
    if n > 1:
        mk = lab == (1+np.argmax(ndimage.sum(mk, lab, range(1, n+1))))
    return mk


def signed_dist(mk, xy, touched=None):
    """Signed distance to REAL silhouette; frame-edge cuts contribute none."""
    h, w = mk.shape
    bnd = mk & ~ndimage.binary_erosion(mk)
    edge = np.zeros_like(mk)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True
    if touched is not None:
        touched.append(bool((mk & edge).any()))
    real = bnd & ~edge
    if not real.any():
        return None
    d = ndimage.distance_transform_edt(~real)
    sd = np.where(mk, -1.0, 1.0) * d
    eh = float(xy[:, 1].max()-xy[:, 1].min())
    px = np.clip(np.round(xy[:, 0]).astype(int), 0, w-1)
    py = np.clip(np.round(xy[:, 1]).astype(int), 0, h-1)
    return sd[py, px]/eh*100.0


def usable(mk, xy, h, w):
    if not (0.02 < mk.mean() < 0.95):
        return False
    hull = xy[ConvexHull(xy).vertices]
    yy, xx = np.mgrid[0:h, 0:w]
    inh = MPath(hull).contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(h, w)
    return (mk & inh).sum()/max(inh.sum(), 1) >= 0.60


A, B, da, db, touch_a, touch_b = [], [], 0, 0, [], []
for c, i in enumerate(pick):
    im = Image.open(R/"data/manual"/rows[int(i)]["image_file"]).convert("RGB")
    w, h = im.size
    xy = L[int(i)]*np.array([w, h])
    if not np.isfinite(xy).all() or (xy[:, 1].max()-xy[:, 1].min()) < 8:
        continue
    mk = segment(im, xy)
    if usable(mk, xy, h, w):
        s = signed_dist(mk, xy, touch_a)
        A.append(s) if s is not None else None
    else:
        da += 1
    lo, hi = xy.min(0), xy.max(0); p = PAD*(hi-lo)
    x0, y0 = int(max(0, lo[0]-p[0])), int(max(0, lo[1]-p[1]))
    x1, y1 = int(min(w, hi[0]+p[0])), int(min(h, hi[1]+p[1]))
    if x1-x0 < 16 or y1-y0 < 16:
        db += 1; continue
    sc = 4.0
    cim = im.crop((x0, y0, x1, y1))
    cim = cim.resize((int((x1-x0)*sc), int((y1-y0)*sc)), Image.LANCZOS)
    cxy = (xy-np.array([x0, y0]))*sc
    mk2 = segment(cim, cxy)
    if usable(mk2, cxy, cim.size[1], cim.size[0]):
        s = signed_dist(mk2, cxy, touch_b)
        B.append(s) if s is not None else None
    else:
        db += 1
    if (c+1) % 50 == 0:
        print(f"  {c+1}/{len(pick)}", flush=True)

A, B = np.array(A), np.array(B)
print(f"\nfull frame : n={len(A)}  dropped {da}  "
      f"(mask touched frame on {sum(touch_a)}, no longer a rejection)")
print(f"cropped x4 pad {PAD:.0%}: n={len(B)}  dropped {db}  "
      f"(mask touched crop border on {sum(touch_b)})")
print(f"\n{'':<20}{'full frame':>24}{'cropped':>24}")
print(f"{'':<20}{'mean   median  inside':>24}{'mean   median  inside':>24}")
for nm, idx in (("OUTER (0-19)", OUTER), ("  helix", g["helix"]),
                ("  ear_lobe", g["ear_lobe"]), ("INNER control", INNER)):
    a, b = A[:, idx], B[:, idx]
    print(f"{nm:<20}{np.abs(a).mean():>8.2f}{np.median(np.abs(a)):>8.2f}"
          f"{(a<0).mean()*100:>7.0f}%{np.abs(b).mean():>9.2f}"
          f"{np.median(np.abs(b)):>8.2f}{(b<0).mean()*100:>7.0f}%")
print(f"\nseparation (inner median / outer median): "
      f"full {np.median(np.abs(A[:,INNER]))/np.median(np.abs(A[:,OUTER])):.1f}x   "
      f"cropped {np.median(np.abs(B[:,INNER]))/np.median(np.abs(B[:,OUTER])):.1f}x")
