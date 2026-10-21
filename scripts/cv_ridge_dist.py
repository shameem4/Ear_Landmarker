"""Distance from each landmark to the nearest TRACED RIDGE, not the response
at its own pixel.

The percentile version understated sato badly: ridges are 1-2 px wide, so an
annotation sitting one pixel off the crest reads as no response at all, while
the picture plainly shows the filter drawing the antihelix and both crura.
The question for tracing is "is there a line near this point", which is the
same question the Canny column answered -- so measure it the same way.

Ridge pixels are the top-q% of the in-ear response, which adapts per image
instead of fixing a threshold across very different exposures.
"""
import sys, csv, pathlib, collections, numpy as np
from PIL import Image
import cv2
from scipy import ndimage
from scipy.spatial import ConvexHull
from matplotlib.path import Path as MPath
from skimage.filters import sato
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from scheme import GROUPS, GROUP_NAMES  # was ear3d.scheme; identical, moved at the repo split
R = pathlib.Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/Ear_Landmarker")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
Q = float(sys.argv[2]) if len(sys.argv) > 2 else 85.0
g = {n: list(range(lo, hi)) for n, (lo, hi) in zip(GROUP_NAMES, GROUPS)}
L = np.load(R/"data/manual/landmarks.npy")
rows = list(csv.DictReader(open(R/"data/manual/manifest.csv")))
rng = np.random.default_rng(0)
pick = rng.choice(len(rows), size=min(N, len(rows)), replace=False)
D = collections.defaultdict(list); C = []; blank = 0
for c, i in enumerate(pick):
    im = Image.open(R/"data/manual"/rows[int(i)]["image_file"]).convert("RGB")
    w, h = im.size; xy = L[int(i)]*np.array([w, h])
    if not np.isfinite(xy).all():
        continue
    eh = float(xy[:, 1].max()-xy[:, 1].min())
    if eh < 24:
        continue
    gray = np.asarray(im.convert("L"), float)/255.0
    s = 2.0*eh/100.0
    val = sato(gray, sigmas=[max(s*.6, 1.), s, s*1.6], black_ridges=True)
    hull = xy[ConvexHull(xy).vertices]
    yy, xx = np.mgrid[0:h, 0:w]
    inh = MPath(hull).contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(h, w)
    pool = val[inh]
    if pool.size < 50:
        continue
    ridge = val >= np.percentile(pool, Q)
    if not ridge.any():
        blank += 1; continue
    dr = ndimage.distance_transform_edt(~ridge)/eh*100.0
    px = np.clip(np.round(xy[:, 0]).astype(int), 0, w-1)
    py = np.clip(np.round(xy[:, 1]).astype(int), 0, h-1)
    for nm, idx in g.items():
        D[nm].append(float(np.mean(dr[py[idx], px[idx]])))
    iy, ix = np.nonzero(inh)
    sel = rng.choice(len(iy), size=min(400, len(iy)), replace=False)
    C.append(float(np.mean(dr[iy[sel], ix[sel]])))
    if (c+1) % 50 == 0:
        print(f"  {c+1}/{len(pick)}", flush=True)
ctl = float(np.mean(C))
print(f"\nn={len(C)} images   top {100-Q:.0f}% of in-ear response = ridge   "
      f"({blank} blank)")
print("\ndistance from a landmark to the nearest sato ridge, % of ear height")
print(f"  {'group':<24}{'landmarks':>11}{'control':>10}{'ratio':>8}")
for r, nm, v in sorted((np.mean(D[nm])/ctl, nm, np.mean(D[nm]))
                       for nm in GROUP_NAMES):
    tag = "   <-- traced" if r < 0.70 else ("   <-- weak" if r < 0.85 else "")
    print(f"  {nm:<24}{v:>10.2f}%{ctl:>9.2f}%{r:>8.2f}{tag}")
