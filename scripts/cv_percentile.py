"""Redo the ridge test as a PERCENTILE, and show the absolute levels.

A ratio of medians explodes when the control median is near zero, which is
exactly what frangi does on a flat patch -- a 42x ratio can mean "0.004 vs
0.0001", i.e. nothing. The honest measure is where a landmark's response
falls within the distribution of responses INSIDE THE SAME EAR: 50 means
the filter sees nothing there, 90 means it sees a strong structure.
"""
import sys, csv, pathlib, collections, numpy as np
from PIL import Image
import cv2
from scipy.spatial import ConvexHull
from matplotlib.path import Path as MPath
from skimage.filters import sato, frangi
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from scheme import GROUPS, GROUP_NAMES  # was ear3d.scheme; identical, moved at the repo split
R = pathlib.Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/Ear_Landmarker")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
g = {n: list(range(lo, hi)) for n, (lo, hi) in zip(GROUP_NAMES, GROUPS)}
L = np.load(R/"data/manual/landmarks.npy")
rows = list(csv.DictReader(open(R/"data/manual/manifest.csv")))
rng = np.random.default_rng(0)
pick = rng.choice(len(rows), size=min(N, len(rows)), replace=False)
pct = collections.defaultdict(list); absn = collections.defaultdict(list)
for c, i in enumerate(pick):
    im = Image.open(R/"data/manual"/rows[int(i)]["image_file"]).convert("RGB")
    w, h = im.size; xy = L[int(i)]*np.array([w, h])
    if not np.isfinite(xy).all():
        continue
    eh = float(xy[:, 1].max()-xy[:, 1].min())
    if eh < 24:
        continue
    gray = np.asarray(im.convert("L"), float)/255.0
    s = 2.0*eh/100.0; sg = [max(s*0.6, 1.0), s, s*1.6]
    F = {"sato_ridge": sato(gray, sigmas=sg, black_ridges=False),
         "sato_valley": sato(gray, sigmas=sg, black_ridges=True),
         "frangi": frangi(gray, sigmas=sg, black_ridges=True)}
    hull = xy[ConvexHull(xy).vertices]
    yy, xx = np.mgrid[0:h, 0:w]
    inh = MPath(hull).contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(h, w)
    px = np.clip(np.round(xy[:, 0]).astype(int), 0, w-1)
    py = np.clip(np.round(xy[:, 1]).astype(int), 0, h-1)
    for fn, arr in F.items():
        pool = arr[inh]
        if pool.size < 50:
            continue
        absn[(fn, "control")].append(float(np.median(pool)))
        for nm, idx in g.items():
            v = arr[py[idx], px[idx]]
            pct[(fn, nm)].append(float(np.mean(
                [np.searchsorted(np.sort(pool), q)/len(pool)*100 for q in v])))
            absn[(fn, nm)].append(float(np.median(v)))
    if (c+1) % 50 == 0:
        print(f"  {c+1}/{len(pick)}", flush=True)
print(f"\nn={len(absn[('frangi','control')])} images")
print("\nabsolute medians -- is the control actually near zero?")
for fn in ("sato_ridge", "sato_valley", "frangi"):
    print(f"  {fn:<13} control {np.mean(absn[(fn,'control')]):.5f}   "
          f"helix {np.mean(absn[(fn,'helix')]):.5f}   "
          f"concha {np.mean(absn[(fn,'concha')]):.5f}")
print("\nPERCENTILE of the landmark response within the same ear (50 = nothing)")
print(f"  {'group':<24}{'sato_ridge':>12}{'sato_valley':>13}{'frangi':>9}")
for nm in GROUP_NAMES:
    print(f"  {nm:<24}" + "".join(
        f"{np.mean(pct[(f,nm)]):>12.0f}" for f in
        ("sato_ridge", "sato_valley", "frangi")))
