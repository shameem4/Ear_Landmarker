"""Which ear curves can conventional CV trace, beyond the outer silhouette?

SAM gives the silhouette, so it can only check points 0-19. The antihelix,
the crura, the tragus and the concha rim are RIDGES AND VALLEYS in shading,
not silhouette, and classical filters see those: Canny for edges, Sato and
Frangi for bright ridges and dark valleys at a chosen scale.

THE CONTROL IS THE WHOLE TEST. "Landmarks sit near an edge" means nothing
on a textured ear where edges are everywhere, so every group is compared
against random points drawn INSIDE THE EAR HULL, matched in count, on the
same image and the same filter. A group is only visible to classical CV if
it beats that.
"""
import sys, csv, pathlib, collections, numpy as np
from PIL import Image
import cv2
from scipy import ndimage
from scipy.spatial import ConvexHull
from matplotlib.path import Path as MPath
from skimage.filters import sato, frangi
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from ear3d.scheme import GROUPS, GROUP_NAMES
R = pathlib.Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/Ear_Landmarker")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
g = {n: list(range(lo, hi)) for n, (lo, hi) in zip(GROUP_NAMES, GROUPS)}
L = np.load(R/"data/manual/landmarks.npy")
rows = list(csv.DictReader(open(R/"data/manual/manifest.csv")))
rng = np.random.default_rng(0)
pick = rng.choice(len(rows), size=min(N, len(rows)), replace=False)

acc = collections.defaultdict(list)
ctl = collections.defaultdict(list)
for c, i in enumerate(pick):
    im = Image.open(R/"data/manual"/rows[int(i)]["image_file"]).convert("RGB")
    w, h = im.size
    xy = L[int(i)]*np.array([w, h])
    if not np.isfinite(xy).all():
        continue
    eh = float(xy[:, 1].max()-xy[:, 1].min())
    if eh < 24:
        continue
    gray = np.asarray(im.convert("L"), float)/255.0
    s = 2.0*eh/100.0                        # scale the filters to the ear
    gb = cv2.GaussianBlur((gray*255).astype(np.uint8), (0, 0), max(s*0.6, 0.8))
    med = float(np.median(gb))
    edges = cv2.Canny(gb, int(max(0, 0.66*med)), int(min(255, 1.33*med)))
    dedge = ndimage.distance_transform_edt(edges == 0)/eh*100.0
    sg = [max(s*0.6, 1.0), s, s*1.6]
    rid = sato(gray, sigmas=sg, black_ridges=False)
    val = sato(gray, sigmas=sg, black_ridges=True)
    fra = frangi(gray, sigmas=sg, black_ridges=True)
    hull = xy[ConvexHull(xy).vertices]
    yy, xx = np.mgrid[0:h, 0:w]
    inh = MPath(hull).contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(h, w)
    iy, ix = np.nonzero(inh)
    if len(iy) < 60:
        continue
    sel = rng.choice(len(iy), size=min(400, len(iy)), replace=False)
    cpx, cpy = ix[sel], iy[sel]

    def at(arr, px, py):
        return arr[np.clip(py, 0, h-1), np.clip(px, 0, w-1)]

    px = np.clip(np.round(xy[:, 0]).astype(int), 0, w-1)
    py = np.clip(np.round(xy[:, 1]).astype(int), 0, h-1)
    for nm, idx in g.items():
        acc[("canny", nm)].append(at(dedge, px[idx], py[idx]).mean())
        for fn, arr in (("sato_ridge", rid), ("sato_valley", val), ("frangi", fra)):
            a = at(arr, px[idx], py[idx])
            acc[(fn, nm)].append(float(np.median(a)))
    ctl["canny"].append(at(dedge, cpx, cpy).mean())
    for fn, arr in (("sato_ridge", rid), ("sato_valley", val), ("frangi", fra)):
        ctl[fn].append(float(np.median(at(arr, cpx, cpy))))
    if (c+1) % 50 == 0:
        print(f"  {c+1}/{len(pick)}", flush=True)

n = len(ctl["canny"])
print(f"\nn={n} images\n")
print("CANNY: distance from a landmark to the nearest edge, % of ear height")
print(f"  {'group':<24}{'landmarks':>11}{'control':>10}{'ratio':>8}")
cv_ = np.mean(ctl["canny"])
out = []
for nm in GROUP_NAMES:
    v = np.mean(acc[("canny", nm)])
    out.append((v/cv_, nm, v))
for r, nm, v in sorted(out):
    print(f"  {nm:<24}{v:>10.2f}%{cv_:>9.2f}%{r:>8.2f}"
          f"{'   <-- CV sees this' if r < 0.75 else ''}")
print("\nRIDGE FILTERS: median response at landmarks vs random inside the hull")
print(f"  {'group':<24}{'sato_ridge':>12}{'sato_valley':>13}{'frangi':>10}")
for nm in GROUP_NAMES:
    s = " ".join(f"{np.mean(acc[(f,nm)])/max(np.mean(ctl[f]),1e-12):>11.2f}"
                 for f in ("sato_ridge", "sato_valley", "frangi"))
    print(f"  {nm:<24}{s}")
print("\n(ratios: <1 means closer to an edge / stronger than random "
      "interior points)")
