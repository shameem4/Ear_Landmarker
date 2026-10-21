"""What do the classical filters actually draw on an ear?"""
import sys, csv, pathlib, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
import cv2
from skimage.filters import sato
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from scheme import GROUPS, GROUP_NAMES  # was ear3d.scheme; identical, moved at the repo split
R = pathlib.Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/Ear_Landmarker")
W = pathlib.Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/wrap_work")
g = {n: list(range(lo, hi)) for n, (lo, hi) in zip(GROUP_NAMES, GROUPS)}
HELIX = g["ascending_helix"]+g["descending_helix"]+g["helix"]+g["ear_lobe"]
ANTI = g["ascending_inner_helix"]+g["descending_inner_helix"]+g["inner_helix"]
OTHER = g["tragus"]+g["antitragus"]+g["concha"]+g["inferior_crus"]+g["superior_crus"]
L = np.load(R/"data/manual/landmarks.npy")
rows = list(csv.DictReader(open(R/"data/manual/manifest.csv")))
rng = np.random.default_rng(3)
pick = [int(i) for i in rng.choice(len(rows), 4, replace=False)]
fig, ax = plt.subplots(4, 4, figsize=(13, 13), dpi=120)
for r, i in enumerate(pick):
    im = Image.open(R/"data/manual"/rows[i]["image_file"]).convert("RGB")
    w, h = im.size; xy = L[i]*np.array([w, h])
    eh = float(xy[:, 1].max()-xy[:, 1].min()); s = 2.0*eh/100.0
    gray = np.asarray(im.convert("L"), float)/255.0
    gb = cv2.GaussianBlur((gray*255).astype(np.uint8), (0, 0), max(s*.6, .8))
    med = float(np.median(gb))
    edges = cv2.Canny(gb, int(max(0, .66*med)), int(min(255, 1.33*med)))
    val = sato(gray, sigmas=[max(s*.6, 1.), s, s*1.6], black_ridges=True)
    for c, (nm, img, cmap) in enumerate((
            ("image", np.asarray(im), None), ("Canny", edges, "gray"),
            ("sato valley", val, "magma"), ("overlay", np.asarray(im), None))):
        a = ax[r, c]
        a.imshow(img, cmap=cmap); a.set_axis_off()
        if c == 3:
            a.contour(edges > 0, levels=[.5], colors="#17d4ff", linewidths=.7)
            for idx, col, lab in ((HELIX, "#ff2d2d", "helix/lobe"),
                                  (ANTI, "#2dff6a", "antihelix"),
                                  (OTHER, "#ffd21e", "tragus/concha/crura")):
                a.plot(xy[idx, 0], xy[idx, 1], ".", ms=4, c=col, label=lab)
            if r == 0:
                a.legend(fontsize=6, loc="lower right", framealpha=.75)
        if r == 0:
            a.set_title(nm, fontsize=10)
fig.suptitle("Classical CV on the ear: Canny edges and dark-ridge (valley) response",
             fontsize=13)
fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(W/"tools/cv_cases.png")
print("wrote", W/"tools/cv_cases.png")
