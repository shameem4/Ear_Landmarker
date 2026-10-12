"""What do the classical filters actually draw on an ear?

Columns are the three channels qa_ridge_cv.py scores, plus Canny for the
record: it produced a completely blank map on a smooth, evenly-lit ear and
is kept only so that result stays visible.

LOOK AT THIS BEFORE BELIEVING A RATIO. A broad region that happens to cover
the hair scores beautifully against in-hull controls while tracing nothing;
that is how a blurred-intensity channel once appeared to find the ascending
helix. The hit mask is drawn so the difference between a thin line on the
anatomy and a blob on the hair is visible at a glance, and each panel
prints the fraction of its hits lying outside the landmark hull.

    python scripts/qa_ridge_show.py --out /tmp/cv_cases.png
    python scripts/qa_ridge_show.py --depth-dir /tmp/depth150 --out ...
"""
import argparse
import csv
import pathlib
import sys

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from PIL import Image                                # noqa: E402
from scipy.spatial import ConvexHull                 # noqa: E402
from matplotlib.path import Path as MPath            # noqa: E402
from skimage.filters import sato                     # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from ear3d.scheme import GROUPS, GROUP_NAMES          # noqa: E402

g = {n: list(range(lo, hi)) for n, (lo, hi) in zip(GROUP_NAMES, GROUPS)}
HELIX = g["ascending_helix"] + g["descending_helix"] + g["helix"] + g["ear_lobe"]
ANTI = g["ascending_inner_helix"] + g["descending_inner_helix"] + g["inner_helix"]
OTHER = (g["tragus"] + g["antitragus"] + g["concha"]
         + g["inferior_crus"] + g["superior_crus"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/tmp/cv_cases.png")
    p.add_argument("--depth-dir", default=None)
    p.add_argument("--rows", type=int, default=4)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--q", type=float, default=85.0)
    p.add_argument("--indices", nargs="*", type=int, default=None)
    a = p.parse_args()

    L = np.load(ROOT / "data/manual/landmarks.npy")
    rows = list(csv.DictReader(open(ROOT / "data/manual/manifest.csv")))
    rng = np.random.default_rng(a.seed)
    pick = (a.indices if a.indices else
            [int(i) for i in rng.choice(len(rows), a.rows, replace=False)])
    cols = ["image", "Canny", "sato base", "sato clahe"]
    if a.depth_dir:
        cols.append("sato depth")
    fig, ax = plt.subplots(len(pick), len(cols),
                           figsize=(3.3 * len(cols), 3.3 * len(pick)), dpi=110)
    ax = np.atleast_2d(ax)
    for r, i in enumerate(pick):
        im = Image.open(ROOT / "data/manual" / rows[i]["image_file"]).convert("RGB")
        w, h = im.size
        xy = L[i] * np.array([w, h])
        eh = float(xy[:, 1].max() - xy[:, 1].min())
        s = 2.0 * eh / 100.0
        sg = [max(s * .6, 1.), s, s * 1.6]
        gray = np.asarray(im.convert("L"), float) / 255.0
        u8 = (gray * 255).astype(np.uint8)
        gb = cv2.GaussianBlur(u8, (0, 0), max(s * .6, .8))
        med = float(np.median(gb))
        edges = cv2.Canny(gb, int(max(0, .66 * med)), int(min(255, 1.33 * med)))
        cl = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(u8) / 255.0
        hull = xy[ConvexHull(xy).vertices]
        yy, xx = np.mgrid[0:h, 0:w]
        inh = MPath(hull).contains_points(
            np.c_[xx.ravel(), yy.ravel()]).reshape(h, w)
        maps = {"sato base": sato(gray, sigmas=sg, black_ridges=True),
                "sato clahe": sato(cl, sigmas=sg, black_ridges=True)}
        if a.depth_dir:
            f = pathlib.Path(a.depth_dir) / f"{i}.npz"
            if f.exists():
                dep = cv2.resize(np.load(f)["depth"].astype(np.float32), (w, h),
                                 interpolation=cv2.INTER_CUBIC)
                d0 = dep[inh]
                maps["sato depth"] = sato((dep - d0.mean()) / (d0.std() + 1e-8),
                                          sigmas=sg, black_ridges=True)
        for c, nm in enumerate(cols):
            axc = ax[r, c]
            axc.set_axis_off()
            if nm == "image":
                axc.imshow(np.asarray(im))
                for idx, col, lab in ((HELIX, "#ff2d2d", "helix/lobe"),
                                      (ANTI, "#2dff6a", "antihelix"),
                                      (OTHER, "#ffd21e", "tragus/concha/crura")):
                    axc.plot(xy[idx, 0], xy[idx, 1], ".", ms=4, c=col, label=lab)
                axc.plot(*hull[list(range(len(hull))) + [0]].T, "-", lw=.7,
                         c="#17d4ff")
                if r == 0:
                    axc.legend(fontsize=6, loc="lower right", framealpha=.75)
                axc.set_title(f"row {i}", fontsize=9)
                continue
            if nm == "Canny":
                axc.imshow(gray, cmap="gray")
                ov = np.zeros((h, w, 4))
                ov[edges > 0] = (1, .2, .1, .6)
                axc.imshow(ov)
                axc.set_title(f"Canny ({'BLANK' if not edges.any() else 'ok'})",
                              fontsize=9)
                continue
            if nm not in maps:
                axc.set_title(f"{nm}: no map", fontsize=9)
                continue
            val = maps[nm]
            hit = val >= np.percentile(val[inh], a.q)
            axc.imshow(gray, cmap="gray")
            ov = np.zeros((h, w, 4))
            ov[hit] = (1, .2, .1, .55)
            axc.imshow(ov)
            for idx, col in ((HELIX, "#ff2d2d"), (ANTI, "#2dff6a"),
                             (OTHER, "#ffd21e")):
                axc.plot(xy[idx, 0], xy[idx, 1], ".", ms=3.5, c=col)
            out = float((hit & ~inh).sum()) / max(1, hit.sum())
            axc.set_title(f"{nm}  (outside hull {out:.0%})", fontsize=9)
    fig.suptitle("Traced structures per channel -- a thin line on the anatomy "
                 "is a trace, a blob on the hair is not", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(a.out)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
