"""Per-image agreement between the GT landmarks and the traced structures.

qa_ridge_cv.py answers "do the annotations sit on real image structure ON
AVERAGE". That is the wrong shape for auditing a corpus: a corpus-wide
ratio of 0.55 is consistent with every image being mediocre and with most
being perfect and a few being wrong. This scores each image on its own and
reports the DISTRIBUTION, so the disagreeing images can be pulled out and
looked at.

Each group is scored on the channel that covers it, measured over 150
images and recorded in RESULTS.md, part 6:

  depth  the rim and the inner helix   clahe  the interior
  base   the crura and the lobe        (concha is scored by nothing)

An image's agreement is the mean over scored landmarks of the distance to
the nearest traced structure, divided by the same image's in-hull control.
Below 0.70 the annotations sit on structure; above 1.0 they are no closer
to a traced edge than a random point inside the ear, which means either a
bad annotation or a channel that failed on that image.

CAVEAT, do not remove: the rim groups sit ON the landmark convex hull
(0.04 of the control distance), so a ratio is only evidence for them when
the channel draws a thin line rather than a broad region. That holds for
the sato channels here and was checked by eye; it did NOT hold for a
blurred-intensity channel that scored well by finding the hair. See the
hull-edge section of RESULTS.md part 6 before adding a channel.

    python scripts/qa_agreement.py --n 500 --depth-dir /tmp/depth500 \
        --out /tmp/agreement.csv --jobs 12
"""
import argparse
import csv
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import cv2
from PIL import Image
from scipy import ndimage
from scipy.spatial import ConvexHull
from matplotlib.path import Path as MPath
from skimage.filters import sato

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from scheme import GROUPS, GROUP_NAMES          # noqa: E402
from qa_common import padding_mask                    # noqa: E402

G = {n: list(range(lo, hi)) for n, (lo, hi) in zip(GROUP_NAMES, GROUPS)}

# which channel covers which group, from the n=150 sweep in RESULTS.md part 6
# concha is in neither table: nothing traces it, so it is never scored
BEST = {
    "helix": "depth", "descending_helix": "depth", "inner_helix": "depth",
    "descending_inner_helix": "depth",
    "ascending_helix": "base", "ear_lobe": "base", "inferior_crus": "base",
    "ascending_inner_helix": "clahe", "superior_crus": "clahe",
    "antitragus": "clahe", "tragus": "clahe", "canal": "clahe",
}
# Pure-CV assignment, for running the whole corpus without a GPU. The four
# groups depth covers best fall back to the better of base/clahe for each,
# which is NOT clahe across the board: base wins the helix (0.67 vs 0.80),
# the descending helix (0.62 vs 0.80) and the antihelix (0.55 vs 0.66),
# and only the inner helix prefers clahe (0.76 vs 0.79).
BEST_CV = dict(BEST, helix="base", descending_helix="base",
               descending_inner_helix="base", inner_helix="clahe")
_ROWS = None
_L = None


def _init():
    global _ROWS, _L
    _ROWS = list(csv.DictReader(open(ROOT / "data/manual/manifest.csv")))
    _L = np.load(ROOT / "data/manual/landmarks.npy")


def score(args):
    i, depth_dir, q, best = args
    if _ROWS is None:
        _init()
    try:
        im = Image.open(ROOT / "data/manual" / _ROWS[i]["image_file"]).convert("RGB")
    except Exception:
        return None
    w, h = im.size
    xy = _L[i] * np.array([w, h])
    if not np.isfinite(xy).all():
        return None
    eh = float(xy[:, 1].max() - xy[:, 1].min())
    if eh < 24:
        return None
    hull = xy[ConvexHull(xy).vertices]
    yy, xx = np.mgrid[0:h, 0:w]
    inh = MPath(hull).contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(h, w)
    if inh.sum() < 50:
        return None
    gray = np.asarray(im.convert("L"), float) / 255.0
    u8 = (gray * 255).astype(np.uint8)
    # a letterbox seam is the sharpest ridge in the frame; exclude it or
    # the filters trace the padding and every landmark near it scores well
    valid = ~padding_mask(gray)
    inh &= valid
    if inh.sum() < 50:
        return None
    s = 2.0 * eh / 100.0
    sg = [max(s * .6, 1.), s, s * 1.6]
    maps = {
        "base": sato(gray, sigmas=sg, black_ridges=True),
        "clahe": sato(cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
                      .apply(u8) / 255.0, sigmas=sg, black_ridges=True),
    }
    depth_sep = np.nan
    f = pathlib.Path(depth_dir) / f"{i}.npz" if depth_dir else None
    if f is not None and f.exists():
        dep = cv2.resize(np.load(f)["depth"].astype(np.float32), (w, h),
                         interpolation=cv2.INTER_CUBIC)
        d0 = dep[inh]
        maps["depth"] = sato((dep - d0.mean()) / (d0.std() + 1e-8),
                             sigmas=sg, black_ridges=True)
        ring = ndimage.binary_dilation(
            inh, iterations=max(2, int(.12 * eh))) & ~inh
        depth_sep = float(abs(dep[inh].mean() - dep[ring].mean())
                          / (dep[inh].std() + 1e-8))

    px = np.clip(np.round(xy[:, 0]).astype(int), 0, w - 1)
    py = np.clip(np.round(xy[:, 1]).astype(int), 0, h - 1)
    rng = np.random.default_rng(i)
    iy, ix = np.nonzero(inh)
    sel = rng.choice(len(iy), size=min(400, len(iy)), replace=False)

    dist, ctrl = {}, {}
    for k, val in maps.items():
        hit = (val >= np.percentile(val[inh], q)) & valid
        if not hit.any():
            continue
        d = ndimage.distance_transform_edt(~hit) / eh * 100.0
        dist[k] = d[py, px]
        ctrl[k] = float(np.mean(d[iy[sel], ix[sel]]))

    num, den, per = [], [], {}
    for nm, ch in best.items():
        if ch not in dist:
            ch = "clahe" if "clahe" in dist else next(iter(dist), None)
        if ch is None:
            continue
        v = float(np.mean(dist[ch][G[nm]]))
        per[nm] = v / max(ctrl[ch], 1e-6)
        num.append(v / max(ctrl[ch], 1e-6) * len(G[nm]))
        den.append(len(G[nm]))
    if not den:
        return None
    return dict(idx=i, agree=float(sum(num) / sum(den)), ear_px=eh,
                depth_sep=depth_sep, has_depth=int("depth" in dist),
                **{f"g_{k}": v for k, v in per.items()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--depth-dir", default=None,
                   help="omit for a pure-CV run: base+clahe only, no GPU, "
                        "and the rim groups fall back per BEST_CV")
    p.add_argument("--out", default="/tmp/agreement.csv")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--q", type=float, default=85.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rows", default=None,
                   help="file of row indices to score instead of a random "
                        "sample -- used to score the gated proper-ear subset")
    p.add_argument("--keep-padding", action="store_true",
                   help="do not mask letterbox padding (for reproducing the old, contaminated numbers)")
    a = p.parse_args()
    rows = list(csv.DictReader(open(ROOT / "data/manual/manifest.csv")))
    if a.rows:
        pick = [int(x) for x in
                pathlib.Path(a.rows).read_text().replace(",", " ").split()]
    else:
        pick = [int(i) for i in np.random.default_rng(a.seed)
                .choice(len(rows), size=min(a.n, len(rows)), replace=False)]
    best = BEST if a.depth_dir else BEST_CV
    if not a.depth_dir:
        print("pure-CV run: base+clahe only, no depth channel")
    out = []
    with ProcessPoolExecutor(max_workers=a.jobs, initializer=_init) as ex:
        for n, r in enumerate(ex.map(score,
                                     [(i, a.depth_dir, a.q, best) for i in pick],
                                     chunksize=4), 1):
            if r:
                out.append(r)
            if n % 100 == 0:
                print(f"  {n}/{len(pick)}", flush=True)
    keys = ["idx", "agree", "ear_px", "depth_sep", "has_depth"] + \
           [f"g_{k}" for k in best]
    with open(a.out, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        wtr.writeheader()
        wtr.writerows(out)
    ag = np.array([r["agree"] for r in out])
    print(f"\nn={len(ag)} images   per-image agreement with GT "
          f"(<0.70 on structure, >1.0 no better than random)\n")
    for q in (10, 25, 50, 75, 90, 95):
        print(f"  p{q:<3} {np.percentile(ag, q):.2f}")
    print(f"\n  on structure (<0.70): {100*np.mean(ag < .70):5.1f}%")
    print(f"  marginal (0.70-1.00): {100*np.mean((ag >= .70) & (ag <= 1)):5.1f}%")
    print(f"  DISAGREES    (>1.00): {100*np.mean(ag > 1.0):5.1f}%")
    hd = np.array([r["has_depth"] for r in out], bool)
    print(f"\n  {hd.sum()} of {len(out)} had a depth map")
    print(f"\nper group, median over images")
    for nm in sorted(best, key=lambda n: -np.nanmedian(
            [r.get(f"g_{n}", np.nan) for r in out])):
        v = np.nanmedian([r.get(f"g_{nm}", np.nan) for r in out])
        print(f"  {nm:<24}{v:>8.2f}  via {best[nm]}")
    print(f"\nworst 15 by agreement (look at these):")
    for r in sorted(out, key=lambda r: -r["agree"])[:15]:
        print(f"  row {r['idx']:<7} agree {r['agree']:.2f}  "
              f"ear {r['ear_px']:.0f}px  depth_sep {r['depth_sep']:.2f}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
