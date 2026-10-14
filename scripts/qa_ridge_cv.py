"""Distance from each landmark to the nearest TRACED STRUCTURE, not the
response at its own pixel.

The percentile version understated sato badly: ridges are 1-2 px wide, so an
annotation sitting one pixel off the crest reads as no response at all, while
the picture plainly shows the filter drawing the antihelix and both crura.
The question for tracing is "is there a line near this point", which is the
same question the Canny column answered -- so measure it the same way.

Ridge pixels are the top-q% of the in-ear response, which adapts per image
instead of fixing a threshold across very different exposures.

Three channels, each of which covers groups the others miss:

  base    sato on intensity          the no-model default
  clahe   sato on CLAHE intensity    the interior: tragus, canal, antitragus
  depth   sato on DA3 depth          the rim and inner helix

`depth` needs maps from qa_depth_dump.py and is reported with the gate that
tells you when to believe it -- see RELIABILITY below.

    python scripts/qa_ridge_cv.py --channel base clahe
    python scripts/qa_ridge_cv.py --channel depth --depth-dir /tmp/depth150

READ THE HULL-EDGE COLUMN BEFORE BELIEVING A RESULT. The ratio is against
random in-hull controls, and the four outer groups DEFINE the convex hull,
so they sit on its edge (0.04 of the control distance). Any channel whose
hits are a broad region touching that edge scores ~0 for them no matter
what the region contains, which is how a blurred-intensity "shadow" channel
once appeared to trace the ascending helix when it had in fact found the
hair. A ratio is only meaningful for a group whose hull-edge figure is not
also tiny; the 'edge' column prints it so the trap stays visible.
"""
import argparse
import collections
import csv
import pathlib
import sys

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
from scheme import GROUPS, GROUP_NAMES  # noqa: E402
from qa_common import padding_mask            # noqa: E402

CHANNELS = ("base", "clahe", "depth")


def load_depth(depth_dir, idx, w, h):
    f = pathlib.Path(depth_dir) / f"{idx}.npz"
    if not f.exists():
        return None
    return cv2.resize(np.load(f)["depth"].astype(np.float32), (w, h),
                      interpolation=cv2.INTER_CUBIC)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--q", type=float, default=85.0,
                   help="top (100-q)%% of the in-ear response is a structure")
    p.add_argument("--channel", nargs="+", default=["base"], choices=CHANNELS)
    p.add_argument("--depth-dir", default=None,
                   help="npz maps from qa_depth_dump.py; required for depth")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if "depth" in a.channel and not a.depth_dir:
        p.error("--channel depth needs --depth-dir")

    g = {n: list(range(lo, hi)) for n, (lo, hi) in zip(GROUP_NAMES, GROUPS)}
    L = np.load(ROOT / "data/manual/landmarks.npy")
    rows = list(csv.DictReader(open(ROOT / "data/manual/manifest.csv")))
    rng = np.random.default_rng(a.seed)
    pick = rng.choice(len(rows), size=min(a.n, len(rows)), replace=False)

    D = collections.defaultdict(list)
    C = collections.defaultdict(list)
    E = collections.defaultdict(list)          # distance to the hull edge
    sep = collections.defaultdict(list)        # depth: ear vs its surround
    rim = collections.defaultdict(list)        # depth: per-image rim ratio
    ctl_edge = []
    blank = collections.Counter()
    missing = 0
    for c, i in enumerate(pick):
        i = int(i)
        im = Image.open(ROOT / "data/manual" / rows[i]["image_file"]).convert("RGB")
        w, h = im.size
        xy = L[i] * np.array([w, h])
        if not np.isfinite(xy).all():
            continue
        eh = float(xy[:, 1].max() - xy[:, 1].min())
        if eh < 24:
            continue
        hull = xy[ConvexHull(xy).vertices]
        yy, xx = np.mgrid[0:h, 0:w]
        inh = MPath(hull).contains_points(
            np.c_[xx.ravel(), yy.ravel()]).reshape(h, w)
        if inh.sum() < 50:
            continue
        gray = np.asarray(im.convert("L"), float) / 255.0
        valid = ~padding_mask(gray)
        inh &= valid
        if inh.sum() < 50:
            continue
        s = 2.0 * eh / 100.0
        sg = [max(s * .6, 1.), s, s * 1.6]
        maps = {}
        if "base" in a.channel:
            maps["base"] = sato(gray, sigmas=sg, black_ridges=True)
        if "clahe" in a.channel:
            u8 = (gray * 255).astype(np.uint8)
            cl = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(u8)
            maps["clahe"] = sato(cl / 255.0, sigmas=sg, black_ridges=True)
        if "depth" in a.channel:
            dep = load_depth(a.depth_dir, i, w, h)
            if dep is None:
                missing += 1
            else:
                # normalise over the EAR, so the head and background range
                # cannot set the scale
                d0 = dep[inh]
                maps["depth"] = sato((dep - d0.mean()) / (d0.std() + 1e-8),
                                     sigmas=sg, black_ridges=True)
                ring = ndimage.binary_dilation(
                    inh, iterations=max(2, int(.12 * eh))) & ~inh
                sep["all"].append(float(abs(dep[inh].mean() - dep[ring].mean())
                                        / (dep[inh].std() + 1e-8)))

        px = np.clip(np.round(xy[:, 0]).astype(int), 0, w - 1)
        py = np.clip(np.round(xy[:, 1]).astype(int), 0, h - 1)
        iy, ix = np.nonzero(inh)
        sel = rng.choice(len(iy), size=min(400, len(iy)), replace=False)

        # how far each group sits from the hull edge, the artefact guard
        edge = inh & ~ndimage.binary_erosion(inh)
        de = ndimage.distance_transform_edt(~edge) / eh * 100.0
        for nm, idx in g.items():
            E[nm].append(float(np.mean(de[py[idx], px[idx]])))
        ctl_edge.append(float(np.mean(de[iy[sel], ix[sel]])))

        outer = (g["helix"] + g["ascending_helix"]
                 + g["descending_helix"] + g["ear_lobe"])
        for k, val in maps.items():
            hit = (val >= np.percentile(val[inh], a.q)) & valid
            if not hit.any():
                blank[k] += 1
                continue
            dr = ndimage.distance_transform_edt(~hit) / eh * 100.0
            for nm, idx in g.items():
                D[(k, nm)].append(float(np.mean(dr[py[idx], px[idx]])))
            ctl = float(np.mean(dr[iy[sel], ix[sel]]))
            C[k].append(ctl)
            if k == "depth":
                rim["all"].append(float(np.mean(dr[py[outer], px[outer]]))
                                  / max(ctl, 1e-6))
        if (c + 1) % 50 == 0:
            print(f"  {c+1}/{len(pick)}", flush=True)

    chans = [k for k in a.channel if C[k]]
    n = len(ctl_edge)
    print(f"\nn={n} images   top {100-a.q:.0f}% of the in-ear response = "
          f"a structure")
    if missing:
        print(f"  {missing} images had no depth map and were skipped for it")
    for k in chans:
        if blank[k]:
            print(f"  {k}: {blank[k]} blank")
    ce = float(np.mean(ctl_edge))
    print("\ndistance to the nearest traced structure, ratio to an in-hull "
          "control (<0.70 = traced)")
    print("'edge' is the group's own distance to the hull edge on the same "
          f"scale (control {ce:.2f});")
    print("a group at edge<0.25 is ON the hull and its ratios cannot be "
          "trusted for region-like channels.")
    print(f"\n  {'group':<24}" + "".join(f"{k:>9}" for k in chans)
          + f"{'edge':>9}")
    out = []
    for nm in GROUP_NAMES:
        r = {k: np.mean(D[(k, nm)]) / np.mean(C[k]) for k in chans}
        out.append((r[chans[0]], nm, r, np.mean(E[nm]) / ce))
    for _, nm, r, e in sorted(out, key=lambda t: -t[0]):
        tag = "   <-- ON the hull" if e < .25 else ""
        print(f"  {nm:<24}" + "".join(f"{r[k]:>9.2f}" for k in chans)
              + f"{e:>9.2f}{tag}")

    if rim["all"]:
        rr = np.array(rim["all"])
        ss = np.array(sep["all"][:len(rr)])
        ok = rr < .70
        print(f"\nRELIABILITY of the depth channel, per image")
        print(f"  traced  (rim ratio < 0.70): {100*ok.mean():5.1f}%")
        print(f"  useless (rim ratio > 1.00): {100*np.mean(rr > 1):5.1f}%")
        print(f"  ear-vs-surround depth separation: {ss[ok].mean():.2f} sd "
              f"where traced, {ss[~ok].mean():.2f} sd where not")
        print(f"  -> gate on that separation; the channel knows when it failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
