"""Re-derive the outer-chain endpoints -- BUILT, TESTED, AND WITHDRAWN.

DO NOT RUN THIS ON THE CORPUS. It is kept because the measurements that
killed it are worth more than the tool, and because the idea is an obvious
one to have again.

Why it is wrong: the whole premise was that points 0 and 19 drift OUTSIDE
the ear, measured as distance to the SAM 3 rim. Rendering the masks shows
SAM systematically under-segments the ear, and worst at exactly those two
ends -- it drops the LOBE wholesale on a large share of rows, cutting across
above it, so a correctly placed point 19 reads as far outside. Signed
distances confirm it is systematic rather than random: 65-92% of ALL points
sit outside the mask, median +0.66% of ear height mid-chain rising to +3.24%
at point 0 and +2.04% at point 19.

So "distance to the rim" cannot separate an annotation that drifted outward
from a mask that stopped short -- they are the same measurement. And the
offending points are only 17.9-19.4% outside at the median (p90 24-28%),
which a dropped lobe produces on its own; just 3 rows in 9,843 exceed 40%.
There is no population of grossly misplaced endpoints to fix.

Run on the ends-only rows it did visible HARM: it dragged correct lobe
points up onto the truncated mask and put a kink in the chain. Adding a
continuity term barely moved it (0.212 -> 0.208 of ear length), which is how
it became clear the problem was the target, not the objective.

What IS real is the wholesale bad contour -- 85 rows, 0.86% of the corpus,
where the whole ring sits off the ear. Those were confirmed by eye and need
re-annotating or dropping. Endpoint repair is not the tool for them either,
since welding a correct endpoint onto a wrong chain fixes nothing.

Original docstring follows.

Re-derive the drifting outer-chain endpoints against the corpus convention.

The full-corpus screen shows the outer contour fails at its two ENDS --
point 0 at 4.83% of rows and point 19 at 6.01%, against 0.02% at point 12 in
the middle. Rendering clean rows shows WHY, and it is not a scheme defect:
point 0 belongs at the top-front where the helix meets the temple and point
19 at the bottom-front where the lobe meets the cheek, and 94.6% of the
corpus already places them there. Those two spots are simply the lowest
contrast in the chain, so they are the easiest to let drift outward.

So this does not invent a rule. It measures what the clean rows do and moves
only the rows that fall outside it, in two conservative steps:

  1. PROJECT onto the SAM 3 rim. Clean endpoints sit 0.02-0.03 of ear length
     from the mask boundary; flagged ones sit 0.08-0.10 OUTSIDE it. The
     outer contour is by definition on the silhouette, so this is not a
     judgement call.
  2. CLAMP along the long axis, but only past --sd standard deviations of
     the clean distribution. Clean spread is 0.077 (point 0) and 0.094
     (point 19) of ear length, which is real anatomical variation -- forcing
     every ear to the median would destroy it. Only genuine outliers move.

  3. KEEP THE CHAIN CONTINUOUS. Steps 1 and 2 alone put the endpoint on the
     rim at a position its NEIGHBOUR does not agree with, which trades a
     point in the hair for a visible kink -- clearly worse on the renders
     for point 19, whose lobe end often has an earring or merges into the
     jaw. The endpoint is therefore also pulled towards where the chain is
     already heading, extrapolated from the two points before it, so it
     continues the contour instead of jumping along the rim.

A point already inside the clean range keeps its along-axis position and is
merely pulled onto the rim.

THIS ONLY APPLIES TO ROWS WHOSE MID-CHAIN IS CLEAN. Of the 532 flagged rows
151 are ends-only, 296 have 1-3 mid-chain points off as well, and 85 are
wholesale bad contours. Repairing the ends of a wholesale bad contour just
welds a correct endpoint onto a wrong chain; those need re-annotating or
dropping, not fixing.

    python scripts/fix_chain_endpoints.py --rows runs/sam3_full/bad_rows.txt \
        --out runs/endpoint_fix --dry-run
"""
import argparse
import csv
import json
import pathlib
import sys

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.spatial import ConvexHull
from matplotlib.path import Path as MPath
from transformers import Sam3Model, Sam3Processor

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from scheme import GROUPS, GROUP_NAMES          # noqa: E402

# measured over 1,500 clean rows, in the mask frame, as a fraction of the
# ear's long-axis extent -- see qa_endpoint_convention.py
CONVENTION = {0: (0.331, 0.077), 19: (-0.507, 0.094)}
ENDPOINTS = (0, 19)


def ear_frame(mk, xy):
    """Origin, long axis and scale from the MASK, never the landmarks.

    Orientation is taken from the ascending-helix group against the lobe
    group rather than from point 0 alone, because point 0 is one of the two
    points this script exists to fix.
    """
    ys, xs = np.nonzero(mk)
    P = np.c_[xs, ys].astype(np.float64)
    cen = P.mean(0)
    _, _, Vt = np.linalg.svd(P - cen, full_matrices=False)
    e1 = Vt[0]
    if ((xy[0:4] - cen) @ e1).mean() < ((xy[14:20] - cen) @ e1).mean():
        e1 = -e1
    proj = (P - cen) @ e1
    return cen, e1, float(proj.max() - proj.min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", default="runs/endpoint_fix")
    ap.add_argument("--sd", type=float, default=2.0,
                    help="clamp along-axis only beyond this many sd of the "
                         "clean distribution")
    ap.add_argument("--continuity", type=float, default=1.0,
                    help="weight on continuing the chain's own direction; 0 "
                         "reproduces the kinking first version")
    ap.add_argument("--dry-run", action="store_true",
                    help="measure and report, write no landmarks")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = Sam3Processor.from_pretrained("facebook/sam3")
    model = Sam3Model.from_pretrained("facebook/sam3",
                                      dtype=torch.float16).to(dev).eval()
    L = np.load(ROOT / "data/manual/landmarks.npy")
    rows = list(csv.DictReader(open(ROOT / "data/manual/manifest.csv")))
    pick = [int(x) for x in pathlib.Path(a.rows).read_text().split()]
    print(f"{len(pick)} rows to re-derive  (dry-run={a.dry_run})")

    moved, skipped, recs = 0, 0, []
    newL = L.copy()
    for c, i in enumerate(pick):
        try:
            im = Image.open(ROOT / "data/manual" / rows[i]["image_file"]).convert("RGB")
        except Exception:
            skipped += 1; continue
        w, h = im.size
        xy = L[i] * np.array([w, h])
        if not np.isfinite(xy).all():
            skipped += 1; continue
        inp = proc(images=im, text="ear", return_tensors="pt").to(dev)
        with torch.inference_mode():
            o = model(**inp)
        try:
            r = proc.post_process_instance_segmentation(
                o, threshold=0.3, mask_threshold=0.5, target_sizes=[(h, w)])[0]
            masks = np.asarray(r["masks"].cpu())
        except Exception:
            skipped += 1; continue
        if len(masks) == 0:
            skipped += 1; continue
        yy, xx = np.mgrid[0:h, 0:w]
        try:
            hull = xy[ConvexHull(xy).vertices]
        except Exception:
            skipped += 1; continue
        inhull = MPath(hull).contains_points(
            np.c_[xx.ravel(), yy.ravel()]).reshape(h, w)
        best, mk = -1.0, None
        for m_ in masks:
            m_ = np.asarray(m_) > 0
            if m_.ndim != 2 or not m_.any():
                continue
            v = (m_ & inhull).sum() / max((m_ | inhull).sum(), 1)
            if v > best:
                best, mk = v, m_
        if mk is None:
            skipped += 1; continue
        lab, n = ndimage.label(mk)
        if n == 0:
            skipped += 1; continue
        sizes = ndimage.sum(mk, lab, range(1, n + 1))
        if sizes.max() / max(sizes.sum(), 1.0) < 0.80:
            skipped += 1; continue
        mk = lab == (1 + np.argmax(sizes))
        cen, e1, scale = ear_frame(mk, xy)
        if scale < 8:
            skipped += 1; continue
        e2 = np.array([-e1[1], e1[0]])
        edge = mk & ~ndimage.binary_erosion(mk)
        ey, ex = np.nonzero(edge)
        if len(ey) < 16:
            skipped += 1; continue
        B = np.c_[ex, ey].astype(np.float64)
        Bu = (B - cen) @ e1 / scale
        Bv = (B - cen) @ e2 / scale
        rec = {"idx": i}
        for j in ENDPOINTS:
            mu, sd = CONVENTION[j]
            lo, hi = mu - a.sd * sd, mu + a.sd * sd
            p = xy[j]
            pu = float((p - cen) @ e1) / scale
            pv = float((p - cen) @ e2) / scale
            # candidate rim points: same side of the long axis as the point,
            # so the chain cannot jump across the ear
            side = Bv * np.sign(pv) >= 0 if abs(pv) > 1e-6 else np.ones(len(Bv), bool)
            cand = side & (Bu >= lo) & (Bu <= hi)
            if not cand.any():                       # convention band empty
                cand = side
            if not cand.any():
                cand = np.ones(len(Bu), bool)
            tgt_u = float(np.clip(pu, lo, hi))
            # where the chain is already heading: extrapolate the last two
            # good points, so the endpoint CONTINUES the contour instead of
            # jumping to some other part of the rim
            nb = (xy[2], xy[1]) if j == 0 else (xy[17], xy[18])
            step = nb[1] - nb[0]
            pred = nb[1] + step
            pred_u = float((pred - cen) @ e1) / scale
            pred_v = float((pred - cen) @ e2) / scale
            # on the rim, within the band, agreeing with both the clamped
            # along-axis position and the chain's own direction
            cost = ((Bu[cand] - tgt_u) ** 2 + 0.25 * (Bv[cand] - pv) ** 2
                    + a.continuity * ((Bu[cand] - pred_u) ** 2
                                      + (Bv[cand] - pred_v) ** 2))
            k = np.nonzero(cand)[0][int(np.argmin(cost))]
            new = B[k]
            rec[f"old_u{j}"] = pu
            rec[f"new_u{j}"] = float((new - cen) @ e1) / scale
            rec[f"shift{j}"] = float(np.linalg.norm(new - p)) / scale
            rec[f"clamped{j}"] = bool(pu < lo or pu > hi)
            newL[i, j] = new / np.array([w, h])
        recs.append(rec)
        moved += 1
        if (c + 1) % 100 == 0:
            print(f"  {c+1}/{len(pick)}", flush=True)

    json.dump(recs, open(out / "endpoint_fix.json", "w"))
    sh = {j: np.array([r[f"shift{j}"] for r in recs]) for j in ENDPOINTS}
    cl = {j: np.mean([r[f"clamped{j}"] for r in recs]) for j in ENDPOINTS}
    print(f"\nre-derived {moved} rows, skipped {skipped} (no usable mask)")
    for j in ENDPOINTS:
        print(f"  point {j:>2}: median shift {np.median(sh[j]):.3f} of ear length, "
              f"p90 {np.percentile(sh[j],90):.3f}, "
              f"along-axis clamped on {100*cl[j]:.0f}% of rows")
    if not a.dry_run:
        np.save(out / "landmarks_fixed.npy", newL)
        print(f"\nwrote {out/'landmarks_fixed.npy'}  "
              f"(data/manual/landmarks.npy is NOT modified)")
    else:
        print("\ndry run: no landmarks written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
