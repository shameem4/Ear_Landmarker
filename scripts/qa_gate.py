"""Pick the ears SAM 3 is ENTITLED to judge, then judge the GT on those.

The session's central finding is that a segmenter delineates visible ear
while the annotation runs through hair, so SAM and the annotation disagree
by construction wherever something occludes the rim. That makes SAM
useless as a corpus-wide auditor -- but only because it is being asked
about rows outside its competence.

So scope it. On rows with no occluder, SAM's deficit is a pure RIND (0.0%
deep intrusion, 91% coverage, measured), i.e. its boundary IS the rim.
There, a GT point far from that boundary is genuinely suspect.

The gate uses IMAGE and INSTRUMENT properties only, never the GT. Gating
on GT irregularity and then judging the GT would be circular; irregularity
is kept as a SECOND, independent detector so the two can be cross-checked.

  occlusion   covered >= 0.85. Measured against fresh intrusion depth on
              348 rows: r = -0.852, and this threshold excludes 100% of
              rows whose deep intrusion exceeds 5% of the ear.
  mask sanity bloat in [0.7, 1.6] -- the mask is neither a sliver nor the
              whole head
  focus       variance of Laplacian on the ear region, measured after
              resizing so ear height = 128px, so it is scale-free
  size        ear at least 48px tall in the original

GT irregularity, computed for every row and used only as the second
opinion: spacing coefficient of variation along each strip, and the
worst turning angle (a kink).

    python scripts/qa_gate.py runs/gate
"""
import csv
import json
import pathlib
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ear3d.scheme import STRIPS, STRIP_NAMES

OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
COV_MIN, BLOAT_LO, BLOAT_HI, EH_MIN = 0.85, 0.7, 1.6, 48.0

L = np.load(ROOT / "data/manual/landmarks.npy")
rows = list(csv.DictReader(open(ROOT / "data/manual/manifest.csv")))
S = {r["idx"]: r for r in json.load(open(ROOT / "runs/sam3_full/sam2_silhouette.json"))}
OUTER = list(range(20))


def focus_and_size(i):
    """Scale-free sharpness on the ear region, and the ear's pixel height."""
    try:
        im = Image.open(ROOT / "data/manual" / rows[i]["image_file"]).convert("L")
    except Exception:
        return np.nan, np.nan
    g = L[i]
    if not np.isfinite(g).all():
        return np.nan, np.nan
    w, h = im.size
    xy = g[:20] * np.array([w, h])
    eh = float(xy[:, 1].max() - xy[:, 1].min())
    if eh < 8:
        return np.nan, eh
    x0, y0 = np.clip([xy[:, 0].min(), xy[:, 1].min()], 0, None)
    x1 = min(xy[:, 0].max(), w); y1 = min(xy[:, 1].max(), h)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return np.nan, eh
    crop = im.crop((int(x0), int(y0), int(x1), int(y1)))
    k = 128.0 / eh                      # normalise scale before measuring blur
    nw, nh = max(8, round(crop.width * k)), max(8, round(crop.height * k))
    a = np.asarray(crop.resize((nw, nh), Image.LANCZOS), float) / 255.0
    return float(ndimage.laplace(a).var()), eh


def irregularity(i):
    """GT-intrinsic defects: uneven spacing, and the worst kink."""
    g = L[i]
    if not np.isfinite(g).all():
        return np.nan, np.nan
    eh = g[:20, 1].max() - g[:20, 1].min()
    if eh <= 0:
        return np.nan, np.nan
    cv, kink = [], []
    for lo, hi in STRIPS:
        q = g[lo:hi]
        d = np.linalg.norm(np.diff(q, axis=0), axis=1)
        if len(d) > 1 and d.mean() > 1e-9:
            cv.append(d.std() / d.mean())
        if hi - lo >= 3:
            v = np.diff(q, axis=0)
            n = np.linalg.norm(v, axis=1, keepdims=True)
            u = v / np.maximum(n, 1e-9)
            cosang = np.clip((u[:-1] * u[1:]).sum(1), -1, 1)
            kink.append(float(np.degrees(np.arccos(cosang)).max()))
    return (float(np.mean(cv)) if cv else np.nan,
            float(np.max(kink)) if kink else np.nan)


rec = []
for i in range(len(rows)):
    s = S.get(i)
    f, eh = focus_and_size(i)
    cv, kink = irregularity(i)
    d = dict(idx=i, eh=eh, focus=f, cv=cv, kink=kink,
             covered=s["covered"] if s else np.nan,
             bloat=s["bloat"] if s else np.nan,
             outer_abs=s["outer_abs"] if s else np.nan)
    if s:
        per = np.abs(np.array(s["per"])[OUTER])
        d["gt_med"] = float(np.median(per))
        d["gt_p90"] = float(np.percentile(per, 90))
        d["gt_max"] = float(per.max())
        d["gt_n15"] = int((per > 15).sum())
    rec.append(d)
    if (i + 1) % 2000 == 0:
        print(f"  {i+1}/{len(rows)}", flush=True)

json.dump(rec, open(OUT / "gate.json", "w"))
A = {k: np.array([r.get(k, np.nan) for r in rec], float) for k in
     ("eh", "focus", "cv", "kink", "covered", "bloat", "gt_med", "gt_p90",
      "gt_max", "gt_n15")}

has = np.isfinite(A["covered"])
foc_thr = np.nanpercentile(A["focus"][has], 10)        # drop the blurriest 10%
gate = (has & (A["covered"] >= COV_MIN)
        & (A["bloat"] >= BLOAT_LO) & (A["bloat"] <= BLOAT_HI)
        & (A["eh"] >= EH_MIN) & (A["focus"] >= foc_thr))
print(f"\n{len(rec)} rows, {has.sum()} with a SAM mask")
for nm, m in (("covered >= 0.85", has & (A["covered"] >= COV_MIN)),
              ("bloat in range", has & (A["bloat"] >= BLOAT_LO) & (A["bloat"] <= BLOAT_HI)),
              ("ear >= 48px", has & (A["eh"] >= EH_MIN)),
              ("in focus (top 90%)", has & (A["focus"] >= foc_thr))):
    print(f"  {nm:<22} {m.sum():5d}  ({100*m.sum()/has.sum():.0f}% of masked)")
print(f"  {'ALL (proper ears)':<22} {gate.sum():5d}  "
      f"({100*gate.sum()/len(rec):.0f}% of the corpus)")

print(f"\nGT distance to the SAM rim, outer chain, % of ear height")
print(f"{'':22}{'median':>9}{'p90':>9}{'p99':>9}")
for nm, m in (("proper ears", gate), ("everything else", has & ~gate)):
    print(f"{nm:<22}{np.median(A['gt_med'][m]):>9.2f}"
          f"{np.percentile(A['gt_med'][m],90):>9.2f}"
          f"{np.percentile(A['gt_med'][m],99):>9.2f}")
print("\n(on proper ears the spread IS the measurement error, so its tail"
      "\n is a real candidate list rather than a list of hard images.)")
