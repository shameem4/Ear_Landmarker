"""GT correctness for the gated "proper ear" subset, from three sources.

No single instrument covers all 55 points, and the session established why
each one fails outside its competence. So score each strip with the
instrument that can actually see it, and keep the sources independent so
agreement between them means something:

  outer chain 0-19   SAM 3 silhouette distance. On gated rows the mask
                     deficit is a pure rind, so its boundary IS the rim
                     and this resolves to ~1% of ear height.
  interior 20-54     sato ridge agreement (qa_agreement.py, pure-CV
                     channels), as a ratio to the image's own in-hull
                     control. Below 0.70 the points sit on traced
                     structure; above 1.0 they are no closer to an edge
                     than a random interior point.
  all 55             GT-intrinsic irregularity: worst turning angle and
                     spacing coefficient of variation. Uses no image at
                     all, so it cannot share a failure mode with either.

CONCHA IS NOT SCORED BY ANYTHING. It is in neither ridge table and it is
not on the silhouette. 15 of 55 points therefore have no check here, and
that is a gap to state rather than paper over.

A row is flagged when at least TWO of the three independent signals put it
in their tail, which is what makes a flag evidence rather than one
instrument's noise.

    python scripts/qa_gt_score.py runs/gate
"""
import csv
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = pathlib.Path(sys.argv[1])
OUTER = list(range(20))

S = {r["idx"]: r for r in json.load(open(ROOT / "runs/sam3_full/sam2_silhouette.json"))}
G = {r["idx"]: r for r in json.load(open(OUT / "gate.json"))}
A = {}
for r in csv.DictReader(open(OUT / "agreement.csv")):
    A[int(r["idx"])] = {k: (float(v) if v not in ("", "nan") else np.nan)
                        for k, v in r.items() if k != "idx"}
pick = [int(x) for x in (OUT / "proper_rows.txt").read_text().split()]
print(f"{len(pick)} proper ears; agreement for {len(set(pick) & set(A))}")

rec = []
for i in pick:
    s, g, a = S.get(i), G[i], A.get(i)
    if s is None or a is None:
        continue
    per = np.abs(np.array(s["per"])[OUTER])
    rec.append(dict(idx=i,
                    sil=float(np.median(per)), sil_max=float(per.max()),
                    agree=a["agree"],
                    kink=g["kink"], cv=g["cv"]))
V = {k: np.array([r[k] for r in rec], float) for k in rec[0] if k != "idx"}
idx = np.array([r["idx"] for r in rec])
print(f"{len(rec)} rows with all three signals\n")

print(f"{'signal':<34}{'median':>9}{'p90':>9}{'p99':>9}")
for nm, k in (("silhouette dist (outer), % ear", "sil"),
              ("ridge agreement ratio (interior)", "agree"),
              ("worst kink, degrees", "kink"),
              ("spacing CV", "cv")):
    v = V[k][np.isfinite(V[k])]
    print(f"{nm:<34}{np.median(v):>9.2f}{np.percentile(v,90):>9.2f}"
          f"{np.percentile(v,99):>9.2f}")

# each signal's own tail: top 2%, so a 2-of-3 hit is far above chance
def tail(k, hi=True):
    v = V[k]
    t = np.nanpercentile(v, 98 if hi else 2)
    return (v > t) if hi else (v < t)

f_sil, f_agr, f_irr = tail("sil"), tail("agree"), (tail("kink") | tail("cv"))
n = len(rec)
votes = f_sil.astype(int) + f_agr.astype(int) + f_irr.astype(int)
print(f"\n{'tail of each signal (top 2%)':<34}{'rows':>7}")
for nm, m in (("silhouette", f_sil), ("ridge agreement", f_agr),
              ("irregularity", f_irr)):
    print(f"  {nm:<32}{m.sum():>7}")
print(f"\n{'agreement between them':<34}{'rows':>7}{'expected':>10}{'enrich':>8}")
for v in (2, 3):
    m = votes >= v
    # expected under independence
    p = (f_sil.mean(), f_agr.mean(), f_irr.mean())
    if v == 2:
        e = n * (p[0]*p[1]*(1-p[2]) + p[0]*p[2]*(1-p[1])
                 + p[1]*p[2]*(1-p[0]) + p[0]*p[1]*p[2])
    else:
        e = n * p[0]*p[1]*p[2]
    print(f"  >= {v} of 3 signals{'':<16}{m.sum():>7}{e:>10.1f}"
          f"{m.sum()/max(e,1e-9):>7.1f}x")

flag = idx[votes >= 2]
(OUT / "gt_flagged.txt").write_text(" ".join(map(str, sorted(flag.tolist()))))
print(f"\n{len(flag)} rows flagged by >=2 signals -> {OUT/'gt_flagged.txt'}")
print(f"  that is {100*len(flag)/n:.2f}% of the proper-ear subset")
print("\nNOTE: the concha border (15 of 55 points) is scored by NOTHING here.")
