"""Corpus-wide report on misrouted outer contours, from a SAM 3 screen.

Turns qa_silhouette_sam3.py's output into the things a decision needs: the
rate, which split the flagged rows fall in, which groups carry the defect,
and a row list ready for re-annotation or for the drop experiment's
drop-worst arm.

    python scripts/qa_bad_contour_report.py runs/sam3_full \
        --tol 15 --min 3 --out runs/sam3_full/bad_rows.txt
"""
import argparse
import csv
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from ear3d.scheme import GROUPS, GROUP_NAMES          # noqa: E402

G = {n: list(range(lo, hi)) for n, (lo, hi) in zip(GROUP_NAMES, GROUPS)}
OUTER = (G["ascending_helix"] + G["descending_helix"]
         + G["helix"] + G["ear_lobe"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--tol", type=float, default=15.0)
    p.add_argument("--min", type=int, default=3)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    R = json.load(open(pathlib.Path(a.run_dir) / "sam2_silhouette.json"))
    rows = list(csv.DictReader(open(ROOT / "data/manual/manifest.csv")))
    D = np.abs(np.array([r["per"] for r in R])[:, OUTER])
    idx = np.array([r["idx"] for r in R])
    n_off = (D > a.tol).sum(1)
    flag = n_off >= a.min

    print(f"screened {len(rows)} rows, {len(R)} returned a usable mask "
          f"({100*len(R)/len(rows):.1f}%)")
    print(f"\nmisrouted outer points (> {a.tol:.0f}% of ear height off the rim)")
    print(f"  per point                 : {100*np.mean(D > a.tol):.2f}%")
    print(f"  median per-point distance : {np.median(D):.2f}%")
    for k in (2, 3, 5, 8):
        m = n_off >= k
        print(f"  rows with >={k} such points : {100*m.mean():5.2f}%  "
              f"({int(m.sum())} rows)")

    splits = {}
    for s in ("train", "val", "test"):
        f = ROOT / f"data/manual/{s}.csv"
        if f.exists():
            splits[s] = {int(r["idx"]) for r in csv.DictReader(open(f))}
    if splits:
        print(f"\nflagged rows by split (>={a.min} misrouted)")
        for s, ids in splits.items():
            inflag = sum(1 for i in idx[flag] if int(i) in ids)
            scr = sum(1 for i in idx if int(i) in ids)
            print(f"  {s:<6} {inflag:>5} flagged of {scr:>6} screened "
                  f"({100*inflag/max(scr,1):.2f}%)")

    print(f"\nshare of each group's points off the rim")
    for nm in ("ascending_helix", "ear_lobe", "descending_helix", "helix"):
        sub = [OUTER.index(j) for j in G[nm]]
        print(f"  {nm:<20}{100*np.mean(D[:, sub] > a.tol):6.2f}%")

    # per-landmark, to see whether specific indices dominate
    worst = np.argsort(-(D > a.tol).mean(0))[:6]
    print(f"\nworst individual landmarks (index: % of rows where it is off)")
    for j in worst:
        print(f"  point {OUTER[j]:>3}  {100*(D[:, j] > a.tol).mean():5.2f}%")

    if a.out:
        order = np.argsort(-n_off)
        sel = [int(i) for i in idx[order] if n_off[list(idx).index(i)] >= a.min]
        pathlib.Path(a.out).write_text("\n".join(map(str, sel)))
        print(f"\nwrote {len(sel)} flagged row ids -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
