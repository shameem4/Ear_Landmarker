"""Build the train CSVs for the drop experiment, from the Gemini verdicts.

Which rows to drop is not the SAM screen. That flagged 532 rows (5.40%) and
the controlled Gemini measurement showed it over-counts by 2.8x, so dropping
all of it would throw away mostly-good hard images -- the failure mode
already recorded for the agreement screen.

The drop set is therefore the rows Gemini INDEPENDENTLY disputed: an
annotated landmark sitting more than --tol % of ear height outside Gemini's
whole-ear box. 15% was the threshold set by eye on the 15-18% band
(precision 0.87) and is 3x the instrument's own ~5% noise floor.

Three arms are written, because a drop experiment without a random arm
cannot distinguish "removed bad labels" from "removed hard examples" or
from run-to-run noise. Same design as the flagged-vs-control measurement:
the comparison carries the claim, not the absolute number.

  full          data/manual/train.csv              (untouched)
  drop_disputed train_drop_disputed.csv            (Gemini-disputed removed)
  drop_random   train_drop_random.csv              (as many random rows removed)

val/test are NOT touched -- they must stay identical or the NMEs are not
comparable across arms.

    python scripts/qa_make_drop_splits.py --tol 15
"""
import argparse
import csv
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "data/manual"


def excursion(box, pts):
    """Max distance of any landmark outside the box, in % of ear height.

    Box is normalised [x0, y0, x1, y1]; pts are normalised too, so the
    vertical extent of the outer chain is the ear-height unit.
    """
    x0, y0, x1, y1 = box
    eh = float(pts[:20, 1].max() - pts[:20, 1].min())
    if not np.isfinite(eh) or eh <= 0:
        return np.nan
    dx = np.maximum(x0 - pts[:, 0], pts[:, 0] - x1)
    dy = np.maximum(y0 - pts[:, 1], pts[:, 1] - y1)
    return float(np.maximum(dx, dy).max()) / eh * 100


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tol", type=float, default=15.0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    L = np.load(D / "landmarks.npy")
    train = list(csv.DictReader(open(D / "train.csv")))
    train_idx = {int(r["idx"]) for r in train}

    stats = {}
    for arm in ("flagged", "control"):
        B = json.load(open(ROOT / f"runs/measure/{arm}/boxes_gemini_nothink.json"))
        e = {int(k): excursion(v, L[int(k)]) for k, v in B.items()
             if np.isfinite(L[int(k)]).all()}
        stats[arm] = e
        v = np.array([x for x in e.values() if np.isfinite(x)])
        print(f"{arm:<8} n={len(v):4d}  median excursion {np.median(v):5.2f}%  "
              f"past {a.tol:.0f}%: {(v > a.tol).sum():3d} ({100*(v>a.tol).mean():.1f}%)")

    # the drop set: disputed AND in train. Dropping val/test would change the
    # yardstick and make the arms incomparable.
    disputed = {i for i, x in stats["flagged"].items()
                if np.isfinite(x) and x > a.tol}
    drop = sorted(disputed & train_idx)
    print(f"\ndisputed rows {len(disputed)}, of which in train: {len(drop)}")
    # control-arm false positives, for the record: these are rows the SAM
    # screen did NOT flag yet Gemini disputes, i.e. what this drop set misses
    miss = {i for i, x in stats["control"].items()
            if np.isfinite(x) and x > a.tol} & train_idx
    print(f"control rows past tol and in train (missed by the screen): {len(miss)}")

    rng = np.random.default_rng(a.seed)
    pool = sorted(train_idx - set(drop))
    rand = set(rng.choice(pool, len(drop), replace=False).tolist())

    hdr = train[0].keys()
    for name, remove in (("drop_disputed", set(drop)), ("drop_random", rand)):
        out = D / f"train_{name}.csv"
        keep = [r for r in train if int(r["idx"]) not in remove]
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(hdr))
            w.writeheader(); w.writerows(keep)
        print(f"wrote {out.name}: {len(keep)} rows (-{len(train)-len(keep)})")
    (ROOT / "runs/measure/drop_disputed_rows.txt").write_text(
        " ".join(map(str, drop)))
    print(f"\nrow list -> runs/measure/drop_disputed_rows.txt")


if __name__ == "__main__":
    main()
