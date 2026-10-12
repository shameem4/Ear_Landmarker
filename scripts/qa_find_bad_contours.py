"""Rank rows by how badly the ANNOTATED outer contour leaves the ear's edge.

This is a different defect from the one qa_agreement.py measures, and the
two do not overlap: low agreement means the IMAGE is hard (occlusion, blur,
exposure), while this means the LABEL is wrong -- the helix points do not
follow the helix. Measured on 200 agreement-worst rows against 200
agreement-best, the rate of misrouted points was the SAME in both (4.84% vs
4.70%), so filtering a corpus on agreement would drop the wrong rows.

Scoring: each of the 20 outer landmarks gets its distance to the SAM mask
boundary in % of ear height. A row's score is how many of them sit further
than --tol off the rim. Counting beats averaging here -- a contour that is
right for 16 points and wanders badly for 4 averages out to nothing.

THE MASK IS THE WEAK LINK, so three gates run before any row is scored and
a failed gate drops the row rather than condemning its landmarks:

  covered < 0.60   SAM under-segmented, took a fragment of the ear
  bloat   > 2.0    SAM took the whole crop -- this one passes a
                   covers-the-hull test perfectly, which is how a batch of
                   "landmarks off the rim" turned out to be the image border
  largest < 0.80   the mask is confetti, whose rim is nowhere

ALSO GATE ON --max-bloat. Even inside the 2.0 cap the flag count tracks mask
bloat at rho +0.474, because a mask carrying some hair pushes the "rim"
outward and makes a correct contour look misrouted. The flag rate by band,
over 989 screened rows:

    bloat 1.0-1.2   2.0% flagged      bloat 1.4-1.6    7.4%
    bloat 1.2-1.4   1.1% flagged      bloat 1.6-2.0   24.2%

So the headline rate depends almost entirely on where the mask gate sits.
On tight masks (bloat < 1.4) 10 of 424 rows flag, and by eye about two
thirds of those are genuinely bad -- a true rate near 1.5%. Reading the
ungated number instead gives 12%, which is mask failure, not annotation
failure.

Even so this is a SCREEN and not a filter: send the output to a human, or to
qa_silhouette_show.py, before dropping anything.

    python scripts/qa_find_bad_contours.py /tmp/sam_screen --tol 15 --min 3
"""
import argparse
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
    p.add_argument("run_dir", help="dir holding sam2_silhouette.json")
    p.add_argument("--tol", type=float, default=15.0,
                   help="a point further than this %% of ear height off the "
                        "rim is misrouted")
    p.add_argument("--min", type=int, default=3,
                   help="flag a row with at least this many misrouted points")
    p.add_argument("--max-bloat", type=float, default=1.4,
                   help="ignore rows whose mask exceeds this multiple of the "
                        "landmark hull; above ~1.4 the flags are mask bloat")
    p.add_argument("--out", default=None, help="write flagged row ids here")
    a = p.parse_args()
    R = json.load(open(pathlib.Path(a.run_dir) / "sam2_silhouette.json"))
    if not R:
        print("no scored rows -- every mask failed its gates")
        return 1
    bloat = np.array([r.get("bloat", 1.0) for r in R])
    keep = bloat <= a.max_bloat
    print(f"{int((~keep).sum())} of {len(R)} rows dropped for mask bloat "
          f"> {a.max_bloat}")
    R = [r for r, k in zip(R, keep) if k]
    if not R:
        print("no rows left after the bloat gate")
        return 1
    D = np.abs(np.array([r["per"] for r in R])[:, OUTER])
    idx = np.array([r["idx"] for r in R])
    n_off = (D > a.tol).sum(1)
    flag = n_off >= a.min
    print(f"n={len(R)} rows with a trusted mask")
    print(f"  outer points >{a.tol:.0f}% off the rim : {100*np.mean(D > a.tol):.2f}%")
    print(f"  median per-point distance          : {np.median(D):.2f}%")
    for k in (2, 3, 5, 8):
        print(f"  rows with >={k} misrouted points    : "
              f"{100*np.mean(n_off >= k):5.1f}%  ({int((n_off >= k).sum())} rows)")
    order = np.argsort(-n_off)
    print(f"\ntop 25 candidates (look before dropping):")
    for j in order[:25]:
        print(f"  row {idx[j]:<7} {n_off[j]:>2} of {len(OUTER)} points off  "
              f"max {D[j].max():5.1f}%  mask covers {R[j]['covered']:.2f} "
              f"bloat {R[j].get('bloat', float('nan')):.2f}")
    if a.out:
        pathlib.Path(a.out).write_text(
            "\n".join(str(int(i)) for i in idx[order][flag[order]]))
        print(f"\nwrote {int(flag.sum())} flagged rows -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
