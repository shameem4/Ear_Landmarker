"""Build the canonical 2D shape of each contour, to seed a VLM with.

Seeding a VLM with a row's OWN annotation destroys the thing that makes
the check worth having: a model that anchors on what it is shown will
bless a bad contour. Seeding with the canonical SHAPE instead keeps the
output independent of the annotation being judged, while template point j
still corresponds to landmark j, so displacement stays defined per point
-- which a blind polygon cannot give.

Built from the clean rows only (the flagged ones are excluded, since the
point is a template of what an ear SHOULD look like), in one handedness:
the corpus is a 55/45 mix of left and right ears and the outer chain's
signed area separates them exactly, so the minority is mirrored in before
averaging and the consumer mirrors back per row.

Normalised by the outer chain's centroid and height, with no rotation --
ears sit roughly upright in these crops and rotating to a PCA axis would
bake one corpus's head pose into the template.

    python scripts/qa_build_shape_template.py runs/shape_template
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ear3d.scheme import STRIP_NAMES, STRIPS

OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
BAD = ROOT / "runs/sam3_full/bad_rows.txt"

L = np.load(ROOT / "data/manual/landmarks.npy")
ok = np.isfinite(L).all((1, 2))
if BAD.exists():
    bad = {int(x) for x in BAD.read_text().split()}
    ok &= ~np.isin(np.arange(len(L)), list(bad))
    print(f"excluded {len(bad)} flagged rows")
idx = np.flatnonzero(ok)
X = L[idx].copy()

# winding sign of the outer chain == handedness
x, y = X[:, :20, 0], X[:, :20, 1]
A = 0.5 * np.sum(x * np.roll(y, -1, 1) - np.roll(x, -1, 1) * y, axis=1)
flip = A > 0 if (A > 0).sum() < (A < 0).sum() else A < 0
X[flip, :, 0] = -X[flip, :, 0]
print(f"mirrored {flip.sum()} of {len(X)} to a single handedness")

c = X[:, :20].mean(1, keepdims=True)
h = (X[:, :20, 1].max(1) - X[:, :20, 1].min(1))[:, None, None]
Z = (X - c) / h

mean = np.median(Z, axis=0)            # median: one bad row cannot drag it
sd = Z.std(0).mean()
print(f"template from {len(Z)} ears, mean per-point spread {sd:.3f} of ear height")

# express in 0-1000 with the ear centred, which is what the prompt wants
span = max(np.ptp(mean[:20, 0]), np.ptp(mean[:20, 1]))
T = (mean - mean[:20].mean(0)) / span * 620 + 500

out = {"note": "right-handed template; mirror x for the other ear",
       "space": "0-1000, ear centred, outer chain spans ~620",
       "strips": {}}
for (lo, hi), nm in zip(STRIPS, STRIP_NAMES):
    out["strips"][nm] = dict(
        lo=lo, hi=hi,
        points=[[round(float(p[1]), 1), round(float(p[0]), 1)]   # y, x
                for p in T[lo:hi]])
    print(f"  {nm:<14} points {lo}-{hi-1}  ({hi-lo})")

import json
json.dump(out, open(OUT / "template.json", "w"), indent=1)
np.save(OUT / "template_xy.npy", T)
print(f"\nwrote {OUT/'template.json'}")

# ---------------------------------------------------------------------------
# Also render the template as an IMAGE, in both handednesses.
#
# The numeric seed makes the model build a 55-point shape in its head before
# it can fit anything, and makes it decide handedness from one line of prose
# on every call. Two labelled pictures let it match and pick instead.
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
for side in ("right", "left"):
    P = T.copy()
    if side == "left":
        P[:, 0] = 1000.0 - P[:, 0]
    fig, ax = plt.subplots(figsize=(4.2, 4.6))
    for (lo, hi), nm, c in zip(STRIPS, STRIP_NAMES, COL):
        q = P[lo:hi]
        ax.plot(q[:, 0], q[:, 1], "-o", color=c, ms=5, lw=2.2,
                label=f"{nm}  ({lo}-{hi-1})")
        for j in range(lo, hi):
            if j == lo or j == hi - 1 or (j - lo) % 5 == 4:
                ax.annotate(str(j), P[j], fontsize=7, color=c,
                            xytext=(4, 3), textcoords="offset points")
    ax.set_title(f"{side.upper()} ear template", fontsize=13)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xlim(150, 900); ax.set_ylim(900, 100)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / f"template_{side}.png", dpi=130)
    plt.close(fig)
    print(f"wrote {OUT/f'template_{side}.png'}")
