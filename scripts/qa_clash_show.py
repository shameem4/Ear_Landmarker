"""Render the flagged inner/outer clash rows -- look before believing the z.

Original outer chain in blue, corrected outer in red, inner helix in cyan
(unchanged by the outer correction), and the worst inner point ringed.
The standing rule applies: the z-score picks the rows, the eye decides.

    python scripts/qa_clash_show.py runs/corrected cross 12 runs/corrected/cross.png
"""
import csv
import importlib.util as ilu
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
_sp = ilu.spec_from_file_location("_cl", ROOT / "scripts/qa_outer_inner_clash.py")
_cl = ilu.module_from_spec(_sp); _sp.loader.exec_module(_cl)

RUN = pathlib.Path(sys.argv[1])
KIND = sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 12
PNG = pathlib.Path(sys.argv[4]) if len(sys.argv) > 4 else RUN / f"{KIND}.png"

L = np.load(ROOT / "data/manual/landmarks.npy")
rows = list(csv.DictReader(open(ROOT / "data/manual/manifest.csv")))
z = np.load(RUN / "outer.npz")
pos = {int(i): k for k, i in enumerate(z["idx"])}
cz = np.load(RUN / "clearance.npz")
want = [int(x) for x in (RUN / f"clash_{KIND}.txt").read_text().split()][:N]
if not want:
    print(f"no rows in clash_{KIND}.txt"); sys.exit(0)

cols = min(4, len(want)); rws = int(np.ceil(len(want) / cols))
fig, ax = plt.subplots(rws, cols, figsize=(3.2 * cols, 3.4 * rws))
ax = np.atleast_1d(ax).ravel()
for p, i in enumerate(want):
    k = pos[i]
    im = Image.open(ROOT / "data/manual" / rows[i]["image_file"]).convert("RGB")
    w, h = im.size
    g = L[i] * np.array([w, h])
    c = g.copy(); c[:20] = z["pts"][k] * np.array([w, h])
    A = ax[p]
    A.imshow(im)
    for pts, col, lab in ((g[:20], "#2a6fdb", "outer orig"),
                          (c[:20], "#e03030", "outer corrected")):
        q = np.r_[pts, pts[:1]]
        A.plot(q[:, 0], q[:, 1], "-", color=col, lw=1.2, label=lab)
    A.plot(g[_cl.INNER][:, 0], g[_cl.INNER][:, 1], "-o", color="#00c8c8",
           lw=1.2, ms=2.0, label="inner helix")
    after = cz["after"][k]
    j = int(np.nanargmin(after)) if KIND in ("cross", "close") else int(np.nanargmax(after))
    A.plot(*g[20 + j], "o", mfc="none", mec="yellow", ms=11, mew=1.6)
    A.set_title(f"{i}  clr {np.nanmin(cz['before'][k]):.1f}->{np.nanmin(after):.1f}%"
                f"\npt {20+j}: {cz['before'][k][j]:.1f}->{after[j]:.1f}%",
                fontsize=8)
    A.axis("off")
for p in range(len(want), len(ax)):
    ax[p].axis("off")
ax[0].legend(fontsize=6, loc="lower right")
fig.suptitle(f"inner/outer clash: {KIND} (clearance, % of ear height)",
             fontsize=11)
fig.tight_layout()
fig.savefig(PNG, dpi=115)
print(f"wrote {PNG}")
