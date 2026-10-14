"""First look at the per-point error trajectories: does the split pay?

Two questions, in order:
  1. do NORMAL and TANGENTIAL error behave differently over training? If they
     do not, storing them apart bought nothing and one per-keypoint loss (what
     REMIND and the NeoPose paper use) is enough.
  2. do points differ in trajectory SHAPE, not just final value? The method
     rests on clean points falling steadily and bottoming out late while
     inconsistent ones stall early -- if shape carries no information beyond
     the endpoint, the whole trajectory apparatus reduces to the residual
     screen already built, and that screen is cheaper.

    python scripts/qa_trajectory_look.py runs/trajectories/traj_s42
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
D = pathlib.Path(sys.argv[1])
fs = sorted(D.glob("epoch_*.npz"))
print(f"{len(fs)} epochs")

z0 = np.load(fs[0])
order = np.argsort(z0["idx"])
idx = z0["idx"][order]
R, K, E = len(idx), 55, len(fs)
N = np.zeros((R, K, E), np.float32)
T = np.zeros((R, K, E), np.float32)
V = np.zeros((R, K), np.float32)
pos = {int(i): r for r, i in enumerate(idx)}
for e, f in enumerate(fs):
    z = np.load(f)
    row = np.array([pos[int(i)] for i in z["idx"]])
    N[row, :, e] = z["d_normal"]
    T[row, :, e] = z["d_tangential"]
    V += z["visible"]
V /= E
print(f"{R} rows x {K} points x {E} epochs")

# ---- 1. do the two components behave differently? -----------------------
mn, mt = N.mean((0, 1)), T.mean((0, 1))
print(f"\n{'epoch':>7}{'normal':>10}{'tangential':>13}{'ratio t/n':>11}")
for e in [0, 1, 2, 5, 10, 25, 50, 100, E - 1]:
    if e < E:
        print(f"{e:>7}{mn[e]:>10.4f}{mt[e]:>13.4f}{mt[e]/max(mn[e],1e-9):>11.2f}")
drop_n = 1 - mn[-1] / mn[0]
drop_t = 1 - mt[-1] / mt[0]
print(f"\n  normal fell     {100*drop_n:5.1f}% from epoch 0")
print(f"  tangential fell {100*drop_t:5.1f}% from epoch 0")
print(f"  epoch of minimum: normal {int(mn.argmin())}, tangential {int(mt.argmin())}")

# ---- 2. shape vs endpoint, per point ------------------------------------
def feats(A):
    """Endpoint, epoch of minimum, and how much of the fall happened late."""
    tail = A[:, :, -20:].mean(2)                      # final level
    head = A[:, :, :5].mean(2)                        # starting level
    emin = A.argmin(2).astype(np.float32) / max(E - 1, 1)
    mid = A[:, :, E // 2]
    late = (mid - tail) / np.maximum(head - tail, 1e-6)   # fraction fallen late
    return tail, emin, late

tn, en, ln = feats(N)
tt, et, lt = feats(T)
ok = V > 0.9
print(f"\nper-point, on the {ok.sum()} points supervised in >90% of epochs")
for nm, (t, e, l) in (("normal", (tn, en, ln)), ("tangential", (tt, et, lt))):
    print(f"  {nm:<11} final {np.median(t[ok]):.4f}   "
          f"epoch-of-min (frac of run) median {np.median(e[ok]):.2f}   "
          f"late-fall fraction median {np.median(l[ok]):.2f}")

# does shape add anything the endpoint does not?
from scipy.stats import spearmanr
for nm, (t, e, l) in (("normal", (tn, en, ln)), ("tangential", (tt, et, lt))):
    a, b, c = t[ok], e[ok], l[ok]
    print(f"  {nm:<11} spearman(final, epoch-of-min) {spearmanr(a,b).statistic:+.3f}"
          f"   spearman(final, late-fall) {spearmanr(a,c).statistic:+.3f}")

# ---- 3. do the known groups differ? -------------------------------------
def rows_of(path):
    p = ROOT / path
    if not p.exists():
        return np.array([], int)
    v = [int(x) for x in p.read_text().replace("#", " ").split()
         if x.isdigit()]
    return np.array([pos[i] for i in v if i in pos])

groups = {
    "hand-dropped (no ear)": rows_of("qa_lists/drop_rows.txt"),
    "landmarker-disagree": rows_of("runs/remark/disagree_hi.txt"),
}
rng = np.random.default_rng(0)
groups["random control"] = rng.choice(R, 400, replace=False)
print(f"\n{'group':<24}{'n':>6}{'final N':>10}{'final T':>10}"
      f"{'ep-of-min':>11}{'late-fall':>11}")
for nm, r in groups.items():
    if not len(r):
        continue
    m = ok[r]
    print(f"{nm:<24}{len(r):>6}{np.median(tn[r][m]):>10.4f}"
          f"{np.median(tt[r][m]):>10.4f}{np.median(en[r][m]):>11.2f}"
          f"{np.median(ln[r][m]):>11.2f}")
np.savez_compressed(D.parent / "traj_summary.npz", idx=idx, tn=tn, tt=tt,
                    en=en, et=et, ln=ln, lt=lt, vis=V, mn=mn, mt=mt)
print(f"\nwrote {D.parent/'traj_summary.npz'}")
