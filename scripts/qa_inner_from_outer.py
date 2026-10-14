"""How much does the outer chain tell us about the other 35 points?

Two different questions hide behind "adjust the inner helix from the
outer", and they need different answers:

  PREDICT    hide points 20-54 and reconstruct them from 0-19 through the
             PDM. This asks how much INDEPENDENT information the interior
             annotation carries. If prediction is as good as the
             annotation, the interior is nearly redundant.
  PROPAGATE  carry the interior along with a correction already applied to
             the outer chain, without replacing it. This is what you want
             operationally, because it never discards a good interior
             annotation -- it only moves it the way its neighbourhood moved.

Prediction here is the standard conditional fit: align the model to the
observed points, solve the shape coefficients from those points alone,
then read off the unobserved ones. Scored against three baselines so the
number means something -- the mean shape (no conditioning), a similarity
fit of the mean (pose but no deformation), and the corpus spread.

Evaluated on rows held out of the PDM build is not possible (it was built
on all clean rows), so this reports the IN-SAMPLE figure and says so: it
is an upper bound on how well prediction can do.

    python scripts/qa_inner_from_outer.py
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scheme import STRIPS, STRIP_NAMES

P = np.load(ROOT / "runs/pdm/pdm.npz")
MEAN, MODES, VAR = P["mean"], P["modes"], P["var"]
L = np.load(ROOT / "data/manual/landmarks.npy")
bad = {int(x) for x in (ROOT / "runs/sam3_full/bad_rows.txt").read_text().split()}
ok = np.isfinite(L).all((1, 2)) & ~np.isin(np.arange(len(L)), list(bad))
idx = np.flatnonzero(ok)
rng = np.random.default_rng(0)
idx = rng.choice(idx, 1500, replace=False)
OBS = list(range(20))
HID = list(range(20, 55))


def similarity(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(d0.T @ s0)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1, d]) @ Vt
    sc = (S[0] + d * S[1]) / max((s0 ** 2).sum(), 1e-9)
    return sc, R, mu_d - sc * (R @ mu_s)


def fit(obs_pts, K=12, iters=4):
    """Conditional fit: pose + K shape coeffs from the observed points."""
    b = np.zeros(K)
    for _ in range(iters):
        model = MEAN + np.einsum("k,kij->ij", b, MODES[:K])
        sc, R, t = similarity(model[OBS], obs_pts)
        inv = (R.T @ (obs_pts - t).T).T / max(sc, 1e-9)
        A = MODES[:K][:, OBS, :].reshape(K, -1).T       # (40, K)
        y = (inv - MEAN[OBS]).reshape(-1)
        b = np.linalg.lstsq(A, y, rcond=None)[0]
        b = np.clip(b, -3 * np.sqrt(VAR[:K]), 3 * np.sqrt(VAR[:K]))
    model = MEAN + np.einsum("k,kij->ij", b, MODES[:K])
    sc, R, t = similarity(model[OBS], obs_pts)
    return sc * (R @ model.T).T + t


res = {k: [] for k in ("pred", "meanfit", "corpus")}
per_strip = {nm: [] for nm in STRIP_NAMES}
for i in idx:
    g = L[i].copy()
    # work in a handedness-normalised frame, as the PDM was built
    x, y = g[:20, 0], g[:20, 1]
    if 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) > 0:
        g[:, 0] = -g[:, 0]
    eh = float(g[:20, 1].max() - g[:20, 1].min())
    if eh <= 0:
        continue
    out = fit(g[OBS])
    sc, R, t = similarity(MEAN[OBS], g[OBS])
    mf = sc * (R @ MEAN.T).T + t
    res["pred"].append(np.median(np.linalg.norm(out[HID] - g[HID], axis=1)) / eh * 100)
    res["meanfit"].append(np.median(np.linalg.norm(mf[HID] - g[HID], axis=1)) / eh * 100)
    for (lo, hi), nm in zip(STRIPS, STRIP_NAMES):
        if lo >= 20:
            per_strip[nm].append(
                np.median(np.linalg.norm(out[lo:hi] - g[lo:hi], axis=1)) / eh * 100)

print(f"n={len(res['pred'])} rows, PDM fitted to points 0-19 only\n")
print(f"{'reconstruction of points 20-54':<38}{'median':>9}{'p90':>9}")
for nm, k in (("conditional PDM fit (12 modes)", "pred"),
              ("mean shape, similarity-fitted", "meanfit")):
    v = np.array(res[k])
    print(f"  {nm:<36}{np.median(v):>9.2f}{np.percentile(v,90):>9.2f}")
print(f"\n{'per strip, conditional fit':<38}{'median':>9}")
for nm in STRIP_NAMES:
    if per_strip[nm]:
        print(f"  {nm:<36}{np.median(per_strip[nm]):>9.2f}")
print("\n(% of ear height. IN-SAMPLE: the PDM was built on these rows, so"
      "\n these are an UPPER BOUND on predictive accuracy.)")
print(f"\nFor scale: the GT sits {0.93:.2f}% from the SAM rim on gated rows,"
      "\nand the shipped landmarker is 1.0-1.7% from the annotation.")
