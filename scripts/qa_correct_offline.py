"""Re-correct the outer chain from CACHED mask boundaries, with the two guards.

The corpus-wide pass showed the v1 corrector damaging 669 of 6,107 rows
(11%), in two distinct modes the renders separated:

  blow-up    the correspondence maps an OPEN 20-point chain onto the
             CLOSED mask boundary loop, and the monotone unwrap can
             traverse the wrong way round -- across the mask's front edge,
             where the chain has no business being. The corrected chain
             then detours across the ear or bulges past the lobe.
  clipping   the rim at the top of the helix is under hair, SAM stops at
             the visible rim, and the correction pulls the rim down onto
             the inner helix. covered >= 0.85 does not catch this because
             a thin band along one arc costs almost no area, and raising
             the threshold is not a fix: 0.97 cuts the damage rate to 0.5%
             but throws away 81% of the good rows, and 1.00 admits 2 rows
             out of 6,107 because SAM's lobe cutoff is conceptual, so
             coverage is never complete.

So both guards are LOCAL:

  evidence   a point whose nearest boundary vertex is further than MAXD is
             not being corrected by anything, it is being teleported --
             leave it. Offsets are clamped to MAXOFF, and a row whose chain
             length changes by more than LEN_TOL is rejected outright.
  clearance  the inner helix, which the correction never touches, is the
             witness. No point may move if it drives an inner-helix
             clearance below that index's 1st percentile in the ORIGINAL
             corpus. Offsets of the outer points nearest the violated
             inner point are zeroed, and it re-checks.

The clearance guard turns the witness into a constraint, so it can no
longer be an independent check of its own output. The honest validation
after this is the render, and the count of rows the guard had to touch.

    python scripts/qa_correct_offline.py runs/corrected
"""
import argparse
import csv
import importlib.util as ilu
import pathlib
import sys

import numpy as np
from scipy import ndimage

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load(name, path):
    sp = ilu.spec_from_file_location(name, ROOT / path)
    m = ilu.module_from_spec(sp); sp.loader.exec_module(m)
    return m


_cl = _load("_cl", "scripts/qa_outer_inner_clash.py")

TOL = 2.0        # % of ear height; below this a point is left alone
SMOOTH = 1.0     # gaussian sigma along the chain, in points
MAXD = 20.0      # % of ear height; no boundary within this = no evidence
MAXOFF = 10.0    # % of ear height; the largest honest single correction
LEN_TOL = 0.25   # chain length may not change by more than this


def propose(gt20, B, eh):
    """Corrected chain and offsets, with the evidence guards. None = reject."""
    d2 = ((gt20[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    j = np.argmin(d2, axis=1)
    dmin = np.sqrt(d2[np.arange(len(gt20)), j]) / eh * 100
    n = len(B)
    k = np.unwrap(j / n * 2 * np.pi) / (2 * np.pi) * n
    from sklearn.isotonic import IsotonicRegression
    order = 1 if k[-1] >= k[0] else -1
    k = IsotonicRegression(increasing=(order == 1)).fit_transform(
        np.arange(len(k)), k)
    tgt = B[np.clip(np.round(k).astype(int) % n, 0, n - 1)]

    t = np.gradient(gt20, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    nrm = np.c_[-t[:, 1], t[:, 0]]
    off = ((tgt - gt20) * nrm).sum(1)
    off = ndimage.gaussian_filter1d(off, SMOOTH, mode="nearest")
    off = off / eh * 100                                  # now in % of eh
    off = np.clip(off, -MAXOFF, MAXOFF)
    off[np.abs(off) <= TOL] = 0.0
    off[dmin > MAXD] = 0.0                                # no evidence there
    return off, nrm


def chain_len(p):
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def apply_guard(gt55, off, nrm, eh, floor, win=2, iters=6):
    """Zero the offsets that drive an inner-helix clearance below floor."""
    vetoed = np.zeros(len(off), bool)
    for _ in range(iters):
        cand = gt55.copy()
        cand[:20] = gt55[:20] + nrm * (off * eh / 100)[:, None]
        c = _cl.clearance(cand, eh)
        bad = np.flatnonzero(np.nan_to_num(c - floor, nan=1.0) < 0)
        if not len(bad):
            return cand, off, vetoed
        # which outer segment each violated inner point leans on
        _, _, kk, _ = _cl.nearest_on_open(cand[_cl.INNER][bad], cand[:20])
        hit = np.zeros(len(off), bool)
        for kj in kk:
            lo, hi = max(0, kj - win), min(len(off), kj + 2 + win)
            hit[lo:hi] = True
        hit &= off != 0.0
        if not hit.any():
            break                       # violation is not ours to fix
        off = off.copy(); off[hit] = 0.0
        vetoed |= hit
    cand = gt55.copy()
    cand[:20] = gt55[:20] + nrm * (off * eh / 100)[:, None]
    return cand, off, vetoed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--src", default="outer.npz")
    ap.add_argument("--out", default="outer_v2.npz")
    a = ap.parse_args()
    RUN = pathlib.Path(a.run)

    L = np.load(ROOT / "data/manual/landmarks.npy")
    rows = list(csv.DictReader(open(ROOT / "data/manual/manifest.csv")))
    z = np.load(RUN / a.src)
    idx = z["idx"].astype(int)
    # The boundaries are cached when available. When they are not -- the v1
    # pass predates the cache -- the applied offsets are RECOVERED from the
    # corrected points by projecting onto the same normals propose() builds,
    # which is exact. The only guard lost that way is the dmin evidence test,
    # and the MAXOFF clamp already covers the case it was built for: a point
    # with no boundary near it gets teleported, i.e. a huge offset.
    has_bnd = "bnd" in z.files and len(z["bnd"])
    bnd = np.split(z["bnd"], np.cumsum(z["blen"])[:-1]) if has_bnd else None
    print("boundaries cached" if has_bnd else "no cached boundaries: "
          "recovering offsets from the corrected points")

    G, EH, WH = [], [], []
    for k, i in enumerate(idx):
        w, h = int(rows[i]["width"]), int(rows[i]["height"])
        g = L[i] * np.array([w, h])
        G.append(g); EH.append(float(g[:20, 1].max() - g[:20, 1].min()))
        WH.append((w, h))
    # the floor: 1st percentile of the ORIGINAL clearance, per inner index
    C0 = np.array([_cl.clearance(G[k], EH[k]) for k in range(len(idx))])
    floor = np.nanpercentile(C0, 1, axis=0)
    floor = np.maximum(floor, 0.0)
    print("clearance floor per inner point (% of ear height):")
    print("  " + "  ".join(f"{20+j}:{floor[j]:.1f}" for j in range(len(floor))))

    out_pts, out_off, rej, nveto, nmoved = [], [], 0, 0, 0
    for k, i in enumerate(idx):
        g, eh = G[k], EH[k]
        t = np.gradient(g[:20], axis=0)
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
        nrm = np.c_[-t[:, 1], t[:, 0]]
        if bnd is not None:
            off, nrm = propose(g[:20], bnd[k].astype(float) * np.array(WH[k]), eh)
        else:
            prev = z["pts"][k] * np.array(WH[k])
            off = np.clip(((prev - g[:20]) * nrm).sum(1) / eh * 100,
                          -MAXOFF, MAXOFF)
            off[np.abs(off) <= TOL] = 0.0
        prop = g[:20] + nrm * (off * eh / 100)[:, None]
        if not (1 - LEN_TOL <= chain_len(prop) / max(chain_len(g[:20]), 1e-9)
                <= 1 + LEN_TOL):
            rej += 1
            out_pts.append(g[:20] / np.array(WH[k])); out_off.append(np.zeros(20))
            continue
        cand, off2, vetoed = apply_guard(g, off, nrm, eh, floor)
        nveto += int(vetoed.any())
        nmoved += int((off2 != 0).any())
        out_pts.append(cand[:20] / np.array(WH[k])); out_off.append(off2)
    O = np.array(out_off)
    kw = dict(bnd=z["bnd"], blen=z["blen"]) if bnd is not None else {}
    np.savez(RUN / a.out, idx=idx, pts=np.array(out_pts), off=O, **kw)
    print(f"\n{len(idx)} rows")
    print(f"  rejected on chain length   {rej}")
    print(f"  clearance guard fired on   {nveto}")
    print(f"  rows with any point moved  {nmoved} ({100*nmoved/len(idx):.0f}%)")
    print(f"  points moved               {100*(O != 0).mean():.1f}%")
    print(f"wrote {RUN/a.out}")


if __name__ == "__main__":
    main()
