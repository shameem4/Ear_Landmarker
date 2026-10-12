#!/usr/bin/env python3
"""Validate a hand-placed scheme extension before any of it is propagated.

Run this on the Wrap project after new points have been placed by eye. It
answers four separate questions and does not mix them up:

  IS IT WELL FORMED. Every point decodes through wrap_barycentric to a point ON
  the template, indices are contiguous, nothing is duplicated, and the original
  55 are where they were. The barycentric convention is P = u*a + v*b +
  (1-u-v)*c and reading it the other way once displaced every landmark in the
  corpus by up to a triangle while every internal statistic stayed happy, so it
  is checked explicitly rather than assumed.

  WHAT DOES EACH NEW POINT BUY. Under the 476-ear shape model, each point's
  value is how much expected surface error it removes. Two numbers are
  reported and they answer different questions:
      alone      -- added to the 55 by itself. Is this point informative?
      marginal   -- added last, after every other new point. Is it informative
                    GIVEN the others, or is a neighbour already saying it?
  A point with a high `alone` and a near-zero `marginal` is redundant, not bad.

  HOW CLOSE IS IT TO THE BEST POSSIBLE. The same greedy that would pick sites
  by information alone is run to the same count, and its total is quoted as a
  ceiling. Anatomical placement is expected to fall short of it -- the point of
  the comparison is to show by how much, not to argue the ceiling should win.
  A set reaching most of the ceiling is doing well; one reaching a fraction of
  it is leaving real information on the table and is worth another pass.

  WHERE IS IT STILL BLIND. The residual uncertainty map after the new points,
  so a second round can be aimed.

WHAT THIS CANNOT TELL YOU is whether a point is findable in an image. That is
the gate that comes after propagation: train with the extended scheme and see
which points localise. A point can be informative about shape and invisible to
a camera, and only the trained model separates those.

    python scripts/validate_marks.py --project .../mark_mean_v5.wrap
"""
import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ear3d.scheme import group_of                               # noqa: E402
from ear3d.wrap import load_mesh, wrap_barycentric              # noqa: E402

TPL = pathlib.Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/wrap_work/templates")


def read_points(path, V, F):
    d = json.load(open(path))
    sel = [n for n in d["nodes"].values() if n["nodeType"] == "SelectPoints"]
    if not sel:
        sys.exit(f"{path}: no SelectPoints node")
    raw = sel[0]["params"]["points"]["value"]
    items = sorted(((int(k[:2]), k, q) for k, q in raw.items()))
    P = np.array([wrap_barycentric(V, F, int(q["triangleInd"]),
                                   float(q["u"]), float(q["v"]))
                  for _, _, q in items])
    return [i for i, _, _ in items], [k for _, k, _ in items], items, P


def sampling_rows(items, faces, nv):
    S = np.zeros((len(items), nv))
    for r, (_, _, q) in enumerate(items):
        t, u, v = int(q["triangleInd"]), float(q["u"]), float(q["v"])
        S[r, faces[t]] += [u, v, 1.0 - u - v]
    return S


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", default=str(TPL / "mark_mean_v5.wrap"))
    p.add_argument("--base", default=str(TPL / "mark_mean_v4.wrap"),
                   help="the 55-point project the extension started from")
    p.add_argument("--pca", default=str(TPL / "ear_pca_v7_all_mm.npz"))
    p.add_argument("--mesh", default=str(TPL / "ear_mean_v4.obj"))
    p.add_argument("--sigma", type=float, default=0.0106)
    p.add_argument("--margin", type=float, default=0.0,
                   help="mm from the template boundary to exclude from the "
                        "objective. DEFAULTS TO 0, i.e. score everything. An "
                        "earlier version excluded 5 mm, on the wrong belief "
                        "that variance near the boundary was a cutting "
                        "artefact; measured, PRIOR variance correlates -0.040 "
                        "with distance to the boundary while POSTERIOR "
                        "variance correlates -0.512, so the ear varies there "
                        "just as much and it is landmark coverage that is "
                        "missing. Excluding it hid the region most in need of "
                        "points.")
    p.add_argument("--out", default="runs/marks_validation.png")
    a = p.parse_args()

    V, F = load_mesh(a.mesh)
    z = np.load(a.pca, allow_pickle=True)
    pcaV, faces = z["mean"], z["faces"]
    nv = len(pcaV)
    if len(V) != nv or not np.array_equal(F, faces):
        sys.exit("template topology does not match the PCA")
    C = z["components"].reshape(-1, nv, 3)
    var = z["variance"]
    ear = float(np.ptp(pcaV, axis=0).max())
    tpl_ear = float(np.ptp(V, axis=0).max())
    sigma = a.sigma * ear
    Cf = np.ascontiguousarray(C.transpose(1, 2, 0))
    # The objective counts only surface we actually care about. Vertex
    # variance within a few mm of the template's open boundary is dominated by
    # WHERE EACH EAR WAS CUT off its head, which differs per subject and is not
    # a property of the ear: measured, uncertainty correlates -0.507 with
    # distance to that boundary and the top decile sits 0.9 mm from it against
    # 7.4 mm for the rest. Scoring it rewards points that pin down the cut.
    import collections as _c
    ec = _c.Counter()
    for f in F:
        for x, y in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            ec[(min(x, y), max(x, y))] += 1
    bnd = sorted({x for (x, y), n in ec.items() if n == 1 for x in (x, y)})
    from scipy.spatial import cKDTree as _KD
    d2b = _KD(V[bnd]).query(V)[0]
    keep = d2b >= a.margin
    if not keep.any():
        sys.exit(f"--margin {a.margin} excludes the whole template")
    print(f"objective over {keep.sum()}/{len(V)} vertices "
          f"(>= {a.margin:.0f} mm from the cut boundary)\n")
    M = np.einsum("vam,van->mn", Cf[keep], Cf[keep])
    nv_obj = int(keep.sum())

    idx, keys, items, P = read_points(a.project, V, F)
    bidx, _, bitems, BP = read_points(a.base, V, F)
    n_new = len(idx) - len(bidx)
    print(f"{a.project}\n  {len(idx)} points, {len(bidx)} in the base "
          f"-> {n_new} new\n")

    # ---- well-formedness -------------------------------------------------
    bad = []
    if idx != list(range(len(idx))):
        bad.append(f"indices are not 0..{len(idx) - 1}: got {idx[:5]}...")
    if max(idx) > 99:
        bad.append("index above 99; Wrap keys are parsed as int(key[:2])")
    for r, (_, k, q) in enumerate(items):
        t, u, v = int(q["triangleInd"]), float(q["u"]), float(q["v"])
        w = np.array([u, v, 1.0 - u - v])
        if w.min() < -1e-6 or w.max() > 1 + 1e-6:
            bad.append(f"{k}: barycentric {w.round(4)} is outside its triangle")
        got = wrap_barycentric(V, F, t, u, v)
        if np.linalg.norm(got - P[r]) > 1e-9:
            bad.append(f"{k}: does not round-trip")
    dup = [(keys[i], keys[j])
           for i in range(len(P)) for j in range(i + 1, len(P))
           if np.linalg.norm(P[i] - P[j]) < 1e-4 * tpl_ear]
    for i, j in dup:
        bad.append(f"{i} and {j} are the same point")
    moved = [(keys[j], float(np.linalg.norm(P[j] - BP[j])))
             for j in range(len(BP)) if np.linalg.norm(P[j] - BP[j]) > 1e-9]
    if moved:
        bad.append(f"{len(moved)} of the original 55 moved, worst "
                   f"{max(m for _, m in moved):.3f} mm ({moved[0][0]} ...)")
    print("WELL FORMED" if not bad else "PROBLEMS:")
    for b in bad:
        print(f"  ! {b}")
    if not bad:
        print(f"  indices 0..{max(idx)}, all round-trip, none coincident, "
              f"original 55 unmoved")
    if n_new <= 0:
        print("\nno new points to score")
        return

    # spacing
    from scipy.spatial import cKDTree
    t_old = cKDTree(P[:len(bidx)])
    dn, _ = t_old.query(P[len(bidx):])
    print(f"\n  new to nearest existing: min {dn.min():.2f} mm, "
          f"median {np.median(dn):.2f} mm (ear {tpl_ear:.0f} mm)")
    if len(P) - len(bidx) > 1:
        t_new = cKDTree(P[len(bidx):])
        dd, _ = t_new.query(P[len(bidx):], k=2)
        print(f"  new to nearest new:      min {dd[:, 1].min():.2f} mm, "
              f"median {np.median(dd[:, 1]):.2f} mm")

    # ---- information -----------------------------------------------------
    S = sampling_rows(items, faces, nv)
    n0 = len(bidx)

    def info(rows):
        A = np.einsum("kv,vam->kam", rows, Cf).reshape(-1, len(C))
        return A.T @ A / sigma ** 2

    def total(Pm):
        return float(np.trace(np.linalg.inv(Pm) @ M))

    P0 = np.diag(1.0 / var)
    base_inf = info(S[:n0]) + P0
    base = total(base_inf)
    full_inf = info(S) + P0
    full = total(full_inf)
    prior = float(np.trace(np.diag(var) @ M))

    def pct(t):
        return 100 * np.sqrt(t / nv_obj) / ear

    print(f"\nSURFACE UNCERTAINTY, % of ear size")
    print(f"  knowing nothing      {pct(prior):.3f}")
    print(f"  the {n0} base points   {pct(base):.3f}")
    print(f"  with your {n_new} added  {pct(full):.3f}"
          f"   ({100 * (1 - np.sqrt(full / base)):.1f}% better)")

    print(f"\nPER POINT (uncertainty removed, % of the base total)")
    print(f"{'point':<26s}{'alone':>9s}{'marginal':>10s}")
    rows = []
    for r in range(n0, len(idx)):
        alone = base - total(base_inf + info(S[r:r + 1]))
        others = [q for q in range(n0, len(idx)) if q != r]
        wo = total(info(np.vstack([S[:n0], S[others]])) + P0)
        marg = wo - full
        rows.append((keys[r], 100 * alone / base, 100 * marg / base))
        print(f"  {keys[r]:<24s}{100 * alone / base:8.2f}%{100 * marg / base:9.2f}%")
    red = [k for k, al, mg in rows if mg < 0.1 and al > 0.3]
    if red:
        print(f"\n  {len(red)} point(s) informative alone but redundant given "
              f"the others: {', '.join(red)}")
    weak = [k for k, al, mg in rows if al < 0.2]
    if weak:
        print(f"  {len(weak)} point(s) buy almost nothing even alone: "
              f"{', '.join(weak)}")

    # ---- ceiling ---------------------------------------------------------
    inf_g = base_inf.copy()
    for _ in range(n_new):
        Sig = np.linalg.inv(inf_g)
        G = Sig @ M @ Sig
        Av = np.einsum("vam,mn,vbn->vab", Cf, Sig, Cf)
        Bv = np.einsum("vam,mn,vbn->vab", Cf, G, Cf)
        I3 = np.eye(3) * sigma ** 2
        gain = np.array([np.trace(np.linalg.solve(I3 + Av[v], Bv[v]))
                         for v in range(nv)])
        b = int(np.argmax(gain))
        inf_g = inf_g + Cf[b].T @ Cf[b] / sigma ** 2
    ceil = total(inf_g)
    got = base - full
    best = base - ceil
    print(f"\nAGAINST THE INFORMATION-OPTIMAL {n_new} POINTS")
    print(f"  optimal would reach  {pct(ceil):.3f}% of ear")
    print(f"  yours reaches        {pct(full):.3f}%")
    print(f"  -> your anatomical set captures {100 * got / best:.0f}% of the "
          f"uncertainty the best {n_new} sites could remove")
    print("  (anatomical placement is expected to fall short; this says by how"
          "\n   much, and a low number is a reason for another pass, not for"
          "\n   using the optimal set)")

    # ---- residual map ----------------------------------------------------
    Sig = np.linalg.inv(full_inf)
    u_after = np.einsum("vam,mn,van->v", Cf, Sig, Cf)
    Sig0 = np.linalg.inv(base_inf)
    u_before = np.einsum("vam,mn,van->v", Cf, Sig0, Cf)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    tri = Triangulation(V[:, 2], V[:, 1], F)
    fig, ax = plt.subplots(1, 2, figsize=(13, 8))
    vmax = 100 * np.sqrt(u_before.max()) / ear
    for k, (u_, ttl) in enumerate([(u_before, f"the {n0} base points"),
                                   (u_after, f"with your {n_new} added")]):
        tp = ax[k].tripcolor(tri, 100 * np.sqrt(u_) / ear, shading="gouraud",
                             cmap="inferno", vmin=0, vmax=vmax)
        ax[k].plot(P[:n0, 2], P[:n0, 1], "o", mfc="none", mec="#39ff88",
                   ms=5, mew=1.1, label="existing")
        if k:
            ax[k].plot(P[n0:, 2], P[n0:, 1], "*", c="#2dd4ff", ms=13,
                       mec="k", mew=.5, label="new")
            for r in range(n0, len(idx)):
                ax[k].annotate(str(idx[r]), (P[r, 2], P[r, 1]), fontsize=6,
                               color="#2dd4ff", xytext=(4, 3),
                               textcoords="offset points")
        ax[k].set_aspect("equal")
        ax[k].axis("off")
        ax[k].set_title(f"{ttl}  ({pct(base if k == 0 else full):.2f}% of ear)",
                        fontsize=11)
        ax[k].legend(loc="lower left", fontsize=8)
        fig.colorbar(tp, ax=ax[k], fraction=.04)
    fig.suptitle("Surface uncertainty before and after the extension "
                 "(same colour scale)", fontsize=12)
    fig.tight_layout()
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
