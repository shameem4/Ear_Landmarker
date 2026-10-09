#!/usr/bin/env python3
"""Choose N new landmark sites on the template and write a Wrap project.

Greedy D-optimal-style selection against the 476-ear shape model: the site that
removes the most expected surface error is taken, the posterior is updated, and
the next is chosen knowing the first was taken -- so the set spreads itself and
nothing has to be allocated to regions by hand.

THE OBJECTIVE IS tr(Sigma M), NOT A SUM OVER VERTICES. Writing total
uncertainty as

    sum_v trace(C_v Sigma C_v^T) = trace(Sigma M),   M = sum_v C_v^T C_v

makes M a constant of the mesh, computed once. Adding a landmark at vertex v is
then a rank-3 update, and Woodbury gives the reduction in closed form as

    trace( (sigma^2 I_3 + C_v Sigma C_v^T)^-1 C_v (Sigma M Sigma) C_v^T )

which is a 3x3 determinant problem per candidate instead of an 80x80 inverse.
The naive loop is about a teraflop for 44 points; this is seconds. The two
agree to machine precision, which the --check flag verifies.

DELIBERATELY OVER-PROVISIONED. Sites are free in 3D -- they propagate through
the existing fits to every ear and every render with no new annotation -- so it
is cheaper to place too many and prune the ones that will not train than to
guess the right set now. Expect the tail to be worthless: the first two sites
remove ~9% of total uncertainty each and the fortieth well under 1%.

INDICES STAY TWO DIGITS. Wrap keys are parsed as `int(key[:2])` throughout this
codebase, so the scheme cannot exceed 100 points without changing every reader.
55 + 44 = 99 is the ceiling this script will respect.

    python scripts/propose_new_marks.py --add 44 --out suggest_ext.wrap
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pca", default=str(TPL / "ear_pca_v4_all.npz"))
    p.add_argument("--project", default=str(TPL / "mark_mean_v4.wrap"))
    p.add_argument("--mesh", default=str(TPL / "ear_mean_v2.obj"))
    p.add_argument("--sigma", type=float, default=0.0106)
    p.add_argument("--add", type=int, default=44)
    p.add_argument("--min-sep", type=float, default=0.02,
                   help="minimum spacing between sites, fraction of ear size; "
                        "stops the greedy stacking points on one hot spot")
    p.add_argument("--check", action="store_true",
                   help="verify the Woodbury objective against the direct one")
    p.add_argument("--out", default=str(TPL / "suggest_ext.wrap"))
    a = p.parse_args()

    import trimesh

    z = np.load(a.pca, allow_pickle=True)
    pcaV, faces = z["mean"], z["faces"]
    nv = len(pcaV)
    C = z["components"].reshape(-1, nv, 3)
    var = z["variance"]
    ear = float(np.ptp(pcaV, axis=0).max())
    sigma = a.sigma * ear
    Cf = np.ascontiguousarray(C.transpose(1, 2, 0))       # (nv, 3, modes)

    V, F = load_mesh(a.mesh)
    if len(V) != nv or not np.array_equal(F, faces):
        sys.exit("template topology does not match the PCA")
    tpl_ear = float(np.ptp(V, axis=0).max())

    d = json.load(open(a.project))
    sel = [n for n in d["nodes"].values() if n["nodeType"] == "SelectPoints"][0]
    pts = sel["params"]["points"]["value"]
    n_old = len(pts)
    P = np.zeros((n_old, 3))
    for k, q in pts.items():
        P[int(k[:2])] = wrap_barycentric(V, F, int(q["triangleInd"]),
                                         float(q["u"]), float(q["v"]))
    if n_old + a.add > 100:
        sys.exit(f"{n_old} + {a.add} exceeds the two-digit key limit")

    # sampling matrix for the existing points, in the PCA frame
    S = np.zeros((n_old, nv))
    for k, q in pts.items():
        t, u, v = int(q["triangleInd"]), float(q["u"]), float(q["v"])
        S[int(k[:2]), faces[t]] = [u, v, 1.0 - u - v]

    M = np.einsum("vam,van->mn", Cf, Cf)                  # constant of the mesh
    A = np.einsum("kv,vam->kam", S, Cf).reshape(-1, len(C))
    P_inf = A.T @ A / sigma ** 2 + np.diag(1.0 / var)

    def total(Pm):
        return float(np.trace(np.linalg.inv(Pm) @ M))

    base = total(P_inf)
    prior = float(np.trace(np.diag(var) @ M))
    print(f"template {nv} verts, {len(C)} modes, {n_old} existing points")
    print(f"sigma {a.sigma} x ear = {sigma:.5f} (shape space)\n")
    print(f"uncertainty knowing nothing  {np.sqrt(prior / nv) / ear * 100:.2f}% of ear")
    print(f"uncertainty with {n_old} points  "
          f"{np.sqrt(base / nv) / ear * 100:.2f}% of ear\n")

    chosen, cur = [], base
    minsep = a.min_sep * tpl_ear
    taken_xyz = list(P)
    for step in range(a.add):
        Sig = np.linalg.inv(P_inf)
        G = Sig @ M @ Sig
        Avv = np.einsum("vam,mn,vbn->vab", Cf, Sig, Cf)       # (nv,3,3)
        Bvv = np.einsum("vam,mn,vbn->vab", Cf, G, Cf)
        red = np.empty(nv)
        I3 = np.eye(3) * sigma ** 2
        for v in range(nv):
            red[v] = np.trace(np.linalg.solve(I3 + Avv[v], Bvv[v]))
        # spacing: never put a site on top of an existing or chosen one
        T = np.asarray(taken_xyz)
        far = np.linalg.norm(V[:, None, :] - T[None], axis=2).min(1) > minsep
        red[~far] = -np.inf
        best = int(np.argmax(red))
        if not np.isfinite(red[best]) or red[best] <= 0:
            print("  no admissible site left")
            break
        if a.check and step == 0:
            Cv = Cf[best]
            direct = total(P_inf + Cv.T @ Cv / sigma ** 2)
            err = abs((cur - direct) - red[best]) / max(red[best], 1e-30)
            print(f"  [check] Woodbury vs direct relative error {err:.2e}")
        Cv = Cf[best]
        P_inf = P_inf + Cv.T @ Cv / sigma ** 2
        new = cur - red[best]
        chosen.append(best)
        taken_xyz.append(V[best])
        if step < 5 or (step + 1) % 10 == 0 or step == a.add - 1:
            print(f"  +{step + 1:2d}  vertex {best:5d}  {cur:.5f} -> {new:.5f}"
                  f"   ({100 * red[best] / cur:.2f}% of what remained)")
        cur = new

    final = total(P_inf)
    print(f"\n{n_old} -> {n_old + len(chosen)} points: uncertainty "
          f"{np.sqrt(base / nv) / ear * 100:.2f}% -> "
          f"{np.sqrt(final / nv) / ear * 100:.2f}% of ear "
          f"({100 * (1 - np.sqrt(final / base)):.1f}% reduction)")

    # name each new site by the existing group it sits nearest to
    from scipy.spatial import cKDTree
    tree = cKDTree(P)
    for i, v in enumerate(chosen):
        dist, j = tree.query(V[v])
        key = f"{n_old + i:02d}_near_{group_of(int(j))}"
        tri = int(np.argmax((F == v).any(axis=1)))
        w = np.zeros(3)
        w[list(F[tri]).index(v)] = 1.0
        pts[key] = {"triangleInd": tri, "u": float(w[0]), "v": float(w[1])}

    # verify every new entry decodes back to the vertex we meant
    for i, v in enumerate(chosen):
        q = pts[f"{n_old + i:02d}_near_{group_of(int(tree.query(V[v])[1]))}"]
        got = wrap_barycentric(V, F, q["triangleInd"], q["u"], q["v"])
        if np.linalg.norm(got - V[v]) > 1e-9:
            sys.exit(f"site {i} does not round-trip -- barycentric convention")
    print(f"all {len(chosen)} new sites round-trip through wrap_barycentric")

    json.dump(d, open(a.out, "w"), indent=4)
    print(f"wrote {a.out}  ({len(pts)} points)")
    np.save(ROOT / "runs" / "new_mark_vertices.npy", np.array(chosen))


if __name__ == "__main__":
    main()
