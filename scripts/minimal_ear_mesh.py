#!/usr/bin/env python3
"""How few vertices does it take to describe an ear? A FaceMesh-style answer.

The question is NOT the one scripts/where_to_add_landmarks.py asks. That one
scores a point by how much it pins down the shape model's coefficients, which
presumes the model is the right description. This asks the direct thing: pick K
vertices of the template, reconstruct all 1188 from them, and measure how far
the reconstruction is from the real ear. If a set of vertices can rebuild every
ear in the corpus, it describes an ear -- whatever basis is used in between.

SELECTION IS Q-DEIM (Drmac and Gugercin, 2016): column-pivoted QR on the
leading shape modes picks interpolation sites that keep the reconstruction
well conditioned, and it is the standard tool for exactly this problem. Greedy
re-fitting at every candidate would cost far more and buy little. The three
coordinates of a vertex are selected together, since a landmark is a point and
not a scalar.

EVALUATION IS HELD OUT, BY SUBJECT. Sites are chosen and the reconstruction
map is fitted on the training ears only, then both are applied to subjects the
selection never saw. Choosing sites and scoring them on the same ears measures
memorisation: the corpus has 478 ears and 1188 vertices, so a map fitted on all
of them can look far better than it is.

THE FLOOR IS REAL AND IS REPORTED. Reconstruction runs through a shape basis
truncated at r modes, so no number of sites beats what r modes can express.
That floor is printed beside the curve; a K whose error has reached it is
limited by the basis, not by having too few points.

    python scripts/minimal_ear_mesh.py --kmax 120
"""
import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ear3d.wrap import base_topology, load_mesh, rigid_fit      # noqa: E402

TPL = pathlib.Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/wrap_work/templates")
CORPUS = pathlib.Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/wrap_work/corpus")


def load_corpus(dirs, Vt, Ft, cache):
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["V"], list(z["names"])
    V, names = [], []
    for d in dirs:
        for f in sorted(pathlib.Path(d).glob("*.obj")):
            Vi, Fi = load_mesh(str(f))
            Vi, _ = base_topology(Vi, Fi, Vt, Ft, f.name)
            V.append(Vi)
            names.append(f.stem)
    V = np.asarray(V)
    np.savez_compressed(cache, V=V, names=np.array(names))
    return V, names


def procrustes(V, iters=5):
    """Remove pose AND scale, as build_ear_pca does, so this is about shape."""
    A = V.copy()
    mean = A[0]
    for _ in range(iters):
        for i in range(len(A)):
            R, t, s = rigid_fit(A[i], mean)
            A[i] = s * A[i] @ R.T + t
        mean = A.mean(0)
    return A, mean


def visible_face_on(V, F, axis=0, sign=-1.0, tol=0.5):
    """Vertices on the surface a camera sees from the lateral view.

    The 55-point scheme exists because a frontal photograph is all an
    annotator ever had, and a point hidden behind the helix rim has no
    definition in that view. A mesh meant to be found in an image inherits the
    same constraint, so candidates are restricted to the first surface a ray
    along the view axis meets. `sign` is -1 because -x is lateral on this
    template: the helix rim sits at x = -11 and the canal at x = -2.
    """
    import trimesh
    m = trimesh.Trimesh(vertices=V, faces=F, process=False)
    d = np.zeros(3)
    d[axis] = -sign                       # ray travels INWARD, toward the head
    span = float(np.ptp(V, axis=0).max())
    origins = V.copy()
    origins[:, axis] += sign * span       # start outside, on the viewer's side
    hit, idx_ray, _ = m.ray.intersects_location(
        origins, np.tile(d, (len(V), 1)), multiple_hits=False)
    vis = np.zeros(len(V), bool)
    for pt, i in zip(hit, idx_ray):
        if abs(pt[axis] - V[i, axis]) <= tol:
            vis[i] = True
    return vis


def qdeim(B, k, allowed=None):
    """Pivoted QR over VERTICES: the three coords of a site stay together."""
    # B: (modes, nv, 3) -> score each vertex by its rows jointly
    modes, nv, _ = B.shape
    chosen = []
    R = B.reshape(modes, nv * 3).copy()
    # work on the vertex-grouped Gram: greedy max-volume by column norms
    Q = np.zeros((0, modes))
    for _ in range(k):
        M = R.reshape(modes, nv, 3)
        score = np.einsum("mva,mva->v", M, M)
        score[chosen] = -np.inf
        if allowed is not None:
            score[~allowed] = -np.inf
        v = int(np.argmax(score))
        chosen.append(v)
        # deflate the mode space by what this vertex explains
        Av = M[:, v, :]                                  # (modes, 3)
        q, _ = np.linalg.qr(Av)
        Q = np.vstack([Q, q.T])
        proj = q @ (q.T @ R.reshape(modes, -1))
        R = (R.reshape(modes, -1) - proj).reshape(modes, nv * 3)
    return np.array(chosen)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--template", default=str(TPL / "ear_mean_v4.obj"))
    p.add_argument("--fits", nargs="+",
                   default=[str(CORPUS / "faceform2_sonicom"),
                            str(CORPUS / "faceform2_hutubs")])
    p.add_argument("--modes", type=int, default=120)
    p.add_argument("--kmax", type=int, default=120)
    p.add_argument("--cache", default="runs/corpus_verts.npz")
    p.add_argument("--sigma", type=float, default=0.0106,
                   help="landmark noise as a fraction of ear size")
    p.add_argument("--visible-only", action="store_true",
                   help="restrict sites to what a face-on camera can see")
    p.add_argument("--out", default="runs/minimal_mesh.png")
    a = p.parse_args()

    Vt, Ft = load_mesh(a.template)
    V, names = load_corpus(a.fits, Vt, Ft, pathlib.Path(a.cache))
    print(f"{len(V)} ears, {V.shape[1]} vertices each")

    A, mean = procrustes(V)
    ear = float(np.ptp(mean, axis=0).max())
    subj = np.array([n.rsplit("_", 1)[0] for n in names])
    uniq = np.unique(subj)
    rng = np.random.default_rng(0)
    test_s = set(rng.choice(uniq, max(1, len(uniq) // 4), replace=False))
    te = np.array([s in test_s for s in subj])
    tr = ~te
    print(f"{tr.sum()} train ears / {te.sum()} test ears "
          f"({len(uniq) - len(test_s)}/{len(test_s)} subjects), "
          f"ear size {ear:.1f} mm\n")

    X = A - mean
    Xtr = X[tr].reshape(tr.sum(), -1)
    U, S, Wt = np.linalg.svd(Xtr, full_matrices=False)
    r = min(a.modes, len(S))
    B = Wt[:r].reshape(r, -1, 3)
    cum = np.cumsum(S ** 2) / (S ** 2).sum()
    print(f"basis: {r} modes, {100 * cum[r - 1]:.1f}% of training variance")

    # floor: what r modes can express on held-out ears, knowing EVERY vertex
    Xte = X[te].reshape(te.sum(), -1)
    coef = Xte @ Wt[:r].T
    floor = np.linalg.norm(
        (Xte - coef @ Wt[:r]).reshape(te.sum(), -1, 3), axis=2).mean()
    print(f"floor from {r} modes, all 1188 vertices known: "
          f"{floor:.3f} mm ({100 * floor / ear:.2f}% of ear)\n")

    vis = visible_face_on(Vt, Ft)
    print(f"visible face-on: {vis.sum()}/{len(Vt)} template vertices "
          f"({100 * vis.sum() / len(Vt):.0f}%)")
    allowed = vis if a.visible_only else None
    sigma_obs = a.sigma * ear
    print(f"assumed landmark noise {a.sigma} x ear = {sigma_obs:.2f} mm\n")
    order = qdeim(B, a.kmax, allowed)
    ks = [k for k in (6, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120)
          if k <= a.kmax]
    print(f"{'K':>5s}{'mean err':>11s}{'% of ear':>10s}{'p95':>9s}")
    curve = []
    for k in ks:
        idx = order[:k]
        Ak = B[:, idx, :].reshape(r, -1).T                   # (3k, r)
        obs = Xte.reshape(te.sum(), -1, 3)[:, idx, :].reshape(te.sum(), -1)
        # Ridge with the MODE VARIANCES as the prior, not plain least
        # squares. At 3K = r the normal equations are square and singular, and
        # an earlier version read 2.02 mm at K=40 against 1.05 at K=30 purely
        # from that -- below it lstsq silently returns a min-norm solution,
        # which is a different estimator, so the curve was not comparable
        # across K. A prior makes one estimator work at every K.
        lam = sigma_obs ** 2 / (S[:r] ** 2 / (tr.sum() - 1))
        c = np.linalg.solve(Ak.T @ Ak + np.diag(lam), Ak.T @ obs.T).T
        rec = (c @ Wt[:r]).reshape(te.sum(), -1, 3)
        e = np.linalg.norm(rec - Xte.reshape(te.sum(), -1, 3), axis=2)
        curve.append((k, e.mean(), np.percentile(e, 95)))
        print(f"{k:5d}{e.mean():11.3f}{100 * e.mean() / ear:9.2f}%"
              f"{np.percentile(e, 95):9.3f}")

    np.savez("runs/minimal_mesh.npz", order=order, curve=np.array(curve),
             mean=mean, faces=Ft, ear=ear, floor=floor)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = np.array(curve)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))
    ax[0].plot(c[:, 0], 100 * c[:, 1] / ear, "o-", label="mean")
    ax[0].plot(c[:, 0], 100 * c[:, 2] / ear, "s--", label="95th pct", alpha=.7)
    ax[0].axhline(100 * floor / ear, color="k", ls=":",
                  label=f"{r}-mode floor")
    ax[0].set_xlabel("vertices kept (K)")
    ax[0].set_ylabel("held-out reconstruction error, % of ear size")
    ax[0].set_title("How few vertices describe an ear")
    ax[0].grid(alpha=.3)
    ax[0].legend()
    from matplotlib.tri import Triangulation
    tri = Triangulation(mean[:, 2], mean[:, 1], Ft)
    ax[1].triplot(tri, lw=.25, color="#bbbbbb")
    for k, col, s in [(60, "#2dd4ff", 26), (30, "#ffb000", 46), (15, "#ff2d95", 70)]:
        ax[1].scatter(mean[order[:k], 2], mean[order[:k], 1], s=s, c=col,
                      edgecolors="k", linewidths=.4, label=f"first {k}", zorder=3)
    ax[1].set_aspect("equal")
    ax[1].axis("off")
    ax[1].legend(loc="lower left", fontsize=8)
    ax[1].set_title("selection order (nested: 15 ⊂ 30 ⊂ 60)")
    fig.tight_layout()
    fig.savefig(a.out, dpi=140)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
