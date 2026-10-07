"""Non-rigid template fitting: wrap a template mesh onto a scan.

This is Amberg et al., "Optimal Step Nonrigid ICP Algorithms for Surface
Registration" (CVPR 2007) -- the algorithm behind the class of tool Faceform
Wrap belongs to, and the reference this implementation is measured against.

WHY IT EXISTS. The 3D path lifts 2D landmarks through a depth buffer, and that
is not dependable: the points land on the scalp or the wrong fold depending on
the viewpoint. Template fitting answers a different question. It never consults
the 2D model -- it slides a known mesh onto the scan until the surfaces agree,
so the correspondence comes from geometry. Mark the landmarks ONCE on the
template and every wrapped scan carries them.

THE MODEL. Each template vertex i gets its own 3x4 affine X_i, and the fit
minimises three terms:

    data       sum_i w_i |X_i [v_i;1] - u_i|^2      pull each vertex to the scan
    stiffness  alpha sum_(i,j) |(X_i - X_j) G|^2    neighbours deform alike
    landmarks  beta  sum_l |X_l [v_l;1] - a_l|^2    hold the anchors

Everything is linear in X, so each iteration is one sparse least-squares solve
rather than a gradient descent -- deterministic, and no step size to tune. The
"optimal step" of the title is that alpha is ANNEALED: start stiff, so the
template moves almost rigidly and cannot fold itself into a local minimum, then
relax so it can take on the scan's detail. Starting at the flexible end finds a
different, worse answer, which is the usual way this algorithm is got wrong.

ON GPU. Deliberately not. The template is ~1k vertices, so the linear system is
~5k unknowns and solves in milliseconds; the nearest-neighbour lookup is a
k-d tree query of 1k points against at most a few hundred thousand, which is
microseconds. There is nothing here for a GPU to do, and a torch dependency
would buy latency rather than spend it.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

__all__ = ["load_mesh", "edges_of", "rigid_fit", "rigid_icp", "nicp",
           "project_to_surface", "bary_to_xyz", "boundary_vertices",
           "smooth_border", "DEFAULT_SCHEDULE"]


# Stiffness, from very stiff to free. Each entry is run to convergence before
# the next begins. The tail matters: the last few values are what lets the
# template enter the concha and follow the helix fold.
DEFAULT_SCHEDULE = (50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05)


def load_mesh(path):
    """Read a mesh as (vertices, triangles).

    Through trimesh, not Open3D: the ear templates are QUAD meshes, and
    Open3D's OBJ reader returns zero vertices for them without raising.
    """
    import trimesh
    m = trimesh.load(path, process=False, force="mesh")
    return np.asarray(m.vertices, float), np.asarray(m.faces, int)


def bary_to_xyz(V, F, pts):
    """Barycentric picks [(triangle, u, v), ...] -> 3D, the way Wrap stores them."""
    out = np.empty((len(pts), 3))
    for k, (ti, u, v) in enumerate(pts):
        a, b, c = V[F[int(ti)]]
        out[k] = (1.0 - u - v) * a + u * b + v * c
    return out


def edges_of(F):
    """Unique undirected edges of a triangle mesh, as an (m, 2) array."""
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e.sort(axis=1)
    return np.unique(e, axis=0)


def boundary_vertices(F):
    """Vertices on an open edge of the mesh.

    They are why an unanchored fit SHRINKS. A border vertex has no true
    correspondence -- the template's edge is a cut through the anatomy, not a
    feature of it -- so pulling it to its nearest scan point always drags it
    inward, the rim creeps in, and the fit ends up lying perfectly on the
    surface while covering only the middle of the ear. The surface gap reports
    that as excellent, which is why it has to be excluded rather than measured.
    """
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e.sort(axis=1)
    uniq, cnt = np.unique(e, axis=0, return_counts=True)
    return np.unique(uniq[cnt == 1])


def vertex_normals(V, F):
    """Area-weighted vertex normals."""
    tn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    N = np.zeros_like(V)
    for c in range(3):
        np.add.at(N, F[:, c], tn)
    n = np.linalg.norm(N, axis=1, keepdims=True)
    return N / np.where(n > 0, n, 1.0)


def rigid_fit(src, dst, with_scale=True):
    """Similarity transform taking `src` onto `dst` for corresponding points.

    Returns (R, t, s) with  x -> s * x @ R.T + t.  The reflection guard matters:
    without it an SVD can return a mirror, which silently turns a left ear into
    a right one and is invisible until a shape model averages the two.
    """
    sc, dc = src.mean(0), dst.mean(0)
    S, D = src - sc, dst - dc
    U, sig, Vt = np.linalg.svd(S.T @ D)
    R = (U @ Vt).T
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = (U @ Vt).T
    s = float(sig.sum() / (S ** 2).sum()) if with_scale else 1.0
    return R, dc - s * sc @ R.T, s


def rigid_icp(Vt, Ft, Vs, Fs, iters=30, min_cos=0.5, trim=0.8, with_scale=True,
              verbose=False):
    """Refine a similarity against the SURFACE, not just the anchors.

    A handful of hand-placed anchors determines a 7-DOF similarity only barely
    -- three pairs give nine equations for seven unknowns, and each pair carries
    its own placement error -- so the pose that comes out of them can be off
    even when every anchor is sensible. The whole surface has thousands of
    constraints and settles it.

    `trim` keeps only that fraction of the closest matches each iteration. The
    template covers the ear and the scan may be a whole head, so the worst
    matches are vertices reaching for cheek or scalp; letting them vote drags
    the template off the ear.
    """
    V = np.asarray(Vt, float).copy()
    Ns = vertex_normals(Vs, Fs)
    tree = cKDTree(Vs)
    R_t, t_t, s_t = np.eye(3), np.zeros(3), 1.0
    for i in range(iters):
        d, idx = tree.query(V)
        ok = np.einsum("ij,ij->i", vertex_normals(V, Ft), Ns[idx]) > min_cos
        if ok.sum() < 10:
            break
        cut = np.quantile(d[ok], trim)
        ok &= d <= cut
        R, t, s = rigid_fit(V[ok], Vs[idx][ok], with_scale=with_scale)
        Vn = s * V @ R.T + t
        step = float(np.abs(Vn - V).max())
        V = Vn
        R_t, t_t, s_t = R @ R_t, s * t_t @ R.T + t, s * s_t
        if step < 1e-6 * float(np.ptp(Vt, axis=0).max()):
            break
    if verbose:
        print(f"  rigid icp: {i + 1} iters, total scale {s_t:.4f}, "
              f"median gap {np.median(d[ok]):.4f}")
    return V, (R_t, t_t, s_t)


def _design(V):
    """Sparse D with row i = [v_i, 1] in block i, so (D X)_i = X_i [v_i;1]."""
    n = len(V)
    rows = np.repeat(np.arange(n), 4)
    cols = (4 * np.repeat(np.arange(n), 4) + np.tile(np.arange(4), n))
    vals = np.hstack([np.c_[V, np.ones(n)]]).ravel()
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, 4 * n))


def nicp(Vt, Ft, Vs, Fs, anchors=None, schedule=DEFAULT_SCHEDULE,
         gamma=1.0, beta=10.0, max_iter=12, tol=1e-3,
         max_dist=None, min_cos=0.5, free_border=True, verbose=True):
    """Fit template (Vt, Ft) onto scan (Vs, Fs). Returns the deformed vertices.

    `anchors` is (template_vertex_indices, target_xyz) -- the handful of point
    pairs that tell the fit which way round the ear goes. They are weighted by
    `beta` and decay as the stiffness relaxes: they are there to steer the
    coarse stages, and once the surfaces agree the geometry knows better than
    four hand-placed points do.

    A correspondence is REJECTED (weight 0) when the scan normal disagrees with
    the template normal by more than `min_cos`, or when it is further than
    `max_dist`. Without that, vertices at the ear's rim grab the skin behind it
    and drag the helix flat -- the single most common way this fails.
    """
    # WORK IN A NORMALISED FRAME. The design row is [v, 1], so the constant
    # competes with the coordinates: at |v| ~ 0.1 it dominates and at |v| ~ 50 it
    # is noise. gamma, which balances the translation column of each affine
    # against its linear part, therefore means something different at every
    # scale -- and the same mesh in metres and in millimetres fits differently.
    # It cost a corpus run: ears in normalised head units came out mangled while
    # the identical pipeline on millimetre meshes was fine.
    Vt = np.asarray(Vt, float)
    _c = Vt.mean(0)
    _k = float(np.ptp(Vt, axis=0).max())
    if not np.isfinite(_k) or _k <= 0:
        raise RuntimeError("template has no extent")
    Vt = (Vt - _c) / _k
    Vs = (np.asarray(Vs, float) - _c) / _k
    if max_dist is not None:
        max_dist = max_dist / _k
    E = edges_of(Ft)
    n, m = len(Vt), len(E)
    border = boundary_vertices(Ft) if free_border else np.array([], int)
    Ns = vertex_normals(Vs, Fs)
    tree = cKDTree(Vs)
    if max_dist is None:
        # A fraction of the TEMPLATE, never of the scan. Scaling it by the scan
        # means the same template is allowed a 21-unit reach when wrapped onto a
        # cropped ear and 61 onto a whole head -- so on a full head the rim
        # happily grabs the cheek behind it and the fit slides off the ear.
        max_dist = 0.25 * float(np.ptp(Vt, axis=0).max())

    # stiffness: alpha * kron(M, G), M the node-arc incidence matrix of the edges
    G = sp.diags([1.0, 1.0, 1.0, gamma])
    M = sp.csr_matrix(
        (np.hstack([np.ones(m), -np.ones(m)]),
         (np.tile(np.arange(m), 2), np.hstack([E[:, 0], E[:, 1]]))), shape=(m, n))
    MG = sp.kron(M, G).tocsr()
    Z = np.zeros((4 * m, 3))

    # D is built from the ORIGINAL template vertices and never changes: X_i is
    # the map from where vertex i STARTED, not from where it currently sits.
    # Rebuilding it from the deformed vertices each iteration would compose the
    # deformation with itself and run away.
    D = _design(Vt)
    if anchors is not None:
        li, la = anchors
        li = np.asarray(li, int)
        la = (np.asarray(la, float) - _c) / _k
        DL = D[li]

    V = Vt.copy()
    for stage, alpha in enumerate(schedule):
        # the anchors hand over to the surface as the template loosens
        b = beta * (alpha / schedule[0]) ** 0.5
        for it in range(max_iter):
            d, idx = tree.query(V)
            U = Vs[idx]
            Nv = vertex_normals(V, Ft)
            w = ((d < max_dist)
                 & (np.einsum("ij,ij->i", Nv, Ns[idx]) > min_cos)).astype(float)
            w[border] = 0.0          # the border is carried by stiffness alone
            if w.sum() < 4:
                raise RuntimeError("every correspondence was rejected; the "
                                   "template is probably not aligned to the scan")
            W = sp.diags(w)
            A = sp.vstack([alpha * MG, W @ D]).tocsr()
            B = np.vstack([Z, w[:, None] * U])
            if anchors is not None:
                A = sp.vstack([A, b * DL]).tocsr()
                B = np.vstack([B, b * la])
            # normal equations: A is tall and very sparse, and A^T A is ~5k
            # square, so one sparse solve beats an iterative least squares.
            AtA = (A.T @ A).tocsc()
            AtA += sp.eye(AtA.shape[0], format="csc") * 1e-9   # keep it posdef
            Xn = np.asarray(spsolve(AtA, A.T @ B))
            Vn = D @ Xn
            step = float(np.abs(Vn - V).max())
            V = Vn
            if step < tol * float(np.ptp(Vt, axis=0).max()):
                break
        if verbose:
            print(f"  stage {stage + 1}/{len(schedule)} alpha {alpha:6.2f}: "
                  f"{it + 1} iters, {int(w.sum())}/{n} matched, "
                  f"median gap {_k * np.median(d[w > 0]):.4f}")
    return V * _k + _c


def project_to_surface(V, Vs, Fs, max_move=None):
    """Snap vertices onto the nearest point of the scan surface.

    The counterpart of Wrap's ProjectGeom, and the reason surface distance is
    NOT a measure of a wrap's quality: after this step it is near zero by
    construction, whether or not the template landed on the right anatomy.
    """
    import trimesh
    mesh = trimesh.Trimesh(vertices=Vs, faces=Fs, process=False)
    close, dist, _ = trimesh.proximity.closest_point(mesh, V)
    if max_move is not None:
        keep = dist <= max_move
        out = V.copy()
        out[keep] = close[keep]
        return out, dist
    return close, dist


def smooth_border(V, F, iters=12, rings=2):
    """Laplacian-smooth the border ring of a mesh, leaving the interior alone.

    An averaged corpus is rough around its edge and nowhere else, because the
    border is the one part the fit does not pull onto the scan -- it has no true
    correspondence, so it is carried by stiffness and lands somewhere slightly
    different in every subject. Averaging those leaves a wobble, worst where the
    template's edge crosses the lobe-to-cheek junction, which has no landmark to
    settle it.

    Smoothing it discards nothing: there is no measurement there to lose. The
    interior, which IS driven by the scans, is not touched.
    """
    V = np.asarray(V, float).copy()
    import collections
    adj = collections.defaultdict(set)
    for t in F:
        for i in range(3):
            adj[t[i]].add(t[(i + 1) % 3])
            adj[t[i]].add(t[(i + 2) % 3])
    band = set(boundary_vertices(F).tolist())
    for _ in range(rings - 1):
        band |= {n for v in list(band) for n in adj[v]}
    band = sorted(band)
    nbr = [sorted(adj[v]) for v in band]
    for _ in range(iters):
        upd = np.array([V[n].mean(0) for n in nbr])
        V[band] = 0.5 * V[band] + 0.5 * upd
    return V
