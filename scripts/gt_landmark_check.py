#!/usr/bin/env python3
"""Score multi-view landmarks against the wrap's own 3D landmarks, point for point.

Surface distance -- what scan_depth_check and moge_multiview report -- measures
distance to ANY surface, so a landmark that slides along the ear, or off it onto
the scalp, scores well. Measured here across 21 views of pp10_right, surface
distance and paired error rank the views with a correlation of +0.01: it tells
you essentially nothing about which view is accurate.

The wrap gives a reference with CORRESPONDENCE. Every landmark is a (triangle,
barycentric) address on the template, so a wrapped ear carries all 95 of them at
known indices, and point i can be compared with point i. That catches tangential
error and escapes, which no surface metric can.

TWO TRAPS, both of which produced plausible-looking nonsense before being found:

  THE BARYCENTRICS INDEX THE TEMPLATE TOPOLOGY (1188 vertices, 2296 faces) --
  wrapped2_*, NOT the faceform2_* fit, which is a different mesh with 13776
  faces. Applying them to the fit still lands every point ON the fit surface, so
  the result looks fine by surface distance (3.2 mm) while actually scattering
  the landmarks at random: pairwise-distance correlation against a true contour
  was 0.04, and the consecutive-step test showed 23 jumps instead of 3.

  A RIGHT EAR IS STORED IN LEFT FORM, mirrored about z. The mirror has to be
  undone BEFORE the cut-to-head matrix, exactly as tf_mm_*/README says. Skipping
  it leaves the ear on the head but visibly wrong: p90 8.43 mm against 4.78.

WHAT THE REFERENCE CANNOT DO. The wrap residual is ~2 mm normal to the surface,
and the landmarks sit on the wrap, so both this reference and our estimate lie
~1.9 mm from the scan. The disagreement therefore lives almost entirely
TANGENTIALLY, along the contour, which is the one direction the wrap residual
does not constrain. So a paired number here is an upper bound on our error and
cannot be split between us and the reference. It is a valid RANKING instrument,
because every view is scored against the same fixed reference, and a valid
detector of gross displacement. It is not an accuracy figure.

    python scripts/gt_landmark_check.py --cons data/mv_hutubs/pp10_e1 \
        --subject pp10 --ear right
"""
import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
WRAP = ROOT.parent / "wrap_work"

# The template topology the barycentrics address. Asserted, not assumed: the
# wrong mesh fails silently and only shows up as scrambled correspondence.
TEMPLATE_FACES = 2296

STRIP_NAME = {0: "outer helix", 20: "inner helix", 35: "concha/tragus",
              50: "lobe"}


def gt_landmarks(dataset, subject, ear, n_points):
    """The wrap's landmarks for one ear, in that head scan's own frame, in mm."""
    z = np.load(WRAP / f"templates/lm3d_v5_{dataset}_mm.npz", allow_pickle=True)
    tri, bary, strips = z["triangle"], z["barycentric"], z["strips"]

    import open3d as o3d
    wrapped = WRAP / f"corpus/wrapped2_{dataset}/{subject}_{ear}.obj"
    m = o3d.io.read_triangle_mesh(str(wrapped))
    V, F = np.asarray(m.vertices), np.asarray(m.triangles)
    if len(F) != TEMPLATE_FACES:
        sys.exit(f"{wrapped} has {len(F)} faces, expected the template's "
                 f"{TEMPLATE_FACES} -- the barycentrics index the WRAPPED "
                 f"template, not a fit mesh")

    tf = np.load(WRAP / f"corpus/tf_mm_{dataset}/{subject}_{ear}.npz",
                 allow_pickle=True)
    M, to_mm = tf["matrix"], float(tf["to_mm"])
    X = np.einsum("ij,ijk->ik", bary, V[F[tri]]) * to_mm
    if ear == "right":
        X = X * np.array([1.0, 1.0, -1.0])      # undo the left-form mirror
    X = X @ M[:3, :3].T + M[:3, 3]
    return X[:n_points], strips, str(tf["source"])


def contour_check(P, strips, label):
    """A correctly ordered landmark set jumps only BETWEEN strips."""
    step = np.linalg.norm(np.diff(P, axis=0), axis=1)
    want = {int(b) - 1 for _, b in strips if int(b) - 1 < len(step)}
    got = set(np.where(step > 3.0 * np.median(step))[0].tolist())
    stray = got - want
    if stray:
        print(f"  WARNING {label}: {len(stray)} jump(s) inside a strip at "
              f"{sorted(stray)} -- correspondence is probably scrambled")
    return np.median(step)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cons", required=True, help="a moge_multiview output dir")
    p.add_argument("--subject", required=True)
    p.add_argument("--ear", required=True, choices=["left", "right"])
    p.add_argument("--dataset", default="hutubs")
    p.add_argument("--n-points", type=int, default=55)
    p.add_argument("--per-view", action="store_true")
    a = p.parse_args()

    import open3d as o3d
    G, strips, src = gt_landmarks(a.dataset, a.subject, a.ear, a.n_points)
    c = np.load(pathlib.Path(a.cons) / "consensus.npz", allow_pickle=True)

    head = o3d.io.read_triangle_mesh(src)
    Vh = np.asarray(head.vertices)
    # metres or millimetres: no head is under 10 mm tall by any measure
    Vh = Vh * (1000.0 if float(np.ptp(Vh, axis=0).max()) < 10.0 else 1.0)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(
        o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(Vh),
                                  head.triangles)))

    def surf(P):
        return scene.compute_distance(
            o3d.core.Tensor(np.ascontiguousarray(P, np.float32))).numpy()

    print(f"{a.dataset} {a.subject}_{a.ear}, reference from the wrap")
    contour_check(G, strips, "reference")
    print(f"  reference sits {np.median(surf(G)):.2f} mm from the scan "
          f"(the wrap residual -- our estimate cannot beat this normal to "
          f"the surface)\n")

    print(f"{'estimator':<16}{'GT med':>8}{'GT p90':>8}{'GT max':>8}"
          f"{'surf med':>10}")
    for est in ["cons_median", "cons_ransac", "cons_geomedian"]:
        if est not in c.files:
            continue
        P = c[est] * 1000.0
        e = np.linalg.norm(P - G, axis=1)
        print(f"{est.removeprefix('cons_'):<16}{np.median(e):>8.2f}"
              f"{np.percentile(e, 90):>8.2f}{e.max():>8.2f}"
              f"{np.median(surf(P)):>10.2f}")

    P = c["cons_median"] * 1000.0
    u = np.triu_indices(len(G), 1)
    DG = np.linalg.norm(G[:, None] - G[None], axis=2)
    DP = np.linalg.norm(P[:, None] - P[None], axis=2)
    print(f"\nshape agreement: pairwise-distance corr "
          f"{np.corrcoef(DG[u], DP[u])[0, 1]:.3f} -- below ~0.9 means the "
          f"correspondence is wrong, not the geometry")

    print("\nby strip (median estimator):")
    for s, b in strips:
        s, b = int(s), int(b)
        if s >= len(G):
            break
        e = np.linalg.norm(P[s:min(b, len(G))] - G[s:min(b, len(G))], axis=1)
        print(f"  {s:2d}-{min(b, len(G)) - 1:2d} {STRIP_NAME.get(s, ''):<14}"
              f"median {np.median(e):5.2f}  max {e.max():5.2f}")

    if a.per_view:
        rows = []
        for E, t in zip(c["est"] * 1000.0, [str(t) for t in c["tags"]]):
            ok = np.isfinite(E).all(1)
            if ok.sum() < 10:
                continue
            rows.append((t, np.median(surf(E[ok])),
                         np.median(np.linalg.norm(E[ok] - G[ok], axis=1))))
        print(f"\n{'view':<12}{'surf med':>10}{'GT med':>9}")
        for t, s, g in sorted(rows, key=lambda r: r[2]):
            print(f"{t:<12}{s:>10.2f}{g:>9.2f}")
        S = np.argsort(np.argsort([r[1] for r in rows]))
        Q = np.argsort(np.argsort([r[2] for r in rows]))
        print(f"\nrank correlation of the two metrics over {len(rows)} views: "
              f"{np.corrcoef(S, Q)[0, 1]:+.3f}")
        print("near zero means surface distance cannot tell a good view from "
              "a bad one, and should not be used to choose between them")


if __name__ == "__main__":
    main()
