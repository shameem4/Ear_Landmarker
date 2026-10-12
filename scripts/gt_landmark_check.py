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

THE REFERENCE IS THE EAR'S OWN SURFACE. The wrap fits the cut ear to 0.002 mm
median (corpus/fits*.csv, 0.003% of the ear), and the cut inverts onto the
head's own vertices to 3e-05 mm, so once the frames are composed correctly the
landmarks sit 0.06 mm from the scan. There is no "wrap residual" to deduct: a
number from this script is our error, not a budget shared with the reference.

THREE TRAPS, each of which produced a plausible-looking result first. All
three share a signature -- the landmarks stay ON the surface, so a
surface-distance check passes while the answer is wrong.

  THE BARYCENTRICS ADDRESS THE CANONICAL MESH, in corpus/canonical_esc/. Applied
  to the faceform2_* fit, which has different topology, every point still
  lands on that fit's surface, so it reads 3.2 mm and looks like ordinary
  error while actually scattering the landmarks at random. The giveaway is
  structural, not metric: pairwise-distance correlation against a true contour
  fell to 0.04 and the consecutive-step test showed 23 jumps instead of 3.

  EACH WRAP GENERATION PAIRS WITH ITS OWN CUT GENERATION. wrapped2_* sits on
  ears_hutubs at 0.002 mm and on ears4_hutubs at 0.856 mm; wrapped3_* is the
  other way round. Crossing them looks exactly like a 0.9 mm wrap residual.

  THE CUT FRAME IS A NORMALISED HEAD, per dataset -- hutubs cuts are relative
  to a centred, unit-radius head and sonicom's to raw vertices.
  corpus/tf_mm_*/matrix omits the scan's centroid and is 5.56 mm adrift on
  pp10. This is why the chain is composed once in
  scripts/canonical_to_scan.py and only read here.

A right ear is additionally fitted and stored in LEFT form, mirrored about z,
which must be undone before the cut's matrix.

    python scripts/gt_landmark_check.py --cons data/mv_hutubs/pp10_e1 \
        --subject pp10 --ear right
"""
import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
WRAP = ROOT.parent / "wrap_work"
sys.path.insert(0, str(ROOT))

def _strip_names():
    """From ear3d.scheme, never retyped. Hardcoding them here had 35-49 as
    'concha/tragus' and 50-54 as 'lobe'; they are the concha border and the
    superior crus, and the 55-point scheme has no lobe point at all."""
    from ear3d.scheme import STRIPS, STRIP_NAMES
    return {int(a): n.replace("_", " ") for (a, _), n in zip(STRIPS, STRIP_NAMES)}


STRIP_NAME = _strip_names()

# How far from the scan the finished reference is allowed to be. The wrap fits
# the cut ear to 0.002 mm and the cut inverts onto the head's own vertices
# exactly, so anything above a few hundredths of a millimetre means a link in
# the chain is wrong -- which is the failure this whole function exists to
# prevent recurring. Measured on hutubs pp10: 0.058 mm median, 0.42 max.
SURFACE_TOL_MM = 1.0


# The marks the reference is built from. v6_spacing keeps the hand-marked
# curve and takes only the SPACING along it from the shipped landmarker, which
# is the half of the old disagreement that was a convention rather than an
# error. Scored on hutubs pp10_right, RANSAC consensus: 6.08 mm against the
# hand marks, 3.61 against these. Pass --marks to compare with another file.
DEFAULT_MARKS = WRAP / "templates/mark_mean_v6_spacing.wrap"


def read_marks(path):
    """(triangle, barycentric, names) from a Wrap SelectPoints project or npz.

    The same convention as scripts/wrap_marks_to_npz.py: u weights the
    triangle's first vertex, v the second, the remainder the third.
    """
    path = pathlib.Path(path)
    if path.suffix == ".npz":
        z = np.load(path, allow_pickle=True)
        return (z["triangle"], z["barycentric"],
                [str(n) for n in z["point_names"]])
    import json
    d = json.loads(path.read_text())
    sel = [n for n in d["nodes"].values() if n["nodeType"] == "SelectPoints"]
    if not sel:
        sys.exit(f"{path}: no SelectPoints node")
    pts = sel[0]["params"]["points"]["value"]
    n = len(pts)
    tri, bary, names = np.zeros(n, int), np.zeros((n, 3)), [""] * n
    for k, q in pts.items():
        j = int(k[:2])
        names[j] = k
        u, v = float(q["u"]), float(q["v"])
        tri[j], bary[j] = int(q["triangleInd"]), (u, v, 1.0 - u - v)
    return tri, bary, names


def gt_landmarks(dataset, subject, ear, n_points, verify=True, marks=None):
    """The wrap's landmarks for one ear, in that scan's own frame, in mm.

    BUILT FROM THE MARKS, NOT FROM A CACHE. A landmark is a (triangle,
    barycentric) address on the template, and every canonical mesh carries the
    template's topology, so the address evaluates directly on this ear's
    canonical mesh. lm3d_v5_*_mm.npz was that evaluation stored, which meant a
    re-marking left a stale copy behind; now there is one source.

    ONE LOOKUP FOR THE FRAME. corpus/canon2scan/<name>.npz holds a single 4x4
    taking canonical millimetres to this scan's, verified once by
    scripts/canonical_to_scan.py -- all 476 ears land on their scan at 0.0 mm
    and it comes out a pure rigid motion. Composing that by hand is what
    produced three bugs in an afternoon, each of which left the landmarks ON
    the surface and so passed a surface-distance check.
    """
    import open3d as o3d
    from ear3d.scheme import STRIPS

    name = f"{subject}_{ear}"
    tri, bary, _ = read_marks(marks or DEFAULT_MARKS)
    gp = WRAP / f"corpus/canonical_esc/{dataset}/{name}.obj"
    if not gp.exists():
        sys.exit(f"{gp} not found")
    m = o3d.io.read_triangle_mesh(str(gp))
    V, F = np.asarray(m.vertices), np.asarray(m.triangles)
    if tri.max() >= len(F):
        sys.exit(f"the marks address triangle {tri.max()} but {gp.name} has "
                 f"{len(F)} -- wrong template topology")
    L = np.einsum("ij,ijk->ik", bary, V[F[tri]])

    tp = WRAP / f"corpus/canon2scan/{name}.npz"
    if not tp.exists():
        sys.exit(f"{tp} not found -- run scripts/canonical_to_scan.py")
    t = np.load(tp, allow_pickle=True)
    T = t["matrix"]
    L = L @ T[:3, :3].T + T[:3, 3]

    if verify:
        head = o3d.io.read_triangle_mesh(str(t["scan"]))
        Vh = np.asarray(head.vertices)
        Vh = Vh * (1000.0 if float(np.ptp(Vh, axis=0).max()) < 10.0 else 1.0)
        sc = o3d.t.geometry.RaycastingScene()
        sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(
            o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(Vh),
                                      head.triangles)))
        d = sc.compute_distance(o3d.core.Tensor(
            np.ascontiguousarray(L, np.float32))).numpy()
        if np.median(d) > SURFACE_TOL_MM:
            sys.exit(f"the reference lands {np.median(d):.3f} mm from "
                     f"{t['scan']}, over the {SURFACE_TOL_MM} mm tolerance "
                     f"-- a link in the chain is wrong, NOT the data")
    return L[:n_points], np.asarray(STRIPS), str(t["scan"])


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
    p.add_argument("--marks", default="",
                   help="a .wrap or .npz of marks; default mark_mean_v6_spacing")
    p.add_argument("--reparam", action="store_true",
                   help="split the error into a labelling-convention part and "
                        "the rest, by letting each strip's reference slide "
                        "along its own arc")
    a = p.parse_args()

    import open3d as o3d
    G, strips, src = gt_landmarks(a.dataset, a.subject, a.ear, a.n_points,
                                  marks=a.marks or None)
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
          f"-- it IS the ear's surface, so what follows is our error with "
          f"nothing to deduct\n")

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

    if a.reparam:
        # WHY THIS IS NOT JUST FITTING THE ERROR AWAY. A rigid fit to the
        # whole set absorbs these offsets as an 11.9 degree rotation, which
        # is what they were first reported as; per-strip the apparent angles
        # are -11, -9, -2 and -23 degrees, so there is no one rotation and the
        # cause is points sitting further along each contour. Depth is not
        # involved: lifting the same 2D points with the renderer's own depth
        # buffer gives the same offsets.
        print("\nSLIDING THE REFERENCE ALONG EACH CONTOUR")
        print(f"{'strip':<16}{'as-is':>8}{'fitted':>8}{'alpha':>8}{'beta':>8}"
              f"{'fixed':>8}")
        tot = {"as": [], "fit": [], "fix": []}
        for s_, b_ in strips:
            s_, b_ = int(s_), int(min(b_, len(G)))
            if s_ >= len(G):
                break
            Gs, Ps = G[s_:b_], P[s_:b_]
            e0 = np.linalg.norm(Ps - Gs, axis=1)
            rf, q = _reparam_fit(Gs, Ps)
            al, be = ARC_FIX.get(s_, (1.0, 0.0))
            ef = np.linalg.norm(Ps - _resample(Gs, len(Ps), al, be), axis=1)
            print(f"  {STRIP_NAME.get(s_, ''):<14}{np.median(e0):>8.2f}"
                  f"{rf:>8.2f}{q[0]:>8.3f}{q[1]:>+8.3f}{np.median(ef):>8.2f}")
            tot["as"] += list(e0); tot["fix"] += list(ef)
        print(f"  {'all':<14}{np.median(tot['as']):>8.2f}{'':>8}{'':>8}{'':>8}"
              f"{np.median(tot['fix']):>8.2f}")
        print("'fixed' uses ONE correction for every ear, not one fitted here."
              "\nWhat remains after it is error no relabelling can explain.")

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
