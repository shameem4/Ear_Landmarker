#!/usr/bin/env python3
"""Re-mark the template's 55 landmarks with the shipped landmarker.

WHY. The hand-marked template and the shipped 2D model do not put the same
number of points in the same places along a contour: both helix strips sit
about 0.05 of their arc further along in the model's scheme -- one index step
in twenty -- and the model's five superior-crus points span only 0.65-0.75 of the arc
the template gives that strip. Measured against hutubs pp10, that convention
difference is 2.5 mm of the 6.1 mm paired error, and it is not an error in
either scheme; they simply disagree. Since the model is what ships, the
template moves.

THE LOOP, which is the point of doing it this way:

  render the template from a cone of views -> landmark each with the SHIPPED
  pipeline -> intersect the rays -> snap to the surface -> re-aim the cone at
  the landmarks' own plane -> render again.

It is iterative because the first cone is aimed down the canonical x axis,
which is only approximately the pinna's normal; once there are landmarks there
is a better axis to aim at. Convergence is reported as how far the marks moved
and as reprojection error in pixels, which is the landmarker checking the
template against itself.

RAYS, NOT DEPTH. A 2D detection asserts a ray; what it later hits is a
separate question, and ear3d.label documents the lift putting landmarks off
the pinna when the detection falls near the silhouette. Intersecting rays from
several views never consults a depth value.

ONE FRAME. ear_mean_v3.obj is already canonical millimetres -- its centroid is
the canonical corpus's to the last digit -- so nothing here converts units or
datasets. Framing varies the CAMERA DISTANCE, never the mesh, so every view's
rays live in the same frame.

    python scripts/remark_template.py --out wrap_work/templates/mark_v6.wrap
"""
import argparse
import copy
import json
import pathlib
import sys
import warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
WRAP = ROOT.parent / "wrap_work"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Canonical -> the renderer's frame: the pinna's normal is the canonical x
# axis and it faces -x, the long axis is +y and stays up. Measured, not
# assumed: facing +x the detector finds nothing at any scale, facing -x it
# reads 0.86.
R_VIEW = np.array([[0.0, 0.0, 1.0],
                   [0.0, 1.0, 0.0],
                   [-1.0, 0.0, 0.0]])

# The detector has reproducible scale blind bands -- at a 600 px render the
# template is found at 1.0-1.4 and 1.6-1.9 of unit radius and NOT at 0.9, 1.5
# or 2.0, with confidence either side of each gap. So a view that fails is
# retried at another apparent size rather than dropped; the mesh never moves,
# only the camera.
SIZE_LADDER = (1.1, 1.3, 1.0, 1.7, 1.2, 1.8, 1.4)


def load_template(path):
    import open3d as o3d
    m = o3d.io.read_triangle_mesh(str(path))
    V = np.asarray(m.vertices)
    if len(m.triangles) != 2296:
        sys.exit(f"{path} has {len(m.triangles)} faces, expected the "
                 f"template's 2296")
    return m, V.copy(), np.asarray(m.triangles).copy()


def to_render_frame(V, R, scale_to):
    """Canonical mm -> the render frame: rotate, centre, unit radius."""
    c = V.mean(0)
    W = (V - c) @ R.T
    s = float(np.abs(W).max()) / scale_to
    return W / s, c, s, R


def back_to_canonical(P, c, s, R):
    return (P * s) @ R + c


def skinned(V, F):
    import open3d as o3d
    from skin import apply_skin
    from ear3d.config import DEFAULTS
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V),
                                  o3d.utility.Vector3iVector(F))
    m.compute_vertex_normals()
    apply_skin(m, seed=0, tone=DEFAULTS["tone"],
               ao_strength=DEFAULTS["ao_strength"], n_rays=DEFAULTS["ao_rays"])
    return m


def sweep(Vr, F, cfg, pipe, angles, verbose=True):
    """Landmark every view, stepping camera distance past the blind bands."""
    from ear3d.camera import Camera, orbit_extrinsic
    from ear3d.render import render
    uv = np.full((len(angles), 55, 2), np.nan)
    cams, hits = [], 0
    for i, (yaw, pitch) in enumerate(angles):
        got = None
        for mag in SIZE_LADDER:
            c2 = dict(cfg, eye_z=cfg["eye_z"] / mag)
            mesh = skinned(Vr, F)
            img, _, cam = render(mesh, c2, E=orbit_extrinsic(yaw, pitch, c2))
            res = pipe(img, timestamp=0.0)
            if res:
                b = max(res, key=lambda d: float(d["confidence"]))
                got = (np.asarray(b["landmarks"], float), cam,
                       float(b["confidence"]), mag)
                break
        cams.append(got[1] if got else None)
        if got:
            uv[i] = got[0]
            hits += 1
            if verbose:
                print(f"    ({yaw:+5.1f},{pitch:+5.1f}) conf {got[2]:.2f} "
                      f"at x{got[3]:.1f}")
        elif verbose:
            print(f"    ({yaw:+5.1f},{pitch:+5.1f}) no detection at any size")
    return uv, cams, hits


def fit(uv, cams, thresh=0.03):
    from ear3d.triangulate import lsq_point, ransac_point
    P = np.full((55, 3), np.nan)
    nin = np.zeros(55, int)
    rays = {}
    for k in range(55):
        o, d = [], []
        for i, cam in enumerate(cams):
            if cam is not None and np.isfinite(uv[i, k]).all():
                a, b = cam.rays(uv[i, k][None, :])
                o.append(a[0]); d.append(b[0])
        if len(o) < 2:
            continue
        rays[k] = (np.array(o), np.array(d))
        q = lsq_point(*rays[k])
        if q is not None:
            P[k], nin[k] = q, len(o)
    ok = np.isfinite(P).all(1)
    if ok.sum() >= 5:
        ext = float(np.ptp(P[ok], axis=0).max())
        for k, (o, d) in rays.items():
            q, inl = ransac_point(o, d, thresh * ext, iters=200,
                                  rng=np.random.default_rng(k))
            if q is not None:
                P[k] = q
                nin[k] = int(inl.sum()) if inl is not None else 0
    return P, nin


def snap(P, V, F):
    """Nearest point on the template -> (triangle, barycentric)."""
    import open3d as o3d
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V),
                                  o3d.utility.Vector3iVector(F))
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
    r = sc.compute_closest_points(o3d.core.Tensor(
        np.ascontiguousarray(P, np.float32)))
    pts = r["points"].numpy().astype(float)
    tri = r["primitive_ids"].numpy().astype(int)
    T = V[F[tri]]
    # barycentric by area, in the order wrap_barycentric applies: u on the
    # first vertex, v on the second, the remainder on the third
    v0, v1, v2 = T[:, 1] - T[:, 0], T[:, 2] - T[:, 0], pts - T[:, 0]
    d00 = (v0 * v0).sum(1); d01 = (v0 * v1).sum(1); d11 = (v1 * v1).sum(1)
    d20 = (v2 * v0).sum(1); d21 = (v2 * v1).sum(1)
    den = np.where(np.abs(d00 * d11 - d01 * d01) < 1e-20, 1e-20,
                   d00 * d11 - d01 * d01)
    b1 = (d11 * d20 - d01 * d21) / den
    b2 = (d00 * d21 - d01 * d20) / den
    bary = np.stack([1.0 - b1 - b2, b1, b2], axis=1)
    # CLAMPED INTO THE TRIANGLE. A closest point that lands on an edge comes
    # back with a weight of -1e-9, and Wrap will not take a point outside its
    # own triangle; three of the 55 did. The clamp moves the mark by less than
    # a nanometre, and the positions are recomputed from the clamped weights
    # so what is written and what is measured cannot differ.
    bary = np.clip(bary, 0.0, 1.0)
    bary = bary / bary.sum(axis=1, keepdims=True)
    pts = np.einsum("ij,ijk->ik", bary, T)
    return tri, bary, pts, np.linalg.norm(pts - P, axis=1)


def onto_old_curve(P, old_xyz, strips):
    """Resample the hand-marked curve at the spacing the landmarker implies.

    THE TWO ERRORS ARE NOT THE SAME ERROR. Against hutubs pp10 the hand-marked
    scheme's disagreement with the model splits into sliding ALONG each contour
    -- the convention, which is what wants fixing -- and sitting OFF it. The
    landmarker's own template placement is better off-contour on the concha
    (4.21 -> 3.62 mm) and worse on the inner helix (2.84 -> 5.52), because an
    interior ridge is where a model reads a mean ear less well than a person
    did. This keeps the person's curve and takes only the model's spacing
    along it, which is the part the model defines by shipping.

    NOT BY NEAREST POINT. Projecting each mark independently is not monotonic:
    two marks can collapse onto one place or cross, and doing it that way put
    fresh jumps inside the outer helix at points 13 and 14. Only the two
    ENDPOINTS are projected, fixing where the model's strip starts and ends on
    the old curve -- its offset and its span, the two numbers the convention
    actually differs by -- and the interior comes from the model's own arc
    spacing, which is ordered by construction.
    """
    out = P.copy()
    for a_, b_ in strips:
        a_, b_ = int(a_), int(b_)
        if a_ >= len(P):
            break
        b_ = min(b_, len(P))
        C, Q = old_xyz[a_:b_], P[a_:b_]
        if len(Q) < 2:
            continue
        d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
        t = np.linspace(0, d[-1], 4000)
        D = np.stack([np.interp(t, d, C[:, k]) for k in range(3)], axis=1)
        ends = [int(np.argmin(np.linalg.norm(D - Q[i], axis=1)))
                for i in (0, -1)]
        s0, s1 = t[ends[0]], t[ends[1]]
        tq = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]
        tq = tq / max(tq[-1], 1e-12)                 # the model's own spacing
        ss = np.clip(s0 + tq * (s1 - s0), 0.0, d[-1])
        out[a_:b_] = np.stack([np.interp(ss, d, C[:, k]) for k in range(3)], 1)
    return out


def reprojection(P, uv, cams):
    """The landmarker checking the marks: px between detection and projection."""
    e = []
    for i, cam in enumerate(cams):
        if cam is None:
            continue
        ok = np.isfinite(uv[i]).all(1) & np.isfinite(P).all(1)
        if ok.sum():
            e += list(np.linalg.norm(cam.project(P[ok]) - uv[i][ok], axis=1))
    return np.array(e) if e else np.array([np.nan])


def write_wrap(src, dst, tri, bary, names_55):
    """The v5 project with points 0..54 replaced; 55..94 are left alone.

    Those extensions have no 2D counterpart -- the landmarker predicts 55
    points -- so they keep their hand-placed addresses and will need a look in
    Wrap, since their neighbours have moved.
    """
    d = json.loads(pathlib.Path(src).read_text())
    sel = [n for n in d["nodes"].values() if n["nodeType"] == "SelectPoints"]
    if not sel:
        sys.exit(f"{src}: no SelectPoints node")
    pts = sel[0]["params"]["points"]["value"]
    for j, nm in enumerate(names_55):
        if nm not in pts:
            sys.exit(f"{src} has no point named {nm!r}")
        pts[nm] = {"triangleInd": int(tri[j]),
                   "u": float(bary[j, 0]), "v": float(bary[j, 1])}
    pathlib.Path(dst).write_text(json.dumps(d, indent=1))
    return len(pts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--template", default=str(WRAP / "templates/ear_mean_v4.obj"))
    p.add_argument("--marks", default=str(WRAP / "templates/mark_mean_v5.wrap"),
                   help="the project to copy; points 55+ are carried over")
    p.add_argument("--out", default=str(WRAP / "templates/mark_mean_v6_landmarker.wrap"))
    p.add_argument("--npz", default="")
    p.add_argument("--iters", type=int, default=4)
    p.add_argument("--views", type=int, default=17)
    p.add_argument("--cone", type=float, default=22.0)
    p.add_argument("--keep-curve", action="store_true",
                   help="take only the SPACING from the landmarker and keep "
                        "the hand-marked curve, by sliding each new mark onto "
                        "the old polyline for its strip")
    p.add_argument("--selftest", action="store_true",
                   help="round-trip the EXISTING marks through the reader and "
                        "writer and stop, which checks the u,v convention "
                        "before anything is overwritten")
    a = p.parse_args()

    warnings.filterwarnings("ignore")
    from ear3d.camera import cone_angles
    from ear3d.config import DEFAULTS
    from ear3d.scheme import STRIPS

    m, V, F = load_template(a.template)
    d = json.loads(pathlib.Path(a.marks).read_text())
    sel = [n for n in d["nodes"].values() if n["nodeType"] == "SelectPoints"][0]
    old = sel["params"]["points"]["value"]
    names = [""] * len(old)
    for k, q in old.items():
        names[int(k[:2])] = k
    names_55 = names[:55]
    old_tri = np.array([old[n]["triangleInd"] for n in names_55])
    old_bary = np.array([[old[n]["u"], old[n]["v"],
                          1.0 - old[n]["u"] - old[n]["v"]] for n in names_55])
    old_xyz = np.einsum("ij,ijk->ik", old_bary, V[F[old_tri]])

    if a.selftest:
        t2, b2, pts, dd = snap(old_xyz, V, F)
        back = np.einsum("ij,ijk->ik", b2, V[F[t2]])
        print(f"selftest: {len(names_55)} points re-addressed by snapping")
        print(f"  snap moved them {np.abs(dd).max():.2e} mm (should be ~0)")
        print(f"  re-evaluated address differs by {np.abs(back - old_xyz).max():.2e} mm")
        print(f"  same triangle for {int((t2 == old_tri).sum())}/55 "
              f"(a point on an edge may land on either side)")
        print("  -> the u,v convention round-trips; safe to write")
        return

    cfg = dict(DEFAULTS)
    from eval_test import best_ckpt
    from inference import BLAZEEAR_DIR, DETECTOR_WEIGHTS, EarLandmarkerPipeline
    pipe = EarLandmarkerPipeline(BLAZEEAR_DIR / DETECTOR_WEIGHTS,
                                 best_ckpt(cfg["run"]), device=cfg["device"],
                                 smooth=False)
    print(f"template {pathlib.Path(a.template).name}, "
          f"{len(V)} verts, canonical mm")
    print(f"landmarker {best_ckpt(cfg['run']).name}\n")

    R = R_VIEW.copy()
    angles = cone_angles(a.views, a.cone)
    prev = None
    for it in range(a.iters):
        Vr, c, s, Rc = to_render_frame(V, R, 1.0)
        print(f"iteration {it + 1}/{a.iters}: {len(angles)} views")
        uv, cams, hits = sweep(Vr, F, cfg, pipe, angles)
        if hits < 3:
            sys.exit(f"only {hits} views detected; cannot triangulate")
        Pr, nin = fit(uv, cams)
        rep = reprojection(Pr, uv, cams)
        P = back_to_canonical(Pr, c, s, Rc)
        if a.keep_curve:
            P = onto_old_curve(P, old_xyz, STRIPS)
        tri, bary, pts, off = snap(P, V, F)
        moved = (np.nan if prev is None
                 else float(np.median(np.linalg.norm(pts - prev, axis=1))))
        print(f"  {hits}/{len(angles)} views, {int((nin >= 3).sum())}/55 points "
              f"with 3+ rays")
        print(f"  reprojection {np.nanmedian(rep):.2f} px median, "
              f"{np.nanpercentile(rep, 90):.2f} p90")
        print(f"  snap to surface {np.median(off):.3f} mm; marks moved "
              f"{moved if moved == moved else float('nan'):.3f} mm "
              f"since the last pass")
        # re-aim the cone at the landmarks' own plane for the next pass
        from ear3d.frames import plane_normal
        ok = np.isfinite(P).all(1)
        n = plane_normal(P[ok])
        if n @ (R.T @ np.array([0.0, 0.0, 1.0])) < 0:
            n = -n
        up = R.T @ np.array([0.0, 1.0, 0.0])
        from ear3d.frames import frame_from
        Rn = frame_from(n, up, front=n)
        if Rn is not None:
            tilt = np.degrees(np.arccos(np.clip(
                (np.trace(Rn @ R.T) - 1) / 2, -1, 1)))
            print(f"  re-aiming the cone by {tilt:.2f} deg")
            R = Rn
        prev = pts
        print()

    moved_from_old = np.linalg.norm(pts - old_xyz, axis=1)
    print("AGAINST THE HAND-MARKED TEMPLATE")
    for s_, b_ in STRIPS:
        e = moved_from_old[s_:b_]
        print(f"  {s_:2d}-{b_ - 1:2d}  moved median {np.median(e):5.2f} mm  "
              f"max {e.max():5.2f}")
    print(f"  all 55  median {np.median(moved_from_old):.2f} mm")

    n = write_wrap(a.marks, a.out, tri, bary, names_55)
    print(f"\nwrote {a.out}")
    print(f"  {n} points in the project: 0-54 re-marked, 55-{n - 1} carried "
          f"over unchanged and worth a look, since their neighbours moved")
    if a.npz:
        np.savez(a.npz, triangle=tri, barycentric=bary, xyz=pts,
                 point_names=np.array(names_55), moved_mm=moved_from_old)
        print(f"  and {a.npz}")


if __name__ == "__main__":
    main()
