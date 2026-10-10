#!/usr/bin/env python3
"""Multi-view: lift the shipped 2D with a depth map in several views, then
RANSAC the 3D coordinates into one consensus set.

WHY MORE VIEWS SHOULD HELP, and what they cannot fix. A single view's depth
map fails in specific, view-dependent ways: a landmark whose 2D drifts a
pixel off the pinna reads the scalp behind it, and a surface nearly parallel
to the line of sight is where a depth map is least certain. Both depend on
where the camera is, so a second camera disagrees -- which makes the failure
detectable instead of silent. What no amount of views fixes is an error in
the 2D itself that is consistent across views, because every view then agrees
on the wrong answer.

THE CONSENSUS IS RANSAC, NOT AN AVERAGE. A mean over views is dragged by the
scalp reads, which are not small: they sit an ear-depth behind the true
surface, so one of them in seven views moves the mean several millimetres.
RANSAC takes the largest subset of views that agree to within a tolerance and
averages only those, so a scalp read is excluded rather than diluted. The size
of that subset is kept per landmark, and it is the useful by-product: a point
on which only two of seven views agree is one to distrust, which no
single-view pipeline can tell you.

    # 1. render the views and dump them
    python scripts/moge_multiview.py --mesh ... --ear 0 --out DIR
    # 2. a depth map per view (separate interpreter; see moge_infer.py)
    for p in DIR/view*/render.png; do .venv_cu130/bin/python \
        scripts/moge_infer.py $p ${p%/*}/moge.npz --fov-x 50.0; done
    # 3. combine
    python scripts/moge_multiview.py --mesh ... --ear 0 --out DIR --combine
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ear3d.backproject import backproject, backproject_snapped    # noqa: E402
from ear3d.config import DEFAULTS                                 # noqa: E402
from ear3d.frames import (frame_from, head_pose, in_frame,        # noqa: E402
                          load_head, plane_normal)
from ear3d.label import landmark_whole_frame                      # noqa: E402
from ear3d.render import render                                   # noqa: E402
from ear3d.scheme import STRIPS, STRIP_NAMES                      # noqa: E402
from scan_depth_check import (corpus_z, flag_scalp,               # noqa: E402
                              write_review_obj)


def rot(axis, deg):
    """Rotation about a unit axis, Rodrigues."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    t = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)


def ransac_point(P, tol):
    """Largest subset of view estimates agreeing to `tol`, and its mean.

    Exhaustive over seeds rather than randomised: there are a handful of
    views, so every candidate can be tried and the result is deterministic.
    A random sample would buy nothing here except irreproducibility.
    """
    ok = np.isfinite(P).all(1)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return None, 0, np.nan
    if len(idx) == 1:
        return P[idx[0]], 1, 0.0
    best, bestn = None, -1
    for i in idx:
        d = np.linalg.norm(P[idx] - P[i], axis=1)
        inl = idx[d <= tol]
        if len(inl) > bestn:
            best, bestn = inl, len(inl)
    c = P[best].mean(0)
    spread = float(np.median(np.linalg.norm(P[best] - c, axis=1)))
    return c, int(bestn), spread


def coord_median(P):
    """Per-axis median. Robust and trivial, but NOT rotation-equivariant:
    it is computed in the scan's axes, so turning the scan would move the
    answer. Kept because it is the obvious meaning of "median" and, on this
    data, is barely distinguishable from the estimator that does not have
    that flaw -- which is worth knowing rather than assuming."""
    ok = np.isfinite(P).all(1)
    return (np.median(P[ok], axis=0) if ok.any() else None)


def geometric_median(P, iters=128, eps=1e-9):
    """The point minimising the sum of distances -- the real L1 centre.

    Rotation-equivariant, unlike a per-axis median, and the right robust
    centre for 3D points. Weiszfeld's iteration, started from the per-axis
    median so a handful of steps suffice; the eps guard is for the case where
    the iterate lands exactly on a sample, where the weight would divide by
    zero.
    """
    ok = np.isfinite(P).all(1)
    X = P[ok]
    if len(X) == 0:
        return None
    if len(X) <= 2:
        return X.mean(0)
    c = np.median(X, axis=0)
    for _ in range(iters):
        w = 1.0 / np.maximum(np.linalg.norm(X - c, axis=1), eps)
        nc = (w[:, None] * X).sum(0) / w.sum()
        if np.linalg.norm(nc - c) < 1e-7:
            break
        c = nc
    return c


def support(P, c, tol):
    """How many views sit within tol of an estimate, and their spread.

    A median gives no inlier set, so this is what replaces RANSAC's count as
    the per-landmark confidence. It is the same question asked of whatever
    centre was chosen, so the three estimators stay comparable.
    """
    if c is None or not np.isfinite(c).all():
        return 0, np.nan
    d = np.linalg.norm(P - c, axis=1)
    ok = np.isfinite(d)
    inl = ok & (d <= tol)
    return int(inl.sum()), (float(np.median(d[inl])) if inl.any() else np.nan)


def locate_and_frame(mesh, cfg, pipe, ear, scalp_tol):
    """The pinna frame, exactly as scan_depth_check derives it."""
    pose = head_pose(mesh, cfg)
    if pose is None:
        sys.exit("MediaPipe found no face on this scan")
    lat, vert, tr_r, tr_l = pose
    left = ear % 2 == 0
    front, up = (lat if left else -lat), vert
    C1 = np.asarray(tr_l if left else tr_r, float)
    R1 = frame_from(front, up, front=front)
    g1 = in_frame(mesh, R1, C1, 1.0)
    _, d1, c1, lm1, conf = landmark_whole_frame(g1, cfg, pipe)
    if lm1 is None:
        sys.exit("the pipeline found no ear face-on; try the other --ear")
    P1, hit1, _ = backproject_snapped(d1, lm1, c1, cfg)
    _, scalp1 = flag_scalp(P1, hit1, c1, scalp_tol)
    good1 = hit1 & ~scalp1
    if good1.sum() < 10:
        sys.exit("too few on-pinna rays to fit a pinna plane")
    P1w = (P1[good1] @ R1) + C1
    nrm = plane_normal(P1w)
    C = P1w.mean(0)
    R0 = frame_from(nrm, up, front=front)
    loc = (P1w - C) @ R0.T
    return nrm, up, front, C, float(np.ptp(loc[:, 1])), conf


def frame_one_view(mesh, cfg, pipe, nrm, up, front, C, ear_h, yaw, pitch,
                   fill, K, ts):
    """Render one view, zoomed on the ear and aimed at it.

    The camera is turned about the ear, not the head: `nrm` is the pinna's own
    normal, so yaw 0 is the view straight into the ear and the sweep stays
    symmetric about it rather than about some arbitrary head axis.
    """
    right = np.cross(up, nrm)
    if np.linalg.norm(right) < 1e-9:
        return None
    right /= np.linalg.norm(right)
    fv = rot(up, yaw) @ (rot(right, pitch) @ np.asarray(nrm, float))
    R = frame_from(fv, up, front=front)
    if R is None:
        return None
    half = cfg["eye_z"] * np.tan(np.radians(cfg["vfov"]) / 2)
    # THE DETECTOR HAS SCALE BLIND BANDS -- measured: zero detections over
    # whole 2% windows of zoom, with confidence 0.94-0.98 either side, and
    # unaffected by pipe.reset(). A view is not actually unusable when the
    # detector refuses one framing of it, so a handful of nearby framings are
    # tried before giving up. Without this, 10 of 21 views were being thrown
    # away including the straight-on one, which is the last view anybody
    # would want to lose.
    for fmul in (1.0, 0.94, 1.06, 0.88, 1.12, 0.82, 1.18):
        scale = ear_h / (2 * half * 0.40)
        Cv = np.array(C, float)
        img = dep = cam = lm = None
        ok = True
        for _ in range(3):                  # zoom, then aim, then confirm
            g = in_frame(mesh, R, Cv, scale)
            img, dep, cam = render(g, cfg)
            ts[0] += 1.0
            r = pipe(img, timestamp=ts[0])
            if not r:
                ok = False
                break
            lm = np.asarray(max(r, key=lambda d: float(d["confidence"]))
                            ["landmarks"], float)[:K]
            W = img.shape[1]
            (x0, y0), (x1, y1) = lm.min(0), lm.max(0)
            need = max(x1 - x0, y1 - y0) / W / (fill * fmul)
            dxp, dyp = cam.cx - (x0 + x1) / 2, cam.cy - (y0 + y1) / 2
            if abs(need - 1) < 0.02 and max(abs(dxp), abs(dyp)) < 0.01 * W:
                break
            zc = float(np.mean(dep[np.isfinite(dep)]))
            u = np.array([dxp * zc / cam.fx, dyp * zc / cam.fy, 0.0]) @ cam.R
            Cv = Cv - (u * scale) @ R
            scale *= need
        if ok and lm is not None:
            return dict(R=R, C=Cv, scale=scale, img=img, depth=dep, cam=cam,
                        lm=lm, yaw=yaw, pitch=pitch, fill_used=fill * fmul)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", required=True)
    p.add_argument("--ear", type=int, default=0)
    p.add_argument("--out", required=True, help="LOCAL ONLY: gitignored")
    p.add_argument("--combine", action="store_true")
    p.add_argument("--yaws", type=float, nargs="*",
                   default=[-24, -16, -8, 0, 8, 16, 24])
    p.add_argument("--pitches", type=float, nargs="*", default=[-12, 0, 12])
    p.add_argument("--fill", type=float, default=0.55)
    p.add_argument("--n-points", type=int, default=55)
    p.add_argument("--scalp-tol", type=float, default=0.25)
    p.add_argument("--consensus", default="median",
                   choices=["median", "geomedian", "ransac"],
                   help="which robust centre to export. All three are "
                        "computed and compared either way")
    p.add_argument("--tol", type=float, default=2.0,
                   help="inlier distance IN MILLIMETRES, converted to scan "
                        "units from the head's measured size. The "
                        "failure being rejected is a scalp read, which sits a "
                        "whole ear-depth out, so this does not have to be "
                        "tight to catch one")
    p.add_argument("--no-skin", action="store_true")
    p.add_argument("--texture-obj", default="")
    p.add_argument("--export-obj", default="")
    p.add_argument("--sphere", type=float, default=0.012)
    a = p.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    import open3d as o3d
    from PIL import Image
    from eval_test import best_ckpt
    from inference import (BLAZEEAR_DIR, DETECTOR_WEIGHTS,
                           EarLandmarkerPipeline)

    cfg = dict(DEFAULTS, mesh=a.mesh, ear=a.ear, skin=not a.no_skin)
    K = a.n_points
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    mesh = load_head(cfg["mesh"])
    if mesh is None:
        sys.exit(f"could not read {cfg['mesh']}")
    if cfg["skin"]:
        from skin import apply_skin
        apply_skin(mesh, seed=cfg["skin_seed"], tone=cfg["tone"],
                   ao_strength=cfg["ao_strength"], n_rays=cfg["ao_rays"])
    ck = best_ckpt(cfg["run"])
    pipe = EarLandmarkerPipeline(BLAZEEAR_DIR / DETECTOR_WEIGHTS, ck,
                                 device=cfg["device"], smooth=False)

    # the scan's own coordinates, for everything that leaves this script
    src = a.texture_obj or (a.mesh if a.mesh.lower().endswith(".obj") else "")
    V0 = np.asarray(o3d.io.read_triangle_mesh(src).vertices) if src else None
    c0 = V0.mean(0) if src else None
    s0 = float(np.abs(V0 - c0).max()) if src else 1.0

    # WHAT A SCAN UNIT IS WORTH IN MILLIMETRES, decided from the head's own
    # size rather than assumed. This matters more than it looks: BJ002 is in
    # mm and the HUTUBS heads are in METRES, and with mm assumed every
    # distance printed as 0.00 while --tol 2.0 silently meant two METRES, so
    # all 55 landmarks came back with full support and the table read as a
    # perfect result. A head is 150-350 mm tall by any measure, so an extent
    # under 10 cannot be millimetres.
    mm = 1.0
    if V0 is not None:
        ext = float(np.ptp(V0, axis=0).max())
        mm = 1000.0 if ext < 10.0 else 1.0
        print(f"units:  head extent {ext:.3f} scan units -> "
              f"{'METRES' if mm == 1000 else 'millimetres'} "
              f"({ext * mm:.0f} mm tall); distances below are mm")
    # --tol is quoted in mm, so it has to come back to scan units to be
    # compared against coordinates
    tol = a.tol / mm

    if not a.combine:
        nrm, up, front, C, ear_h, conf = locate_and_frame(
            mesh, cfg, pipe, a.ear, a.scalp_tol)
        print(f"locate: {'left' if a.ear % 2 == 0 else 'right'} ear, "
              f"detector conf {conf:.3f}, ear height {ear_h:.4f}")
        views, ts = [], [0.0]
        for pitch in a.pitches:
            for yaw in a.yaws:
                v = frame_one_view(mesh, cfg, pipe, nrm, up, front, C, ear_h,
                                   yaw, pitch, a.fill, K, ts)
                tag = f"y{yaw:+.0f}_p{pitch:+.0f}"
                if v is None:
                    print(f"  view {tag}: no detection, skipped")
                    continue
                d = out / f"view_{tag}"
                d.mkdir(exist_ok=True)
                Image.fromarray(v["img"]).save(d / "render.png")
                W = v["img"].shape[1]
                fov = float(np.degrees(2 * np.arctan(W / (2 * v["cam"].fx))))
                np.savez_compressed(
                    d / "frame.npz", depth=v["depth"].astype(np.float32),
                    lm=v["lm"], R=v["R"], C=v["C"], scale=v["scale"],
                    fx=v["cam"].fx, fy=v["cam"].fy, cx=v["cam"].cx,
                    cy=v["cam"].cy, camR=v["cam"].R, camt=v["cam"].t,
                    fov_x=fov, yaw=yaw, pitch=pitch)
                (x0, y0), (x1, y1) = v["lm"].min(0), v["lm"].max(0)
                print(f"  view {tag}: ear {(x1-x0)/W:.3f} x {(y1-y0)/W:.3f} "
                      f"of frame, fov_x {fov:.2f}")
                views.append(tag)
        (out / "views.json").write_text(json.dumps(views, indent=1))
        print(f"\n{len(views)} views under {out}. Next, a depth map each:")
        print(f"  for p in {out}/view_*/render.png; do "
              f"/mnt/14BE47C2BE479ADE/Code/exrcise_trainer/.venv_cu130/bin/"
              f"python scripts/moge_infer.py $p ${{p%/*}}/moge.npz "
              f"--fov-x 50.0; done")
        print(f"then re-run with --combine")
        return

    # ---------------- combine ------------------------------------------
    tags = json.loads((out / "views.json").read_text())
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(
        o3d.io.read_triangle_mesh(src)))

    est = np.full((len(tags), K, 3), np.nan)
    kept = []
    for vi, tag in enumerate(tags):
        d = out / f"view_{tag}"
        if not (d / "moge.npz").exists():
            print(f"  {tag}: no moge.npz, skipped")
            continue
        F = np.load(d / "frame.npz", allow_pickle=True)
        M = np.load(d / "moge.npz")
        mg = M["depth"].astype(float)
        lm, W = F["lm"], mg.shape[1]
        fx_m = float(M["intrinsics"][0, 0]) * W
        ix = np.clip(np.rint(lm[:, 0]).astype(int), 0, W - 1)
        iy = np.clip(np.rint(lm[:, 1]).astype(int), 0, mg.shape[0] - 1)
        zm = mg[iy, ix]
        # z in the corpus's units, then back along each ray. The standoff
        # comes from the view's own depth buffer, as everywhere else in this
        # project: a crop cannot determine absolute distance and the model
        # never claims it. It fixes distance only, never shape.
        zq = corpus_z(np.where(np.isfinite(zm), zm, np.nanmedian(zm)),
                      fx_m, float(W))
        dep = F["depth"].astype(float)
        fxr, fyr = float(F["fx"]), float(F["fy"])
        cxr, cyr = float(F["cx"]), float(F["cy"])
        camR, camt = F["camR"], F["camt"]
        # THE STANDOFF IS THE DEPTH AT THE EAR, not the mean of the frame.
        # Using the frame mean looks harmless head-on, where most of what is
        # visible is near the ear anyway, and is badly wrong at an oblique
        # view, where the frame fills with cheek and scalp at quite different
        # distances. Measured: it put the negative-yaw views 7-15 mm off the
        # surface -- a radial error, uniform across all 55 landmarks, which is
        # the signature of a wrong standoff rather than of bad depth. The
        # scalp reads are dropped first, for the same reason they are dropped
        # everywhere else: they sit an ear-depth behind and would drag it.
        zs = dep[np.clip(np.rint(lm[:, 1]).astype(int), 0, dep.shape[0] - 1),
                 np.clip(np.rint(lm[:, 0]).astype(int), 0, dep.shape[1] - 1)]
        fz = np.isfinite(zs)
        if fz.sum() < 5:
            print(f"  {tag}: too few landmarks on the surface, skipped")
            continue
        med, span = np.median(zs[fz]), np.ptp(zs[fz])
        keep = fz & (zs <= med + 0.25 * max(span, 1e-9))
        zbar = float(np.mean(zs[keep]))
        zcp = zbar * (1.0 + zq * W / fxr)
        pc = np.stack([(lm[:, 0] - cxr) / fxr * zcp,
                       (lm[:, 1] - cyr) / fyr * zcp, zcp], axis=1)
        Pv = (pc - camt) @ camR
        # this view's own frame, inverted, into scan coordinates
        Pw = (((Pv * float(F["scale"])) @ F["R"]) + F["C"]) * s0 + c0
        est[vi] = Pw
        kept.append(tag)

    if not kept:
        sys.exit("no views had a depth map; run moge_infer first")
    print(f"\n{len(kept)} views combined: {', '.join(kept)}")

    # all three estimators, always, because they cost nothing next to the
    # renders and the depth maps, and the comparison is the point
    ests = {}
    for name, fn in (("ransac", lambda P: ransac_point(P, tol)[0]),
                     ("median", coord_median),
                     ("geomedian", geometric_median)):
        C = np.full((K, 3), np.nan)
        N = np.zeros(K, int)
        S = np.full(K, np.nan)
        for k in range(K):
            c = fn(est[:, k, :])
            if c is not None:
                C[k] = c
                N[k], S[k] = support(est[:, k, :], c, tol)
        ests[name] = (C, N, S)
    cons, nin, spr = ests[a.consensus]

    def dist_to_surface(P):
        """Distance from each point to the scan surface, IN MILLIMETRES."""
        ok = np.isfinite(P).all(1)
        d = np.full(len(P), np.nan)
        if ok.any():
            d[ok] = sc.compute_distance(
                o3d.core.Tensor(P[ok].astype(np.float32))).numpy()
        return d * mm

    dc = dist_to_surface(cons)
    print(f"\n{'':<26s}{'median':>8s}{'p90':>8s}{'max':>8s}"
          f"{'<=' + str(int(a.tol)) + 'mm':>8s}")

    def row(lbl, d):
        f = np.isfinite(d)
        print(f"  {lbl:<24s}{np.median(d[f]):8.2f}{np.percentile(d[f], 90):8.2f}"
              f"{d[f].max():8.2f}{100 * (d[f] <= a.tol).mean():7.0f}%")

    for vi, tag in enumerate(kept):
        row(f"single view {tag}", dist_to_surface(est[tags.index(tag)]))
    row("mean over views", dist_to_surface(np.nanmean(est, axis=0)))
    for name in ("ransac", "median", "geomedian"):
        row(("-> " if name == a.consensus else "   ") + name,
            dist_to_surface(ests[name][0]))
    # how far the two medians actually differ, which is the only thing that
    # decides whether the rotation-equivariance argument matters here
    dm = np.linalg.norm(ests["median"][0] - ests["geomedian"][0], axis=1)
    fm = np.isfinite(dm)
    print(f"\ncoord median vs geometric median: median "
          f"{np.median(dm[fm]) * mm:.3f} mm apart, max {dm[fm].max() * mm:.3f}")
    print(f"\nsupport per landmark ({a.consensus}): median "
          f"{np.median(nin):.0f} of "
          f"{len(kept)}, {(nin >= 3).sum()}/{K} with 3 or more, "
          f"{(nin <= 1).sum()} with 1 or none")
    print(f"agreement within the consensus set: median "
          f"{np.nanmedian(spr) * mm:.2f} mm")
    lone = np.where(nin <= 1)[0]
    if len(lone):
        print(f"landmarks the views do NOT agree on: {list(lone)}")

    np.savez(out / "consensus.npz", cons=cons, nin=nin, spread=spr,
             est=est, tags=np.asarray(kept), dist=dc,
             estimator=a.consensus, tol=a.tol,
             **{f"cons_{n}": v[0] for n, v in ests.items()})
    if a.export_obj:
        col = {True: (0.20, 0.95, 0.65), False: (1.00, 0.45, 0.10)}
        # named for the estimator ACTUALLY used, so an obj cannot claim to be
        # a RANSAC result when --consensus asked for a median
        sets = {
            f"moge_{a.consensus}": (
                col[True],
                [cons[k] if nin[k] >= 3 else None for k in range(K)]),
            f"moge_{a.consensus}_WEAK": (
                col[False],
                [cons[k] if nin[k] < 3 else None for k in range(K)]),
        }
        oo, mm, tt = write_review_obj(
            a.export_obj, src, sets, a.sphere * ear_h_scan(s0),
            lambda i: next(n for (lo, hi), n in zip(STRIPS, STRIP_NAMES)
                           if lo <= i < hi))
        print(f"\nreview mesh -> {oo}  ({mm.name}, texture "
              f"{tt.name if tt else 'NONE'})")
        for k, (c, pts) in sets.items():
            print(f"    {k:<20s} {sum(q is not None for q in pts):3d}  "
                  f"rgb {tuple(round(x, 2) for x in c)}")


def ear_h_scan(s0):
    """Sphere radius reference: the ear is about 55 mm, so a fraction of that
    is the right size for a marker regardless of the frame maths above."""
    return 55.0


if __name__ == "__main__":
    main()
