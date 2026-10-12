#!/usr/bin/env python3
"""Score a depth head against a REAL head scan's own geometry.

WHY THIS IS NOT THE BACK-PROJECTION PATH. Back-projection existed because the
2D model gave no depth, so depth had to be recovered from the mesh -- and that
path is not dependable: lateral 2D error becomes amplified depth error through
a narrow view cone, and rays passing just outside the pinna land on the scalp
an ear-depth too deep. A depth head predicts z directly, so the mesh stops
being the SOURCE of depth and becomes an independent CHECK on it.

That check is worth having because this scan is a subject from none of the
training sources, and because the camera is exact: the render's own float32
view-space depth buffer is the rasterisation that produced the pixels, so the
reference needs no camera estimation and no triangulation.

WHAT IT CAN AND CANNOT MEASURE. The scan carries no landmark annotation, so
PLACEMENT cannot be scored -- there is nothing to say where point 29 belongs.
What is measured is whether the predicted depth profile matches the real
surface at the pixels the model itself chose. That conflates placement error
with depth error and the two cannot be separated here; it is still a real
measurement of depth on an unseen head, which nothing else available provides.

THE SILHOUETTE CASE IS REPORTED, NOT HIDDEN. A landmark a pixel or two outside
the pinna samples the scalp behind it, which is one ear-depth too far and would
quietly dominate the error. Those points are counted and excluded from the
headline number, and the count is part of the result.

    python scripts/scan_depth_check.py --mesh supersecret/BJ002/BJ002.OBJ
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ear3d.backproject import backproject, backproject_snapped   # noqa: E402
from ear3d.config import DEFAULTS                                # noqa: E402
from ear3d.frames import (frame_from, head_pose, in_frame,       # noqa: E402
                          load_head, plane_normal)
from ear3d.label import landmark_whole_frame                     # noqa: E402
from ear3d.render import render                                  # noqa: E402
from ear3d.scheme import STRIPS, STRIP_NAMES                     # noqa: E402

TPL = pathlib.Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/wrap_work/templates")


def flag_scalp(P, hit, cam, tol):
    """Which back-projected points sit behind the pinna rather than on it.

    A ray passing a pixel or two outside the pinna hits the SCALP, about one
    ear-depth too far. The point looks right face-on -- the error is purely in
    depth -- so nothing in the 2D overlay reveals it, and it has to be found
    from the depth it came back with. Returns (camera-space z, scalp mask).
    """
    zc = np.full(len(P), np.nan)
    zc[hit] = (P[hit] @ cam.R.T + cam.t)[:, 2]
    if hit.sum() < 3:
        return zc, np.zeros(len(P), bool)
    med = float(np.nanmedian(zc))
    span = float(np.nanmax(zc) - np.nanmin(zc))
    return zc, hit & (zc > med + tol * max(span, 1e-9))


def snap_front_surface(P, src_obj, fwd, max_dist, probe=None):
    """Move each point the SHORTEST distance onto the camera-facing surface.

    MoGe SETS THE POSITION; this only tidies it. The alternative -- snapping
    along the camera ray -- would replace the predicted depth with the depth
    buffer's, which is back-projection again: all that survives a ray snap is
    x and y, so a MoGe dot and a mean-ear dot land in the same place. A
    nearest-point snap keeps the prediction and corrects it by the minimum
    displacement, which on this scan is about a millimetre.

    FRONT OF THE EAR, NOT THE BACK. A dot sitting inside the pinna is roughly
    equidistant from its anterior and posterior surfaces, so an unrestricted
    nearest-point query can land on the back. Only triangles whose normal
    faces the camera are offered, which removes the posterior sheet from the
    search entirely rather than trying to detect the mistake afterwards.

    `max_dist` leaves anything further than that where it is. A dot a long way
    from the surface is not a rounding error to be tidied -- it is usually a
    landmark whose 2D fell off the pinna -- and dragging it onto the nearest
    triangle would manufacture a plausible wrong answer and hide the failure.

    `probe` is points KNOWN to lie on the visible surface (back-projected
    through the depth buffer). The normals of a scan are not guaranteed
    outward, and if they point the wrong way this function would silently
    offer the posterior sheet and snap everything to the back of the ear. So
    the kept subset is verified against the probe first and the orientation is
    flipped if it is wrong, which turns an invisible failure into a printed
    line. Returns (snapped, distance, moved mask, diagnostic string).
    """
    import open3d as o3d

    m = o3d.io.read_triangle_mesh(str(src_obj))
    m.compute_triangle_normals()
    V = np.asarray(m.vertices)
    F = np.asarray(m.triangles)
    N = np.asarray(m.triangle_normals)
    fwd = np.asarray(fwd, float)
    fwd = fwd / np.linalg.norm(fwd)

    def scene_of(mask):
        sub = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V),
                                        o3d.utility.Vector3iVector(F[mask]))
        sc = o3d.t.geometry.RaycastingScene()
        sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(sub))
        return sc

    facing = (N @ fwd) < 0
    note = f"{facing.mean():.0%} of triangles face the camera"
    if probe is not None and len(probe):
        q = o3d.core.Tensor(np.asarray(probe, np.float32))
        da = float(np.median(scene_of(facing).compute_distance(q).numpy()))
        db = float(np.median(scene_of(~facing).compute_distance(q).numpy()))
        if db < da:
            facing = ~facing
            note += f"; ORIENTATION FLIPPED (probe {db:.2f} vs {da:.2f} mm)"
        else:
            note += f"; orientation verified on probe ({da:.2f} mm)"

    res = scene_of(facing).compute_closest_points(
        o3d.core.Tensor(np.asarray(P, np.float32)))
    cp = res["points"].numpy().astype(float)
    d = np.linalg.norm(cp - np.asarray(P, float), axis=1)
    moved = d <= max_dist
    out = np.asarray(P, float).copy()
    out[moved] = cp[moved]
    return out, d, moved, note


def wire_tubes(V, F, radius, target):
    """A decimated mesh's edges as REAL TRIANGLES, not obj `l` polylines.

    WHY NOT `l`. An obj can carry a polyline and this script used to rely on
    that for the strip contours and the error sticks, but a lot of viewers
    silently ignore `l` and draw nothing -- which is exactly what happened
    here: 47 error sticks were in the file and none of them appeared. Anything
    that must be SEEN therefore has to be triangles.

    Each surviving edge becomes a three-sided tube, so it is visible from
    every angle; a flat sliver quad would vanish when viewed edge-on. The
    mesh is decimated first because the scan is 210k triangles and a tube per
    edge on that would be millions.
    """
    import open3d as o3d

    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))
    m = m.simplify_quadric_decimation(target_number_of_triangles=int(target))
    V2 = np.asarray(m.vertices)
    F2 = np.asarray(m.triangles)
    E = np.concatenate([F2[:, [0, 1]], F2[:, [1, 2]], F2[:, [2, 0]]], 0)
    E = np.unique(np.sort(E, axis=1), axis=0)

    A, B = V2[E[:, 0]], V2[E[:, 1]]
    d = B - A
    L = np.linalg.norm(d, axis=1, keepdims=True)
    keep = L[:, 0] > 1e-9
    A, B, d, L = A[keep], B[keep], d[keep], L[keep]
    d = d / L
    # any vector not parallel to the edge, to start the perpendicular frame
    tmp = np.tile([0.0, 0.0, 1.0], (len(d), 1))
    par = np.abs((d * tmp).sum(1)) > 0.9
    tmp[par] = [1.0, 0.0, 0.0]
    u = np.cross(d, tmp)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    w = np.cross(d, u)

    ang = np.arange(3) * (2 * np.pi / 3)
    off = (np.cos(ang)[None, :, None] * u[:, None, :] +
           np.sin(ang)[None, :, None] * w[:, None, :]) * radius
    verts = np.concatenate([A[:, None, :] + off, B[:, None, :] + off], 1)

    n = len(A)
    base = np.arange(n)[:, None] * 6
    i0, i1, i2 = [base + k for k in (0, 1, 2)]
    j0, j1, j2 = [base + k for k in (3, 4, 5)]
    tris = np.concatenate([
        np.concatenate([i0, i1, j1], 1), np.concatenate([i0, j1, j0], 1),
        np.concatenate([i1, i2, j2], 1), np.concatenate([i1, j2, j1], 1),
        np.concatenate([i2, i0, j0], 1), np.concatenate([i2, j0, j2], 1),
    ], 0)
    return verts.reshape(-1, 3), tris


def write_review_obj(out_obj, src_obj, sets, radius, strip_of,
                     links=None, wire=None):
    """Textured head + one named sphere per landmark, for a 3D viewer.

    The head is re-emitted from the ORIGINAL obj so its uvs and texture come
    through untouched -- the vertex-coloured ply the renderer uses is a
    lossy bake and is not what anyone wants to review against.

    Every landmark is its own `o` group with its own material, so a viewer
    lists them by name and they can be clicked individually rather than
    arriving as one soup of triangles.

    Two kinds of POLYLINE go in alongside the spheres, because 275 spheres on
    one ear is unreadable and the spheres alone cannot show WHICH way a point
    is wrong:
      <name>_strips  the scheme's four contours in order, so a set that has
                     lost the ear's anatomy looks wrong instead of merely
                     scattered
      <name>_error   one segment from each prediction to the surface point at
                     the same pixel, so the depth error is a stick with a
                     direction rather than a gap to be judged by eye
    """
    import shutil
    import open3d as o3d

    src_obj = pathlib.Path(src_obj)
    out_obj = pathlib.Path(out_obj)
    out_obj.parent.mkdir(parents=True, exist_ok=True)

    # the texture, named as it is ON DISK; the scan's own .mtl disagrees in
    # case with the file beside it, which breaks on a case-sensitive viewer
    tex = None
    mtl_src = src_obj.with_suffix(".MTL")
    if mtl_src.exists():
        want = next((l.split(maxsplit=1)[1].strip()
                     for l in open(mtl_src, errors="ignore")
                     if l.lower().startswith("map_kd")), None)
        if want:
            tex = next((q for q in src_obj.parent.iterdir()
                        if q.name.lower() == pathlib.Path(want).name.lower()),
                       None)
    if wire:
        tex = None
    if tex is not None and not (out_obj.parent / tex.name).exists():
        shutil.copy2(tex, out_obj.parent / tex.name)

    sph = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=6)
    sv = np.asarray(sph.vertices)
    sf = np.asarray(sph.triangles)

    mtl = out_obj.with_suffix(".mtl")
    with open(mtl, "w") as f:
        if wire:
            # A WIREFRAME HEAD, so a landmark that has gone SUBSURFACE is
            # visible instead of hidden behind an opaque face. Coloured warm,
            # not grey, to stay distinguishable from the landmarks while still
            # reading as the head. No texture: the atlas cannot survive
            # decimation, and the point of this mode is to see through.
            f.write(f"newmtl head\nKa 0 0 0\n"
                    f"Kd {wire[1][0]:.3f} {wire[1][1]:.3f} {wire[1][2]:.3f}\n"
                    f"Ks 0.1 0.1 0.1\nNs 20\n")
        else:
            f.write("newmtl head\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nNs 100\n")
            if tex is not None:
                f.write(f"map_Kd {tex.name}\n")
        for name, (col, _) in sets.items():
            f.write(f"\nnewmtl {name}\nKa 0 0 0\n"
                    f"Kd {col[0]:.3f} {col[1]:.3f} {col[2]:.3f}\n"
                    f"Ks 0.2 0.2 0.2\nNs 50\n")
        # The error sticks get their OWN colour rather than the prediction's.
        # Sharing it made them read as more of the same cloud, when the whole
        # reason they exist is to be seen AGAINST both point sets. Amber is
        # picked to sit away from the prediction green, the truth blue and the
        # skin tones of the textured head, which is the full palette here.
        if links:
            f.write("\nnewmtl error\nKa 0 0 0\nKd 1.000 0.750 0.000\n"
                    "Ks 0.2 0.2 0.2\nNs 50\n")

    nv = 0
    with open(out_obj, "w") as f:
        f.write(f"# head scan + predicted landmarks, for review\n")
        f.write(f"mtllib {mtl.name}\n\no head\nusemtl head\n")
        if wire:
            src = o3d.io.read_triangle_mesh(str(src_obj))
            wv, wf = wire_tubes(np.asarray(src.vertices),
                                np.asarray(src.triangles), wire[0], wire[2])
            for v in wv:
                f.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
            for t in wf:
                f.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")
            nv = len(wv)
        else:
            for ln in open(src_obj, errors="ignore"):
                if ln.startswith("v ") or ln.startswith("vt "):
                    f.write(ln)
                    nv += ln.startswith("v ")
                elif ln.startswith("f "):
                    f.write(ln)
        for name, (_, pts) in sets.items():
            for i, P in enumerate(pts):
                if P is None or not np.isfinite(P).all():
                    continue
                f.write(f"\no {name}_{i:02d}_{strip_of(i)}\nusemtl {name}\n")
                for v in sv + P:
                    f.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
                for t in sf:
                    f.write(f"f {nv + t[0] + 1} {nv + t[1] + 1} "
                            f"{nv + t[2] + 1}\n")
                nv += len(sv)
        for name, (_, pts) in sets.items():
            keep = [i for i, q in enumerate(pts)
                    if q is not None and np.isfinite(q).all()]
            if len(keep) < 2:
                continue
            f.write(f"\no {name}_strips\nusemtl {name}\n")
            base = nv
            at = {}
            for i in keep:
                at[i] = nv
                q = pts[i]
                f.write(f"v {q[0]:.5f} {q[1]:.5f} {q[2]:.5f}\n")
                nv += 1
            for lo, hi in STRIPS:
                run = [at[i] for i in range(lo, hi) if i in at]
                if len(run) > 1:
                    f.write("l " + " ".join(str(v + 1) for v in run) + "\n")
        for name, pairs in (links or {}).items():
            pr = [(x, y) for x, y in pairs
                  if x is not None and y is not None
                  and np.isfinite(x).all() and np.isfinite(y).all()]
            if not pr:
                continue
            f.write(f"\no {name}_error\nusemtl error\n")
            for x, y in pr:
                f.write(f"v {x[0]:.5f} {x[1]:.5f} {x[2]:.5f}\n")
                f.write(f"v {y[0]:.5f} {y[1]:.5f} {y[2]:.5f}\n")
                f.write(f"l {nv + 1} {nv + 2}\n")
                nv += 2
    return out_obj, mtl, tex


def corpus_z(zc, fx, crop_px):
    """View-space depth -> the corpus's root-relative z, in crop-width units.

    zn = (zc - zbar) * fx / (crop_px * zbar), the definition data/depth_dataset
    documents. It depends on fx and zbar only through their ratio -- pixels per
    unit length at the ear -- which is what a scaled-orthographic fit observes,
    so it is directly comparable to a predicted z.
    """
    zb = float(np.mean(zc))
    return (zc - zb) * fx / (crop_px * zb)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", required=True)
    p.add_argument("--ear", type=int, default=0, help="0 = subject left, 1 = right")
    p.add_argument("--ckpt", default="runs/pash_0.0/ckpt/last.ckpt",
                   help="depth head; default is the Route C recipe that scored "
                        "best on real photographs")
    p.add_argument("--n-points", type=int, default=55)
    p.add_argument("--occupancy", type=float, default=0.677,
                   help="fraction of frame HEIGHT the ear should fill. The "
                        "corpus frames at 0.354 x 0.677 and a depth head is "
                        "sensitive to it -- a crop round-trip alone cost "
                        "z_skill 0.802 -> 0.584 -- so it is matched here")
    p.add_argument("--scalp-tol", type=float, default=0.25,
                   help="a sampled depth this far behind the median, as a "
                        "fraction of the sampled span, is the scalp beside the "
                        "pinna rather than the ear")
    p.add_argument("--no-skin", action="store_true")
    p.add_argument("--compare-flip", action="store_true",
                   help="export head 1 TWICE, as predicted and with z "
                        "negated, so the two can be judged side by side in a "
                        "viewer against the same surface reference")
    p.add_argument("--flip-z", action="store_true",
                   help="DIAGNOSTIC: negate every predicted z before scoring. "
                        "If a score improves, that prediction's depth is "
                        "sign-inverted relative to the scan")
    p.add_argument("--feat-ckpt", default="",
                   help="head 3b: the shipped 2D frozen, lifter on top")
    p.add_argument("--shape-ckpt", default="",
                   help="a head 2 (shape) checkpoint. Lifted two ways: its "
                        "own decoded x,y,z, and its COEFFICIENTS with pose "
                        "solved from the shipped 2D -- which on photographs "
                        "was worth +0.62 to +0.86 z_skill over the regressed "
                        "pose, so the two are worth seeing side by side")
    p.add_argument("--export-obj", default="",
                   help="write a textured .obj of the head with a named "
                        "sphere at every landmark, for review in a 3D viewer")
    p.add_argument("--texture-obj", default="",
                   help="the ORIGINAL textured .obj to embed; defaults to "
                        "--mesh when that is itself an .obj")
    p.add_argument("--sphere", type=float, default=0.012,
                   help="sphere radius as a fraction of ear height")
    p.add_argument("--zoom-fill", type=float, default=0.0,
                   help="with --zoom: make the ear fill this fraction of the "
                        "frame's LONGER extent, instead of the corpus "
                        "0.354 x 0.677. Only meaningful with --no-head1 -- "
                        "the corpus occupancy exists because head 1 is "
                        "framing-sensitive, and a depth map is not, so with "
                        "head 1 gone the ear can simply be made as large as "
                        "fits and the extra pixels spent on relief")
    p.add_argument("--no-head1", action="store_true",
                   help="do not load or score head 1 at all. It is the "
                        "incumbent depth head and a useful baseline, but it "
                        "is NOT part of the shipped-2D + depth-map flow, and "
                        "keeping it forces the framing to the corpus "
                        "occupancy it was trained on -- which is what makes "
                        "--zoom fight the detector. Without it the framing is "
                        "free")
    p.add_argument("--zoom", action="store_true",
                   help="frame the ear by MOVING THE CAMERA instead of "
                        "cropping the render. Same geometry -- a crop and a "
                        "narrowed field of view give the same fx/width "
                        "ratio, which is the only thing corpus z depends on "
                        "-- but it removes the crop box, removes putting a "
                        "depth map back on the render grid, makes the z unit "
                        "simply the image width, and renders the ear at full "
                        "resolution instead of inside a sub-window")
    p.add_argument("--snap-surface", type=float, default=0.0,
                   metavar="MM",
                   help="after lifting, move each exported landmark the "
                        "SHORTEST distance onto the camera-facing surface, "
                        "leaving anything further than MM where it is. The "
                        "prediction still sets the position; this is a bounded "
                        "tidy-up, not a re-derivation -- a snap along the "
                        "camera ray would discard the predicted depth and be "
                        "back-projection again. Scores are reported for the "
                        "UNSNAPPED points, because a point snapped onto the "
                        "surface is on the surface by construction and cannot "
                        "be graded against it")
    p.add_argument("--wireframe", action="store_true",
                   help="emit the head as a coloured WIREFRAME instead of a "
                        "textured surface, so a landmark that has gone "
                        "subsurface is visible rather than hidden behind an "
                        "opaque face. Built from real triangles (a tube per "
                        "edge), because many viewers ignore obj `l` lines")
    p.add_argument("--wire-tris", type=int, default=8000,
                   help="decimate the head to this many triangles before "
                        "turning its edges into tubes; the scan is ~210k, and "
                        "a tube per edge on that is millions of triangles")
    p.add_argument("--wire-radius", type=float, default=0.003,
                   help="tube radius as a fraction of ear height, same "
                        "convention as --sphere")
    p.add_argument("--wire-color", default="0.80,0.62,0.48",
                   help="r,g,b for the wireframe")
    p.add_argument("--export-only", default="",
                   help="comma-separated set names to keep in --export-obj "
                        "(e.g. moge_lift,scan_surface). Every method scored "
                        "still appears in the TABLE; this only thins the obj, "
                        "because four overlapping 55-sphere sets on one ear "
                        "is unreadable in a viewer. scan_surface is the truth "
                        "reference, so keep it unless the error sticks alone "
                        "are enough")
    p.add_argument("--moge-npz", default="",
                   help="a monocular depth map from scripts/moge_infer.py, "
                        "scored as the depth source at the SHIPPED 2D "
                        "model's own landmark pixels -- no learned depth "
                        "head anywhere in that path. The map may be of the "
                        "full render or of the corpus crop; a crop's map is "
                        "put back on the render grid first")
    p.add_argument("--dump-render", default="",
                   help="write the render, the corpus crop, the EXACT depth "
                        "buffer, this camera and the shipped 2D to a "
                        "directory, so another depth source can be scored on "
                        "the identical pixels without reproducing any of the "
                        "locating and framing above. Analysis only -- nothing "
                        "in this script reads it back")
    p.add_argument("--out", default="supersecret/out",
                   help="LOCAL ONLY: a gitignored directory")
    a = p.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    import open3d as o3d
    import torch
    from PIL import Image
    from model.depth_module import EarDepth3DModule
    from eval_test import best_ckpt
    from inference import (BLAZEEAR_DIR, DETECTOR_WEIGHTS,
                           EarLandmarkerPipeline)
    from scripts.fit_photo_depth import weak_perspective

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
    if ck is None:
        sys.exit(f"no checkpoint for run {cfg['run']!r} under runs/checkpoints/")
    # smooth=False: the tracker is for video, and on a single still it would
    # smooth a one-frame track against wall-clock time and shift the landmarks.
    pipe = EarLandmarkerPipeline(BLAZEEAR_DIR / DETECTOR_WEIGHTS, ck,
                                 device=cfg["device"], smooth=False)
    print(f"landmarker {ck.name}")

    pose = head_pose(mesh, cfg)
    if pose is None:
        sys.exit("MediaPipe found no face on this scan")
    lat, vert, tr_r, tr_l = pose
    left = a.ear % 2 == 0
    front, up = (lat if left else -lat), vert
    C1 = np.asarray(tr_l if left else tr_r, float)
    R1 = frame_from(front, up, front=front)
    g1 = in_frame(mesh, R1, C1, 1.0)
    _, depth1, cam1, lm1, conf = landmark_whole_frame(g1, cfg, pipe)
    if lm1 is None:
        sys.exit("the pipeline found no ear face-on; try --ear 1")
    P1, hit1, _ = backproject_snapped(depth1, lm1, cam1, cfg)
    # THE SCALP POINTS MUST NOT REACH THE PLANE FIT. The pinna plane sets the
    # view direction for the render the depth head is scored on, and a point an
    # ear-depth behind the ear tilts that plane. Measured on this scan: leaving
    # them in gave a 0.572-of-width framing and an ear whose landmark box came
    # out 1.17 tall-to-wide against a real ear's ~1.9, i.e. the view was oblique
    # without anything saying so.
    _, scalp1 = flag_scalp(P1, hit1, cam1, a.scalp_tol)
    good1 = hit1 & ~scalp1
    print(f"locate: {'left' if left else 'right'} ear, detector conf "
          f"{conf:.3f}, {int(hit1.sum())}/{len(hit1)} rays hit, "
          f"{int(scalp1.sum())} on the scalp, {int(good1.sum())} used for the "
          f"pinna plane")
    if good1.sum() < 10:
        sys.exit("too few on-pinna rays to fit a pinna plane")
    P1w = (P1[good1] @ R1) + C1

    R2 = frame_from(plane_normal(P1w), up, front=front)
    C2 = P1w.mean(0)
    loc = (P1w - C2) @ R2.T
    ear_h = float(np.ptp(loc[:, 1]))
    half = cfg["eye_z"] * np.tan(np.radians(cfg["vfov"]) / 2)
    # Deliberately LOOSE: this only has to get the ear fully into frame with
    # margin. The framing that matters is set below from the 2D landmarks, not
    # from here -- sizing the crop off P1w would feed the back-projection's
    # errors straight into it, and on this scan 19 of those 55 points are on
    # the scalp, which stretched the box to 0.572 of frame width against the
    # corpus's 0.354.
    scale = ear_h / (2 * half * 0.40)
    g2 = in_frame(mesh, R2, C2, scale)
    img2, depth2, cam2 = render(g2, cfg)

    # ---- crop to the CORPUS framing, from the 2D landmarks ---------------
    # The 2D model is the reliable part of this pipeline (0.0149 NME), so the
    # ear's apparent extent comes from it. A depth head is sensitive to
    # occupancy, so this is matched rather than inherited.
    from PIL import Image as _I

    def landmark_render(im, ts, soft=False):
        r = pipe(im, timestamp=ts)
        if not r:
            if soft:
                return None
            sys.exit("the pipeline found no ear in the pinna-frame render")
        return np.asarray(max(r, key=lambda d: float(d["confidence"]))
                          ["landmarks"], float)[:K]

    lm2 = landmark_render(img2, 0.0)

    if a.zoom:
        # THE CAMERA DOES THE FRAMING, so there is no crop and no second
        # coordinate system: the z unit becomes the image width, a depth map
        # arrives already on the render's own grid, and the ear is rendered at
        # full resolution instead of inside a sub-window. Geometrically this
        # is the same view -- a crop and a narrowed field of view give the
        # same fx/width ratio, which is all corpus z depends on.
        #
        # THE DETECTOR HAS SCALE BLIND BANDS, which is the one thing that
        # makes this harder than it looks. Measured on this scan by sweeping
        # the zoom in 2% steps: zero detections anywhere in 0.90-0.98 or
        # 0.76-0.80 of the starting scale, while the scales on either side
        # return confidence 0.94-0.98. It is not a size limit and it is not
        # tracker state -- pipe.reset() changes nothing and the pattern
        # repeats exactly. The corpus occupancy lands inside the second band,
        # which is why the crop path never hit this: it only ever shows the
        # detector a WIDE framing and crops afterwards.
        #
        # So the target is tried, and if the detector refuses it the nearest
        # framing that works is used and the deviation is printed. That
        # matters for head 1, which is framing-sensitive (a crop round-trip
        # alone cost it 0.802 -> 0.584), and not for MoGe, which was never
        # trained on this corpus and only wants resolution.
        W0 = img2.shape[1]
        (q0, r0), (q1, r1) = lm2.min(0), lm2.max(0)
        if a.zoom_fill > 0:
            need = max(q1 - q0, r1 - r0) / W0 / a.zoom_fill
        else:
            need = max((q1 - q0) / W0 / 0.354,
                       (r1 - r0) / W0 / a.occupancy)
        print(f"zoom:   ear fills {(q1 - q0) / W0:.3f} x "
              f"{(r1 - r0) / W0:.3f} of the frame at the start; "
              f"target needs x{need:.4f}")
        base_scale, got = scale, None
        # the target first, then outward in 2% steps either side of it
        ladder = [1.0]
        for k in range(1, 9):
            ladder += [1 + 0.02 * k, 1 - 0.02 * k]
        for f in ladder:
            cand = base_scale * need * f
            g2t = in_frame(mesh, R2, C2, cand)
            it, dt, ct = render(g2t, cfg)
            lt = landmark_render(it, 100.0 + 10 * abs(f - 1) + f, soft=True)
            if lt is None:
                continue
            got = (cand, it, dt, ct, lt, f)
            break
        if got is None:
            sys.exit("no zoom within +-16% of the target framing is one the "
                     "detector will accept; run without --zoom")
        scale, img2, depth2, cam2, lm2, f = got
        H0, W0 = depth2.shape
        crop_box = (0, 0, W0, H0)
        img_c = img2
        side = size = float(W0)
        (bx0, by0), (bx1, by1) = lm2.min(0), lm2.max(0)
        note = "on target" if f == 1.0 else f"OFF TARGET by {100*(f-1):+.0f}%"
        print(f"frame:  ear fills {(bx1 - bx0) / side:.3f} of frame width, "
              f"{(by1 - by0) / side:.3f} of height "
              f"(corpus 0.354 x 0.677), no crop -- {note}")
        if f != 1.0:
            print(f"        the detector rejected the exact target framing; "
                  f"head 1's score below is at a framing it was not trained "
                  f"for and is not comparable with its crop-path number")

        # AIM THE CAMERA, not just zoom it. The corpus framing is occupancy
        # AND centring -- a crop puts the ear's box dead centre by
        # construction -- and matching only the occupancy leaves the ear up
        # to 6% of the frame off centre, which is enough to cost head 1 a
        # lot (0.277 -> 0.083 on the left ear). So the look-at point is
        # nudged until the ear's box centre lands on the principal point.
        #
        # Solved by iteration rather than in closed form: the correction is
        # a small translation of C2, and iterating makes the result
        # self-verifying instead of resting on getting a chain of frame
        # conventions right first time. Converges in two passes.
        for it in range(1, 5):
            (cx0, cy0), (cx1, cy1) = lm2.min(0), lm2.max(0)
            ecx, ecy = (cx0 + cx1) / 2, (cy0 + cy1) / 2
            dxp, dyp = cam2.cx - ecx, cam2.cy - ecy
            if max(abs(dxp), abs(dyp)) < 0.004 * W0:
                break
            zc_ear = float(np.mean(depth2[np.isfinite(depth2)]))
            u = np.array([dxp * zc_ear / cam2.fx,
                          dyp * zc_ear / cam2.fy, 0.0]) @ cam2.R
            C2t = C2 - (u * scale) @ R2
            g2t = in_frame(mesh, R2, C2t, scale)
            it2, dt2, ct2 = render(g2t, cfg)
            lt2 = landmark_render(it2, 300.0 + it, soft=True)
            if lt2 is None:
                print(f"        aim {it}: detector refused the recentred "
                      f"framing, keeping the previous one")
                break
            C2, img2, depth2, cam2, lm2 = C2t, it2, dt2, ct2, lt2
            img_c = img2
            print(f"        aim {it}: ear centre off by "
                  f"({dxp / W0:+.3f}, {dyp / W0:+.3f}) of the frame, corrected")
        (bx0, by0), (bx1, by1) = lm2.min(0), lm2.max(0)
        ec = ((bx0 + bx1) / 2 / W0, (by0 + by1) / 2 / W0)
        print(f"        ear centre now ({ec[0]:.3f}, {ec[1]:.3f}) of the "
              f"frame; a crop centres at (0.500, 0.500)")

        fov_x = float(np.degrees(2 * np.arctan(W0 / (2 * cam2.fx))))
        print(f"        true fov_x {fov_x:.3f} deg (fx {cam2.fx:.1f} px), "
              f"told to MoGe rather than guessed")
    else:
        fov_x = float(np.degrees(2 * np.arctan(depth2.shape[1]
                                               / (2 * cam2.fx))))
        (bx0, by0), (bx1, by1) = lm2.min(0), lm2.max(0)
        side = max((bx1 - bx0) / 0.354, (by1 - by0) / a.occupancy)
        cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
        x0, y0 = cx - side / 2, cy - side / 2
        H0, W0 = depth2.shape
        if x0 < 0 or y0 < 0 or x0 + side > W0 or y0 + side > H0:
            print(f"note: the corpus crop runs off the {W0}px render "
                  f"(x0 {x0:.0f}, side {side:.0f}); it is clamped, so "
                  f"occupancy is approximate")
            x0, y0 = max(0.0, min(x0, W0 - 1)), max(0.0, min(y0, H0 - 1))
            side = min(side, W0 - x0, H0 - y0)
        crop_box = (int(round(x0)), int(round(y0)),
                    int(round(x0 + side)), int(round(y0 + side)))
        img_c = np.asarray(_I.fromarray(img2).crop(crop_box))
        size = float(side)        # crop width in RENDER pixels: the z unit
        print(f"frame:  ear fills {(bx1 - bx0) / side:.3f} of crop width, "
              f"{(by1 - by0) / side:.3f} of height (corpus 0.354 x 0.677)")

    if a.dump_render:
        # EVERYTHING A SECOND DEPTH SOURCE NEEDS, and nothing it would have to
        # re-derive. The point is that a monocular depth model can be scored
        # against this scan on the SAME pixels this harness scores a depth
        # head on -- same locate, same pinna frame, same corpus framing --
        # so the two numbers are comparable and no framing difference can be
        # blamed for a gap. depth2 is the renderer's own float32 view-space
        # buffer, i.e. exact truth for these pixels, not an estimate.
        dr = pathlib.Path(a.dump_render)
        dr.mkdir(parents=True, exist_ok=True)
        _I.fromarray(img2).save(dr / "render_full.png")
        _I.fromarray(img_c).save(dr / "render_crop.png")
        np.savez_compressed(
            dr / "frame.npz", depth_full=depth2.astype(np.float32),
            crop_box=np.asarray(crop_box), side=side, lm2=lm2,
            fx=cam2.fx, fy=cam2.fy, cx=cam2.cx, cy=cam2.cy,
            R=cam2.R, t=cam2.t, ear=a.ear, mesh=str(a.mesh),
            fov_x=fov_x, zoom=bool(a.zoom))
        print(f"dump:   {dr}/render_full.png, render_crop.png, frame.npz")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t = np.asarray(Image.fromarray(img_c).resize((192, 192), Image.BILINEAR),
                   np.float32).transpose(2, 0, 1)[None] / 255.0
    xy = z_pred = uv = None
    if not a.no_head1:
        m = EarDepth3DModule.load_from_checkpoint(
            a.ckpt, map_location=dev).eval()
        dev = next(m.parameters()).device
        with torch.no_grad():
            pr = m(torch.from_numpy((t - 0.5) / 0.5).float().to(dev)
                   ).view(-1, 3).cpu().numpy()[:K]
        xy, z_pred = pr[:, :2], pr[:, 2] - pr[:, 2].mean()
        # xy is a fraction OF THE CROP; the depth buffer is the full render
        uv = np.stack([crop_box[0] + xy[:, 0] * side,
                       crop_box[1] + xy[:, 1] * side], axis=1)

    def sample_surface(uvq):
        """Where the scan's surface actually is, at these pixels."""
        Pq, hitq = backproject(depth2, uvq, cam2)
        zcq, scalpq = flag_scalp(Pq, hitq, cam2, a.scalp_tol)
        return Pq, hitq, zcq, scalpq, (hitq & ~scalpq)

    def lift(uvq, zq, zbar):
        """Pixel + PREDICTED z -> 3D. The ray comes from x,y; the distance
        along it comes from z and nothing else. zbar only fixes the overall
        scale, which a crop cannot determine and the model never claims."""
        zcp = zbar * (1.0 + zq * side / cam2.fx)
        pc = np.stack([(uvq[:, 0] - cam2.cx) / cam2.fx * zcp,
                       (uvq[:, 1] - cam2.cy) / cam2.fy * zcp, zcp], axis=1)
        return (pc - cam2.t) @ cam2.R

    mu = np.load(TPL / "lm_basis_v8_all_55_unitheight.npz", allow_pickle=True)
    mean_u = mu["mean"][:K] - mu["mean"][:K].mean(0)
    comp_u = mu["components"][:, :K]

    # The shipped pipeline's own 2D, in crop coordinates. This is the reliable
    # half of the system (0.0149 NME) and it is what a deployed MediaPipe-style
    # pipeline would feed the solver, so the fitted methods use it rather than
    # a depth head's x,y.
    uv_pipe = lm2.copy()
    xy_pipe = (lm2 - np.array(crop_box[:2])) / side

    methods = [] if a.no_head1 else [
        ("head1_depth", (0.95, 0.18, 0.58), uv, z_pred,
         "head 1: regressed depth")]

    if a.moge_npz:
        # THE SHIPPED 2D, LIFTED BY A GENERAL DEPTH MODEL. x and y are
        # uv_pipe, i.e. the 0.0149-NME model that actually ships; z is read
        # off the depth map at those same pixels. Nothing here was trained on
        # ears, and nothing is fitted to this scan: corpus z is invariant to a
        # global scale on the map, and a metric point map carries no unknown
        # shift, so the raw depth is usable as it stands.
        Mz = np.load(a.moge_npz)
        mgd = Mz["depth"].astype(float)
        # fx from the width of the image MoGe ACTUALLY RAN ON. The crop is
        # saved at its native size, so that width is already in render
        # pixels and the fx/side ratio corpus_z needs stays consistent.
        fx_m = float(Mz["intrinsics"][0, 0]) * mgd.shape[1]
        if mgd.shape != depth2.shape:
            fullm = np.full(depth2.shape, np.nan)
            fullm[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]] = (
                np.asarray(_I.fromarray(mgd).resize(
                    (crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]),
                    _I.NEAREST)))
            mgd = fullm
        # NEAREST, not bilinear: a bilinear tap across the silhouette blends
        # the pinna with whatever is behind it and returns a depth that is on
        # neither surface.
        ix = np.clip(np.rint(uv_pipe[:, 0]).astype(int), 0, mgd.shape[1] - 1)
        iy = np.clip(np.rint(uv_pipe[:, 1]).astype(int), 0, mgd.shape[0] - 1)
        zm = mgd[iy, ix]
        if not np.isfinite(zm).all():
            zm = np.where(np.isfinite(zm), zm, np.nanmedian(zm))
        z_moge = corpus_z(zm, fx_m, size)
        methods += [("moge_lift", (0.20, 0.95, 0.65), uv_pipe, z_moge,
                     "shipped 2D + MoGe-2 depth map (no ear training)")]
    if a.compare_flip:
        # same pixels, so the same rays and the same zbar: the two sets differ
        # ONLY in the sign of the predicted depth
        methods += [("head1_zflipped", (1.00, 0.85, 0.10), uv, -z_pred,
                     "head 1: z NEGATED (diagnostic)")]

    if a.feat_ckpt:
        from model.depth_module import EarFeatureLifter3DModule
        fm = EarFeatureLifter3DModule.load_from_checkpoint(
            a.feat_ckpt, map_location=dev).eval().to(dev)
        with torch.no_grad():
            xyf, _, zpf = fm.from_image(
                torch.from_numpy((t - 0.5) / 0.5).float().to(dev))
        xy_f = xyf.cpu().numpy()[0][:K]
        z_f = zpf.cpu().numpy()[0][:K]
        z_f = z_f - z_f.mean()
        uv_f = np.stack([crop_box[0] + xy_f[:, 0] * side,
                         crop_box[1] + xy_f[:, 1] * side], axis=1)
        methods += [("head3b_lifter", (0.65, 0.35, 0.90), uv_f, z_f,
                     "head 3b: frozen shipped 2D + lifter")]

    if a.shape_ckpt:
        from model.depth_module import EarShape3DModule
        sm = EarShape3DModule.load_from_checkpoint(
            a.shape_ckpt, map_location=dev).eval().to(dev)
        with torch.no_grad():
            prm = sm.model.parameters_from(
                torch.from_numpy((t - 0.5) / 0.5).float().to(dev))
            dec = sm.model.decode(prm).cpu().numpy()[0][:K]
        coef = prm.cpu().numpy()[0][:sm.model.n_modes]
        xy_s, z_s = dec[:, :2], dec[:, 2] - dec[:, 2].mean()
        uv_s = np.stack([crop_box[0] + xy_s[:, 0] * side,
                         crop_box[1] + xy_s[:, 1] * side], axis=1)
        shp = mean_u + np.einsum("m,mkj->kj", coef, comp_u[:len(coef)])
        shp = shp - shp.mean(0)
        methods += [
            ("shape_e2e", (1.00, 0.55, 0.10), uv_s, z_s,
             "shape head: REGRESSED pose, its own x,y"),
            ("shape_fit", (0.25, 0.55, 0.95), uv_pipe,
             weak_perspective(shp, xy_pipe)[0],
             "shape head: coefficients + SOLVED pose, shipped 2D"),
        ]
        print(f"shape head: predicted coefficient sd {coef.std():.3f} "
              f"(gt is 1.0)")

    if not a.compare_flip:
        # --compare-flip is about head 1's sign alone; a third prediction set
        # only crowds the viewer
        methods += [("meanear_fit", (0.20, 0.85, 0.85), uv_pipe,
                     weak_perspective(mean_u, xy_pipe)[0],
                     "mean ear + SOLVED pose, shipped 2D")]

    # ---- score each at ITS OWN pixels, then lift it ----------------------
    print(f"\n{'method':<44s} {'usable':>7s} {'signal':>8s} {'z_skill':>8s}"
          f" {'corr':>6s}")
    sets, scored = {}, {}
    for name, col, uvq, zq, label in methods:
        if a.flip_z:
            zq = -zq
        _, hitq, zcq, scalpq, goodq = sample_surface(uvq)
        if goodq.sum() < 10:
            print(f"  {label:<42s} too few usable landmarks")
            continue
        ztq = corpus_z(zcq[goodq], cam2.fx, side)
        ztq = ztq - ztq.mean()
        zpq = zq[goodq] - zq[goodq].mean()
        maeq = float(np.abs(zpq - ztq).mean())
        baseq = float(np.abs(ztq).mean())
        corrq = float(np.corrcoef(zpq, ztq)[0, 1])
        print(f"  {label:<42s} {int(goodq.sum()):5d}/{K} {baseq:8.4f} "
              f"{1 - maeq / baseq:8.3f} {corrq:6.3f}")
        zbarq = float(np.mean(zcq[goodq]))
        sets[name] = (col, [q for q in lift(uvq, zq, zbarq)])
        scored[name] = dict(uv=uvq, good=goodq, scalp=scalpq, zt=ztq,
                            zp=zpq, mae=maeq, base=baseq, zc=zcq)
    print(f"  {'predict z = 0':<42s} {'-':>5s}    {'-':>8s} {0.0:8.3f}"
          f" {'-':>6s}")

    # THE SURFACE REFERENCE IS THE SHIPPED 2D's PIXELS. It used to be head
    # 1's, which was wrong in two ways: it made a depth head mandatory even
    # for a flow that does not use one, and in the export it paired every
    # prediction's error stick with the surface point at a DIFFERENT pixel.
    # The reference now follows whichever set is primary, preferring the
    # lifted depth map, because that is the flow being reviewed.
    if not scored:
        sys.exit("nothing to score: --no-head1 with no --moge-npz leaves only "
                 "the mean-ear fit, which is not a depth source")
    ref_name = ("moge_lift" if "moge_lift" in scored
                else "head1_depth" if "head1_depth" in scored
                else next(iter(scored)))
    ref = scored[ref_name]
    P2, hit2, zc, scalp, good = sample_surface(ref["uv"])
    zt_g, zp = ref["zt"], ref["zp"]
    mae, base = ref["mae"], ref["base"]
    xy_ref = (ref["uv"] - np.array(crop_box[:2])) / side
    zf_all = weak_perspective(mean_u, xy_ref)[0]
    zf = zf_all[good] - zf_all[good].mean()
    mf = float(np.abs(zf - zt_g).mean())
    print(f"\nreference set for the figure and the export: {ref_name} "
          f"(at the pixels that set produced)")

    np.savez(out / "scan_depth.npz", xy=xy_ref, z_true_good=zt_g, zp=zp,
             good=good, hit=hit2, scalp=scalp, uv=ref["uv"], conf=conf,
             mae=mae, base=base, mean_ear_mae=mf, ref=ref_name)

    if a.export_obj:
        src = a.texture_obj or (a.mesh if a.mesh.lower().endswith(".obj") else "")
        if not src:
            sys.exit("--export-obj needs --texture-obj: the mesh being "
                     "rendered is not an .obj, and the point of the export is "
                     "the original texture")
        # EVERY TRANSFORM, INVERTED, so the spheres land in the scan's own
        # coordinates and the result overlays the original file.
        #   load_head : V1 = (V0 - c) / s0
        #   in_frame  : V2 = ((V1 - C2) @ R2.T) / scale
        V0 = np.asarray(o3d.io.read_triangle_mesh(src).vertices)
        c0 = V0.mean(0)
        s0 = float(np.abs(V0 - c0).max())

        def to_scan(P2):
            return (((np.atleast_2d(P2) * scale) @ R2) + C2) * s0 + c0

        surf = [to_scan(P2[i])[0] if hit2[i] else None for i in range(K)]
        osets = {k: (c, [to_scan(q)[0] for q in pts])
                 for k, (c, pts) in sets.items()}
        osets["scan_surface"] = ((0.15, 0.50, 1.00),
                                 [v if (hit2[i] and not scalp[i]) else None
                                  for i, v in enumerate(surf)])
        osets["scan_surface_SCALP"] = ((1.0, 1.0, 1.0),
                                       [v if scalp[i] else None
                                        for i, v in enumerate(surf)])
        if a.snap_surface > 0:
            # the camera's forward axis, carried into the scan's coordinates:
            # a direction picks up R2 and the scales, not the translations
            fwd = (cam2.R.T @ np.array([0.0, 0.0, 1.0])) @ R2
            # points the depth buffer SAW, so known to be on the visible
            # surface -- these verify the normal orientation
            Pp, hp = backproject(depth2, uv_pipe, cam2)
            probe = np.array([to_scan(Pp[i])[0] for i in range(K) if hp[i]])
            print(f"\nsnap to front surface (max {a.snap_surface:.2f} "
                  f"scan units):")
            for nm in [k for k in osets
                       if not k.startswith("scan_surface")]:
                col, pts = osets[nm]
                idx = [i for i, q in enumerate(pts)
                       if q is not None and np.isfinite(q).all()]
                if not idx:
                    continue
                Q = np.array([pts[i] for i in idx])
                Qs, dd, mv, note = snap_front_surface(
                    Q, src, fwd, a.snap_surface, probe=probe)
                new = list(pts)
                for j, i in enumerate(idx):
                    new[i] = Qs[j]
                osets[nm] = (col, new)
                # THE PRE-SNAP DISTANCE IS A MEASUREMENT, not a
                # by-product, and a better-behaved one than z_skill in two
                # ways: it is in scan units, and it keeps ALL the points
                # rather than dropping the ones whose 2D left the pinna. So
                # the median is reported over everything; the median over the
                # moved subset alone is bounded by the limit and so is biased
                # low by construction.
                print(f"  {nm:<16s} {note}")
                print(f"  {'':<16s} distance to front surface BEFORE the "
                      f"snap: median {np.median(dd):.2f}, "
                      f"p90 {np.percentile(dd, 90):.2f}, max {dd.max():.2f}")
                print(f"  {'':<16s} snapped {int(mv.sum())}/{len(idx)} "
                      f"(within {a.snap_surface:.1f}), median move "
                      f"{np.median(dd[mv]) if mv.any() else 0:.2f}; "
                      f"{int((~mv).sum())} left alone")

        if a.export_only:
            want = [w.strip() for w in a.export_only.split(",") if w.strip()]
            missing = [w for w in want if w not in osets]
            if missing:
                sys.exit(f"--export-only {missing} not among "
                         f"{sorted(osets)}")
            osets = {k: v for k, v in osets.items() if k in want}

        rad = a.sphere * ear_h * s0
        # AN ERROR STICK WITHOUT ITS TRUTH POINT MEANS NOTHING. Each stick
        # runs from a prediction to the surface point at the same pixel, so
        # if scan_surface has been excluded from the export the sticks are
        # measuring against something that is not in the file; they are
        # dropped with it rather than left pointing into empty space.
        links = {k: [(q, surf[i]) for i, q in enumerate(pts)
                     if hit2[i] and not scalp[i]]
                 for k, (_, pts) in osets.items()
                 if k not in ("scan_surface", "scan_surface_SCALP")} \
            if "scan_surface" in osets else {}
        wire = None
        if a.wireframe:
            wire = (a.wire_radius * ear_h * s0,
                    tuple(float(v) for v in a.wire_color.split(",")),
                    a.wire_tris)
        oo, mm, tt = write_review_obj(a.export_obj, src, osets, rad,
                                      lambda i: next(n for (lo, hi), n
                                                     in zip(STRIPS, STRIP_NAMES)
                                                     if lo <= i < hi),
                                      links=links, wire=wire)
        print(f"\nreview mesh -> {oo}")
        print(f"  {mm.name}, texture {tt.name if tt else 'NONE'}")
        print(f"  spheres r={rad:.2f} scan units:")
        for k, (c, pts) in osets.items():
            print(f"    {k:<22s} {sum(q is not None for q in pts):3d}  "
                  f"rgb {tuple(round(x, 2) for x in c)}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    idx = np.arange(K)[good]
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.6))
    ax[0].imshow(img_c)
    cu = xy_ref * side          # crop-local pixels, for drawing
    for lo, hi in STRIPS:
        ax[0].plot(cu[lo:hi, 0], cu[lo:hi, 1], "-", c="#ff2d95", lw=1.4)
    ax[0].plot(cu[:, 0], cu[:, 1], ".", c="#ff2d95", ms=3)
    if scalp.any():
        ax[0].plot(cu[scalp, 0], cu[scalp, 1], "o", mfc="none", mec="w",
                   ms=9, label="sampled the scalp")
        ax[0].legend(fontsize=6, loc="lower right")
    ax[0].set_title(f"render of the scan + head 1 (det conf {conf:.2f})",
                    fontsize=8)
    ax[0].axis("off")
    ax[1].plot(idx, zt_g, "-o", c="#2ca02c", ms=3, label="scan surface")
    ax[1].plot(idx, zp, "-o", c="#ff2d95", ms=3, label=ref_name)
    ax[1].plot(idx, zf, "-o", c="#5599ff", ms=3, label="mean ear + pose")
    ax[1].set_xlabel("landmark index")
    ax[1].set_ylabel("z (crop widths)")
    ax[1].set_title("depth profile along the scheme", fontsize=8)
    ax[1].legend(fontsize=6)
    ax[1].grid(alpha=.25)
    ax[2].plot(xy_ref[good, 0], zt_g, "o", c="#2ca02c", ms=4, label="scan")
    ax[2].plot(xy_ref[good, 0], zp, "o", c="#ff2d95", ms=4, label=ref_name)
    ax[2].set_xlabel("x")
    ax[2].set_ylabel("z")
    ax[2].invert_yaxis()
    ax[2].set_title("from above (x, z)", fontsize=8)
    ax[2].legend(fontsize=6)
    ax[2].grid(alpha=.25)
    fig.suptitle(
        f"{pathlib.Path(a.mesh).stem}: predicted depth against the scan's own "
        f"geometry.  {ref_name} z_skill {1 - mae / base:.3f}, mean-ear fit "
        f"{1 - mf / base:.3f}, {int(good.sum())}/{K} usable landmarks."
        "\nPlacement is NOT scored: the scan has no annotation. This asks "
        "whether the depth matches the surface at the pixels the model chose.",
        fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(out / "scan_depth.png", dpi=140)
    print(f"\nwrote {out}/scan_depth.png and scan_depth.npz  (both gitignored)")


if __name__ == "__main__":
    main()
