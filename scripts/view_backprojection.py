"""Standalone, editable: 3D head -> face-on render -> landmarker -> back-projection.

    1. load a head mesh (HUTUBS .ply / any .ply/.obj head) or a lone ear
    2. find the ear with the detector, build the ear's OWN plane frame, and
       render it face-on at the occupancy the model was trained at
    3. run the landmarker on that render -> 55 points in crop pixels
    4. back-project each point by ray-cast onto the mesh -> 55 points in 3D
    5. optionally snap them onto the ear surface
    6. open an interactive window -- mesh + 3D landmarks + the four linestrips --
       and write a 2D overlay PNG of the same prediction beside it

SELF-CONTAINED BY DESIGN. Every geometry stage is inlined below rather than
imported, so you can hack on any of it without perturbing ingest_render3d.py or
the eval scripts. The only project imports are the two networks (EarDetector,
LandmarkPredictor) and the checkpoint picker, which are the model, not geometry.
The flip side: fixes made here do NOT propagate back to the pipeline, and fixes
made there do not arrive here. If a change proves out, port it deliberately.

KNOWN LIMITATION, and the reason this viewer exists. A camera ray that passes
just outside the pinna hits the SCALP behind it, so that landmark lands about one
ear-depth too deep. It looks correct face-on -- the error is purely in depth --
and only separates when you orbit. Snapping helps a ray that merely grazed; it
cannot help one that genuinely struck the head, because the nearest surface to
that hit IS the scalp it already hit. Orbit to ~60 deg to see which points are
wrong, and watch the black spheres (ray missed) and white ones (chain-repaired).

Controls: drag to orbit, scroll to zoom, R resets the view, Q or Escape closes.

Usage:
    python scripts/view_backprojection.py
    python scripts/view_backprojection.py --mesh /path/pp16_3DheadMesh.ply --ear 1
    python scripts/view_backprojection.py --audioear 0 --rays
    python scripts/view_backprojection.py --no-window      # just stats + PNG
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import tempfile
import zipfile
from pathlib import Path

# MUST COME BEFORE `import open3d`, and it is not optional on a Wayland session.
# Open3D's interactive window is the legacy GLFW/GLEW visualizer. On Wayland it
# fails outright -- "Failed to initialize GLEW", then "Failed creating OpenGL
# window" -- and draw_geometries returns having shown nothing. GLFW has to be
# pushed onto its X11 path, which XWayland then serves, and BOTH of these are
# needed: clearing WAYLAND_DISPLAY alone still fails. Verified on this machine.
#
# Timing matters as much as the values. Setting them after the first
# OffscreenRenderer has run is too late -- Open3D has initialised its renderer by
# then and the window still fails. Hence module scope, above the import.
# Offscreen rendering itself is unaffected either way: that path is EGL.
if os.environ.get("WAYLAND_DISPLAY") and os.environ.get("DISPLAY"):
    os.environ.pop("WAYLAND_DISPLAY", None)
    os.environ["XDG_SESSION_TYPE"] = "x11"

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------- CONFIG ----
# Third-party research data; not in any repo. Override with EAR3D_DIR.
EAR3D_DIR = Path(os.environ.get("EAR3D_DIR", ROOT.parent / "clean_3d_data"))

CONFIG = dict(
    # --- what to load -------------------------------------------------------
    mesh=str(EAR3D_DIR / "3D head meshes" / "pp12_3DheadMesh.ply"),
    audioear=None,          # instead: index into the AudioEar3D zip, e.g. 0
    ear=0,                  # which detected ear: 0 = highest confidence, 1 = other side

    # --- model --------------------------------------------------------------
    run="manual_occ_s42",   # checkpoint under runs/checkpoints/<run>/
    device="cuda",

    # --- camera / framing (these mirror the training pipeline) --------------
    size=600,               # full render resolution before cropping
    eye_z=3.0,              # camera at (0, 0, eye_z) looking at the origin
    vfov=50.0,              # vertical field of view, degrees
    occupancy=0.777,        # ear extent / crop side. This is TRAIN_OCCUPANCY.
    crop_out=192,           # crop is resized to this before the landmarker
    frame_expand=1.0,       # detector box -> CROP FRAMING region. Keep at 1.0:
                            # expanding this puts scalp in the crop and the
                            # predicted landmarks come out larger than the ear.
    target_expand=2.5,      # detector box -> RAYCAST / SNAP region. Generous on
                            # purpose: a tight box truncates the pinna, so rim
                            # rays miss and rim landmarks snap onto the cut edge.
    depth_frac=0.22,        # keep only vertices within this fraction of mesh
                            # extent of the frontmost depth in the box -- a 2D
                            # box otherwise selects the whole column through the
                            # skull (measured: 21% of the head).

    # --- back-projection ----------------------------------------------------
    raycast_on="patch",     # "patch" = the target_expand region, "full" = all of it
    snap=True,
    snap_in_plane_all=False,    # force the in-plane snap on EVERY point, not just misses
    snap_tol=2.5,           # chain-break tolerance, in median strip steps
    snap_passes=3,

    # --- display ------------------------------------------------------------
    sphere_frac=0.022,      # landmark sphere radius, in mesh units (ear spans ~1)
    show_rays=False,        # draw the camera ray to each landmark
    show_patch=False,       # draw the raycast target region as a point cloud
    # the render derives from third-party 3D data, so this defaults OUTSIDE the repo
    overlay_png=str(Path(tempfile.gettempdir()) / "backprojection_overlay.png"),
    interactive=True,
)
# ---------------------------------------------------------------------------

# 55-point iBUG ear scheme, four ordered linestrips.
STRIPS = [(0, 20), (20, 35), (35, 50), (50, 55)]
STRIP_NAMES = ["outer_helix", "inner_helix", "concha_border", "superior_crus"]
STRIP_COLOURS = [(0.95, 0.25, 0.25), (0.25, 0.65, 0.95),
                 (0.30, 0.85, 0.35), (0.98, 0.80, 0.15)]


# ============================================================ camera ========
# project() and unproject_rays() are exact inverses for this camera. Change one
# and you must change the other, or the back-projection lands somewhere else.

def project(P, cfg):
    P = np.atleast_2d(np.asarray(P, float))
    z = cfg["eye_z"] - P[:, 2]
    t = np.tan(np.radians(cfg["vfov"]) / 2.0)
    s = cfg["size"]
    return np.stack([(P[:, 0] / np.maximum(z * t, 1e-9) + 1) / 2 * s,
                     (1 - P[:, 1] / np.maximum(z * t, 1e-9)) / 2 * s], axis=1)


def unproject_rays(uv, cfg):
    t = np.tan(np.radians(cfg["vfov"]) / 2.0)
    s = cfg["size"]
    x = (uv[:, 0] / s * 2 - 1) * t
    y = (1 - uv[:, 1] / s * 2) * t
    d = np.stack([x, y, -np.ones(len(uv))], axis=1)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return np.tile(np.array([0.0, 0.0, cfg["eye_z"]]), (len(uv), 1)), d


def render(mesh, cfg, size=None, background=(0.5, 0.5, 0.5)):
    """Offscreen render (EGL -- works headless). Returns HxWx3 uint8."""
    size = size or cfg["size"]
    r = o3d.visualization.rendering.OffscreenRenderer(size, size)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    r.scene.add_geometry("m", mesh, mat)
    r.scene.set_background([*background, 1.0])
    r.scene.scene.set_sun_light([-0.3, -0.4, -0.9], [1.0, 1.0, 1.0], 95000)
    r.scene.scene.enable_sun_light(True)
    r.setup_camera(cfg["vfov"], np.zeros(3, np.float32),
                   np.array([0, 0, cfg["eye_z"]], np.float32),
                   np.array([0, 1, 0], np.float32))
    img = np.asarray(r.render_to_image())[:, :, :3]
    del r
    return img


def raycast(mesh, uv, cfg):
    """Exact 2D -> 3D. Returns (points, hit mask); misses are NaN."""
    t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(t)
    o, d = unproject_rays(uv, cfg)
    ans = scene.cast_rays(o3d.core.Tensor(np.hstack([o, d]).astype(np.float32)))
    dist = ans["t_hit"].numpy()
    hit = np.isfinite(dist)
    P = np.full((len(uv), 3), np.nan)
    P[hit] = o[hit] + d[hit] * dist[hit, None]
    return P, hit


# ============================================================ ear finding ===

def load_head(path):
    """Head mesh, centred and scaled to unit radius."""
    m = o3d.io.read_triangle_mesh(str(path))
    if len(m.vertices) == 0:
        return None
    V = np.asarray(m.vertices)
    V = (V - V.mean(0)) / np.abs(V - V.mean(0)).max()
    m.vertices = o3d.utility.Vector3dVector(V)
    m.compute_vertex_normals()
    return m


def _probe(mesh, front, up, dist, size, cfg):
    r = o3d.visualization.rendering.OffscreenRenderer(size, size)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    r.scene.add_geometry("m", mesh, mat)
    r.scene.set_background([0.5, 0.5, 0.5, 1.0])
    r.scene.scene.set_sun_light([-0.4, -0.3, -0.9], [1, 1, 1], 100000)
    r.scene.scene.enable_sun_light(True)
    eye = np.asarray(front, float) * dist
    r.setup_camera(cfg["vfov"], np.zeros(3, np.float32), eye.astype(np.float32),
                   np.asarray(up, np.float32))
    img = np.asarray(r.render_to_image())[:, :, :3]
    del r
    return img


def find_ears(mesh, det, cfg, probe_size=400, dist=2.4):
    """Locate ears by sweeping the detector around the head.

    Geometric rules do not transfer between datasets -- HUTUBS and SONICOM store
    heads on different axes -- so the ear is found by what an ear detector sees.
    """
    out = []
    for axis in (0, 1, 2):
        for sgn in (1, -1):
            front = np.zeros(3); front[axis] = sgn
            up = np.array([0.0, 0.0, 1.0]) if axis != 2 else np.array([0.0, 1.0, 0.0])
            b = det.detect(_probe(mesh, front, up, dist, probe_size, cfg))
            if not len(b):
                continue
            h, w = b[0, 2] - b[0, 0], b[0, 3] - b[0, 1]
            if (h * w) / (probe_size ** 2) > 0.25:   # a whole-head box is a false positive
                continue
            out.append((float(b[0, 4]), front, up, b[0, :4].copy()))
    out.sort(key=lambda t: -t[0])
    return out[:2]


def expand_box(box, factor, size):
    ymin, xmin, ymax, xmax = box
    cy, cx = (ymin + ymax) / 2.0, (xmin + xmax) / 2.0
    hh, hw = (ymax - ymin) * factor / 2.0, (xmax - xmin) * factor / 2.0
    return np.array([max(0.0, cy - hh), max(0.0, cx - hw),
                     min(float(size), cy + hh), min(float(size), cx + hw)])


def ear_region(mesh, front, up, box, cfg, factor, dist=2.4, size=400):
    """Boolean mask of mesh vertices in the detector's ear box, front surface only.

    Returns a MASK, not points: callers index the mesh with it. Rebuilding a mask
    by value-matching rounded coordinates has silently mismatched here before.
    """
    box = expand_box(box, factor, size) if factor != 1.0 else np.asarray(box, float)
    V = np.asarray(mesh.vertices)
    z = np.asarray(front, float); z = z / np.linalg.norm(z)
    y = np.asarray(up, float) - z * (np.asarray(up, float) @ z)
    y /= max(np.linalg.norm(y), 1e-9)
    P = V @ np.stack([np.cross(y, z), y, z]).T
    t = np.tan(np.radians(cfg["vfov"]) / 2.0)
    zc = np.maximum((dist - P[:, 2]) * t, 1e-9)
    uu = (P[:, 0] / zc + 1) / 2 * size
    vv = (1 - P[:, 1] / zc) / 2 * size
    ymin, xmin, ymax, xmax = box
    inside = (uu >= xmin) & (uu <= xmax) & (vv >= ymin) & (vv <= ymax)
    if inside.sum() < 200:
        return None
    front_z = np.percentile(P[inside, 2], 99)
    window = cfg["depth_frac"] * max(np.ptp(V, axis=0).max(), 1e-6)
    sel = inside & (P[:, 2] > front_z - window)
    return sel if sel.sum() >= 200 else inside


def pinna_frame(sel_pts, front, up):
    """Rotation taking the EAR's own surface normal onto +Z (toward the camera).

    The pinna's plane is not the head's lateral plane: over 14 HUTUBS subjects
    they differ by 8.1 deg on average (sd 3.9, range 1-13), varying per subject.
    Labelling along the lateral axis would bake that tilt into the result.
    """
    if sel_pts is None or len(sel_pts) < 200:
        return None, None
    C = sel_pts.mean(0)
    _, _, vt = np.linalg.svd(sel_pts - C, full_matrices=False)
    n = vt[2]
    if n @ np.asarray(front, float) < 0:
        n = -n
    z = n / np.linalg.norm(n)
    y = np.asarray(up, float) - z * (np.asarray(up, float) @ z)
    if np.linalg.norm(y) < 1e-6:
        return None, None
    y /= np.linalg.norm(y)
    return np.stack([np.cross(y, z), y, z]), C      # world -> pinna frame


def framed_crop(mesh, ear_pts, cfg):
    """Render, then crop so the EAR fills `occupancy` of the frame.

    `ear_pts` must be the ear, not the mesh. The head is deliberately left in the
    render for context -- a floating ear teaches the model that a hard silhouette
    marks its boundary -- but framing to the whole head gives 0.23 occupancy
    against the 0.777 the model was trained at.
    """
    img = render(mesh, cfg)
    uv = project(np.asarray(ear_pts, float), cfg)
    side = max(np.ptp(uv[:, 0]), np.ptp(uv[:, 1])) / cfg["occupancy"]
    cx = (uv[:, 0].min() + uv[:, 0].max()) / 2
    cy = (uv[:, 1].min() + uv[:, 1].max()) / 2
    l, t, s = int(round(cx - side / 2)), int(round(cy - side / 2)), int(round(side))
    if s < 48:
        return None, None
    out = cfg["crop_out"]
    cv = Image.new("RGB", (s, s), (128, 128, 128))       # grey-128 pad, as in training
    src = Image.fromarray(img)
    sx0, sy0 = max(0, l), max(0, t)
    sx1, sy1 = min(cfg["size"], l + s), min(cfg["size"], t + s)
    if sx1 <= sx0 or sy1 <= sy0:
        return None, None
    cv.paste(src.crop((sx0, sy0, sx1, sy1)), (sx0 - l, sy0 - t))
    return cv.resize((out, out), Image.BILINEAR), (l, t, s, out)


def crop_to_full(pts, box):
    l, t, s, out = box
    return np.stack([pts[:, 0] / out * s + l, pts[:, 1] / out * s + t], axis=1)


# ============================================================ snapping ======

def snap_in_plane(points, V, hit):
    """Snap within the image plane only: adjust x/y, never push the point in/out.

    A plain 3D nearest-neighbour snap can pull a landmark BACKWARDS into the scalp
    directly behind the pinna -- same image position, slightly further away, and
    genuinely the nearest surface in 3D. That is the one direction the move must
    not take: the 2D prediction is what the model is good at, depth is what it
    cannot see. A ray that HIT already returns the frontmost surface at exactly
    that image position, so it is left alone.
    """
    P = np.asarray(points, float).copy()
    need = ~np.asarray(hit, bool)
    if not need.any():
        return P
    _, j = cKDTree(V[:, :2]).query(P[need][:, :2], k=1)
    P[need] = V[np.atleast_1d(j)]
    return P


def _neighbours(k):
    for a, b in STRIPS:
        if a <= k < b:
            return [j for j in (k - 1, k + 1) if a <= j < b]
    return []


def snap_chain(points, V, tol, max_passes):
    """Repair landmarks that broke their strip's spacing. Returns (points, mask).

    The 55 points are four ordered linestrips, so a correctly placed landmark
    sits roughly one step from its sequence neighbours; one pinned to the skull is
    an outlier against that spacing while its neighbours are not. Structural on
    purpose -- it needs no pinna/scalp segmentation, which is the part that has
    proven unreliable. It also cannot see a scalp hit that happens to stay
    plausibly spaced, which is the residual error this viewer shows.
    """
    tree = cKDTree(V)
    P = np.asarray(points, float).copy()
    repaired = np.zeros(len(P), bool)
    for _ in range(max_passes):
        step = {}
        for a, b in STRIPS:
            d = np.linalg.norm(np.diff(P[a:b], axis=0), axis=1)
            ok = d[~repaired[a + 1:b]] if repaired[a + 1:b].any() else d
            step[(a, b)] = np.median(ok) if len(ok) else np.median(d)
        flagged = []
        for k in range(len(P)):
            nb = [j for j in _neighbours(k) if not repaired[j]]
            if not nb:
                continue
            ab = next((a, b) for a, b in STRIPS if a <= k < b)
            if np.mean([np.linalg.norm(P[k] - P[j]) for j in nb]) > tol * step[ab]:
                flagged.append(k)
        if not flagged:
            break
        for k in flagged:
            nb = [j for j in _neighbours(k) if j not in flagged]
            if not nb:
                continue
            P[k] = V[tree.query(np.mean([P[j] for j in nb], axis=0))[1]]
            repaired[k] = True
    return P, repaired


# ============================================================ display ======

def spheres(points, radius, colours):
    out = o3d.geometry.TriangleMesh()
    for p, c in zip(points, colours):
        if not np.all(np.isfinite(p)):
            continue
        s = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=8)
        s.translate(p)
        s.paint_uniform_color(c)
        out += s
    out.compute_vertex_normals()
    return out


def strip_lines(points):
    pts, idx, col = [], [], []
    for i, (a, b) in enumerate(STRIPS):
        for k in range(a, b - 1):
            if np.all(np.isfinite(points[k])) and np.all(np.isfinite(points[k + 1])):
                idx.append([len(pts), len(pts) + 1])
                pts += [points[k], points[k + 1]]
                col.append(STRIP_COLOURS[i])
    return _lineset(pts, idx, col)


def ray_lines(points, cfg):
    """Camera ray to each landmark: shows where the back-projection came from."""
    o, d = unproject_rays(project(points, cfg), cfg)
    pts, idx = [], []
    for i, p in enumerate(points):
        if not np.all(np.isfinite(p)):
            continue
        idx.append([len(pts), len(pts) + 1])
        pts += [o[i] + d[i] * 0.5, p]
    return _lineset(pts, idx, [(0.55, 0.55, 0.55)] * len(idx))


def _lineset(pts, idx, col):
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.array(pts) if pts else np.zeros((0, 3)))
    ls.lines = o3d.utility.Vector2iVector(np.array(idx) if idx else np.zeros((0, 2), int))
    ls.colors = o3d.utility.Vector3dVector(np.array(col) if col else np.zeros((0, 3)))
    return ls


def write_overlay(crop, lm, path):
    """The 2D prediction on the render it came from, in the same strip colours."""
    img = crop.convert("RGB").resize((512, 512), Image.BILINEAR)
    s = 512 / crop.width
    d = ImageDraw.Draw(img)
    for i, (a, b) in enumerate(STRIPS):
        c = tuple(int(255 * v) for v in STRIP_COLOURS[i])
        d.line([tuple(p) for p in (lm[a:b] * s)], fill=c, width=2)
        for x, y in lm[a:b] * s:
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=c, outline=(0, 0, 0))
    img.save(path)
    print(f"wrote {path}")



# ============================================================ pipeline =====

def load_subject(cfg):
    """Returns (mesh in the pinna frame, ear points, target vertex indices).

    The mesh is placed in the EAR's own plane frame and scaled so the ear spans
    about one unit, so pose zero means "camera perpendicular to this pinna" for
    every subject alike rather than perpendicular to the head's lateral plane.
    """
    if cfg["audioear"] is not None:
        mesh = load_audioear(int(cfg["audioear"]), cfg)
        V = np.asarray(mesh.vertices)
        # A lone ear is already oriented and unit-scaled, and there is no scalp to
        # separate from, so every vertex is a valid raycast target.
        return mesh, V, np.arange(len(V))

    from inference import BLAZEEAR_DIR, DETECTOR_WEIGHTS, EarDetector
    det = EarDetector(BLAZEEAR_DIR / DETECTOR_WEIGHTS, "cpu", 0.5)
    mesh = load_head(cfg["mesh"])
    if mesh is None:
        sys.exit(f"could not read {cfg['mesh']}")
    found = find_ears(mesh, det, cfg)
    if not found:
        sys.exit("the detector found no ear on this head")
    print("detector found " + ", ".join(f"conf={c:.2f}" for c, *_ in found))
    conf, front, up, box = found[int(cfg["ear"]) % len(found)]
    print(f"using ear {cfg['ear']} (conf {conf:.2f})")

    frame_mask = ear_region(mesh, front, up, box, cfg, cfg["frame_expand"])
    target_mask = ear_region(mesh, front, up, box, cfg, cfg["target_expand"])
    if frame_mask is None or target_mask is None:
        sys.exit("too few vertices in the ear box")
    V = np.asarray(mesh.vertices)
    R, C = pinna_frame(V[frame_mask], front, up)
    if R is None:
        sys.exit("pinna plane fit failed")

    base = copy.deepcopy(mesh)
    scale = float(np.abs((V[frame_mask] - C) @ R.T).max())
    base.vertices = o3d.utility.Vector3dVector(((V - C) @ R.T) / scale)
    base.compute_vertex_normals()
    Vn = np.asarray(base.vertices)
    print(f"ear region: {frame_mask.sum()} vertices framed, "
          f"{target_mask.sum()} in the raycast target")
    return base, Vn[frame_mask], np.flatnonzero(target_mask)


def load_audioear(index, cfg, depth=9, max_dist=12.0):
    """One AudioEar3D ear, Poisson-reconstructed into a canonically oriented mesh.

    max_dist culls invented surface by distance to the nearest SCANNED point, in
    units of mean point spacing, and should be LOOSE. The two populations separate
    cleanly -- real surface at or under ~4x spacing, Poisson's spurious sheets at
    ~30x -- so 12x sits in the empty gap. Tighter settings (a density quantile,
    or 3.5x) sculpted holes into real anatomy, and a landmark ray through a hole
    is silently dropped.
    """
    zp = EAR3D_DIR / "AudioEar3D.zip"
    z = zipfile.ZipFile(str(zp))
    members = sorted(x for x in z.namelist() if x.endswith(".ply"))
    member = members[index]
    print(f"AudioEar3D ear {index}: {member}  (of {len(members)})")
    fd, tmp = tempfile.mkstemp(suffix=".ply")
    os.write(fd, z.read(member))
    os.close(fd)
    try:
        pcd = o3d.io.read_point_cloud(tmp)
    finally:
        os.unlink(tmp)
    # These PLYs already carry scanner normals -- use them. Re-estimating is worse
    # (the scanner knows its own view direction) and the orientation pass is
    # order-dependent, so it made reconstruction vary between runs on one input.
    if not pcd.has_normals():
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=40))
        pcd.orient_normals_consistent_tangent_plane(30)
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    pts = np.asarray(pcd.points)
    tree = cKDTree(pts)
    spacing = tree.query(pts, k=2)[0][:, 1].mean()
    mesh.remove_vertices_by_mask(
        tree.query(np.asarray(mesh.vertices))[0] > max_dist * spacing)
    V = np.asarray(mesh.vertices)
    if len(V) == 0:
        sys.exit("ear failed to reconstruct")
    V = V - V.mean(0)
    _, s, vt = np.linalg.svd(V, full_matrices=False)
    axes = vt[np.argsort(-s)]
    R = np.stack([axes[1], axes[0], axes[2]])   # smallest extent -> +Z, largest -> +Y
    if np.linalg.det(R) < 0:
        R[2] *= -1
    V = (V @ R.T)
    mesh.vertices = o3d.utility.Vector3dVector(V / np.abs(V).max())
    mesh.compute_vertex_normals()
    return mesh


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mesh", default=None, help="head .ply/.obj")
    p.add_argument("--audioear", type=int, default=None, help="index into AudioEar3D.zip")
    p.add_argument("--ear", type=int, default=None, help="which detected ear, 0 or 1")
    p.add_argument("--run", default=None, help="checkpoint run name")
    p.add_argument("--device", default=None)
    p.add_argument("--no-snap", action="store_true")
    p.add_argument("--snap-all", action="store_true", help="in-plane snap every point")
    p.add_argument("--rays", action="store_true", help="draw the camera rays")
    p.add_argument("--patch", action="store_true", help="draw the raycast target region")
    p.add_argument("--no-window", action="store_true", help="stats and PNG only")
    a = p.parse_args()

    cfg = dict(CONFIG)
    if a.mesh:
        cfg["mesh"], cfg["audioear"] = a.mesh, None
    if a.audioear is not None:
        cfg["audioear"] = a.audioear
    for k, v in (("ear", a.ear), ("run", a.run), ("device", a.device)):
        if v is not None:
            cfg[k] = v
    cfg["snap"] = cfg["snap"] and not a.no_snap
    cfg["snap_in_plane_all"] |= a.snap_all
    cfg["show_rays"] |= a.rays
    cfg["show_patch"] |= a.patch
    cfg["interactive"] = cfg["interactive"] and not a.no_window

    # --- 1. the subject, in its own pinna frame ------------------------------
    mesh, ear_pts, target_idx = load_subject(cfg)
    target_V = np.asarray(mesh.vertices)[target_idx]

    # --- 2. face-on render at the trained occupancy --------------------------
    crop, box = framed_crop(mesh, ear_pts, cfg)
    if crop is None:
        sys.exit("the ear framed to under 48 px -- wrong ear region?")
    print(f"face-on crop: {box[2]} px at occupancy {cfg['occupancy']}")

    # --- 3. landmark it ------------------------------------------------------
    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_test import best_ckpt
    from inference import LandmarkPredictor
    ck = best_ckpt(cfg["run"])
    if ck is None:
        sys.exit(f"no checkpoint for run {cfg['run']!r} under runs/checkpoints/")
    lm, conf = LandmarkPredictor(ck, cfg["device"]).predict(
        np.asarray(crop), with_confidence=True)
    lm = np.asarray(lm, float)
    print(f"landmarker {ck.name}: confidence mean {float(np.mean(conf)):.3f} "
          f"min {float(np.min(conf)):.3f}")

    # --- 4. back-project -----------------------------------------------------
    if cfg["raycast_on"] == "patch" and len(target_idx) < len(mesh.vertices):
        target = mesh.select_by_index(target_idx)
        target.compute_vertex_normals()
    else:
        target = mesh
    P3, hit = raycast(target, crop_to_full(lm, box), cfg)
    print(f"ray-cast: {int(hit.sum())}/{len(P3)} rays hit")
    # A missed ray has no 3D position. Park it on the target centroid so the snap
    # has something finite to move; its sphere is drawn black either way.
    P3 = np.where(np.isfinite(P3), P3, target_V.mean(0))

    # --- 5. snap -------------------------------------------------------------
    repaired = np.zeros(len(P3), bool)
    if cfg["snap"]:
        before = P3.copy()
        h = np.zeros(len(P3), bool) if cfg["snap_in_plane_all"] else hit
        P3 = snap_in_plane(P3, target_V, h)
        # Measure the in-plane move BEFORE the chain repair or the two conflate.
        # A point whose ray missed was parked at the centroid above, so its move
        # is centroid -> surface and is expected to be large.
        step = np.linalg.norm(P3 - before, axis=1)
        moved = step > 1e-9
        P3, repaired = snap_chain(P3, target_V, cfg["snap_tol"], cfg["snap_passes"])
        print(f"snap: {int(moved.sum())} moved in-plane "
              f"(median {np.median(step[moved]) if moved.any() else 0.0:.4f}), "
              f"{int(repaired.sum())} chain-repaired")

    # Depth spread is the tell for a scalp-pinned point: it sits behind the rest.
    print(f"landmark depth z: {P3[:,2].min():+.3f} to {P3[:,2].max():+.3f} "
          f"(target region spans {np.ptp(target_V[:,2]):.3f})")
    for i, (a_, b_) in enumerate(STRIPS):
        print(f"  {STRIP_NAMES[i]:<14s} z {P3[a_:b_,2].min():+.3f} .. "
              f"{P3[a_:b_,2].max():+.3f}   misses {int((~hit[a_:b_]).sum())}"
              f"  repaired {int(repaired[a_:b_].sum())}")

    if cfg["overlay_png"]:
        write_overlay(crop, lm, cfg["overlay_png"])

    # --- 6. viewer -----------------------------------------------------------
    colours = []
    for k in range(len(P3)):
        if repaired[k]:
            colours.append((1.0, 1.0, 1.0))        # chain-repaired
        elif not hit[k]:
            colours.append((0.0, 0.0, 0.0))        # the ray missed entirely
        else:
            colours.append(STRIP_COLOURS[next(i for i, (a_, b_) in enumerate(STRIPS)
                                              if a_ <= k < b_)])
    geoms = [mesh, spheres(P3, cfg["sphere_frac"], colours), strip_lines(P3)]
    if cfg["show_patch"]:
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_V))
        pc.paint_uniform_color((0.2, 0.3, 0.8))
        geoms.append(pc)
    if cfg["show_rays"]:
        geoms.append(ray_lines(P3, cfg))

    if not cfg["interactive"]:
        return
    print("\norbit to about 60 deg to check depth. Q or Escape to close.")
    o3d.visualization.draw_geometries(
        geoms, window_name="back-projected ear landmarks",
        width=1100, height=900, mesh_show_back_face=True)


if __name__ == "__main__":
    main()
