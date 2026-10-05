"""Standalone, editable: 3D head -> face-on render -> landmarker -> back-projection.

    1. load a head mesh (HUTUBS .ply, or any .ply/.obj head)
    2. find the ear with the detector and render the head face-on
    3. hand the WHOLE render to EarLandmarkerPipeline -- the shipped
       detect -> ROI -> refine -> landmark path, which does its own cropping and
       returns 55 points in full-frame pixels
    4. back-project each point by ray-cast onto the whole mesh -> 55 points in 3D
       (then repeat 2-4 in the plane those landmarks define)
    5. open an interactive window -- mesh + 3D landmarks + the four linestrips --
       and write a 2D overlay PNG of the same prediction beside it

SELF-CONTAINED BY DESIGN. Every geometry stage is inlined below rather than
imported, so you can hack on any of it without perturbing ingest_render3d.py or
the eval scripts. The project imports are the networks (EarDetector,
LandmarkPredictor, via EarLandmarkerPipeline), the checkpoint picker, and
scripts/skin.py -- the model and the appearance shader, neither of which is
geometry.
The flip side: fixes made here do NOT propagate back to the pipeline, and fixes
made there do not arrive here. If a change proves out, port it deliberately.

KNOWN LIMITATION, and the reason this viewer exists. A camera ray that passes
just outside the pinna hits the SCALP behind it, so that landmark lands about one
ear-depth too deep. It looks correct face-on -- the error is purely in depth --
and only separates when you orbit to ~60 deg. Black spheres are rays that missed
the mesh entirely. The ray-cast is shown RAW: no snapping, no correction.

Controls: drag to orbit, scroll to zoom, R resets the view, Q or Escape closes.

Usage:
    python scripts/view_backprojection.py
    python scripts/view_backprojection.py --mesh /path/pp16_3DheadMesh.ply --ear 1
    python scripts/view_backprojection.py --rays
    python scripts/view_backprojection.py --no-window      # just stats + PNG
"""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
import tempfile
import time
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from skin import ALL_TONES, apply_skin       # noqa: E402  (appearance, not geometry)

# ---------------------------------------------------------------- CONFIG ----
# Third-party research data; not in any repo. Override with EAR3D_DIR.
EAR3D_DIR = Path(os.environ.get("EAR3D_DIR", ROOT.parent / "clean_3d_data"))

CONFIG = dict(
    # --- what to load -------------------------------------------------------
    mesh=str(EAR3D_DIR / "3D head meshes" / "pp12_3DheadMesh.ply"),
    ear=0,                  # which detected ear: 0 = highest confidence, 1 = other side

    # --- model --------------------------------------------------------------
    run="manual_occ_s42",   # checkpoint under runs/checkpoints/<run>/
    device="cuda",

    # --- head pose ----------------------------------------------------------
    use_mediapipe=True,     # derive the face-on direction from a FaceMesh head
                            # frame instead of the ear detector's 6-view sweep
    face_task=os.environ.get(
        "FACE_LANDMARKER_TASK",
        "/mnt/14BE47C2BE479ADE/Code/landmarking_stuff/landmarker/face_landmarker.task"),

    # --- appearance ---------------------------------------------------------
    skin=True,              # colour the mesh with scripts/skin.py before rendering
    tone=10,                # tone index; None draws a random Fitzpatrick I-VI.
                            # 10 is deeper than real skin and is there because the
                            # landmarker reads the ear most cleanly on it -- see
                            # EXTRA_TONES in scripts/skin.py. Fine here, where the
                            # point is to inspect landmarks; ingest renders its
                            # training images with a realistic tone instead.
    skin_seed=0,            # also drives the blotching and grain
    ao_strength=0.75,       # how hard ambient occlusion darkens cavities
    ao_rays=24,             # rays per vertex. The cost of apply_skin is all here.

    # --- camera / framing (these mirror the training pipeline) --------------
    size=600,               # full render resolution before cropping
    eye_z=3.0,              # camera at (0, 0, eye_z) looking at the origin
    vfov=50.0,              # vertical field of view, degrees

    # NOTE there is no crop setting here. The WHOLE render goes to
    # EarLandmarkerPipeline, which runs the shipped detect -> ROI -> refine ->
    # landmark path and hands back landmarks in full-frame pixels. That path
    # already drives the ROI toward TRAIN_OCCUPANCY (ROI_OCC_TOL/ROI_SATURATED),
    # so framing and occupancy are ITS business, not this script's -- which also
    # means this viewer now shows what inference actually does, rather than what
    # a hand-rolled crop here happened to do.

    # --- back-projection ----------------------------------------------------
    # No snapping. The ray-cast is shown RAW -- what the camera ray actually hit,
    # and nothing else. scripts/snap_landmarks.py still holds the in-plane and
    # chain-repair rules for when they go back in.

    # --- display ------------------------------------------------------------
    sphere_frac=0.022,      # landmark sphere radius, in mesh units (ear spans ~1)
    show_2d=True,           # pop the landmark overlay in its own window
    show_rays=False,        # draw the camera ray to each landmark
    # these derive from third-party 3D data, so they default OUTSIDE the repo
    snapshot_png=str(Path(tempfile.gettempdir()) / "backprojection_snapshot.png"),
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
    """Offscreen render (EGL -- works headless). Returns HxWx3 uint8.

    Open3D only. The Blender/Cycles path was removed: it is ~100x slower, and
    every part of the pipeline it touched -- camera basis, sensor fit, camera
    distance, light falloff -- was a separate silent way to render a different
    view than the labels described. scripts/blender_skin_render.py is gone from
    the tree; recover it from commit 056dd70 if it is ever wanted back.
    """
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







def frame_from(normal, up, front=None):
    """Rotation taking `normal` onto +Z (toward the camera), keeping `up` near +Y."""
    n = np.asarray(normal, float)
    if front is not None and n @ np.asarray(front, float) < 0:
        n = -n
    z = n / max(np.linalg.norm(n), 1e-9)
    y = np.asarray(up, float) - z * (np.asarray(up, float) @ z)
    ny = np.linalg.norm(y)
    if ny < 1e-6:
        return None
    y /= ny
    return np.stack([np.cross(y, z), y, z])            # world -> camera frame


def plane_normal(points):
    """Surface normal of the best-fit plane through the landmarks."""
    P = np.asarray(points, float)
    _, _, vt = np.linalg.svd(P - P.mean(0), full_matrices=False)
    return vt[2]


def in_frame(mesh, R, C, scale):
    """A copy of the mesh rotated into frame R about C and scaled by `scale`."""
    g = copy.deepcopy(mesh)
    g.vertices = o3d.utility.Vector3dVector(
        ((np.asarray(mesh.vertices) - C) @ R.T) / scale)
    g.compute_vertex_normals()
    return g


def landmark_whole_frame(mesh, cfg, pipe):
    """Render the whole head and landmark it with the SHIPPED pipeline.

    No crop is made here. EarLandmarkerPipeline detects the ear, builds and
    refines its own ROI, runs the landmarker and maps the 55 points back into
    full-frame pixels -- which is exactly the space project()/unproject_rays()
    work in, so the result feeds the ray-cast directly with no mapping of ours
    in between. Returns (image, landmarks_px, mean confidence) or (img, None, 0).
    """
    img = render(mesh, cfg)
    res = pipe(img, timestamp=0.0)
    if not res:
        return img, None, 0.0
    # Several ears can be detected on a head render; take the most confident.
    best = max(res, key=lambda d: float(d["confidence"]))
    # NOTE this is the DETECTOR's box confidence, which is what the pipeline
    # returns. It is not the landmarker's per-point confidence and must not be
    # compared against it -- the pipeline does not expose that, because it owns
    # the crop. Reach into pipe.landmarker.predict(..., with_confidence=True) if
    # you need the per-point values back.
    return img, np.asarray(best["landmarks"], float), float(best["confidence"])


# MediaPipe FaceMesh canonical indices.
TRAGION_R, TRAGION_L, FOREHEAD, CHIN = 234, 454, 10, 152


def head_pose(mesh, cfg):
    """Head frame from MediaPipe FaceMesh, in the mesh's own coordinates.

    Returns (lateral, vertical, tragion_R, tragion_L) or None.

    WHY THIS BEATS SWEEPING THE EAR DETECTOR. The sweep renders six axis-aligned
    views and keeps the two best ear boxes, which on a head gives two nearly equal
    confidences and no way to tell which side is which. MediaPipe finds a face on
    exactly ONE of those six views -- on every head tried -- so the frontal
    direction is unambiguous, and the face landmarks then give a real anatomical
    frame rather than whichever axis the dataset stored the head on. The ear-to-ear
    axis comes from the two tragion landmarks, which sit AT the ears.

    IT DOES NOT REMOVE PASS 2, AND IT IS NOT MEASURABLY MORE ACCURATE. Head to
    head over four heads, pass-2 confidence against the detector sweep: 0.931 vs
    0.955, 0.942 vs 0.927, fails vs 0.881, 0.980 vs 0.981. The remaining offset
    from the pinna plane is the same either way (8-35 deg here, 16-37 for the
    sweep). What it buys is that the SIDE is known -- the sweep returns two ear
    boxes at near-equal confidence with no way to tell left from right, so "ear 0"
    is arbitrary and changes between heads. The pinna's own tilt is per-subject
    anatomy that no face-derived frame can supply, which is why pass 2 stays.

    Measured per side (pass-2 confidence / degrees off the pinna plane), the RIGHT
    ear is consistently better: pp16 fails/0.839@12, pp12 0.931@33/0.990@28,
    pp11 0.942@35/0.986@13, pp10 0.980@19/0.982@8. Left and right ears are mirror
    images, so this is most likely an asymmetry in the landmarker rather than in
    this frame -- it is not explained, and worth a look before trusting left-ear
    labels as much as right.
    """
    task = cfg.get("face_task")
    if not task or not Path(task).exists():
        print(f"mediapipe: no face_landmarker.task at {task} -- using the ear sweep")
        return None
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    fl = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(task)), num_faces=1))
    C0 = np.asarray(mesh.vertices).mean(0)
    for axis in (0, 1, 2):
        for sgn in (1, -1):
            front = np.zeros(3); front[axis] = sgn
            up = np.array([0.0, 0.0, 1.0]) if axis != 2 else np.array([0.0, 1.0, 0.0])
            R = frame_from(front, up, front=front)
            if R is None:
                continue
            g = in_frame(mesh, R, C0, 1.0)
            res = fl.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                     data=np.ascontiguousarray(render(g, cfg))))
            if not res.face_landmarks:
                continue
            uv = np.array([[p.x * cfg["size"], p.y * cfg["size"]]
                           for p in res.face_landmarks[0]])
            P, hit = raycast(g, uv, cfg)
            if not hit[[TRAGION_R, TRAGION_L, FOREHEAD, CHIN]].all():
                continue
            # The landmarks come back in THIS frame's coordinates; everything
            # downstream works in the mesh's own, so convert before using them.
            # Skipping this put the ear view 62-70 deg off instead of 10-24.
            P = (P @ R) + C0
            lat = P[TRAGION_L] - P[TRAGION_R]
            lat /= np.linalg.norm(lat)
            vert = P[FOREHEAD] - P[CHIN]
            vert -= lat * (vert @ lat)
            vert /= np.linalg.norm(vert)
            print(f"mediapipe: face on axis {axis}{'+' if sgn > 0 else '-'}, "
                  f"tragion separation {np.linalg.norm(P[TRAGION_L] - P[TRAGION_R]):.3f}")
            return lat, vert, P[TRAGION_R], P[TRAGION_L]
    print("mediapipe: no face found on any of the six views -- using the ear sweep")
    return None


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


_VIEW_2D = r"""
import sys
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
img = plt.imread(sys.argv[1])
fig = plt.figure("landmark placement (2D)", figsize=(6.5, 6.5))
fig.canvas.manager.set_window_title("landmark placement (2D)")
ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(img); ax.axis("off")
plt.show()
"""


def show_image_window(path):
    """Pop the 2D overlay in its own window, beside the 3D one.

    A SEPARATE PROCESS, not a second window in this one. Open3D's
    draw_geometries runs its own blocking event loop, so a matplotlib figure
    opened here would simply freeze -- never redrawing, never responding -- for
    as long as the 3D window is up. A child has its own loop and stays live.
    It also outlives this script, so the overlay is still there to compare
    against after the 3D window is closed; close it yourself when done.
    """
    try:
        helper = Path(tempfile.gettempdir()) / "_view_backprojection_2d.py"
        helper.write_text(_VIEW_2D)
        log = Path(tempfile.gettempdir()) / "_view_backprojection_2d.log"
        # start_new_session puts the child in its own process group, so it is not
        # torn down with this one and the overlay stays up after the 3D window
        # closes. Errors go to a log rather than /dev/null: a window that fails
        # to appear is otherwise completely silent.
        proc = subprocess.Popen([sys.executable, str(helper), str(path)],
                                stdout=open(log, "w"), stderr=subprocess.STDOUT,
                                start_new_session=True)
        print(f"  2D overlay window: pid {proc.pid} (log: {log})")
        return proc
    except Exception as e:                      # a viewer is never worth crashing for
        print(f"  could not open the 2D window ({e}); the PNG is still written")
        return None


def write_overlay(crop, lm, path):
    """The 2D prediction on the render it came from, in the same strip colours."""
    img = crop.convert("RGB").resize((512, 512), Image.BILINEAR)
    s = 512 / crop.width            # the WHOLE frame now, not a crop
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
    """Returns (head mesh at unit radius, front, up) for the chosen ear.

    No ear REGION is computed. An earlier version selected "ear vertices" by
    projecting the mesh into the detector's 2D box and keeping a depth window,
    and used that set for the plane fit, the framing and the ray-cast target. It
    failed silently and often: a 2D box selects the whole column through the
    skull, so the window had to be tight, and on some subjects a wider box pulled
    in nearer geometry, raised the frontmost depth and excluded the ear -- on
    HUTUBS pp16's second ear the 2.5x region came out SMALLER than the 1.0x one
    (8139 vertices against 12226) and 4 of 55 rays hit. The 55 back-projected
    landmarks are a better definition of the ear: they are on it by construction.
    """
    mesh = load_head(cfg["mesh"])
    if mesh is None:
        sys.exit(f"could not read {cfg['mesh']}")
    if cfg["skin"]:
        # Applied ONCE, to the mesh in its original frame. Ambient occlusion is a
        # property of the geometry, not of the camera, so recomputing it per pass
        # would cost the same answer twice -- and the colours survive in_frame(),
        # which only replaces the vertex positions.
        t0 = time.perf_counter()
        tone = cfg["tone"]
        apply_skin(mesh, seed=cfg["skin_seed"], tone=tone,
                   ao_strength=cfg["ao_strength"], n_rays=cfg["ao_rays"])
        shown = ALL_TONES[tone] if tone is not None else np.asarray(
            mesh.vertex_colors)[::997].mean(0)
        print(f"skin: tone {np.round(shown, 2).tolist()}, "
              f"AO {cfg['ao_rays']} rays over {len(mesh.vertices)} vertices "
              f"({time.perf_counter() - t0:.1f}s)")
    # ALL DETECTION RUNS ON THE SKINNED MESH, above. Every detector here reads
    # clay far worse than skin -- measured, 0.835 against 0.94 on the same ear --
    # so colouring first is not cosmetic, it is what the models are good at.
    pose = head_pose(mesh, cfg) if cfg["use_mediapipe"] else None
    if pose is not None:
        lat, vert, tr_r, tr_l = pose
        # ear 0 = the subject's left (along +lateral), ear 1 = their right.
        left = int(cfg["ear"]) % 2 == 0
        front = lat if left else -lat
        print(f"using the {'left' if left else 'right'} ear, from the FaceMesh frame")
        return mesh, front, vert, (tr_l if left else tr_r)

    from inference import BLAZEEAR_DIR, DETECTOR_WEIGHTS, EarDetector
    det = EarDetector(BLAZEEAR_DIR / DETECTOR_WEIGHTS, "cpu", 0.5)
    found = find_ears(mesh, det, cfg)
    if not found:
        sys.exit("neither MediaPipe nor the ear detector found anything on this head")
    print("detector found " + ", ".join(f"conf={c:.2f}" for c, *_ in found))
    conf, front, up, _ = found[int(cfg["ear"]) % len(found)]
    print(f"using ear {cfg['ear']} (probe confidence {conf:.2f})")
    return mesh, front, up, np.asarray(mesh.vertices).mean(0)


def label_two_pass(mesh, front, up, cfg, pipe, centre=None):
    """Face-on landmarks in 3D. Returns (mesh in the pinna frame, P3, hit, conf).

    PASS 1 uses the detector's axis-aligned view direction purely to get a usable
    face-on render, and back-projects its landmarks.
    PASS 2 fits a plane to those 55 points and repeats in that frame. Both passes
    ray-cast against the WHOLE mesh.

    Why the second pass: the pinna's plane is not the head's lateral plane. Over
    14 HUTUBS subjects they differ by 8.1 deg on average (sd 3.9, range 1-13),
    varying per subject, so labelling along the detector's axis would bake a
    subject-dependent tilt into pose zero.
    """
    R = frame_from(front, up, front=front)
    if R is None:
        sys.exit("degenerate detector frame")
    V = np.asarray(mesh.vertices)
    # Centre on the tragion when MediaPipe gave one: the head centroid puts the
    # ear near the frame edge, where the ROI has least room to grow.
    C1 = V.mean(0) if centre is None else np.asarray(centre, float)
    g = in_frame(mesh, R, C1, 1.0)
    img, lm, conf = landmark_whole_frame(g, cfg, pipe)
    if lm is None:
        sys.exit("pass 1: the pipeline found no ear in the face-on render.\n"
                 "  With the MediaPipe frame the SIDE is chosen deterministically, so a\n"
                 "  head whose other ear is easier will still fail here. Try --ear 1,\n"
                 "  or --no-mediapipe to let the detector sweep pick whichever it likes.")
    P1, hit1 = raycast(g, lm, cfg)
    print(f"pass 1: detector conf {conf:.3f}, {int(hit1.sum())}/55 rays hit")
    if hit1.sum() < 10:
        sys.exit("pass 1: too few rays hit to fit a pinna plane")

    P1w = (P1[hit1] @ R) + C1                      # back to world coordinates
    R2 = frame_from(plane_normal(P1w), up, front=front)
    if R2 is None:
        sys.exit("degenerate pinna frame")
    C2 = P1w.mean(0)
    scale = float(np.abs((P1w - C2) @ R2.T).max())
    if not np.isfinite(scale) or scale <= 0:
        sys.exit("degenerate ear scale")
    tilt = np.degrees(np.arccos(np.clip(abs(R[2] @ R2[2]), -1, 1)))
    print(f"pinna plane is {tilt:.1f} deg off the detector's view direction")

    g2 = in_frame(mesh, R2, C2, scale)
    img2, lm2, conf2 = landmark_whole_frame(g2, cfg, pipe)
    if lm2 is None:
        sys.exit("pass 2: the pipeline found no ear in the pinna-frame render")
    P3, hit = raycast(g2, lm2, cfg)
    print(f"pass 2: detector conf {conf2:.3f}, {int(hit.sum())}/55 rays hit")
    return g2, img2, lm2, P3, hit, conf2


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mesh", default=None, help="head .ply/.obj")
    p.add_argument("--ear", type=int, default=None, help="which detected ear, 0 or 1")
    p.add_argument("--tone", type=int, default=None, help="skin tone 0-8, light to deep")
    p.add_argument("--no-skin", action="store_true", help="render bare clay instead")
    p.add_argument("--no-mediapipe", action="store_true", help="use the ear-detector sweep")
    p.add_argument("--run", default=None, help="checkpoint run name")
    p.add_argument("--device", default=None)
    p.add_argument("--rays", action="store_true", help="draw the camera rays")
    p.add_argument("--no-2d", action="store_true", help="skip the 2D overlay window")
    p.add_argument("--no-window", action="store_true", help="stats and PNG only")
    a = p.parse_args()

    cfg = dict(CONFIG)
    if a.mesh:
        cfg["mesh"] = a.mesh
    for k, v in (("ear", a.ear), ("run", a.run), ("device", a.device),
                 ("tone", a.tone)):
        if v is not None:
            cfg[k] = v
    cfg["skin"] = cfg["skin"] and not a.no_skin
    cfg["use_mediapipe"] = cfg["use_mediapipe"] and not a.no_mediapipe
    cfg["show_rays"] |= a.rays
    cfg["show_2d"] = cfg["show_2d"] and not a.no_2d
    cfg["interactive"] = cfg["interactive"] and not a.no_window

    # --- the shipped pipeline, which owns detection, ROI and landmarking -----
    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_test import best_ckpt
    from inference import BLAZEEAR_DIR, DETECTOR_WEIGHTS, EarLandmarkerPipeline
    ck = best_ckpt(cfg["run"])
    if ck is None:
        sys.exit(f"no checkpoint for run {cfg['run']!r} under runs/checkpoints/")
    # smooth=False: the tracker is for video. On a single still it would smooth a
    # one-frame track against wall-clock time and shift the landmarks.
    pipe = EarLandmarkerPipeline(BLAZEEAR_DIR / DETECTOR_WEIGHTS, ck,
                                 device=cfg["device"], smooth=False)
    print(f"landmarker {ck.name}")

    mesh, front, up, centre = load_subject(cfg)
    g, img, lm, P3, hit, conf = label_two_pass(mesh, front, up, cfg, pipe, centre)
    V = np.asarray(g.vertices)

    # A missed ray has no 3D position at all. Park it on the mesh centroid so the
    # arrays stay finite; its sphere is drawn black so it reads as "no result".
    P3 = np.where(np.isfinite(P3), P3, V.mean(0))

    # Depth spread is the tell for a scalp-pinned point: it sits behind the rest.
    print(f"landmark depth z: {P3[:,2].min():+.3f} to {P3[:,2].max():+.3f} "
          f"(ear spans about 1.0 by construction)")
    for i, (a_, b_) in enumerate(STRIPS):
        print(f"  {STRIP_NAMES[i]:<14s} z {P3[a_:b_,2].min():+.3f} .. "
              f"{P3[a_:b_,2].max():+.3f}   misses {int((~hit[a_:b_]).sum())}")

    if cfg["snapshot_png"]:
        # The snapshot the landmarks were read from, saved UNMARKED so it can be
        # re-landmarked or diffed without the overlay in the way.
        Image.fromarray(img).save(cfg["snapshot_png"])
        print(f"wrote {cfg['snapshot_png']}")
    if cfg["overlay_png"]:
        write_overlay(Image.fromarray(img), lm, cfg["overlay_png"])
        if cfg["show_2d"] and cfg["interactive"]:
            show_image_window(cfg["overlay_png"])

    colours = [(0.0, 0.0, 0.0) if not hit[k]          # the ray missed entirely
               else STRIP_COLOURS[next(i for i, (a_, b_) in enumerate(STRIPS)
                                       if a_ <= k < b_)]
               for k in range(len(P3))]
    geoms = [g, spheres(P3, cfg["sphere_frac"], colours), strip_lines(P3)]
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
