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
the eval scripts. The only project imports are the two networks (EarDetector,
LandmarkPredictor) and the checkpoint picker, which are the model, not geometry.
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
    python scripts/view_backprojection.py --renderer open3d --rays
    python scripts/view_backprojection.py --no-window      # just stats + PNG
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
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

    # --- renderer -----------------------------------------------------------
    renderer="blender",     # "blender" = Cycles skin (slow, photo-like),
                            # "open3d" = clay shading (~100x faster)
    tone=None,              # index into SKIN_TONES (0 = lightest, 8 = darkest);
                            # None picks one at random per run
    freckles=0.0,           # they COST landmarker confidence (0.426 -> 0.419)
    # RESULTS.md swept these one factor at a time and found azimuth -40 deg with a
    # shallow energy optimum at 5 W. The ENERGY carries over; the AZIMUTH does
    # not, because that sweep ran before the camera-orientation bug below was
    # found, so its angles were measured about a different axis than the ear's.
    # Re-swept here in the corrected frame, on two heads: +20 is best at 0.407
    # mean confidence, falling monotonically to 0.387 at +80, and -40 gives only
    # 0.362-0.366. Azimuth remains the largest single appearance factor.
    key_azimuth=20.0,       # NEGATIVE swings the key toward the front of the
                            # face, positive behind the head.
    key_elevation=30.0,
    key_type="AREA",        # "SUN" uses a directional light with the Open3D
                            # viewer's own direction instead. See render_blender.
    key_size=0.5,           # key light width, as a fraction of camera distance.
                            # NOT part of the original sweep. The script's own
                            # default of 1.2 makes the source wider than the head,
                            # so shadows wash out and the ear renders FLAT.
                            # Shrinking it raises in-ear contrast (sd 9.7 -> 10.4)
                            # and confidence (+0.004); 0.2 is no better than 0.5.
                            # Much the smaller effect: azimuth is worth +0.030.
    key_energy=20.0,        # at light_ref_dist; scaled by the inverse square below.
                            # RESULTS.md's 5 W optimum was swept at ambient 0.35
                            # with the broken camera. Re-swept at ambient 0.06,
                            # relief peaks at 20 W (0.01727, 143% of the clay
                            # render) with mean brightness 99 against 5 W's 56 --
                            # brighter AND crisper. Past that the highlights blow
                            # out and relief falls back: 60 W 0.01411, 150 W
                            # 0.01079. Detection is 0.94 and 55/55 throughout.
    fill_energy=1.5,
    ambient=0.06,           # world light, and the ONE setting that governs whether
                            # the ear reads as a surface or a flat blob. It lights
                            # from every direction at once, so it fills exactly the
                            # shadows that make relief legible.
                            # CALIBRATED AGAINST REAL PHOTOGRAPHS, not against the
                            # clay render -- clay is itself only 0.47x the relief
                            # of 300 real crops from data/manual, so matching it
                            # was the wrong target. Relief as a fraction of the
                            # real median (0.02868): 0.20 -> 0.39x, 0.12 -> 0.49x,
                            # 0.06 -> 0.64x, 0.03 -> 0.78x. Renders stay BELOW
                            # real at every setting, so lower is better here --
                            # though real crops also carry hair, skin texture and
                            # compression noise, which inflate the measure, so
                            # 1.0x is not a target to chase. At the original 0.35
                            # the pipeline itself half failed: detector confidence
                            # 0.40 and 27.5 of 55 rays hitting, against 0.95 and
                            # 55/55 here.
    sss=0.35,               # subsurface weight. NOT the cause of flatness, though
                            # it looks like it: 0.35 -> 0.0 moves relief by 0.7%
                            # (0.00569 -> 0.00565), against ambient's 63% -> 105%.
    sss_scale=0.012,        # scatter radius in MESH UNITS, so it means different
                            # things at different mesh scales -- the ear spans
                            # ~0.2 units in the pass-1 frame and ~2.1 in pass 2.
    # Those energies were swept with the camera (and so the key light, which sits
    # at dist*1.1) at 2.2. This viewer must put the camera at eye_z=3.0 for the
    # back-projection to be correct, which is 1.86x further and delivers 54% of
    # the irradiance -- ambient then dominates and the ear renders FLAT. Scaling
    # by the inverse square keeps the swept lighting, at the correct distance.
    light_ref_dist=2.2,
    denoise=False,          # Cycles' denoiser is a softening filter: at low
                            # sample counts it cannot separate fine relief from
                            # noise and smooths the antihelix and concha edges
                            # away. Raise `samples` instead.
    samples=48,             # Cycles samples. 48 is enough to landmark; raise for a
                            # cleaner picture, lower to iterate faster.
    blender="blender",      # the executable
    blender_script=str(ROOT / "scripts" / "blender_skin_render.py"),

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
    # the render derives from third-party 3D data, so this defaults OUTSIDE the repo
    overlay_png=str(Path(tempfile.gettempdir()) / "backprojection_overlay.png"),
    interactive=True,
)
# ---------------------------------------------------------------------------

# Fitzpatrick I-VI, lightest to darkest. Tone is the second largest factor in
# how photo-like a render is (0.020 confidence spread), after lighting azimuth.
SKIN_TONES = np.array([
    [0.96, 0.84, 0.76], [0.93, 0.79, 0.69], [0.88, 0.72, 0.60],
    [0.80, 0.63, 0.50], [0.71, 0.54, 0.42], [0.60, 0.44, 0.34],
    [0.48, 0.34, 0.26], [0.36, 0.25, 0.19], [0.27, 0.18, 0.14],
])

# Change of basis from THIS script's camera frame into the one Blender renders.
# Blender's camera sits at +Y looking down -Y with +Z up, and bpy.ops.wm.ply_import
# applies no axis conversion, so an untransformed mesh is rendered from a
# different axis entirely -- measured: our +Z lands UP in the image and our +X
# lands LEFT, i.e. a 90 deg rotation plus a mirror. This maps our (x, y, z) to
# Blender's (-x, z, y), after which markers land within 0.5 px of where project()
# puts them. It is applied ONLY to the copy handed to Blender; the mesh that gets
# ray-cast is never touched.
TO_BLENDER = np.array([[-1.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0],
                       [0.0, 1.0, 0.0]])

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
    """Render face-on. Returns HxWx3 uint8. Dispatches on cfg["renderer"]."""
    if cfg.get("renderer") == "blender":
        img = render_blender(mesh, cfg, size=size)
        if img is not None:
            return img
        print("  blender render failed -- falling back to open3d")
    return render_open3d(mesh, cfg, size=size, background=background)


def render_blender(mesh, cfg, size=None):
    """Cycles skin render: Principled BSDF with subsurface scattering.

    Open3D fakes skin by baking ambient occlusion into vertex colours. It cannot
    do subsurface scattering or specular highlights, and the helix rim is almost
    always the brightest thing in a real ear photograph. Cycles does both, and
    reaches 96% of real-photo landmarker confidence against clay's ~84%.

    The mesh is rotated into Blender's camera frame by TO_BLENDER first, and the
    camera distance is cfg["eye_z"], so the result is pixel-aligned with what
    project() predicts. Both of those are easy to get wrong and silent when wrong.
    """
    size = size or cfg["size"]
    tone = SKIN_TONES[cfg["tone"] if cfg["tone"] is not None
                      else np.random.default_rng().integers(len(SKIN_TONES))]
    falloff = (float(cfg["eye_z"]) / float(cfg["light_ref_dist"])) ** 2
    g = copy.deepcopy(mesh)
    g.vertices = o3d.utility.Vector3dVector(np.asarray(g.vertices) @ TO_BLENDER.T)
    g.compute_vertex_normals()
    with tempfile.TemporaryDirectory() as td:
        mp = os.path.join(td, "m.ply")
        o3d.io.write_triangle_mesh(mp, g)
        args = dict(mesh=mp, ear=[0.0, 0.0, 0.0], dist=float(cfg["eye_z"]),
                    tone=[float(v) for v in tone], out=os.path.join(td, "r"),
                    size=int(size), samples=int(cfg["samples"]),
                    key_energy=float(cfg["key_energy"]) * falloff,
                    fill_energy=float(cfg["fill_energy"]) * falloff,
                    ambient=float(cfg["ambient"]), denoise=bool(cfg["denoise"]),
                    sss=float(cfg["sss"]), sss_scale=float(cfg["sss_scale"]),
                    freckles=float(cfg["freckles"]),
                    key_type=cfg["key_type"], key_size=float(cfg["key_size"]),
                    # Open3D's sun travels (-0.3, -0.4, -0.9) in OUR axes, so it
                    # has to go through the same basis change as the mesh.
                    key_direction=list(np.array([-0.3, -0.4, -0.9]) @ TO_BLENDER.T),
                    key_azimuth=float(cfg["key_azimuth"]),
                    key_elevation=float(cfg["key_elevation"]))
        ap = os.path.join(td, "a.json")
        with open(ap, "w") as f:
            json.dump(args, f)
        r = subprocess.run([cfg["blender"], "-b", "-P", cfg["blender_script"], "--", ap],
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                           timeout=900)
        out = os.path.join(td, "r.png")
        if not os.path.exists(out):
            print("  blender:", r.stderr.decode()[-300:].strip())
            return None
        print(f"  blender: tone {tone.round(2).tolist()}, key azimuth "
              f"{cfg['key_azimuth']:+.0f} deg, key {cfg['key_energy'] * falloff:.1f} W "
              f"({cfg['key_energy']:.1f} W swept x{falloff:.2f} for distance), "
              f"{cfg['samples']} samples")
        return np.asarray(Image.open(out).convert("RGB"))


def render_open3d(mesh, cfg, size=None, background=(0.5, 0.5, 0.5)):
    """Offscreen clay render (EGL -- works headless). Returns HxWx3 uint8."""
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
    from inference import BLAZEEAR_DIR, DETECTOR_WEIGHTS, EarDetector
    det = EarDetector(BLAZEEAR_DIR / DETECTOR_WEIGHTS, "cpu", 0.5)
    mesh = load_head(cfg["mesh"])
    if mesh is None:
        sys.exit(f"could not read {cfg['mesh']}")
    found = find_ears(mesh, det, cfg)
    if not found:
        sys.exit("the detector found no ear on this head")
    print("detector found " + ", ".join(f"conf={c:.2f}" for c, *_ in found))
    conf, front, up, _ = found[int(cfg["ear"]) % len(found)]
    print(f"using ear {cfg['ear']} (probe confidence {conf:.2f})")
    return mesh, front, up


def label_two_pass(mesh, front, up, cfg, pipe):
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
    g = in_frame(mesh, R, V.mean(0), 1.0)
    img, lm, conf = landmark_whole_frame(g, cfg, pipe)
    if lm is None:
        sys.exit("pass 1: the pipeline found no ear in the face-on render")
    P1, hit1 = raycast(g, lm, cfg)
    print(f"pass 1: detector conf {conf:.3f}, {int(hit1.sum())}/55 rays hit")
    if hit1.sum() < 10:
        sys.exit("pass 1: too few rays hit to fit a pinna plane")

    P1w = (P1[hit1] @ R) + V.mean(0)               # back to world coordinates
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
    p.add_argument("--run", default=None, help="checkpoint run name")
    p.add_argument("--device", default=None)
    p.add_argument("--rays", action="store_true", help="draw the camera rays")
    p.add_argument("--renderer", choices=["blender", "open3d"], default=None)
    p.add_argument("--tone", type=int, default=None, help="skin tone 0-8, light to dark")
    p.add_argument("--key-azimuth", type=float, default=None, help="key light, degrees")
    p.add_argument("--key-energy", type=float, default=None, help="key light W, pre-falloff")
    p.add_argument("--key-size", type=float, default=None, help="key light width / distance")
    p.add_argument("--key-type", choices=["AREA", "SUN"], default=None)
    p.add_argument("--samples", type=int, default=None, help="Cycles samples")
    p.add_argument("--no-2d", action="store_true", help="skip the 2D overlay window")
    p.add_argument("--no-window", action="store_true", help="stats and PNG only")
    a = p.parse_args()

    cfg = dict(CONFIG)
    if a.mesh:
        cfg["mesh"] = a.mesh
    for k, v in (("ear", a.ear), ("run", a.run), ("device", a.device),
                 ("renderer", a.renderer), ("tone", a.tone),
                 ("key_azimuth", a.key_azimuth), ("samples", a.samples),
                 ("key_energy", a.key_energy), ("key_size", a.key_size),
                 ("key_type", a.key_type)):
        if v is not None:
            cfg[k] = v
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

    mesh, front, up = load_subject(cfg)
    g, img, lm, P3, hit, conf = label_two_pass(mesh, front, up, cfg, pipe)
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
