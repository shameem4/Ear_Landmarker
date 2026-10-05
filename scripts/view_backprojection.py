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
the mesh entirely; WHITE ones were snapped onto the near lip of a depth cliff
(--no-snap shows the raw back-projection instead).

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
# This now gates EVERY render, not just the interactive window: the one renderer
# is the GLFW visualizer, and a screen capture needs a real GL context.
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
from scipy import ndimage
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
    snap=True,              # place landmarks that sit behind a depth cliff onto
                            # the near lip of that cliff. See snap_to_cliff().
    snap_radius=0.10,       # JITTER RANGE: the search window, as a fraction of
                            # ear extent. At the usual framing the ear spans
                            # ~435 px, so this is a radius of ~43 px (an 87x87
                            # window). This is the whole safety bound -- the
                            # further it reaches, the more a "snap" becomes a
                            # relocation of a landmark the model simply put in
                            # the wrong place. See the measurements by radius in
                            # the commit that set this.
    snap_step=0.30,         # how big a depth step counts as a cliff, in mesh
                            # units with the ear spanning ~1. Scalp-behind-pinna
                            # measures 0.85-1.08; concha bowl structure varies by
                            # ~0.3, so this sits between them. Lower (0.15) drags
                            # concha points forward; higher (0.50) fixes nothing.

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
# ONE RENDERER AND ONE CAMERA, and they are the same object.
#
# camera_ke() produces the intrinsic matrix K and extrinsic E. Those exact arrays
# are handed to Filament via setup_camera(K, E, w, h) AND used to build the
# Camera that projects and back-projects. Nothing re-derives a camera from
# constants, so the forward transform, the inverse transform and the picture
# cannot drift apart -- there is no second definition to drift from.
#
# Two earlier arrangements are what this replaces. Snapshots once went through
# Filament while the 3D window used the legacy GLFW visualizer: same mesh, same
# vertex colours, mean brightness 119.9 against 27.8 and a specular highlight in
# one but not the other, so the landmarker read an image nobody could see.
# Unifying on the LEGACY renderer instead was measured and is much worse -- its
# default lighting costs the pipeline badly (pass 2 failed on 3 of 6 skin tones,
# and the pinna-plane estimate ranged over 16-76 deg against 8-35 here).

MATERIAL = o3d.visualization.rendering.MaterialRecord()
MATERIAL.shader = "defaultLit"


def camera_ke(size, cfg, E=None):
    """(K, E) for this project's view. The single camera definition.

    `E` overrides the face-on extrinsic -- used to re-render from wherever the
    interactive window's camera has been orbited to. K is unchanged, so the
    intrinsics the landmarker and the back-projection see never vary with pose.

    Open3D is OpenCV-style -- x right, y DOWN, z INTO the scene -- while this
    project has y up and the camera at +Z looking back along -Z, so the extrinsic
    flips both y and z. cx/cy are size/2 - 0.5, the pixel-centre convention
    Open3D's own intrinsic requires.
    """
    f = (size / 2) / np.tan(np.radians(cfg["vfov"]) / 2)
    K = np.array([[f, 0.0, size / 2 - 0.5],
                  [0.0, f, size / 2 - 0.5],
                  [0.0, 0.0, 1.0]])
    if E is None:
        E = np.array([[1.0, 0, 0, 0], [0, -1.0, 0, 0],
                      [0, 0, -1.0, cfg["eye_z"]], [0, 0, 0, 1.0]])
    return K, np.asarray(E, float)


# Filament's view matrix is OpenGL-style (y up, -z forward); our extrinsic is
# OpenCV-style (y down, +z forward). Verified exact: FLIP @ get_view_matrix()
# returns the very matrix that was handed to setup_camera.
VIEW_TO_EXTRINSIC = np.diag([1.0, -1.0, -1.0, 1.0])


def extrinsic_of(scene_camera):
    """The extrinsic for whatever the window's camera is looking at right now."""
    return VIEW_TO_EXTRINSIC @ np.asarray(scene_camera.get_view_matrix(), float)


def pose_angles(E):
    """Yaw and pitch of this camera relative to the face-on view, in degrees.

    These are the angles the back-projection is working at: the mesh is fixed in
    its pinna frame, so orbiting the camera IS the pose. Face-on reads 0/0 by
    construction, yaw grows as the camera swings toward +X, pitch as it rises.
    """
    d = E[:3, :3].T @ np.array([0.0, 0.0, 1.0])      # view direction, world frame
    yaw = np.degrees(np.arctan2(-d[0], -d[2]))
    pitch = np.degrees(np.arcsin(np.clip(-d[1], -1.0, 1.0)))
    off = np.degrees(np.arccos(np.clip(-d[2], -1.0, 1.0)))
    return yaw, pitch, off


class Camera:
    """Projection and back-projection for a render, from that render's own K/E."""

    def __init__(self, K, E, img_hw, requested):
        h, w = img_hw
        # A render can come back at a size we did not ask for. K describes the
        # requested size, so rescale it to the pixels actually produced --
        # otherwise every projection is off by that ratio, silently.
        sx, sy = w / requested, h / requested
        self.fx, self.fy = K[0, 0] * sx, K[1, 1] * sy
        self.cx, self.cy = K[0, 2] * sx, K[1, 2] * sy
        self.scaled = (sx != 1.0 or sy != 1.0)
        self.w, self.h = w, h
        self.R, self.t = E[:3, :3], E[:3, 3]
        self.centre = -self.R.T @ self.t

    def project(self, P):
        P = np.atleast_2d(np.asarray(P, float))
        Pc = P @ self.R.T + self.t
        z = np.maximum(Pc[:, 2], 1e-9)
        return np.stack([self.fx * Pc[:, 0] / z + self.cx,
                         self.fy * Pc[:, 1] / z + self.cy], axis=1)

    @classmethod
    def from_gui(cls, scene_camera, img_hw):
        """Camera for a capture taken from inside the GUI window.

        Intrinsics come from the live projection matrix rather than from
        camera_ke, because the widget can be resized and its aspect is whatever
        the window is. Verified against camera_ke on a square window: fx 257.34
        against 257.3, cx/cy 119.5 against size/2 - 0.5.
        """
        P = np.asarray(scene_camera.get_projection_matrix(), float)
        h, w = img_hw
        obj = cls.__new__(cls)
        obj.fx, obj.fy = P[0, 0] * w / 2.0, P[1, 1] * h / 2.0
        obj.cx, obj.cy = w * (1 - P[0, 2]) / 2.0, h * (1 + P[1, 2]) / 2.0
        obj.scaled, obj.w, obj.h = False, w, h
        E = extrinsic_of(scene_camera)
        obj.R, obj.t = E[:3, :3], E[:3, 3]
        obj.centre = -obj.R.T @ obj.t
        return obj

    def rays(self, uv):
        """Pixel -> (origin, unit direction). Exact inverse of project()."""
        uv = np.atleast_2d(np.asarray(uv, float))
        d = np.stack([(uv[:, 0] - self.cx) / self.fx,
                      (uv[:, 1] - self.cy) / self.fy,
                      np.ones(len(uv))], axis=1) @ self.R
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        return np.tile(self.centre, (len(uv), 1)), d


def light_scene(scene, background):
    """The one lighting setup, shared by the snapshot and the 3D window."""
    scene.set_background([*background, 1.0])
    scene.scene.set_sun_light([-0.3, -0.4, -0.9], [1.0, 1.0, 1.0], 95000)
    scene.scene.enable_sun_light(True)


def render(mesh, cfg, size=None, background=(0.5, 0.5, 0.5), E=None):
    """Render, and return the depth buffer and camera that produced it.

    Returns (image HxWx3 uint8, depth float32 HxW, Camera). The depth is
    view-space z, with inf where nothing was drawn. Callers back-project through
    THIS depth and THIS camera; see backproject().
    """
    size = size or cfg["size"]
    r = o3d.visualization.rendering.OffscreenRenderer(size, size)
    r.scene.add_geometry("m", mesh, MATERIAL)
    light_scene(r.scene, background)
    K, E = camera_ke(size, cfg, E)
    r.setup_camera(K, E, size, size)
    img = np.asarray(r.render_to_image())[:, :, :3]
    depth = np.asarray(r.render_to_depth_image(z_in_view_space=True))
    del r
    return img, depth, Camera(K, E, img.shape[:2], size)


def gui_depth_to_view(depth, scene_camera):
    """The GUI's normalised depth buffer -> view-space distance, background inf.

    The window's own capture has no z_in_view_space flag, and its projection has
    an INFINITE far plane: P[2,2] is -1 and P[2,3] is -2*near, so the buffer
    holds d = 1 + near/z_e with z_e negative in front of the camera, and the
    distance is near / (1 - d). Measured against a sphere whose apex is exactly
    2.5 away: 2.5011, matching what the offscreen renderer reports in view space.

    The projection form is asserted rather than assumed -- if Open3D ever ships a
    finite far plane or reverse-Z, this must fail loudly instead of returning
    quietly wrong depths.
    """
    P = np.asarray(scene_camera.get_projection_matrix(), float)
    if not (abs(P[2, 2] + 1.0) < 1e-3 and P[2, 3] < 0):
        raise RuntimeError(f"unexpected projection; cannot linearise depth:\n{P}")
    near = -P[2, 3] / 2.0
    d = np.asarray(depth, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = near / (1.0 - d)
    return np.where(d >= 1.0, np.inf, z)


def cliff_map(depth, step):
    """Pixels where the surface STEPS by more than `step` within a 3x3 window.

    Background (inf) is pushed to a finite far value first, so the pinna's
    silhouette against the head behind it counts as an edge like any other
    occlusion. A RIDGE is not a cliff: the antihelix curves, it does not break,
    so its local max-min stays small and it never qualifies. That distinction is
    what makes this safe where two earlier snaps were not -- both of those used
    "nearest surface in the window", which drags concha-floor points forward onto
    the rim, because a bowl legitimately has nearer surface beside it.

    Returns (is_cliff, near_depth) where near_depth is the 3x3 minimum: the
    foreground lip of whatever edge runs through that pixel.
    """
    d = np.where(np.isfinite(depth), depth, np.nan)
    far = np.nanmax(d) + 1.0 if np.isfinite(np.nanmax(d)) else 1.0
    d = np.where(np.isnan(d), far, d)
    hi = ndimage.maximum_filter(d, size=3)
    lo = ndimage.minimum_filter(d, size=3)
    return (hi - lo) > step, lo


def snap_to_cliff(depth, uv, radius, step):
    """Place landmarks that sit BEHIND a depth cliff onto its near lip.

    For each landmark, look within `radius` px for a depth cliff. If one is
    there AND the landmark is currently on the far side of it, move to the
    nearest cliff pixel and take the near-side depth. A landmark with no cliff
    nearby, or already on the near lip, is left exactly where it is.

    Returns (uv, moved_mask, shift_px, near_depth). The near depth is returned
    rather than applied, because re-sampling the depth buffer AT a cliff pixel is
    a coin flip between the two surfaces it separates -- which is the entire
    failure being corrected. Measured, taking it explicitly moved this from
    fixing 0 of 13 bad landmarks to fixing 11.
    """
    cliff, lo = cliff_map(depth, step)
    H, W = depth.shape
    out = np.asarray(uv, float).copy()
    moved = np.zeros(len(out), bool)
    shift = np.zeros(len(out))
    znear = np.full(len(out), np.nan)
    r = int(np.ceil(radius))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    rad = np.hypot(xx, yy)
    for k, (u, v) in enumerate(out):
        x0, y0 = int(round(u)), int(round(v))
        xs, xe = max(0, x0 - r), min(W, x0 + r + 1)
        ys, ye = max(0, y0 - r), min(H, y0 + r + 1)
        if xe <= xs or ye <= ys:
            continue
        sub = cliff[ys:ye, xs:xe]
        rr = rad[ys - (y0 - r):ye - (y0 - r), xs - (x0 - r):xe - (x0 - r)]
        cand = sub & (rr <= radius)
        if not cand.any():
            continue
        dd = np.where(cand, rr, np.inf)
        iy, ix = np.unravel_index(np.argmin(dd), dd.shape)
        gy, gx = ys + iy, xs + ix
        d_here = depth[min(max(y0, 0), H - 1), min(max(x0, 0), W - 1)]
        # Only move a point that is actually BEHIND the cliff. One already on the
        # near lip belongs there; moving it anyway was the whole source of
        # collateral damage in the previous version (15-17 good points per head
        # displaced, against 3 with this gate).
        if not np.isfinite(d_here) or (d_here - lo[gy, gx]) < 0.5 * step:
            continue
        out[k] = [gx, gy]
        moved[k] = True
        shift[k] = dd[iy, ix]
        znear[k] = lo[gy, gx]
    return out, moved, shift, znear


def backproject_snapped(depth, uv, cam, cfg):
    """backproject(), then the cliff snap if it is enabled. Returns (P, hit, moved)."""
    P, hit = backproject(depth, uv, cam)
    if not cfg.get("snap"):
        return P, hit, np.zeros(len(P), bool)
    uv = np.asarray(uv, float)
    ext = max(np.ptp(uv[:, 0]), np.ptp(uv[:, 1]))
    uv2, moved, _, znear = snap_to_cliff(depth, uv, cfg["snap_radius"] * ext,
                                         cfg["snap_step"])
    use = moved & np.isfinite(znear)
    if use.any():
        u, v, z = uv2[use, 0], uv2[use, 1], znear[use]
        Pc = np.stack([(u - cam.cx) / cam.fx * z, (v - cam.cy) / cam.fy * z, z], axis=1)
        P[use] = (Pc - cam.t) @ cam.R
        hit = hit | use
    return P, hit, moved


def backproject(depth, uv, cam, max_jump=0.02):
    """2D -> 3D from the DEPTH BUFFER of the same render. Returns (P, hit).

    WHY NOT RAY-CAST. Ray-casting means building a second acceleration structure
    over the mesh and intersecting it, which is a second geometry path: it agrees
    with the picture because it is the same mesh seen through the same camera, but
    nothing makes it agree. The depth buffer IS the rasterisation that produced
    the pixels, so the surface found here is by construction the surface the
    landmarker was looking at. It is also free, where the ray-cast was not.

    Precision: the buffer is float32 view-space z, so unlike the 8-bit depth
    round-trip the reference implementation uses to enable inpainting, nothing is
    quantised. Background reads as inf and is reported as a miss rather than
    filled in -- an inpainted depth would invent geometry and hand back a
    plausible, wrong 3D point with nothing marking it.

    `max_jump` guards the one real hazard of sampling a depth buffer at
    sub-pixel positions: bilinear interpolation ACROSS A SILHOUETTE blends a near
    surface with a far one and returns a depth that lies in empty space between
    them. Where the four neighbours disagree by more than this fraction, the
    nearest sample is taken instead of a blend.
    """
    depth = np.asarray(depth, float)
    h, w = depth.shape
    uv = np.atleast_2d(np.asarray(uv, float))
    x0 = np.clip(np.floor(uv[:, 0]).astype(int), 0, w - 2)
    y0 = np.clip(np.floor(uv[:, 1]).astype(int), 0, h - 2)
    fx, fy = uv[:, 0] - x0, uv[:, 1] - y0
    q = np.stack([depth[y0, x0], depth[y0, x0 + 1],
                  depth[y0 + 1, x0], depth[y0 + 1, x0 + 1]], axis=1)
    wts = np.stack([(1 - fx) * (1 - fy), fx * (1 - fy),
                    (1 - fx) * fy, fx * fy], axis=1)
    good = np.isfinite(q)
    hit = good.any(axis=1)
    z = np.full(len(uv), np.nan)
    qf = np.where(good, q, np.nan)
    near = np.nanmin(np.where(good, q, np.inf), axis=1, initial=np.inf)
    far = np.nanmax(np.where(good, q, -np.inf), axis=1, initial=-np.inf)
    blend = hit & good.all(axis=1) & ((far - near) <= max_jump * np.maximum(near, 1e-9))
    with np.errstate(invalid="ignore"):
        z[blend] = (q[blend] * wts[blend]).sum(axis=1)
    # Anything not safely blendable takes its nearest-neighbour sample: the
    # closest of the four that actually has geometry.
    rest = hit & ~blend
    if rest.any():
        pick = np.nanargmin(np.where(good[rest], np.abs(qf[rest] - near[rest, None]),
                                     np.nan), axis=1)
        z[rest] = q[rest, pick]
    P = np.full((len(uv), 3), np.nan)
    Pc = np.stack([(uv[hit, 0] - cam.cx) / cam.fx * z[hit],
                   (uv[hit, 1] - cam.cy) / cam.fy * z[hit],
                   z[hit]], axis=1)
    P[hit] = (Pc - cam.t) @ cam.R        # camera -> world; R is orthonormal
    return P, hit


LINE_MATERIAL = o3d.visualization.rendering.MaterialRecord()
LINE_MATERIAL.shader = "unlitLine"
LINE_MATERIAL.line_width = 2.0


def show(mesh, P3, hit, cfg, pipe, size=900, background=(0.5, 0.5, 0.5),
         title="back-projected ear landmarks", on_tick=None):
    """Interactive window -- SAME renderer and SAME camera as the snapshot.

    Built from gui.Window rather than O3DVisualizer because that class exposes
    neither key events nor a tick callback in the Python bindings, and both are
    needed here.

    Two things it does beyond showing the scene:

      the readout tracks the camera as you orbit, in the angles the
      back-projection actually works at. The mesh is fixed in its pinna frame, so
      camera yaw/pitch IS the pose; face-on reads 0/0 by construction.

      L re-landmarks from where you are standing. The 3D landmarks are removed,
      the mesh alone is re-rendered through the window's CURRENT camera, the
      pipeline runs on that image, and the result is back-projected through the
      depth buffer of that same render. So it answers "what would the model make
      of the ear from here?" rather than re-showing the face-on answer. The
      landmarks it draws are a fresh measurement at this pose, not the pose-zero
      ones rotated.
    """
    gui = o3d.visualization.gui
    rendering = o3d.visualization.rendering
    app = gui.Application.instance
    app.initialize()
    win = app.create_window(title, size, size)

    widget = gui.SceneWidget()
    widget.scene = rendering.Open3DScene(win.renderer)
    widget.scene.add_geometry("mesh", mesh, MATERIAL)
    light_scene(widget.scene, background)
    K, E0 = camera_ke(size, cfg)
    widget.setup_camera(K, E0, size, size, mesh.get_axis_aligned_bounding_box())

    info = gui.Label("")
    panel = gui.Vert(0, gui.Margins(10, 10, 10, 10))
    panel.background_color = gui.Color(0, 0, 0, 0.6)
    panel.add_child(info)
    win.add_child(widget)
    win.add_child(panel)

    def on_layout(ctx):
        rect = win.content_rect
        widget.frame = rect
        pref = panel.calc_preferred_size(ctx, gui.Widget.Constraints())
        panel.frame = gui.Rect(rect.x, rect.y, min(rect.width, 420), pref.height)
    win.set_on_layout(on_layout)

    state = {"drawn": [], "text": None, "busy": False, "snapped": None,
             "note": "face-on, as labelled"}

    def draw_landmarks(P3, hit):
        for n in state["drawn"]:
            widget.scene.remove_geometry(n)
        state["drawn"] = []
        snapped = state.get("snapped")
        colours = [(0.0, 0.0, 0.0) if not hit[k]              # ray missed
                   else (1.0, 1.0, 1.0) if (snapped is not None and len(snapped) == len(P3)
                                            and snapped[k])   # moved to a cliff
                   else STRIP_COLOURS[next(i for i, (a, b) in enumerate(STRIPS)
                                           if a <= k < b)]
                   for k in range(len(P3))]
        widget.scene.add_geometry("lm", spheres(P3, cfg["sphere_frac"], colours),
                                  MATERIAL)
        widget.scene.add_geometry("lmlines", strip_lines(P3), LINE_MATERIAL)
        state["drawn"] = ["lm", "lmlines"]

    draw_landmarks(P3, hit)

    def finish(colour, raw_depth):
        """Landmark the captured frame and redraw. Runs on the main thread."""
        img = np.asarray(colour)[:, :, :3].copy()
        cam = Camera.from_gui(widget.scene.camera, img.shape[:2])
        depth = gui_depth_to_view(np.asarray(raw_depth), widget.scene.camera)
        yaw, pitch, off = pose_angles(extrinsic_of(widget.scene.camera))
        res = pipe(img, timestamp=0.0)
        if not res:
            state["note"] = f"no ear found at yaw {yaw:+.0f} pitch {pitch:+.0f}"
            draw_landmarks(P3, hit)                 # put the previous ones back
            state["busy"] = False
            print(f"  [L] {state['note']}", flush=True)
            return
        best = max(res, key=lambda d: float(d["confidence"]))
        lm = np.asarray(best["landmarks"], float)
        Q, qhit, qsnap = backproject_snapped(depth, lm, cam, cfg)
        state["snapped"] = qsnap
        draw_landmarks(Q, qhit)
        state["note"] = (f"re-landmarked at yaw {yaw:+.0f} pitch {pitch:+.0f}: "
                         f"det {float(best['confidence']):.2f}, {int(qhit.sum())}/55 hit"
                         + (f", {int(qsnap.sum())} snapped" if qsnap.any() else ""))
        state["busy"] = False
        print(f"  [L] {state['note']}", flush=True)
        # Rewriting the overlay is what refreshes the 2D window: it polls this
        # file, so the picture there always matches the pose in the 3D window.
        if cfg["snapshot_png"]:
            tmp = f"{cfg['snapshot_png']}.{os.getpid()}.tmp.png"
            Image.fromarray(img).save(tmp, format="PNG")
            os.replace(tmp, cfg["snapshot_png"])
        if cfg["overlay_png"]:
            write_overlay(Image.fromarray(img), lm, cfg["overlay_png"],
                          note=f"yaw {yaw:+.0f}  pitch {pitch:+.0f}  "
                               f"det {float(best['confidence']):.2f}",
                          announce=False)

    def relandmark():
        """Capture THIS window, through the window's own renderer.

        It must not open an OffscreenRenderer: that is a second Filament context,
        and creating one while the GUI holds the first fails at the driver
        ("failed to create dri2 screen") and takes the process down. The window's
        own scene.render_to_image / render_to_depth_image reuse the live context.

        Both are asynchronous, so the colour capture chains into the depth
        capture, and the work happens once both have arrived. The landmarks are
        removed BEFORE capturing -- they are scene geometry, and a frame with
        them in it would show the landmarker an ear with 55 spheres stuck to it.
        """
        if state["busy"]:
            return
        state["busy"] = True
        for n in state["drawn"]:
            widget.scene.remove_geometry(n)
        state["drawn"] = []

        # Both captures call back from the RENDER thread. Touching the scene
        # from there corrupts the interpreter state outright ("gilstate_tss_set:
        # failed to set current tstate"), so each step hops back to the main
        # thread before doing anything.
        app = gui.Application.instance

        def got_depth(colour, d):
            app.post_to_main_thread(win, lambda: finish(colour, d))

        def ask_depth(c):
            widget.scene.scene.render_to_depth_image(lambda d: got_depth(c, d))

        def got_colour(c):
            app.post_to_main_thread(win, lambda: ask_depth(c))

        widget.scene.scene.render_to_image(got_colour)

    def on_key(e):
        # gui.Window.set_on_key wants a BOOL -- True to stop dispatching. It is
        # gui.Widget.set_on_key_event that takes an EventCallbackResult, and
        # returning one here is not a type error at registration: it crashes
        # inside app.run() the first time any key is pressed.
        if e.type == gui.KeyEvent.Type.DOWN and e.key == gui.KeyName.L:
            relandmark()
            return True
        return False
    win.set_on_key(on_key)

    # `on_tick` exists so the L path can be driven without a human at the
    # keyboard: it is handed the same handles the key binding uses. GUI code that
    # cannot be exercised automatically is how the first version of this shipped
    # broken. It is None in normal use.
    if on_tick is not None:
        state["hook"] = on_tick

    def tick():
        if "hook" in state:
            state["hook"](dict(relandmark=relandmark, state=state, win=win,
                               widget=widget, app=gui.Application.instance))
        yaw, pitch, off = pose_angles(extrinsic_of(widget.scene.camera))
        text = (f"yaw {yaw:+6.1f}\u00b0   pitch {pitch:+6.1f}\u00b0   "
                f"off-axis {off:5.1f}\u00b0\n{state['note']}\n"
                f"[L] re-landmark from this view")
        if text == state["text"]:
            return False
        state["text"] = text
        info.text = text
        return True
    win.set_on_tick_event(tick)

    app.run()


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
            R = frame_from(front, up, front=front)
            if R is None:
                continue
            g = in_frame(mesh, R, np.asarray(mesh.vertices).mean(0), 1.0)
            b = det.detect(render(g, cfg, size=probe_size)[0])
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
    full-frame pixels -- exactly the space the returned Camera works in, so the
    result feeds the ray-cast directly with no mapping of ours in between.

    Returns (image, depth, Camera, landmarks_px, confidence); landmarks None if no ear.
    """
    img, depth, cam = render(mesh, cfg)
    res = pipe(img, timestamp=0.0)
    if not res:
        return img, depth, cam, None, 0.0
    # Several ears can be detected on a head render; take the most confident.
    best = max(res, key=lambda d: float(d["confidence"]))
    # NOTE this is the DETECTOR's box confidence, which is what the pipeline
    # returns. It is not the landmarker's per-point confidence and must not be
    # compared against it -- the pipeline does not expose that, because it owns
    # the crop. Reach into pipe.landmarker.predict(..., with_confidence=True) if
    # you need the per-point values back.
    return img, depth, cam, np.asarray(best["landmarks"], float), float(best["confidence"])


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
            img, depth, cam = render(g, cfg)
            res = fl.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                     data=np.ascontiguousarray(img)))
            if not res.face_landmarks:
                continue
            # MediaPipe normalises to the IMAGE it was handed, so scale by that
            # image's own shape, never by the size we requested.
            h, w = img.shape[:2]
            uv = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]])
            P, hit = backproject(depth, uv, cam)
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


def ray_lines(points, cam):
    """Camera ray to each landmark: shows where the back-projection came from."""
    o, d = cam.rays(cam.project(points))
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
import os, sys
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

path = sys.argv[1]
fig = plt.figure("landmark placement (2D)", figsize=(6.5, 6.5))
fig.canvas.manager.set_window_title("landmark placement (2D)")
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
art = ax.imshow(plt.imread(path))
state = {"m": os.path.getmtime(path)}

# Poll the file rather than being told. The window is a separate process -- it
# has to be, because Open3D's own event loop blocks -- so there is no channel
# back from the viewer. The overlay is written atomically via os.replace, so a
# poll can never catch a half-written file.
def poll():
    try:
        m = os.path.getmtime(path)
        if m == state["m"]:
            return
        img = plt.imread(path)
        state["m"] = m
        art.set_data(img)
        art.set_extent((0, img.shape[1], img.shape[0], 0))
        fig.canvas.draw_idle()
    except Exception:
        pass            # a viewer is never worth crashing for

timer = fig.canvas.new_timer(interval=300)
timer.add_callback(poll)
timer.start()
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


def write_overlay(frame, lm, path, note=None, announce=True):
    """The 2D prediction on the frame it came from, in the same strip colours.

    `note` is stamped into the corner -- the pose it was measured at. Placement
    changes with angle, so an overlay without its angle is an unlabelled sample.

    Written atomically: the 2D viewer polls this file, and os.replace means it
    can only ever see a complete image.
    """
    img = frame.convert("RGB").resize((512, 512), Image.BILINEAR)
    s = 512 / frame.width           # the WHOLE frame, not a crop
    d = ImageDraw.Draw(img)
    for i, (a, b) in enumerate(STRIPS):
        c = tuple(int(255 * v) for v in STRIP_COLOURS[i])
        d.line([tuple(p) for p in (lm[a:b] * s)], fill=c, width=2)
        for x, y in lm[a:b] * s:
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=c, outline=(0, 0, 0))
    if note:
        d.rectangle([0, 0, 8 + 6 * len(note), 18], fill=(0, 0, 0))
        d.text((5, 4), note, fill=(255, 255, 255))
    tmp = f"{path}.{os.getpid()}.tmp.png"
    img.save(tmp, format="PNG")
    os.replace(tmp, path)
    if announce:
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
    """Face-on landmarks in 3D.

    Returns (mesh in the pinna frame, image, Camera, landmarks, P3, hit, conf).

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
    img, depth1, cam1, lm, conf = landmark_whole_frame(g, cfg, pipe)
    if lm is None:
        sys.exit("pass 1: the pipeline found no ear in the face-on render.\n"
                 "  With the MediaPipe frame the SIDE is chosen deterministically, so a\n"
                 "  head whose other ear is easier will still fail here. Try --ear 1,\n"
                 "  or --no-mediapipe to let the detector sweep pick whichever it likes.")
    P1, hit1, _ = backproject_snapped(depth1, lm, cam1, cfg)
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
    img2, depth2, cam2, lm2, conf2 = landmark_whole_frame(g2, cfg, pipe)
    if lm2 is None:
        sys.exit("pass 2: the pipeline found no ear in the pinna-frame render")
    P3, hit, snapped = backproject_snapped(depth2, lm2, cam2, cfg)
    if snapped.any():
        print(f"snap: {int(snapped.sum())} landmarks moved onto a depth cliff")
    print(f"pass 2: detector conf {conf2:.3f}, {int(hit.sum())}/55 rays hit")
    return g2, img2, cam2, lm2, P3, hit, conf2


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mesh", default=None, help="head .ply/.obj")
    p.add_argument("--ear", type=int, default=None, help="which detected ear, 0 or 1")
    p.add_argument("--tone", type=int, default=None, help="skin tone 0-8, light to deep")
    p.add_argument("--no-skin", action="store_true", help="render bare clay instead")
    p.add_argument("--no-mediapipe", action="store_true", help="use the ear-detector sweep")
    p.add_argument("--no-snap", action="store_true", help="show the raw back-projection")
    p.add_argument("--snap-radius", type=float, default=None, help="jitter range, 0-1 of ear extent")
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
    cfg["snap"] = cfg["snap"] and not a.no_snap
    if a.snap_radius is not None:
        cfg["snap_radius"] = a.snap_radius
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
    g, img, cam, lm, P3, hit, conf = label_two_pass(mesh, front, up, cfg, pipe, centre)
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
        write_overlay(Image.fromarray(img), lm, cfg["overlay_png"],
                      note="face-on  yaw +0  pitch +0")
        if cfg["show_2d"] and cfg["interactive"]:
            show_image_window(cfg["overlay_png"])

    if not cfg["interactive"]:
        return
    print("\norbit to about 60 deg to check depth; press L to re-landmark from "
          "wherever you are. Close the window to exit.")
    show(g, P3, hit, cfg, pipe)


if __name__ == "__main__":
    main()
