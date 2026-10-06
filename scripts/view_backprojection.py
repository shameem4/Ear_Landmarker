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

THIS FILE IS THE SINGLE SOURCE FOR THE 3D PATH. The camera, the renderer, the
back-projection, the ear/head framing and the snaps are defined here and nowhere
else; scripts/ingest_render3d.py imports them rather than keeping its own. It
started out self-contained so it could be hacked on freely, and that is exactly
how the path fragmented -- ingest grew a second copy of load_head, find_ears and
in_frame, plus a third camera setup of its own, and the copies drifted.

What it still imports is the model (EarLandmarkerPipeline and the checkpoint
picker) and scripts/skin.py for appearance. Neither is geometry.

Anything added here is inherited by ingest. Anything added THERE that belongs to
the 3D path belongs here instead.
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

# THE 3D PATH LIVES IN ear3d/, NOT HERE. Camera, renderer, back-projection, the
# snaps, the framing and the labelling are imported so that this viewer and
# scripts/ingest_render3d.py run the same code rather than two copies of it.
# What remains below is the viewer itself: configuration, the interactive window
# and the command line.
from ear3d.backproject import (backproject, backproject_snapped,  # noqa: E402,F401
                               reseat)
from ear3d.camera import (Camera, camera_ke, cone_angles,        # noqa: E402,F401
                          extrinsic_of, orbit_extrinsic, pose_angles)
from ear3d.config import DEFAULTS, EAR3D_DIR                     # noqa: E402
from ear3d.draw import (ray_lines, save_agreement_render,        # noqa: E402,F401
                        spheres, strip_lines, write_overlay)
from ear3d.frames import (find_ears, frame_from, head_pose,      # noqa: E402,F401
                          in_frame, load_head, plane_normal)
from ear3d.label import (agreement, collect_rays, fit_rays,   # noqa: E402,F401
                         label_two_pass, multiview, report_agreement,
                         triangulate_landmarks)
from ear3d.render import (LINE_MATERIAL, MATERIAL,               # noqa: E402,F401
                          gui_depth_to_view, light_scene, render)
from ear3d.scheme import STRIPS, STRIP_COLOURS, STRIP_NAMES      # noqa: E402,F401

from skin import ALL_TONES, apply_skin       # noqa: E402  (appearance, not geometry)

CONFIG = dict(DEFAULTS)                      # the shared defaults; see ear3d/config.py

# ---------------------------------------------------------------- CONFIG ----
# Third-party research data; not in any repo. Override with EAR3D_DIR.
EAR3D_DIR = Path(os.environ.get("EAR3D_DIR", ROOT.parent / "clean_3d_data"))

# ---------------------------------------------------------------------------



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































def show(mesh, P3, hit, cfg, pipe, size=900, background=(0.5, 0.5, 0.5),
         title="back-projected ear landmarks", on_tick=None,
         variants=None, start=0):
    """Interactive window -- SAME renderer and SAME camera as the snapshot.

    Built from gui.Window rather than O3DVisualizer because that class exposes
    neither key events nor a tick callback in the Python bindings, and both are
    needed here.

    Two things it does beyond showing the scene:

      the readout tracks the camera as you orbit, in the angles the
      back-projection actually works at. The mesh is fixed in its pinna frame, so
      camera yaw/pitch IS the pose; face-on reads 0/0 by construction.

      1..9 switch between the lifting methods in `variants`, which are all
      computed up front by main(). They have to be: each triangulated method
      needs a dozen offscreen renders, and an OffscreenRenderer cannot be opened
      while this window holds the Filament context. So the keys are a pure
      switch between answers already in hand -- instant, and comparable, because
      every method saw exactly the same renders and the same detections.

      L re-landmarks from where you are standing. The 3D landmarks are removed,
      the mesh alone is re-rendered through the window's CURRENT camera, the
      pipeline runs on that image, and the result is back-projected through the
      depth buffer of that same render. So it answers "what would the model make
      of the ear from here?" rather than re-showing the face-on answer. The
      landmarks it draws are a fresh measurement at this pose, not the pose-zero
      ones rotated. L does not change the selected method -- a single view is all
      it can ever be -- so it reports itself on the note line and leaves the
      method line to the number keys. Press the selected number to get back.
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
        # Give the panel the width the text actually wants. Capping it (it used
        # to be 420px) made the long lines wrap, and the wrapped text then
        # overflowed the height measured for the unwrapped version, which is
        # what put a scrollbar in the readout.
        pref = panel.calc_preferred_size(ctx, gui.Widget.Constraints())
        panel.frame = gui.Rect(rect.x, rect.y, min(rect.width, pref.width),
                               min(rect.height, pref.height))
    win.set_on_layout(on_layout)

    # How the landmarks on screen were produced. Shown because it is not fixed:
    # the number keys swap between methods, and judging what is drawn needs to
    # know which one made it.
    variants = list(variants or [])
    if not variants:
        variants = [{"label": "single face-on lift (depth buffer)",
                     "short": "face-on", "P": P3, "hit": hit, "snapped": None,
                     "angles": None, "method": None, "reseat": False}]
    start = max(0, min(start, len(variants) - 1))
    state = {"drawn": [], "text": None, "busy": False,
             "snapped": variants[start].get("snapped"), "idx": start,
             "method": variants[start]["label"], "note": "as computed"}
    menu = "   ".join(f"[{i+1}]{v['short']}" for i, v in enumerate(variants))

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

    draw_landmarks(variants[start]["P"], variants[start]["hit"])

    bbox = mesh.get_axis_aligned_bounding_box()
    centroid = np.asarray(mesh.vertices).mean(0)

    def absorb(colour, raw_depth):
        """Take one captured view into the job. Runs on the main thread."""
        job = state["job"]
        img = np.asarray(colour)[:, :, :3].copy()
        cam = Camera.from_gui(widget.scene.camera, img.shape[:2])
        res = pipe(img, timestamp=0.0)
        lm, conf = None, float("nan")
        if res:
            best = max(res, key=lambda d: float(d["confidence"]))
            lm = np.asarray(best["landmarks"], float)
            conf = float(best["confidence"])
        job["uv"].append(lm if lm is not None else np.full((55, 2), np.nan))
        job["cams"].append(cam)
        job["det"].append(conf)
        if job["i"] == 0:
            # The first view IS the pose you are standing at, so it is the one
            # the 2D window and the depth buffer come from.
            job["img0"], job["lm0"], job["conf0"] = img, lm, conf
            if raw_depth is not None:
                job["depth"] = gui_depth_to_view(np.asarray(raw_depth),
                                                 widget.scene.camera)
        job["i"] += 1
        job["stage"] = "fit" if job["i"] >= len(job["angles"]) else "set"

    def grab(want_depth):
        """Capture the window's current frame, through the window's own renderer.

        It must not open an OffscreenRenderer: that is a second Filament context,
        and creating one while the GUI holds the first fails at the driver
        ("failed to create dri2 screen") and takes the process down. The window's
        own scene.render_to_image / render_to_depth_image reuse the live context.

        Both call back from the RENDER thread. Touching the scene from there
        corrupts the interpreter state outright ("gilstate_tss_set: failed to set
        current tstate"), so each step hops back to the main thread first.
        """
        app = gui.Application.instance

        def got_depth(c, d):
            app.post_to_main_thread(win, lambda: absorb(c, d))

        def ask_depth(c):
            widget.scene.scene.render_to_depth_image(lambda d: got_depth(c, d))

        def got_colour(c):
            if want_depth:
                app.post_to_main_thread(win, lambda: ask_depth(c))
            else:
                app.post_to_main_thread(win, lambda: absorb(c, None))

        widget.scene.scene.render_to_image(got_colour)

    def fit_job():
        """Every view is in. Fit the selected method and draw the result."""
        job = state.pop("job")
        v = variants[state["idx"]]
        E = job["home"]                             # put the camera back
        widget.setup_camera(K, E, size, size, bbox)
        yaw, pitch = job["base"]
        n_det = int(np.isfinite(job["det"]).sum())

        def give_up(why):
            """Keep what was on screen. Drawing nothing, or a cloud of identical
            points, is not a result -- and a degenerate line set is rejected by
            the renderer outright ("Caught an unknown exception"), which used to
            take the process down from inside the tick callback."""
            state["note"] = f"{why} at yaw {yaw:+.0f} pitch {pitch:+.0f}"
            cur = variants[state["idx"]]             # put the previous ones back
            state["snapped"] = cur.get("snapped")
            draw_landmarks(cur["P"], cur["hit"])
            state["busy"] = False
            print(f"  [L] {state['note']}", flush=True)

        if job["lm0"] is None and v.get("angles") is None:
            give_up("no ear found")
            return

        if v.get("angles") is None:
            Q, qhit, qsnap = backproject_snapped(job["depth"], job["lm0"],
                                                 job["cams"][0], cfg)
            state["snapped"] = qsnap
            detail = (f"det {job['conf0']:.2f}, {int(qhit.sum())}/55 hit"
                      + (f", {int(qsnap.sum())} snapped" if qsnap.any() else ""))
        else:
            # The same fit the number keys offer, on rays captured here rather
            # than at pose zero: the cone is re-centred on where you stand, so
            # it answers "what does this method make of the ear from here?"
            Q, n_in, n_view = fit_rays(np.array(job["uv"]), job["cams"],
                                       v["method"], cfg["tri_thresh"])
            qhit = np.isfinite(Q).all(axis=1)
            if v.get("reseat") and job["depth"] is not None and qhit.any():
                ext = float(np.ptp(Q[qhit], axis=0).max())
                Q, moved = reseat(np.where(qhit[:, None], Q, centroid),
                                  job["depth"], job["cams"][0], ext,
                                  cfg["reseat_tol"])
            if qhit.sum() < 2:
                give_up(f"nothing to triangulate ({n_det}/"
                        f"{len(job['angles'])} views detected)")
                return
            Q = np.where(qhit[:, None], Q, centroid)
            state["snapped"] = None
            detail = (f"{n_det}/{len(job['angles'])} views detected, "
                      f"{int(qhit.sum())}/55 fitted")

        draw_landmarks(Q, qhit)
        # The method line belongs to the number keys, and L now runs whichever
        # one is selected, so it has nothing left to say about method. The note
        # line stays empty on success: the landmarks are the answer.
        state["note"] = ""
        state["busy"] = False
        print(f"  [L] {v['label']} at yaw {yaw:+.0f} pitch {pitch:+.0f}: "
              f"{detail}", flush=True)
        # Rewriting the overlay is what refreshes the 2D window: it polls this
        # file, so the picture there always matches the pose in the 3D window.
        if job["lm0"] is None:
            return
        if cfg["snapshot_png"]:
            tmp = f"{cfg['snapshot_png']}.{os.getpid()}.tmp.png"
            Image.fromarray(job["img0"]).save(tmp, format="PNG")
            os.replace(tmp, cfg["snapshot_png"])
        if cfg["overlay_png"]:
            write_overlay(Image.fromarray(job["img0"]), job["lm0"],
                          cfg["overlay_png"],
                          note=f"yaw {yaw:+.0f}  pitch {pitch:+.0f}  "
                               f"det {job['conf0']:.2f}", announce=False)

    def pump():
        """Advance the capture job one tick. Driven from the tick callback.

        A tick-driven state machine rather than a chain of callbacks because the
        camera has to MOVE between views, and a capture requested in the same
        tick as the move can return the frame from before it. Setting the camera,
        letting a frame go by, then shooting keeps each image honest about the
        pose recorded with it.
        """
        job = state.get("job")
        if not job:
            return False
        if job["stage"] == "set":
            if job["i"] > 0:          # view 0 is where they already stand
                yaw, pitch = job["angles"][job["i"]]
                E = orbit_extrinsic(yaw, pitch, cfg)
                widget.setup_camera(K, E, size, size, bbox)
            job["stage"] = "settle"
            return True
        if job["stage"] == "settle":
            job["stage"] = "shoot"
            return True
        if job["stage"] == "shoot":
            job["stage"] = "capturing"
            grab(job["i"] == 0)       # depth only from the pose they stand at
            return True
        if job["stage"] == "fit":
            fit_job()
            return True
        return False                  # "capturing": waiting on the renderer

    def relandmark():
        """Re-run the SELECTED method from where the camera is now.

        For the single-view variant that is one capture at this pose. For a
        triangulated variant the cone is re-centred on this pose and the window
        walks its own camera through every view, because triangulating needs
        renders and the offscreen renderer is not available while the GUI holds
        the context.

        The landmarks come off the scene BEFORE any capture: they are geometry,
        and a frame with them in it would show the landmarker an ear with 55
        spheres stuck to it.
        """
        if state["busy"]:
            return
        v = variants[state["idx"]]
        yaw, pitch, _ = pose_angles(extrinsic_of(widget.scene.camera))
        offsets = v.get("angles") or [(0.0, 0.0)]
        state["busy"] = True
        state["job"] = {"angles": [(yaw + dy, pitch + dp) for dy, dp in offsets],
                        "i": 0, "stage": "set", "uv": [], "cams": [], "det": [],
                        "depth": None, "img0": None, "lm0": None,
                        "conf0": float("nan"), "base": (yaw, pitch),
                        "home": extrinsic_of(widget.scene.camera)}
        for n in state["drawn"]:
            widget.scene.remove_geometry(n)
        state["drawn"] = []
        state["note"] = (f"[L] {v['label']}: capturing "
                         f"{len(state['job']['angles'])} view(s)...")
        print(f"  {state['note']}", flush=True)

    def select(i):
        """Show variant i. No rendering: the points are already computed."""
        if state["busy"]:
            return
        if not 0 <= i < len(variants):
            state["note"] = (f"[{i+1}] no such method -- "
                             f"{len(variants)} available")
            print(f"  {state['note']}", flush=True)
            return
        v = variants[i]
        state["idx"] = i
        state["snapped"] = v.get("snapped")
        draw_landmarks(v["P"], v["hit"])
        state["method"] = v["label"]
        state["note"] = f"[{i+1}] {v['label']}"
        print(f"  {state['note']}", flush=True)

    def on_key(e):
        # gui.Window.set_on_key wants a BOOL -- True to stop dispatching. It is
        # gui.Widget.set_on_key_event that takes an EventCallbackResult, and
        # returning one here is not a type error at registration: it crashes
        # inside app.run() the first time any key is pressed.
        if e.type != gui.KeyEvent.Type.DOWN:
            return False
        if e.key == gui.KeyName.L:
            relandmark()
            return True
        # The digit KeyNames ARE their ASCII codes (gui.KeyName.ONE == 49), so
        # this covers 1..9 without naming each one.
        if ord("1") <= int(e.key) <= ord("9"):
            select(int(e.key) - ord("1"))
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
            state["hook"](dict(relandmark=relandmark, select=select, state=state,
                               variants=variants, win=win, widget=widget,
                               app=gui.Application.instance))
        busy = pump()
        yaw, pitch, off = pose_angles(extrinsic_of(widget.scene.camera))
        text = (f"yaw {yaw:+6.1f}\u00b0   pitch {pitch:+6.1f}\u00b0   "
                f"off-axis {off:5.1f}\u00b0\n{state['method']}\n"
                + (f"{state['note']}\n" if state["note"] else "")
                + f"{menu}\n[L] re-landmark from this view, same method")
        if text == state["text"]:
            return busy
        state["text"] = text
        info.text = text
        # The panel is only measured during layout, so a readout that gains or
        # loses a line must ask for one. Without this the frame keeps the height
        # measured for the old text and the surplus goes behind a scrollbar.
        win.set_needs_layout()
        return True
    win.set_on_tick_event(tick)

    app.run()


# ============================================================ ear finding ===






















# ============================================================ display ======









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




def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mesh", default=None, help="head .ply/.obj")
    p.add_argument("--ear", type=int, default=None, help="which detected ear, 0 or 1")
    p.add_argument("--tone", type=int, default=None, help="skin tone 0-8, light to deep")
    p.add_argument("--no-skin", action="store_true", help="render bare clay instead")
    p.add_argument("--no-mediapipe", action="store_true", help="use the ear-detector sweep")
    p.add_argument("--no-snap", action="store_true", help="show the raw back-projection")
    p.add_argument("--no-chain", action="store_true", help="skip the link-spacing pass")
    p.add_argument("--no-triangulate", action="store_true",
                   help="lift the single face-on view instead of intersecting rays")
    p.add_argument("--tri-views", type=int, default=None,
                   help="how many views to triangulate from (spread over a cone)")
    p.add_argument("--tri-method", choices=["lsq", "ransac"], default=None)
    p.add_argument("--no-reseat", action="store_true",
                   help="leave triangulated points where the rays put them")
    p.add_argument("--reseat", action="store_true",
                   help="pull triangulated points far off the surface back onto it")
    p.add_argument("--reseat-tol", type=float, default=None,
                   help="how far off counts as far, as a fraction of ear extent")
    p.add_argument("--multiview", action="store_true",
                   help="re-landmark from several views and report per-landmark agreement")
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
    cfg["chain"] = cfg["chain"] and not a.no_chain
    cfg["triangulate"] = ((cfg["triangulate"] or a.tri_views is not None)
                          and not a.no_triangulate)
    if a.tri_views:
        cfg["tri_angles"] = cone_angles(a.tri_views, cfg["tri_cone"])
    if a.tri_method:
        cfg["tri_method"] = a.tri_method
    cfg["reseat"] = ((cfg["reseat"] or a.reseat or a.reseat_tol is not None)
                     and not a.no_reseat)
    if a.reseat_tol is not None:
        cfg["reseat_tol"] = a.reseat_tol
    if cfg["reseat"] and not cfg["triangulate"]:
        # Re-seating only means something for TRIANGULATED points. The face-on
        # lift is read off the depth buffer, so it is on the surface already and
        # re-seating it is a no-op -- asking for one without the other silently
        # did nothing at all.
        print("--reseat implies triangulation; enabling it")
        cfg["triangulate"] = True
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

    # One entry per lifting method, all computed here, switched by the number
    # keys in the window. They share ONE set of renders and detections
    # (collect_rays), so what differs between them is only the fit -- and the
    # renders are what cost, so five methods are barely dearer than one.
    extras = ("  +cliff snap" if cfg["snap"] else "") + ("  +chain" if cfg["chain"] else "")
    # "angles" is what L re-runs: None means a single view, a list of yaw/pitch
    # offsets means a cone, re-centred on wherever the camera is at the time.
    variants = [{"label": "single face-on lift (depth buffer)" + extras,
                 "short": "face-on", "P": P3.copy(), "hit": hit.copy(),
                 "snapped": None, "angles": None, "method": None,
                 "reseat": False}]
    sel = 0
    if cfg["triangulate"]:
        nv = len(cfg["tri_angles"])
        print(f"triangulating from {nv} views (all methods, one set of renders)")
        uv, cams = collect_rays(cfg, g, pipe)
        img_f, depth_f, cam_f = render(g, cfg)
        fits = []
        for m in ("lsq", "ransac"):
            Pm, n_in, n_view = fit_rays(uv, cams, m, cfg["tri_thresh"])
            ok = np.isfinite(Pm).all(axis=1)
            Pm = np.where(ok[:, None], Pm, P3)
            hm = hit | ok
            moved = np.linalg.norm(Pm[ok] - P3[ok], axis=1)
            print(f"  {m}: {int(ok.sum())}/55 triangulated, mean "
                  f"{n_view[ok].mean():.1f} views and {n_in[ok].mean():.1f} inliers; "
                  f"moved from the face-on lift by median {np.median(moved):.3f}, "
                  f"max {moved.max():.3f} (ear spans ~1)")
            fits.append({"label": f"triangulated from {nv} views ({m})" + extras,
                         "short": m, "P": Pm, "hit": hm, "snapped": None,
                         "angles": list(cfg["tri_angles"]), "method": m,
                         "reseat": False})
        # The re-seated pair goes after both plain fits, so 2/3 compare the fits
        # and 4/5 are the same two with the depth buffer given the last word on
        # how far away each point is. Only points well off the surface move.
        tol = cfg["reseat_tol"]
        seats = []
        for f in fits:
            ext = float(np.ptp(f["P"][f["hit"]], axis=0).max())
            Pr, moved_r = reseat(f["P"].copy(), depth_f, cam_f, ext, tol)
            print(f"  {f['short']} + re-seat: {int(moved_r.sum())} landmarks sat "
                  f"more than {100*tol:.0f}% of ear extent off the surface")
            # Say so when it moved nothing: a menu entry that is byte-identical
            # to another one invites reading a difference into it that is not
            # there. At this tolerance on this head, it is a no-op.
            seats.append({"label": f["label"].replace(extras, "")
                          + f", re-seated >{100*tol:.0f}%"
                          + ("" if moved_r.any() else " (moved nothing)") + extras,
                          "short": f"{f['short']}+seat", "P": Pr,
                          "hit": f["hit"], "snapped": None,
                          "angles": list(cfg["tri_angles"]),
                          "method": f["method"], "reseat": True})
        variants += fits + seats
        # Open on whatever the flags asked for, so running with no arguments
        # still shows the default flow -- the keys only add ways to compare.
        sel = (1 if cfg["tri_method"] == "lsq" else 2) + (2 if cfg["reseat"] else 0)
        P3, hit = variants[sel]["P"], variants[sel]["hit"]
    print("methods: " + "   ".join(
        f"[{i+1}]{' <' if i == sel else ' '} {v['label']}"
        for i, v in enumerate(variants)))

    # Depth spread is the tell for a scalp-pinned point: it sits behind the rest.
    # Printed AFTER triangulation and re-seating, so it describes the landmarks
    # actually returned rather than an intermediate the next stage replaces.
    print(f"landmark depth z: {P3[:,2].min():+.3f} to {P3[:,2].max():+.3f} "
          f"(ear spans about 1.0 by construction)")
    for i, (a_, b_) in enumerate(STRIPS):
        print(f"  {STRIP_NAMES[i]:<14s} z {P3[a_:b_,2].min():+.3f} .. "
              f"{P3[a_:b_,2].max():+.3f}   misses {int((~hit[a_:b_]).sum())}")

    if a.multiview:
        # Each view is an independent measurement of the same anatomy. Colour by
        # how far they disagree, so the points worth distrusting are visible
        # rather than buried in a table.
        ext = float(np.ptp(P3[hit], axis=0).max()) if hit.any() else 1.0
        Pv, okv, angles = multiview(g, cfg, pipe)
        spread, centre, nview = agreement(Pv, okv)
        report_agreement(spread, nview, ear_extent=ext)
        print(f"\near extent {ext:.3f}; {len(angles)} views: "
              + ", ".join(f"({y:+.0f},{p:+.0f})" for y, p in angles))
        bad = np.isfinite(spread) & (spread > 0.05 * ext)
        print(f"{int(bad.sum())} of {int(np.isfinite(spread).sum())} landmarks "
              f"disagree by more than 5% of ear extent")
        if cfg["overlay_png"]:
            lo, hi = 0.0, 0.08 * ext
            cols = []
            for k in range(len(spread)):
                if not np.isfinite(spread[k]):
                    cols.append((0.4, 0.4, 0.4))
                    continue
                t = float(np.clip((spread[k] - lo) / max(hi - lo, 1e-9), 0, 1))
                cols.append((t, 1.0 - t, 0.15))      # green = agree, red = not
            P_show = np.where(np.isfinite(centre), centre, P3)
            path = cfg["overlay_png"].replace(".png", "_agreement.png")
            save_agreement_render(g, P_show, cols, cfg, path)
        return

    if a.multiview:
        # Each view is an independent measurement of the same anatomy, so how far
        # they disagree is the closest thing here to a placement check. Colour by
        # it, so the points worth distrusting are visible rather than tabulated.
        ext = float(np.ptp(P3[hit], axis=0).max()) if hit.any() else 1.0
        Pv, okv, angles = multiview(g, cfg, pipe)
        spread, centre, nview = agreement(Pv, okv)
        report_agreement(spread, nview, ear_extent=ext)
        print(f"\near extent {ext:.3f}; {len(angles)} views: "
              + ", ".join(f"({y:+.0f},{p:+.0f})" for y, p in angles))
        fin = np.isfinite(spread)
        print(f"{int((fin & (spread > 0.05 * ext)).sum())} of {int(fin.sum())} "
              f"landmarks disagree by more than 5% of ear extent")
        if cfg["overlay_png"]:
            hi = 0.08 * ext
            cols = [(0.4, 0.4, 0.4) if not np.isfinite(spread[k]) else
                    (lambda t: (t, 1.0 - t, 0.15))(float(np.clip(spread[k] / hi, 0, 1)))
                    for k in range(len(spread))]
            P_show = np.where(np.isfinite(centre), centre, P3)
            save_agreement_render(g, P_show, cols, cfg,
                                  cfg["overlay_png"].replace(".png", "_agreement.png"))
        return

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
    show(g, P3, hit, cfg, pipe, variants=variants, start=sel)


if __name__ == "__main__":
    main()
