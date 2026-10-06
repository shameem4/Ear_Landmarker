"""The renderer: Open3D/Filament, one lighting setup, returning its own camera.

Every render in the 3D path comes through here, including the detector probe
sweep. ingest_render3d.py used to carry its own OffscreenRenderer with a separate
camera setup, which is how the two drifted."""

from __future__ import annotations

import numpy as np
import open3d as o3d

from .camera import Camera, camera_ke

MATERIAL = o3d.visualization.rendering.MaterialRecord()
MATERIAL.shader = "defaultLit"


LINE_MATERIAL = o3d.visualization.rendering.MaterialRecord()
LINE_MATERIAL.shader = "unlitLine"
LINE_MATERIAL.line_width = 2.0


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
