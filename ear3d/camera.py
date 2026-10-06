"""The camera. ONE definition, used by the renderer and the projection alike.

camera_ke() produces (K, E); those same arrays go to Filament's setup_camera AND
build the Camera that projects and back-projects, so the forward transform, the
inverse transform and the picture cannot drift apart. Verified with markers to
0.73 px, and project -> rays round-trips at 9e-16."""

from __future__ import annotations

import numpy as np
import open3d as o3d

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


def orbit_extrinsic(yaw, pitch, cfg):
    """Extrinsic for a camera orbited to (yaw, pitch) about the ear, looking at it."""
    y, p = np.radians(yaw), np.radians(pitch)
    d = cfg["eye_z"]
    C = np.array([d * np.sin(y) * np.cos(p), d * np.sin(p), d * np.cos(y) * np.cos(p)])
    z = -C / np.linalg.norm(C)
    x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    yv = np.cross(z, x)
    R = np.stack([x, yv, z])
    E = np.eye(4)
    E[:3, :3], E[:3, 3] = R, -R @ C
    return E
