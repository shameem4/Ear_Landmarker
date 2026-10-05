"""Render real 3D ear geometry at a known pose.

WHY: perspective augmentation warps a 2D crop by a homography, which reproduces
foreshortening but not parallax or self-occlusion (data/dataset.py says so in its
own docstring). Nothing in the 2D pipeline can tell whether that approximation
matters. A rendered mesh can, because a real ear rotating really does occlude
itself.

WHICH DATA, AND WHY NOT THE OTHERS. Of the sources in clean_3d_data only
AudioEar3D works:
  - AudioEar3D       dense COLOURED point clouds of individual ears, ~90k points
                     at 0.10mm spacing. Needs Poisson reconstruction; the colour
                     is what keeps renders close enough to photographs to be
                     worth anything.
  - SONICOM graded   BEM acoustic simplifications. The pinna is a featureless
                     blob at 2.18mm edge length -- no concha, antihelix or
                     tragus, so there is nothing for 55 landmarks to attach to.
                     Measured: vertex colour variance exactly 0 (STL carries no
                     colour), and the ear DETECTOR boxes the whole head as an ear
                     on these renders at 0.62 confidence.
  - HUTUBS           head meshes; same acoustic-grade limitation.

CAMERA. Fixed at (0, 0, 3) looking at the origin, 50 deg vertical FOV, +Y up.
The MESH rotates, not the camera, so a given yaw/pitch means the same thing for
every subject. project() and unproject_rays() are the matching analytic forms --
keep all three in step if you change any of them.
"""

from __future__ import annotations

import copy
import io
import os
import tempfile
import zipfile

import numpy as np
import open3d as o3d
from pathlib import Path

EYE_Z = 3.0
VFOV = 50.0

# The 3D data is third-party research material and is NOT in this repository --
# it lives outside every repo and is gitignored (see NOTICE and the top-level
# clean_3d_data/README.md). Point EAR3D_DIR at it to run anything here.
EAR3D_DIR = Path(os.environ.get(
    "EAR3D_DIR", Path(__file__).resolve().parents[2] / "clean_3d_data"))
AUDIOEAR3D = str(EAR3D_DIR / "AudioEar3D.zip")


def rot(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Rotation about +Y (yaw) then +X (pitch)."""
    ry, rx = np.radians(yaw_deg), np.radians(pitch_deg)
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    return Rx @ Ry


def project(P: np.ndarray, size: int) -> np.ndarray:
    """World point -> pixel, for the camera below."""
    P = np.atleast_2d(P)
    z = EYE_Z - P[:, 2]
    t = np.tan(np.radians(VFOV) / 2.0)
    return np.stack([(P[:, 0] / np.maximum(z * t, 1e-9) + 1) / 2 * size,
                     (1 - P[:, 1] / np.maximum(z * t, 1e-9)) / 2 * size], axis=1)


def unproject_rays(uv: np.ndarray, size: int):
    """Pixel -> (origin, unit direction). Inverse of project(), to within the ray."""
    t = np.tan(np.radians(VFOV) / 2.0)
    x = (uv[:, 0] / size * 2 - 1) * t
    y = (1 - uv[:, 1] / size * 2) * t
    d = np.stack([x, y, -np.ones(len(uv))], axis=1)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return np.tile(np.array([0.0, 0.0, EYE_Z]), (len(uv), 1)), d


def render(mesh, yaw_deg: float = 0.0, pitch_deg: float = 0.0, size: int = 500,
           background=(0.5, 0.5, 0.5)) -> np.ndarray:
    """Offscreen render (EGL headless). Returns HxWx3 uint8."""
    g = copy.deepcopy(mesh)
    if yaw_deg or pitch_deg:
        g.rotate(rot(yaw_deg, pitch_deg), center=(0, 0, 0))
    r = o3d.visualization.rendering.OffscreenRenderer(size, size)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    r.scene.add_geometry("m", g, mat)
    r.scene.set_background([*background, 1.0])
    r.scene.scene.set_sun_light([-0.3, -0.4, -0.9], [1.0, 1.0, 1.0], 95000)
    r.scene.scene.enable_sun_light(True)
    r.setup_camera(VFOV,
                   np.array([0, 0, 0], dtype=np.float32),
                   np.array([0, 0, EYE_Z], dtype=np.float32),
                   np.array([0, 1, 0], dtype=np.float32))
    img = np.asarray(r.render_to_image())
    del r
    return img


def load_ear(member: str, zip_path: str = AUDIOEAR3D, depth: int = 9, keep: float = 0.6):
    """One AudioEar3D ear, Poisson-reconstructed into a canonically oriented mesh.

    Canonical frame: the ear's smallest principal extent is its surface normal, so
    that axis goes to +Z (toward the camera) and the largest to +Y. Pose zero is
    then "facing the ear" for every subject, independent of how the scan was
    stored. `keep` drops the lowest-density Poisson vertices, which are the
    invented surface that closes a single-sided scan.
    """
    z = zipfile.ZipFile(zip_path)
    fd, tmp = tempfile.mkstemp(suffix=".ply")
    os.write(fd, z.read(member))
    os.close(fd)
    try:
        pcd = o3d.io.read_point_cloud(tmp)
    finally:
        os.unlink(tmp)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=40))
    pcd.orient_normals_consistent_tangent_plane(30)
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, 1.0 - keep))
    V = np.asarray(mesh.vertices)
    if len(V) == 0:
        return None
    V = V - V.mean(0)
    _, s, vt = np.linalg.svd(V, full_matrices=False)
    axes = vt[np.argsort(-s)]
    R = np.stack([axes[1], axes[0], axes[2]])
    if np.linalg.det(R) < 0:
        R[2] *= -1
    V = V @ R.T
    V /= np.abs(V).max()
    mesh.vertices = o3d.utility.Vector3dVector(V)
    mesh.compute_vertex_normals()
    return mesh


def ears(zip_path: str = AUDIOEAR3D) -> list[str]:
    return sorted(x for x in zipfile.ZipFile(zip_path).namelist() if x.endswith(".ply"))
