"""The renderer and the back-projection must not be able to disagree.

scripts/view_backprojection.py defines its camera ONCE, as (K, E), and hands
those same arrays both to Filament's setup_camera and to the Camera that
projects and back-projects. These tests pin that down, because the failure mode
is silent: a camera mismatch does not raise, it just puts every 3D landmark in
the wrong place, and the 2D overlay still looks correct.

This has gone wrong twice. Snapshots once rendered through Filament while the 3D
window used the legacy GLFW visualizer (different lighting entirely), and before
that the Blender path rendered a 90-degree-rotated, mirrored view at the wrong
field of view while the labels described the original.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip("open3d")
import open3d as o3d  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "vbp", ROOT / "scripts" / "view_backprojection.py")
vbp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vbp)

SIZE = 400
CFG = dict(vbp.CONFIG, size=SIZE)
# Asymmetric on purpose: a symmetric set hides a mirrored or transposed camera.
TARGETS = np.array([[0.0, 0.0, 0.0], [0.6, 0.0, 0.0], [-0.45, 0.0, 0.0],
                    [0.0, 0.55, 0.0], [0.0, -0.35, 0.0], [0.3, 0.4, 0.0],
                    [-0.3, 0.2, 0.25]])


def _marker_mesh(radius=0.045):
    m = o3d.geometry.TriangleMesh()
    for t in TARGETS:
        s = o3d.geometry.TriangleMesh.create_sphere(radius)
        s.translate(t)
        m += s
    m.compute_vertex_normals()
    m.paint_uniform_color([0.0, 0.0, 0.0])
    return m


def _centroids(img):
    from scipy import ndimage
    g = np.asarray(img).mean(axis=2)
    lab, n = ndimage.label(g < 90)
    return np.array([ndimage.center_of_mass(g < 90, lab, i + 1)[::-1]
                     for i in range(n)])


@pytest.fixture(scope="module")
def rendered():
    return vbp.render(_marker_mesh(), CFG)        # (image, depth, Camera)


def test_projection_matches_where_the_renderer_actually_drew(rendered):
    """Project() must land on the pixels the renderer put the markers at."""
    img, _, cam = rendered
    found = _centroids(img)
    assert len(found) == len(TARGETS), f"saw {len(found)} markers, expected {len(TARGETS)}"
    for p in cam.project(TARGETS):
        assert np.linalg.norm(found - p, axis=1).min() < 1.5


def test_rays_invert_projection_exactly(rendered):
    """rays() is the exact inverse of projection, not an approximation."""
    _, _, cam = rendered
    rng = np.random.default_rng(0)
    P = rng.uniform(-0.8, 0.8, (500, 3))
    o, d = cam.rays(cam.project(P))
    v = P - o
    perp = np.linalg.norm(v - (v * d).sum(1, keepdims=True) * d, axis=1)
    assert perp.max() < 1e-9


def test_backprojection_returns_the_surface_the_camera_sees(rendered):
    """Back-projecting a marker's own pixel must land back on that marker."""
    _, depth, cam = rendered
    P, hit = vbp.backproject(depth, cam.project(TARGETS), cam)
    assert hit.all(), f"only {hit.sum()} of {len(TARGETS)} points hit"
    # The surface is the sphere's front face, one radius toward the camera from
    # the centre -- never off to one side.
    assert np.linalg.norm(P - TARGETS, axis=1).max() < 0.06


def test_depth_backprojection_agrees_with_an_independent_raycast(rendered):
    """Cross-check the depth buffer against ray-casting the same mesh.

    The script deliberately has only ONE geometry path -- the depth buffer that
    produced the pixels. This test builds the second path that used to be in the
    script and asserts they agree, so the cheap, structurally-safe method is
    pinned against the independent one without shipping it.
    """
    _, depth, cam = rendered
    uv = cam.project(TARGETS)
    P_depth, hit_d = vbp.backproject(depth, uv, cam)

    t = o3d.t.geometry.TriangleMesh.from_legacy(_marker_mesh())
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(t)
    o, d = cam.rays(uv)
    ans = scene.cast_rays(o3d.core.Tensor(np.hstack([o, d]).astype(np.float32)))
    dist = ans["t_hit"].numpy()
    hit_r = np.isfinite(dist)
    P_ray = o + d * dist[:, None]

    assert (hit_d == hit_r).all()
    both = hit_d & hit_r
    assert np.linalg.norm(P_depth[both] - P_ray[both], axis=1).max() < 0.01


def test_background_is_a_miss_not_an_invented_point(rendered):
    """Empty background must report no hit rather than a filled-in depth."""
    _, depth, cam = rendered
    corners = np.array([[2.0, 2.0], [CFG["size"] - 3.0, 2.0]])
    _, hit = vbp.backproject(depth, corners, cam)
    assert not hit.any()


def test_silhouette_is_not_blended_into_empty_space():
    """Sampling across a depth edge must not return a point between surfaces.

    Bilinear interpolation across a silhouette averages a near surface with a far
    one and lands in the gap. The guard takes the nearest sample instead.
    """
    cam = vbp.Camera(*vbp.camera_ke(SIZE, CFG), (SIZE, SIZE), SIZE)
    near, far = 2.0, 2.9
    depth = np.full((SIZE, SIZE), far, dtype=np.float32)
    depth[:, : SIZE // 2] = near                       # a hard vertical edge
    # sample exactly on the edge, where a blend would give (near + far) / 2
    uv = np.array([[SIZE // 2 - 0.5, SIZE // 2]])
    P, hit = vbp.backproject(depth, uv, cam)
    assert hit.all()
    z = cam.R @ (P[0] - cam.centre)
    assert min(abs(z[2] - near), abs(z[2] - far)) < 1e-3, (
        f"blended to {z[2]:.3f}, between {near} and {far}")


def test_camera_rescales_when_the_render_is_not_the_requested_size():
    """A render returned at another size must rescale K, not silently misproject."""
    K, E = vbp.camera_ke(SIZE, CFG)
    full = vbp.Camera(K, E, (SIZE, SIZE), SIZE)
    half = vbp.Camera(K, E, (SIZE // 2, SIZE // 2), SIZE)
    assert not full.scaled and half.scaled
    assert half.fx == pytest.approx(full.fx / 2)
    assert np.allclose(half.project(TARGETS) * 2,
                       full.project(TARGETS), atol=1e-6)


def test_there_is_only_one_camera_definition():
    """Nothing may build a camera except camera_ke()."""
    src = (ROOT / "scripts" / "view_backprojection.py").read_text()
    assert src.count("def camera_ke") == 1
    # and only one geometry path: no ray-casting scene in the script
    assert "RaycastingScene" not in src
    # setup_camera is the renderer handshake; both call sites must pass K/E from
    # camera_ke, never a field-of-view/eye/up form that re-derives the camera.
    for line in src.splitlines():
        if "setup_camera(" in line and "def " not in line:
            assert "K, E" in line, f"camera built some other way: {line.strip()}"
