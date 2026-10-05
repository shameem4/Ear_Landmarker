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


# --- the interactive window's pose readout ----------------------------------
# The window orbits the CAMERA while the mesh stays in its pinna frame, so the
# camera's yaw/pitch are the pose the back-projection is working at. These pin
# the sign conventions, which are the part that silently misleads if wrong.

def _orbit(cfg, C, size=SIZE):
    """Extrinsic for a camera at world position C, looking at the origin."""
    z = -np.asarray(C, float)
    z /= np.linalg.norm(z)                      # +z points INTO the scene
    x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    E = np.eye(4)
    E[:3, :3], E[:3, 3] = R, -R @ np.asarray(C, float)
    return E


def test_face_on_reads_zero():
    _, E0 = vbp.camera_ke(SIZE, CFG)
    yaw, pitch, off = vbp.pose_angles(E0)
    assert abs(yaw) < 1e-6 and abs(pitch) < 1e-6 and abs(off) < 1e-6


def test_yaw_is_positive_toward_plus_x():
    d = CFG["eye_z"]
    yaw, pitch, off = vbp.pose_angles(_orbit(CFG, [d * np.sin(np.radians(30)), 0,
                                                   d * np.cos(np.radians(30))]))
    assert yaw == pytest.approx(30.0, abs=0.1)
    assert abs(pitch) < 0.1
    assert off == pytest.approx(30.0, abs=0.1)


def test_pitch_is_positive_when_the_camera_is_above():
    d = CFG["eye_z"]
    yaw, pitch, off = vbp.pose_angles(_orbit(CFG, [0, d * np.sin(np.radians(25)),
                                                   d * np.cos(np.radians(25))]))
    assert pitch == pytest.approx(25.0, abs=0.1)
    assert abs(yaw) < 0.1


def test_extrinsic_override_actually_moves_the_camera():
    """render(E=...) must render from E, and its Camera must agree with it."""
    mesh = _marker_mesh()
    E = _orbit(CFG, [1.5, 0.0, CFG["eye_z"] * 0.8])
    img_a, _, cam_a = vbp.render(mesh, CFG)
    img_b, _, cam_b = vbp.render(mesh, CFG, E=E)
    assert not np.array_equal(img_a, img_b), "the override did not move the camera"
    # the returned Camera must be the one that drew img_b, not the default
    found = _centroids(img_b)
    assert len(found) == len(TARGETS)
    for p in cam_b.project(TARGETS):
        assert np.linalg.norm(found - p, axis=1).min() < 1.5


def test_view_matrix_round_trips_to_our_extrinsic():
    """extrinsic_of() must invert what setup_camera was given."""
    mesh = _marker_mesh()
    E = _orbit(CFG, [0.9, 0.6, CFG["eye_z"] * 0.85])
    r = o3d.visualization.rendering.OffscreenRenderer(SIZE, SIZE)
    r.scene.add_geometry("m", mesh, vbp.MATERIAL)
    K, _ = vbp.camera_ke(SIZE, CFG)
    r.setup_camera(K, E, SIZE, SIZE)
    back = vbp.extrinsic_of(r.scene.camera)
    del r
    assert np.allclose(back, E, atol=1e-4)


# --- the depth-cliff snap ---------------------------------------------------
# A landmark whose ray grazes past the pinna lands on the head behind it. The
# snap moves such a point onto the near lip of the depth step it is hiding
# behind. The danger is the opposite error -- dragging a point that is DEEPER
# FOR GOOD REASON, like the floor of the concha bowl, forward onto a rim. These
# pin the distinction that makes it safe.

def _step_depth(near=2.0, far=2.9, size=200):
    """Left half near, right half far: one hard occlusion edge down the middle."""
    d = np.full((size, size), far, dtype=np.float32)
    d[:, : size // 2] = near
    return d


def _bowl_depth(size=200, lo=2.4, hi=2.7):
    """A smooth basin: deeper in the middle, no discontinuity anywhere."""
    y, x = np.mgrid[0:size, 0:size]
    r = np.hypot(x - size / 2, y - size / 2) / (size / 2)
    return (lo + (hi - lo) * np.clip(1 - r, 0, 1)).astype(np.float32)[::-1]


def test_cliff_is_found_on_a_step_and_not_on_a_bowl():
    assert vbp.cliff_map(_step_depth(), 0.30)[0].any()
    assert not vbp.cliff_map(_bowl_depth(), 0.30)[0].any(), (
        "a smooth basin must not register as a cliff, or concha points get dragged")


def test_a_point_behind_the_step_snaps_to_the_near_lip():
    d = _step_depth(near=2.0, far=2.9)
    uv = np.array([[110.0, 100.0]])            # 10 px into the far side
    out, moved, shift, znear = vbp.snap_to_cliff(d, uv, radius=40, step=0.30)
    assert moved[0]
    assert znear[0] == pytest.approx(2.0, abs=1e-5), "must take the NEAR side"
    assert shift[0] < 15


def test_a_point_on_the_near_side_is_left_alone():
    d = _step_depth()
    uv = np.array([[90.0, 100.0]])             # already on the near side
    _, moved, _, _ = vbp.snap_to_cliff(d, uv, radius=40, step=0.30)
    assert not moved[0]


def test_a_point_in_a_bowl_is_left_alone():
    """The concha case: deeper than its surroundings, but no step to snap to."""
    d = _bowl_depth()
    uv = np.array([[100.0, 100.0]])            # the deepest part of the basin
    _, moved, _, _ = vbp.snap_to_cliff(d, uv, radius=40, step=0.30)
    assert not moved[0]


def test_a_point_beyond_the_radius_is_left_alone():
    """The radius is the safety bound: far-out points are errors, not grazes."""
    d = _step_depth(near=2.0, far=2.9)
    uv = np.array([[180.0, 100.0]])            # 80 px into the far side
    _, moved, _, _ = vbp.snap_to_cliff(d, uv, radius=20, step=0.30)
    assert not moved[0]


def test_snap_can_be_turned_off():
    d = _step_depth()
    cam = vbp.Camera(*vbp.camera_ke(SIZE, CFG), (SIZE, SIZE), SIZE)
    uv = np.array([[110.0, 100.0], [115.0, 100.0]])
    _, _, moved = vbp.backproject_snapped(d, uv, cam, dict(CFG, snap=False))
    assert not moved.any()
