"""Geometry for showing landmarks, and the 2D overlay image.

Presentation only -- nothing here is used to compute a landmark."""

from __future__ import annotations

import os

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw

from .camera import camera_ke
from .render import MATERIAL, light_scene
from .scheme import STRIPS, STRIP_COLOURS

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


def save_agreement_render(mesh, P, colours, cfg, path, size=620):
    """Render the cross-view median landmarks, coloured by how much views disagree."""
    r = o3d.visualization.rendering.OffscreenRenderer(size, size)
    for i, geom in enumerate([mesh, spheres(P, cfg["sphere_frac"] * 1.3, colours),
                              strip_lines(P)]):
        r.scene.add_geometry(f"g{i}", geom, MATERIAL)
    light_scene(r.scene, (0.5, 0.5, 0.5))
    K, E = camera_ke(size, cfg)
    r.setup_camera(K, E, size, size)
    img = np.asarray(r.render_to_image())[:, :, :3]
    del r
    Image.fromarray(img).save(path)
    print(f"wrote {path}  (green = views agree, red = they do not)")


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
