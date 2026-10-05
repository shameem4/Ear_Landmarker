"""Segment the pinna in IMAGE space, by flood-filling to its depth cliff.

WHY NOT IN 3D. Three geometric attempts failed:
  - depth-gap histogram on the vertex cloud: picked a sliver of the wrong region;
  - cropping the raycast target to a tight box: truncated the ear;
  - standoff from a fitted scalp surface: labels the rim but NOT the concha,
    which is a depression and so does not stand proud of the head at all.
The last is fatal for any "the ear is what sticks out" rule.

WHY IMAGE SPACE WORKS. From a face-on camera the pinna's outer boundary is a
depth CLIFF -- the surface jumps back by the whole standoff to reach the scalp.
The concha causes no such cliff: it is a bowl whose wall runs continuously down
from the rim, so a flood fill that stops only at discontinuities keeps it.

WHY IT IS NEEDED AT ALL. With a generous raycast target every ray hits
something, so "did the ray hit?" stops distinguishing pinna from scalp and any
correction keyed on misses silently never fires.
"""

from __future__ import annotations

import numpy as np
from collections import deque


def depth_map(scene, size, unproject_rays, max_t=8.0):
    """Render a depth map of the submesh from the fixed camera."""
    import open3d as o3d
    ys, xs = np.mgrid[0:size, 0:size]
    uv = np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], axis=1).astype(float)
    o, d = unproject_rays(uv, size)
    t = scene.cast_rays(o3d.core.Tensor(np.hstack([o, d]).astype(np.float32)))["t_hit"].numpy()
    t[~np.isfinite(t)] = np.inf
    t[t > max_t] = np.inf
    return t.reshape(size, size)


def flood_pinna(depth, seed_xy, rel_jump=0.045):
    """Flood from `seed_xy`, refusing to cross a depth jump.

    `rel_jump` is a fraction of the depth map's own spread, so it adapts to how
    far the ear stands off on this subject rather than assuming a fixed distance.
    """
    H, W = depth.shape
    finite = np.isfinite(depth)
    if not finite.any():
        return np.zeros_like(depth, bool)
    spread = np.percentile(depth[finite], 95) - np.percentile(depth[finite], 5)
    thr = max(rel_jump * max(spread, 1e-6), 1e-6)

    sx, sy = int(round(seed_xy[0])), int(round(seed_xy[1]))
    sx, sy = np.clip(sx, 0, W - 1), np.clip(sy, 0, H - 1)
    if not finite[sy, sx]:                       # nudge onto the surface
        ok = np.argwhere(finite)
        sy, sx = ok[np.argmin(np.abs(ok - [sy, sx]).sum(1))]

    mask = np.zeros((H, W), bool)
    mask[sy, sx] = True
    q = deque([(sy, sx)])
    while q:
        y, x = q.popleft()
        dz = depth[y, x]
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < H and 0 <= nx < W and not mask[ny, nx] and finite[ny, nx]:
                if abs(depth[ny, nx] - dz) < thr:
                    mask[ny, nx] = True
                    q.append((ny, nx))
    return mask


def nearest_in_mask(uv, mask):
    """Move each 2D point to the nearest pixel inside `mask`. Returns (uv, moved)."""
    from scipy.spatial import cKDTree
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return np.asarray(uv, float), np.zeros(len(uv), bool)
    tree = cKDTree(np.stack([xs, ys], axis=1))
    P = np.asarray(uv, float)
    inside = np.zeros(len(P), bool)
    for i, (x, y) in enumerate(P):
        xi, yi = int(round(x)), int(round(y))
        if 0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1] and mask[yi, xi]:
            inside[i] = True
    out = P.copy()
    if (~inside).any():
        _, j = tree.query(P[~inside])
        out[~inside] = np.stack([xs[j], ys[j]], axis=1) + 0.5
    return out, ~inside
