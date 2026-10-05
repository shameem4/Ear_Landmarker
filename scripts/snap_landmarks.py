"""Snap ray-cast landmarks onto the mesh, then repair any that broke the chain.

STAGE 1 -- nearest mesh point. A camera ray that passes just outside the pinna
hits the scalp behind it, placing the landmark about one ear-depth too deep.
Snapping to the nearest surface point fixes rays that merely grazed, but NOT a
ray that genuinely struck the head: its nearest surface point is the scalp it
already hit.

STAGE 2 -- repair by neighbour consistency. The 55 points are four ordered
linestrips, so a correctly placed landmark sits roughly one step from its
sequence neighbours. One pinned to the skull is an outlier against that spacing
while its neighbours are not. Those are re-placed at the mesh point nearest the
midpoint of their surviving neighbours.

This is deliberately structural rather than geometric: it needs no pinna/scalp
segmentation, which is the part that has proven unreliable -- a quadratic scalp
fit plus a standoff threshold labelled between 0% and 8% of the region as pinna
depending on subject, where the true figure is 20-35%.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

STRIPS = [(0, 20), (20, 35), (35, 50), (50, 55)]


def snap_in_plane(points, V, hit=None, k=48):
    """Snap within the image plane only: adjust x/y, never push the point in/out.

    A plain 3D nearest-neighbour snap can pull a landmark BACKWARDS into the
    scalp directly behind the pinna -- same image position, slightly further
    away, and genuinely the nearest surface in 3D. That is the one direction the
    move must not take: the 2D prediction is what the model is good at, depth is
    what it cannot see.

    A ray that HIT already returns the frontmost surface at exactly that image
    position, which is what the camera saw, so it is left alone. Only a ray that
    MISSED is moved, and only to the nearest surface in (x, y).

    An earlier version took the frontmost vertex within a lateral RADIUS. At
    0.04 of ear extent that radius reaches the helix rim, so concha landmarks
    were dragged forward onto it -- every landmark moved, all of them forward,
    by a median 7% of ear extent. That is the in/out move this is meant to
    forbid, reintroduced as a side effect.
    """
    P = np.asarray(points, float).copy()
    if hit is None:
        hit = np.zeros(len(P), bool)
    need = ~np.asarray(hit, bool)
    if not need.any():
        return P
    tree2d = cKDTree(V[:, :2])
    d, j = tree2d.query(P[need][:, :2], k=1)
    P[need] = V[np.atleast_1d(j)]
    return P


def _neighbours(k):
    for a, b in STRIPS:
        if a <= k < b:
            return [j for j in (k - 1, k + 1) if a <= j < b]
    return []


def snap(points, mesh_vertices, hit=None, tol=2.5, max_passes=3, in_plane=True):
    """Two-stage snap. Returns (points, repaired_mask).

    `tol` is how many median strip-steps a landmark may sit from its neighbours
    before it is treated as mis-placed.
    """
    tree = cKDTree(mesh_vertices)
    P = (snap_in_plane(points, mesh_vertices, hit=hit) if in_plane
         else mesh_vertices[tree.query(np.asarray(points, float))[1]])   # stage 1

    repaired = np.zeros(len(P), bool)
    for _ in range(max_passes):
        # median step per strip, from the points not already flagged
        step = {}
        for a, b in STRIPS:
            d = np.linalg.norm(np.diff(P[a:b], axis=0), axis=1)
            ok = d[~repaired[a + 1:b]] if repaired[a + 1:b].any() else d
            step[(a, b)] = np.median(ok) if len(ok) else np.median(d)

        flagged = []
        for k in range(len(P)):
            nb = [j for j in _neighbours(k) if not repaired[j]]
            if not nb:
                continue
            a, b = next((a, b) for a, b in STRIPS if a <= k < b)
            d = np.mean([np.linalg.norm(P[k] - P[j]) for j in nb])
            if d > tol * step[(a, b)]:
                flagged.append(k)
        if not flagged:
            break
        for k in flagged:
            nb = [j for j in _neighbours(k) if j not in flagged]
            if not nb:
                continue
            target = np.mean([P[j] for j in nb], axis=0)
            P[k] = mesh_vertices[tree.query(target)[1]]              # stage 2
            repaired[k] = True
    return P, repaired
