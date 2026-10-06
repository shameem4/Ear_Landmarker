"""Fuse 2D detections from many views by intersecting their rays.

WHY THIS AND NOT A SINGLE VIEW. Lifting a landmark by reading the depth buffer
under it takes whatever surface that pixel happens to show. When the detection
sits just outside the pinna, that surface is the head behind it, and the point
lands an ear-depth too deep -- the failure the cliff snap patches locally. The
RAY does not have that problem: it is what the 2D detection actually asserts,
and it is unaffected by what the ray later hits. Intersecting rays from several
views recovers the point without ever consulting a depth value.

This is the standard approach outside this project. Multi-view consensus CNN
(arXiv 1910.06007) renders a face scan from ~100 views, turns each 2D heatmap
peak into a 3D ray and solves for the crossing with least squares plus RANSAC,
reporting ~2 mm -- about an experienced annotator's own repeatability. Learnable
Triangulation (ICCV 2019) replaces the hard RANSAC vote with per-view learned
confidences, for the same reason: views contribute unevenly under occlusion.

RANSAC rather than plain least squares because a landmark the detector put on the
scalp in one view produces a ray that misses the true point entirely, and a least
squares fit over all rays is dragged by it. A vote discards it instead.
"""

from __future__ import annotations

import numpy as np


def lsq_point(origins, dirs):
    """Point minimising the summed squared distance to a set of rays.

    Each ray contributes (I - n n^T), the projector onto the plane perpendicular
    to it; the normal equations are then a 3x3 solve. Returns None when the rays
    are degenerate -- near-parallel rays leave the system ill-conditioned, which
    is exactly the case with a narrow baseline.
    """
    o = np.asarray(origins, float)
    n = np.asarray(dirs, float)
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for oi, ni in zip(o, n):
        P = np.eye(3) - np.outer(ni, ni)
        A += P
        b += P @ oi
    if np.linalg.cond(A) > 1e8:
        return None
    return np.linalg.solve(A, b)


def ray_distance(p, origins, dirs):
    """Perpendicular distance from a point to each ray."""
    v = np.asarray(p, float) - np.asarray(origins, float)
    n = np.asarray(dirs, float)
    return np.linalg.norm(v - (v * n).sum(1, keepdims=True) * n, axis=1)


def ransac_point(origins, dirs, thresh, iters=64, rng=None, min_inliers=2):
    """Robust ray intersection. Returns (point, inlier mask) or (None, None).

    Minimal sets of TWO rays, since two suffice to define a crossing and smaller
    samples make the vote far more likely to find a clean pair. The winning set
    is refit by least squares over all its inliers, which is what actually sets
    the accuracy -- RANSAC only decides who votes.
    """
    o = np.asarray(origins, float)
    n = np.asarray(dirs, float)
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    m = len(o)
    if m < 2:
        return None, None
    if m == 2:
        p = lsq_point(o, n)
        return p, (np.ones(2, bool) if p is not None else None)
    rng = rng or np.random.default_rng(0)
    best_p, best_in = None, np.zeros(m, bool)
    for _ in range(iters):
        i, j = rng.choice(m, 2, replace=False)
        p = lsq_point(o[[i, j]], n[[i, j]])
        if p is None:
            continue
        inl = ray_distance(p, o, n) < thresh
        if inl.sum() > best_in.sum():
            best_p, best_in = p, inl
    # Two rays already define a point, so requiring more than that just sends
    # every sparse landmark down the non-robust fallback.
    if best_p is None or best_in.sum() < min_inliers:
        # No consensus. Fall back to the plain fit rather than returning nothing,
        # and let the caller see the inlier count to judge it.
        p = lsq_point(o, n)
        return p, (np.ones(m, bool) if p is not None else None)
    refit = lsq_point(o[best_in], n[best_in])
    return (refit if refit is not None else best_p), best_in


def triangulate(uv_per_view, cams, thresh, iters=64, seed=0):
    """Per-landmark robust intersection of the rays from every view.

    `uv_per_view` is (V, K, 2) with NaN where a view produced no detection, and
    `cams` the V Camera objects that produced them. Returns (points (K,3),
    inlier counts (K,), views used (K,)).
    """
    uv = np.asarray(uv_per_view, float)
    V, K, _ = uv.shape
    pts = np.full((K, 3), np.nan)
    n_in = np.zeros(K, int)
    n_view = np.zeros(K, int)
    rng = np.random.default_rng(seed)
    for k in range(K):
        o_list, d_list = [], []
        for v in range(V):
            if not np.isfinite(uv[v, k]).all():
                continue
            o, d = cams[v].rays(uv[v, k][None, :])
            o_list.append(o[0])
            d_list.append(d[0])
        n_view[k] = len(o_list)
        if len(o_list) < 2:
            continue
        p, inl = ransac_point(np.array(o_list), np.array(d_list), thresh,
                              iters=iters, rng=rng)
        if p is None:
            continue
        pts[k] = p
        n_in[k] = int(inl.sum()) if inl is not None else 0
    return pts, n_in, n_view
