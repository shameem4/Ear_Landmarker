"""2D -> 3D through the depth buffer of the render that produced the pixels.

Not a ray-cast: a RaycastingScene is a SECOND geometry path, agreeing with the
picture only because it is the same mesh through the same camera, with nothing
enforcing it. The depth buffer IS that rasterisation.

Also holds the two snaps, which only make sense against a depth buffer."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .scheme import STRIPS

def cliff_map(depth, step):
    """Pixels where the surface STEPS by more than `step` within a 3x3 window.

    Background (inf) is pushed to a finite far value first, so the pinna's
    silhouette against the head behind it counts as an edge like any other
    occlusion. A RIDGE is not a cliff: the antihelix curves, it does not break,
    so its local max-min stays small and it never qualifies. That distinction is
    what makes this safe where two earlier snaps were not -- both of those used
    "nearest surface in the window", which drags concha-floor points forward onto
    the rim, because a bowl legitimately has nearer surface beside it.

    Returns (is_cliff, near_depth) where near_depth is the 3x3 minimum: the
    foreground lip of whatever edge runs through that pixel.
    """
    d = np.where(np.isfinite(depth), depth, np.nan)
    far = np.nanmax(d) + 1.0 if np.isfinite(np.nanmax(d)) else 1.0
    d = np.where(np.isnan(d), far, d)
    hi = ndimage.maximum_filter(d, size=3)
    lo = ndimage.minimum_filter(d, size=3)
    return (hi - lo) > step, lo


def snap_to_cliff(depth, uv, radius, step):
    """Place landmarks that sit BEHIND a depth cliff onto its near lip.

    For each landmark, look within `radius` px for a depth cliff. If one is
    there AND the landmark is currently on the far side of it, move to the
    nearest cliff pixel and take the near-side depth. A landmark with no cliff
    nearby, or already on the near lip, is left exactly where it is.

    Returns (uv, moved_mask, shift_px, near_depth). The near depth is returned
    rather than applied, because re-sampling the depth buffer AT a cliff pixel is
    a coin flip between the two surfaces it separates -- which is the entire
    failure being corrected. Measured, taking it explicitly moved this from
    fixing 0 of 13 bad landmarks to fixing 11.
    """
    cliff, lo = cliff_map(depth, step)
    H, W = depth.shape
    out = np.asarray(uv, float).copy()
    moved = np.zeros(len(out), bool)
    shift = np.zeros(len(out))
    znear = np.full(len(out), np.nan)
    r = int(np.ceil(radius))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    rad = np.hypot(xx, yy)
    for k, (u, v) in enumerate(out):
        x0, y0 = int(round(u)), int(round(v))
        xs, xe = max(0, x0 - r), min(W, x0 + r + 1)
        ys, ye = max(0, y0 - r), min(H, y0 + r + 1)
        if xe <= xs or ye <= ys:
            continue
        sub = cliff[ys:ye, xs:xe]
        rr = rad[ys - (y0 - r):ye - (y0 - r), xs - (x0 - r):xe - (x0 - r)]
        cand = sub & (rr <= radius)
        if not cand.any():
            continue
        dd = np.where(cand, rr, np.inf)
        iy, ix = np.unravel_index(np.argmin(dd), dd.shape)
        gy, gx = ys + iy, xs + ix
        d_here = depth[min(max(y0, 0), H - 1), min(max(x0, 0), W - 1)]
        # Only move a point that is actually BEHIND the cliff. One already on the
        # near lip belongs there; moving it anyway was the whole source of
        # collateral damage in the previous version (15-17 good points per head
        # displaced, against 3 with this gate).
        if not np.isfinite(d_here) or (d_here - lo[gy, gx]) < 0.5 * step:
            continue
        out[k] = [gx, gy]
        moved[k] = True
        shift[k] = dd[iy, ix]
        znear[k] = lo[gy, gx]
    return out, moved, shift, znear


def snap_chain(P, uv, depth, cam, cfg):
    """Second pass: re-place landmarks that break their chain's link spacing.

    The 55 points are four ordered linestrips, and within a strip consecutive
    links are roughly equal. Measured over four heads, link/median runs from p10
    0.84 to p90 1.09-1.61 by strip, and only 4 of 204 links exceed 2x. So this
    corrects OUTLIERS; it does not enforce uniformity. That distinction matters:
    concha_border naturally spreads to 1.6x, and flattening it would fight real
    anatomy rather than fix anything. superior_crus is the tight one (p90 1.09).

    A displaced landmark shows up in one of two ways, and testing for only the
    first finds nothing: BOTH links long when it is pushed off the chain, or one
    long and one SHORT when it has slid toward a neighbour. The second is the
    common one here -- pp12's concha border reads 2.0 then 0.3 across one point.
    So the score is total deviation of both links from the strip median, which
    catches either, rather than a test on the shorter link, which caught neither.

    The replacement is searched on the surface itself: candidates are pixels in a
    window around the landmark, back-projected through the depth buffer, scored by
    how close they bring both links to the strip median. So the result is a point
    the camera actually saw, not an interpolated midpoint hanging off the mesh --
    which is what the old vertex-based version produced.
    """
    P = np.asarray(P, float).copy()
    uv = np.asarray(uv, float)
    moved = np.zeros(len(P), bool)
    H, W = depth.shape
    r = int(np.ceil(cfg["chain_radius"] * max(np.ptp(uv[:, 0]), np.ptp(uv[:, 1]))))
    tol = cfg["chain_tol"]
    for _ in range(cfg["chain_passes"]):
        worst, worst_score = None, tol
        for a, b in STRIPS:
            d = np.linalg.norm(np.diff(P[a:b], axis=0), axis=1)
            if len(d) < 3:
                continue
            med = np.median(d)
            for k in range(a, b):
                i = k - a
                links = [d[j] for j in (i - 1, i) if 0 <= j < len(d)]
                if len(links) < 2:
                    continue                      # strip ends have one link only
                score = sum(abs(x - med) for x in links) / max(med, 1e-9)
                if score > worst_score and not moved[k]:
                    worst, worst_score = (k, a, b, med), score
        if worst is None:
            break
        k, a, b, med = worst
        x0, y0 = int(round(uv[k, 0])), int(round(uv[k, 1]))
        xs, xe = max(0, x0 - r), min(W, x0 + r + 1)
        ys, ye = max(0, y0 - r), min(H, y0 + r + 1)
        yy, xx = np.mgrid[ys:ye, xs:xe]
        zz = depth[ys:ye, xs:xe]
        ok = np.isfinite(zz)
        if ok.sum() < 10:
            break
        u, vv, z = xx[ok].ravel(), yy[ok].ravel(), zz[ok].ravel()
        Pc = np.stack([(u - cam.cx) / cam.fx * z, (vv - cam.cy) / cam.fy * z, z], axis=1)
        cand = (Pc - cam.t) @ cam.R
        cost = np.zeros(len(cand))
        for j in (k - 1, k + 1):
            if a <= j < b:
                cost += np.abs(np.linalg.norm(cand - P[j], axis=1) - med)
        P[k] = cand[np.argmin(cost)]
        moved[k] = True
    return P, moved


def backproject_snapped(depth, uv, cam, cfg):
    """backproject(), then the cliff snap if it is enabled. Returns (P, hit, moved)."""
    P, hit = backproject(depth, uv, cam)
    if not cfg.get("snap"):
        return P, hit, np.zeros(len(P), bool)
    uv = np.asarray(uv, float)
    ext = max(np.ptp(uv[:, 0]), np.ptp(uv[:, 1]))
    uv2, moved, _, znear = snap_to_cliff(depth, uv, cfg["snap_radius"] * ext,
                                         cfg["snap_step"])
    use = moved & np.isfinite(znear)
    if use.any():
        u, v, z = uv2[use, 0], uv2[use, 1], znear[use]
        Pc = np.stack([(u - cam.cx) / cam.fx * z, (v - cam.cy) / cam.fy * z, z], axis=1)
        P[use] = (Pc - cam.t) @ cam.R
        hit = hit | use
    if cfg.get("chain"):
        P, cmoved = snap_chain(P, uv2, depth, cam, cfg)
        moved = moved | cmoved
    return P, hit, moved


def backproject(depth, uv, cam, max_jump=0.02):
    """2D -> 3D from the DEPTH BUFFER of the same render. Returns (P, hit).

    WHY NOT RAY-CAST. Ray-casting means building a second acceleration structure
    over the mesh and intersecting it, which is a second geometry path: it agrees
    with the picture because it is the same mesh seen through the same camera, but
    nothing makes it agree. The depth buffer IS the rasterisation that produced
    the pixels, so the surface found here is by construction the surface the
    landmarker was looking at. It is also free, where the ray-cast was not.

    Precision: the buffer is float32 view-space z, so unlike the 8-bit depth
    round-trip the reference implementation uses to enable inpainting, nothing is
    quantised. Background reads as inf and is reported as a miss rather than
    filled in -- an inpainted depth would invent geometry and hand back a
    plausible, wrong 3D point with nothing marking it.

    `max_jump` guards the one real hazard of sampling a depth buffer at
    sub-pixel positions: bilinear interpolation ACROSS A SILHOUETTE blends a near
    surface with a far one and returns a depth that lies in empty space between
    them. Where the four neighbours disagree by more than this fraction, the
    nearest sample is taken instead of a blend.
    """
    depth = np.asarray(depth, float)
    h, w = depth.shape
    uv = np.atleast_2d(np.asarray(uv, float))
    x0 = np.clip(np.floor(uv[:, 0]).astype(int), 0, w - 2)
    y0 = np.clip(np.floor(uv[:, 1]).astype(int), 0, h - 2)
    fx, fy = uv[:, 0] - x0, uv[:, 1] - y0
    q = np.stack([depth[y0, x0], depth[y0, x0 + 1],
                  depth[y0 + 1, x0], depth[y0 + 1, x0 + 1]], axis=1)
    wts = np.stack([(1 - fx) * (1 - fy), fx * (1 - fy),
                    (1 - fx) * fy, fx * fy], axis=1)
    good = np.isfinite(q)
    hit = good.any(axis=1)
    z = np.full(len(uv), np.nan)
    qf = np.where(good, q, np.nan)
    near = np.nanmin(np.where(good, q, np.inf), axis=1, initial=np.inf)
    far = np.nanmax(np.where(good, q, -np.inf), axis=1, initial=-np.inf)
    blend = hit & good.all(axis=1) & ((far - near) <= max_jump * np.maximum(near, 1e-9))
    with np.errstate(invalid="ignore"):
        z[blend] = (q[blend] * wts[blend]).sum(axis=1)
    # Anything not safely blendable takes its nearest-neighbour sample: the
    # closest of the four that actually has geometry.
    rest = hit & ~blend
    if rest.any():
        pick = np.nanargmin(np.where(good[rest], np.abs(qf[rest] - near[rest, None]),
                                     np.nan), axis=1)
        z[rest] = q[rest, pick]
    P = np.full((len(uv), 3), np.nan)
    Pc = np.stack([(uv[hit, 0] - cam.cx) / cam.fx * z[hit],
                   (uv[hit, 1] - cam.cy) / cam.fy * z[hit],
                   z[hit]], axis=1)
    P[hit] = (Pc - cam.t) @ cam.R        # camera -> world; R is orthonormal
    return P, hit


def visible(depth, P, cam, tol=0.012):
    """Which 3D points the camera can actually see, from the depth buffer.

    A point is visible when the surface drawn at its pixel is at the point's own
    distance; if the buffer holds something nearer, the point is behind it. The
    old version cast a ray per landmark into a separate scene, which is the
    second geometry path this package exists to avoid -- and it had to build an
    acceleration structure per pose to do it.
    """
    P = np.asarray(P, float)
    uv = cam.project(P)
    H, W = depth.shape
    x = np.clip(np.round(uv[:, 0]).astype(int), 0, W - 1)
    y = np.clip(np.round(uv[:, 1]).astype(int), 0, H - 1)
    drawn = depth[y, x]
    own = (P @ cam.R.T + cam.t)[:, 2]
    inside = ((uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H))
    return inside & np.isfinite(drawn) & (own - drawn < tol)


def reseat(P, depth, cam, ext, tol):
    """Pull points that sit FAR off the drawn surface back onto it.

    A triangulated landmark is a free 3D point: the rays decide where it goes and
    nothing requires it to lie on the ear. Measured over four heads, the median
    lands on the surface (+0.33% of ear extent, 47% inside / 53% outside, so no
    systematic push) but the p90 is 22% -- one landmark in ten floats well clear
    of the ear or sits buried in the head, and those cluster on the helix rim.

    PARTIAL on purpose. Re-seating everything fixes the surface (p90 19.4% ->
    0.10%) and costs 45% of triangulation's reprojection gain, because it also
    moves points that were only a per-cent or two out -- inside the noise of the
    surface itself. Only points beyond `tol` are moved; the rest keep the depth
    the rays gave them.

    The move is along the viewing axis: same pixel, the depth the renderer drew
    there. Returns (points, moved mask).
    """
    P = np.asarray(P, float).copy()
    uv = cam.project(P)
    H, W = depth.shape
    x = np.clip(np.round(uv[:, 0]).astype(int), 0, W - 1)
    y = np.clip(np.round(uv[:, 1]).astype(int), 0, H - 1)
    drawn = depth[y, x]
    own = (P @ cam.R.T + cam.t)[:, 2]
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    far = inside & np.isfinite(drawn) & (np.abs(drawn - own) > tol * ext)
    if far.any():
        Q, hit = backproject(depth, uv[far], cam)
        idx = np.flatnonzero(far)
        good = hit & np.isfinite(Q).all(axis=1)
        P[idx[good]] = Q[good]
        far[idx[~good]] = False
    return P, far
