"""Place a 2D landmark onto the PINNA in 3D, not onto whatever the ray hits first.

THE PROBLEM. The pinna stands off the skull by ~15-20mm. A landmark predicted
slightly outside the pinna's silhouette -- which happens constantly on the helix
rim and lobe edge, where the annotation sits ON the boundary -- sends its camera
ray past the ear and onto the scalp behind. The resulting 3D point is too deep by
the whole protrusion. Rotate it and it swings far more than it should, which is
how labels end up sprawling off the ear at other poses.

THE FIX, in two parts:
  1. ray-cast against the PINNA ALONE, so the skull is not a candidate surface;
  2. for rays that still miss, take the closest point on the pinna to the ray
     rather than dropping the landmark or letting it fall through.

isolate_pinna() separates ear from skull by depth within the detector's box: the
protrusion forms a distinct front cluster, so the gap in the depth histogram is
the standoff itself.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d


def isolate_pinna(mesh, inside_mask, gap_frac=0.35):
    """Triangles of the protruding ear, split from the skull by the depth gap.

    `inside_mask` marks vertices inside the detector's ear box. Depth is +Z
    toward the camera, so the pinna is the FRONT cluster.
    """
    V = np.asarray(mesh.vertices)
    T = np.asarray(mesh.triangles)
    z = V[:, 2]
    zi = z[inside_mask]
    if len(zi) < 200:
        return None, None
    lo, hi = np.percentile(zi, [1, 99])
    hist, edges = np.histogram(zi, bins=40, range=(lo, hi))
    # Walk back from the front until the histogram thins out: that trough is the
    # standoff between pinna and scalp.
    peak = len(hist) - 1 - int(np.argmax(hist[::-1] > hist.max() * 0.15))
    cut = edges[0]
    for b in range(peak, 0, -1):
        if hist[b] < hist.max() * gap_frac:
            cut = edges[b]
            break
    keep_v = inside_mask & (z >= cut)
    if keep_v.sum() < 150:
        keep_v = inside_mask
    keep_t = keep_v[T].all(axis=1)
    if keep_t.sum() < 100:
        return None, None
    sub = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(T[keep_t]))
    sub.remove_unreferenced_vertices()
    sub.compute_vertex_normals()
    return sub, keep_v


def cast_to_pinna(pinna, origins, dirs, n_samples=48, max_t=6.0):
    """2D rays -> points on the pinna. Returns (points, hit_mask, snapped_mask).

    A ray that strikes the pinna uses that intersection. A ray that misses is
    assigned the closest point on the pinna to the ray, found by sampling along
    it and taking the nearest surface point. `snapped` marks the latter so a
    caller can treat them differently -- they are a reasonable placement, not a
    measurement.
    """
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(pinna))
    rays = o3d.core.Tensor(np.hstack([origins, dirs]).astype(np.float32))
    t_hit = scene.cast_rays(rays)["t_hit"].numpy()
    hit = np.isfinite(t_hit) & (t_hit < max_t)
    P = np.full((len(origins), 3), np.nan)
    P[hit] = origins[hit] + dirs[hit] * t_hit[hit, None]

    miss = ~hit
    snapped = np.zeros(len(origins), bool)
    if miss.any():
        V = np.asarray(pinna.vertices)
        zmid = V[:, 2].mean()
        # sample the ray across the depth range the pinna occupies
        t0 = (origins[miss, 2] - V[:, 2].max()) / np.maximum(-dirs[miss, 2], 1e-6)
        t1 = (origins[miss, 2] - V[:, 2].min()) / np.maximum(-dirs[miss, 2], 1e-6)
        ts = np.linspace(0, 1, n_samples)[None, :]
        tt = t0[:, None] + (t1 - t0)[:, None] * ts
        pts = origins[miss][:, None, :] + dirs[miss][:, None, :] * tt[:, :, None]
        q = o3d.core.Tensor(pts.reshape(-1, 3).astype(np.float32))
        res = scene.compute_closest_points(q)
        cp = res["points"].numpy().reshape(len(tt), n_samples, 3)
        d = np.linalg.norm(cp - pts, axis=2)
        best = np.argmin(d, axis=1)
        P[miss] = cp[np.arange(len(best)), best]
        snapped[miss] = True
    return P, hit, snapped
