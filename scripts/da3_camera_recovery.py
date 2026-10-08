#!/usr/bin/env python3
"""Recover the AudioEar3D photo camera by 3D-to-3D registration, not 2D-to-3D.

The earlier attempt searched for a camera by matching the cloud's colours and
silhouette against the photo. It never converged: the objective was not convex
in pose, a free scale collapsed the ear to a tenth of its size to drive the
residual down, and the recovered yaw scattered over 80 degrees across subjects
that were photographed the same way.

Metric depth changes the problem. DA3's metric model turns the photo into a
point cloud in real units, so instead of solving for a camera that projects 3D
onto 2D, we register one 3D cloud onto another -- a problem with a closed-form
inner step, a good initialisation from the ear mask, and no scale to run away
with. The camera then falls out of the transform.

DA3METRIC-LARGE is apache-2.0 despite the name; only NESTED and the plain
LARGE/GIANT are non-commercial. Verified on the hub.

    ../.venv_da3/bin/python scripts/da3_camera_recovery.py --subject 001 --side left
"""
import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

AE = pathlib.Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/clean_3d_data/AudioEar3D")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subject", default="001")
    p.add_argument("--side", default="left", choices=["left", "right"])
    p.add_argument("--model", default="depth-anything/DA3METRIC-LARGE")
    p.add_argument("--res", type=int, default=504)
    p.add_argument("--device", default="cuda")
    p.add_argument("--save", default="")
    a = p.parse_args()

    import torch
    import trimesh
    from PIL import Image
    from scipy.spatial import cKDTree
    from depth_anything_3.api import DepthAnything3
    from ear3d.wrap import rigid_fit

    d = AE / a.subject
    im = Image.open(d / f"{a.side}.jpg").convert("RGB")
    W, H = im.size
    mask = np.asarray(Image.open(d / f"mask_{a.side}.jpg").convert("L")
                      .resize((W, H))) > 128
    print(f"{a.subject}/{a.side}: photo {W}x{H}, mask covers {100*mask.mean():.0f}%")

    model = DepthAnything3.from_pretrained(a.model).to(a.device).eval()
    with torch.no_grad():
        pr = model.inference([im], process_res=a.res)
    dep = np.asarray(pr.depth)
    dep = dep[0] if dep.ndim == 3 else dep
    K = np.asarray(pr.intrinsics)
    K = K[0] if K.ndim == 3 else K
    print(f"  depth {dep.shape}, metric={bool(pr.is_metric)}, "
          f"range {dep.min():.3f} to {dep.max():.3f}")
    print(f"  intrinsics fx {K[0, 0]:.1f} fy {K[1, 1]:.1f} "
          f"cx {K[0, 2]:.1f} cy {K[1, 2]:.1f}")

    # back-project the EAR ONLY into camera coordinates
    dh, dw = dep.shape
    mm = np.asarray(Image.fromarray(mask).resize((dw, dh))) > 0
    v, u = np.nonzero(mm)
    z = dep[v, u]
    good = np.isfinite(z) & (z > 0)
    u, v, z = u[good], v[good], z[good]
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    P = np.stack([x, y, z], 1)
    print(f"  back-projected {len(P)} ear points, extent "
          f"{np.ptp(P, axis=0).round(1)}")

    cloud = trimesh.load(d / f"{a.side}.ply", process=False)
    Q = np.asarray(cloud.vertices, float)
    print(f"  scan cloud {len(Q)} points, extent {np.ptp(Q, axis=0).round(1)} mm")

    # Scale from EXTENT, then refine rigidly. The scale is never left free:
    # nearest-point distance always rewards shrinking, and letting it float is
    # what collapsed the earlier attempts to a tenth of ear size.
    k0 = float(np.ptp(Q, axis=0).max() / np.ptp(P, axis=0).max())
    X = (P - P.mean(0)) * k0 + Q.mean(0)
    tree = cKDTree(Q)
    best = None
    for yaw in range(0, 360, 45):
        t_ = np.radians(yaw)
        R0 = np.array([[np.cos(t_), 0, np.sin(t_)], [0, 1, 0],
                       [-np.sin(t_), 0, np.cos(t_)]])
        Y = (X - X.mean(0)) @ R0.T + Q.mean(0)
        for _ in range(60):
            dd, idx = tree.query(Y, workers=-1)
            keep = dd <= np.quantile(dd, 0.8)
            R, t, _ = rigid_fit(Y[keep], Q[idx][keep], with_scale=False)
            Y = Y @ R.T + t
        r = float(np.median(tree.query(Y, workers=-1)[0]))
        if best is None or r < best[0]:
            best = (r, Y, yaw)
    r, Y, yaw0 = best
    size = float(np.ptp(Q, axis=0).max())
    print(f"\nregistration: median {r:.2f} mm = {100 * r / size:.2f}% of ear "
          f"(best start yaw {yaw0})")
    print(f"  the photo cloud kept its size: "
          f"{np.ptp(Y, axis=0).max() / np.ptp(Q, axis=0).max():.3f}x the scan")
    ok = r / size < 0.05
    print(f"  -> {'USABLE' if ok else 'NOT usable'}; a camera from this is "
          f"{'worth projecting landmarks through' if ok else 'not trustworthy'}")
    if a.save:
        np.savez(a.save, photo_points=Y, scan_points=Q, residual=r,
                 intrinsics=K, depth=dep)
        print(f"wrote {a.save}")


if __name__ == "__main__":
    main()
