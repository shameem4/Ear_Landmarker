#!/usr/bin/env python3
"""Recover the AudioEar3D photo camera by matching features, then PnP.

Everything tried before matched GLOBAL cues -- silhouette overlap, whole-image
colour correlation -- and both were measured not to determine the pose. The
silhouette has a broad flat maximum: with the handedness fixed and colour
ignored it reaches 0.84 to 0.93 IoU at yaws of 47, 107 and 127 degrees on ears
photographed the same way. Colour managed NCC 0.09 to 0.46.

Local correspondence is a different mechanism. Each match is an independent
2D-to-3D constraint, RANSAC discards the wrong ones rather than averaging them
in, and the pose comes out in closed form -- there is no objective to collapse
and no flat maximum to wander across.

The data suits it unusually well: the AudioEar3D clouds carry PER-POINT RGB
sampled from the very photo we are localising, so a render of the cloud should
look almost like the photo rather than like a synthetic ear.

Intrinsics are unknown, so the focal length is swept and the value with the
most RANSAC inliers is kept -- the inlier count is a far better behaved
objective than any of the global scores, because a wrong focal simply fails to
find a consistent pose.

    python scripts/pnp_camera_recovery.py --subject 001 --side left
"""
import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

AE = pathlib.Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/clean_3d_data/AudioEar3D")


def load_cloud(path):
    import trimesh
    c = trimesh.load(path, process=False)
    raw = c.metadata["_ply_raw"]["vertex"]["data"]
    return (np.asarray(c.vertices, float),
            np.asarray(raw[["nx", "ny", "nz"]].tolist(), float),
            np.asarray(raw[["red", "green", "blue"]].tolist(), np.uint8))


def basis(yaw, pitch, roll=0.0):
    y, p, r = np.radians([yaw, pitch, roll])
    d = np.array([np.cos(p) * np.sin(y), np.sin(p), np.cos(p) * np.cos(y)])
    a = np.array([0, 1.0, 0]) if abs(d[1]) < 0.95 else np.array([1.0, 0, 0])
    u = np.cross(a, d)
    u /= np.linalg.norm(u)
    v = -np.cross(d, u)
    return (np.cos(r) * u + np.sin(r) * v, -np.sin(r) * u + np.cos(r) * v, d)


def render_cloud(V, N, C, yaw, pitch, size):
    """RGB render plus, per pixel, WHICH 3D point drew it."""
    u, v, d = basis(yaw, pitch)
    vis = (N @ d) < -0.1
    P = V[vis]
    idx = np.flatnonzero(vis)
    X, Y, Z = P @ u, P @ v, P @ d
    s = 0.92 * size / max(np.ptp(X), np.ptp(Y))
    px = (X - X.mean()) * s + size / 2
    py = (Y - Y.mean()) * s + size / 2
    xi, yi = np.round(px).astype(int), np.round(py).astype(int)
    ok = (xi >= 0) & (xi < size) & (yi >= 0) & (yi < size)
    xi, yi, Z, idx = xi[ok], yi[ok], Z[ok], idx[ok]
    col = C[vis][ok]
    order = np.argsort(-Z)                       # far first, near overwrites
    img = np.zeros((size, size, 3), np.uint8)
    who = np.full((size, size), -1, np.int64)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            x2, y2 = xi[order] + dx, yi[order] + dy
            k = (x2 >= 0) & (x2 < size) & (y2 >= 0) & (y2 < size)
            img[y2[k], x2[k]] = col[order][k]
            who[y2[k], x2[k]] = idx[order][k]
    return img, who


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subject", default="001")
    p.add_argument("--side", default="left")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--yaws", type=float, nargs="*",
                   default=list(range(-180, 180, 30)))
    p.add_argument("--pitches", type=float, nargs="*", default=[-20, 0, 20])
    p.add_argument("--ratio", type=float, default=0.9)
    p.add_argument("--min-inliers", type=int, default=12)
    p.add_argument("--save", default="")
    a = p.parse_args()

    import cv2
    from PIL import Image

    d = AE / a.subject
    photo = np.asarray(Image.open(d / f"{a.side}.jpg").convert("RGB"))
    H, W = photo.shape[:2]
    V, N, C = load_cloud(d / f"{a.side}.ply")
    print(f"{a.subject}/{a.side}: photo {W}x{H}, cloud {len(V)} points")

    # LOW CONTRAST THRESHOLD on purpose. Ear skin is nearly textureless and
    # these photos are soft -- laplacian variance 63 where a sharp photo is
    # over 500 -- so the default 0.04 finds 61 keypoints on the whole ear.
    sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.004,
                           edgeThreshold=20)
    gph = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY)
    # the photo is small; upsample so SIFT has scale space to work with
    up = 4
    gph = cv2.resize(gph, (W * up, H * up), interpolation=cv2.INTER_CUBIC)
    kp1, des1 = sift.detectAndCompute(gph, None)
    print(f"  photo keypoints {len(kp1)} (upsampled x{up})")
    bf = cv2.BFMatcher()

    best = None
    for yaw in a.yaws:
        for pitch in a.pitches:
            img, who = render_cloud(V, N, C, yaw, pitch, a.size)
            g2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            kp2, des2 = sift.detectAndCompute(g2, None)
            if des2 is None or len(kp2) < 20:
                continue
            m = bf.knnMatch(des1, des2, k=2)
            good = [x for x, y in m if x.distance < a.ratio * y.distance]
            if len(good) < a.min_inliers:
                continue
            pts2d, pts3d = [], []
            for g in good:
                x2, y2 = kp2[g.trainIdx].pt
                j = who[min(int(round(y2)), a.size - 1),
                        min(int(round(x2)), a.size - 1)]
                if j < 0:
                    continue
                pts2d.append(np.array(kp1[g.queryIdx].pt) / up)
                pts3d.append(V[j])
            if len(pts2d) < a.min_inliers:
                continue
            pts2d = np.asarray(pts2d, np.float32)
            pts3d = np.asarray(pts3d, np.float32)
            for f in (0.6, 1.0, 1.5, 2.5, 4.0):
                K = np.array([[f * W, 0, W / 2], [0, f * W, H / 2], [0, 0, 1]],
                             np.float32)
                ok, rvec, tvec, inl = cv2.solvePnPRansac(
                    pts3d, pts2d, K, None, reprojectionError=3.0,
                    iterationsCount=5000, flags=cv2.SOLVEPNP_EPNP)
                n = 0 if inl is None else len(inl)
                if ok and (best is None or n > best[0]):
                    pr, _ = cv2.projectPoints(pts3d[inl[:, 0]], rvec, tvec, K, None)
                    err = float(np.linalg.norm(
                        pr.reshape(-1, 2) - pts2d[inl[:, 0]], axis=1).mean())
                    best = (n, yaw, pitch, f, err, rvec, tvec, K, len(pts2d))
    if best is None:
        print("\nno pose found: no view produced enough consistent matches")
        return
    n, yaw, pitch, f, err, rvec, tvec, K, ncorr = best
    print(f"\nbest: render yaw {yaw:+.0f} pitch {pitch:+.0f}, focal {f:.1f}*W")
    print(f"  {n} RANSAC inliers of {ncorr} correspondences "
          f"({100*n/ncorr:.0f}%), reprojection {err:.2f} px on a {W}px image")
    good = n >= 20 and err < 3.0
    print(f"  -> {'USABLE' if good else 'WEAK'}; "
          f"{'project the landmarks and check' if good else 'not a pose to trust'}")
    if a.save:
        np.savez(a.save, rvec=rvec, tvec=tvec, K=K, inliers=n, reproj=err)
        print(f"wrote {a.save}")


if __name__ == "__main__":
    main()
