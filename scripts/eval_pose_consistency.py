"""How far do a model's landmarks drift when the ear really rotates?

THE PROBLEM THIS SOLVES. Measuring pose behaviour needs landmark ground truth at
a known pose, and two routes to it failed:

  edge response (scripts/eval_gt_free.py's metric) -- works on photographs, where
    the ear sits against a head. On an isolated render against flat grey the
    silhouette is an enormous gradient, so a foreshortened ear rewards any
    contour hugging its outline. It produced the physically impossible result
    that rotating an ear IMPROVES contour quality.

  lifting AudioEar3D's own annotations -- the dataset annotates a photograph and
    ships a point cloud of the same ear, with no camera relating them. Fitting
    the cloud's silhouette to the photo's ear mask reaches ~0.90 IoU and still
    leaves 12px reprojection error, 3.7% of the crop diagonal, against the ~2.9%
    NME it would be judging. Looked fine, measured nothing.

WHAT WORKS: define the landmarks on a render WE produce, so the camera is exact
and no registration is needed. Round-trip error is 0.006% of ear extent against
the 3.7% above.

  1. render the coloured mesh head-on, run the landmarker, take its 55 points
  2. ray-cast them onto the mesh -> exact 3D landmarks
  3. rotate mesh AND landmarks to pose theta, re-render
  4. run the landmarker again; compare to the projected 3D landmarks

Points that rotate out of sight are masked per pose and per model -- scoring a
prediction against a landmark the model cannot see measures nothing.

WHAT THIS MEASURES: self-consistency under pose. Each model is its own reference,
which makes it comparable BETWEEN models ("whose landmarks move least as the ear
turns") and needs no human annotation.

WHAT IT CANNOT MEASURE: accuracy. A model that puts a point in the wrong place at
pose zero bakes that into its own reference and is never charged for it. It also
rewards stability per se, so a model placing points more sharply has a harder
reference to re-hit and can look worse while being better. Do not read this as a
quality ranking.

Usage:
    python scripts/eval_pose_consistency.py --ears 28 v6_persp65 manual_occ_s42
"""

from __future__ import annotations

import argparse
import copy
import pickle
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from render3d_ears import ears, load_ear, project, render, rot, unproject_rays  # noqa: E402
from eval_test import best_ckpt                                                 # noqa: E402
from inference import LandmarkPredictor                                         # noqa: E402

SIZE = 600
TRAIN_OCC = 0.777


def raycast(mesh, uv, size=SIZE):
    """Exact 2D -> 3D. Returns (points, hit mask)."""
    t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(t)
    o, d = unproject_rays(uv, size)
    ans = scene.cast_rays(o3d.core.Tensor(np.hstack([o, d]).astype(np.float32)))
    dist = ans["t_hit"].numpy()
    hit = np.isfinite(dist)
    P = np.full((len(uv), 3), np.nan)
    P[hit] = o[hit] + d[hit] * dist[hit, None]
    return P, hit


def framed_crop(mesh, out=192, size=SIZE):
    """Render and crop so the ear fills the occupancy the model was trained at."""
    img = render(mesh, 0, 0, size=size)[:, :, :3]
    uv = project(np.asarray(mesh.vertices), size)
    side = max(np.ptp(uv[:, 0]), np.ptp(uv[:, 1])) / TRAIN_OCC
    cx = (uv[:, 0].min() + uv[:, 0].max()) / 2
    cy = (uv[:, 1].min() + uv[:, 1].max()) / 2
    l, t, s = int(round(cx - side / 2)), int(round(cy - side / 2)), int(round(side))
    if s < 32:
        return None, None
    cv = Image.new("RGB", (s, s), (128, 128, 128))
    src = Image.fromarray(img)
    sx0, sy0 = max(0, l), max(0, t)
    sx1, sy1 = min(size, l + s), min(size, t + s)
    if sx1 <= sx0 or sy1 <= sy0:
        return None, None
    cv.paste(src.crop((sx0, sy0, sx1, sy1)), (sx0 - l, sy0 - t))
    return cv.resize((out, out), Image.BILINEAR), (l, t, s, out)


def crop_to_full(pts, box):
    l, t, s, out = box
    return np.stack([pts[:, 0] / out * s + l, pts[:, 1] / out * s + t], axis=1)


def visible(mesh_rot, P3, tol=0.012):
    """Is the camera ray to each landmark unobstructed?"""
    Q, hit = raycast(mesh_rot, project(P3, SIZE))
    d = np.full(len(P3), np.inf)
    d[hit] = np.linalg.norm(Q[hit] - P3[hit], axis=1)
    return hit & (d < tol)


def main() -> None:
    p = argparse.ArgumentParser(description="Landmark drift under true 3D rotation")
    p.add_argument("runs", nargs="+")
    p.add_argument("--ears", type=int, default=28)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    YAWS = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50]
    PITCHES = [-30, -20, -10, 0, 10, 20, 30]
    preds = {r: LandmarkPredictor(best_ckpt(r), "cuda") for r in args.runs}

    mem = ears()
    rng = np.random.default_rng(args.seed)
    mem = [mem[i] for i in sorted(rng.choice(len(mem), min(args.ears, len(mem)), replace=False))]

    out = {"yaw": {r: {a: [] for a in YAWS} for r in args.runs},
           "pitch": {r: {a: [] for a in PITCHES} for r in args.runs},
           "visfrac": {"yaw": {a: [] for a in YAWS}, "pitch": {a: [] for a in PITCHES}}}
    ok = 0
    for n, m in enumerate(mem):
        try:
            mesh = load_ear(m)
        except Exception:
            continue
        if mesh is None:
            continue
        ref = {}
        for r in args.runs:
            crop, box = framed_crop(mesh)
            if crop is None:
                break
            lm = np.asarray(preds[r].predict(np.asarray(crop)), dtype=float)
            ref[r] = raycast(mesh, crop_to_full(lm, box))
        if len(ref) != len(args.runs):
            continue
        ok += 1
        for axis, angles in (("yaw", YAWS), ("pitch", PITCHES)):
            for a in angles:
                R = rot(a, 0) if axis == "yaw" else rot(0, a)
                mr = copy.deepcopy(mesh)
                mr.rotate(R, center=(0, 0, 0))
                crop, box = framed_crop(mr)
                if crop is None:
                    continue
                vf = None
                for r in args.runs:
                    P3, hit = ref[r]
                    Pr = P3 @ R.T
                    vis = hit & visible(mr, Pr)
                    if vis.sum() < 10:
                        continue
                    lm = np.asarray(preds[r].predict(np.asarray(crop)), dtype=float)
                    e = np.linalg.norm(crop_to_full(lm, box)[vis] - project(Pr, SIZE)[vis], axis=1)
                    out[axis][r][a].append(e.mean() / box[2])
                    vf = vis.mean() if vf is None else vf
                if vf is not None:
                    out["visfrac"][axis][a].append(vf)
        print(f"  [{n+1}/{len(mem)}] {m}", flush=True)
        # Checkpoint every ear. Poisson plus a raycasting scene per mesh is
        # memory-hungry, and an earlier run was killed at 24 of 28 having written
        # nothing at all; partial results are worth keeping.
        if args.out:
            pickle.dump({"out": out, "yaws": YAWS, "pitches": PITCHES, "n": ok},
                        open(args.out, "wb"))

    print(f"\n{ok} ears. Drift normalised by crop side; 0 deg must be ~0 by construction.\n")
    for axis, angles in (("yaw", YAWS), ("pitch", PITCHES)):
        print(f"--- {axis.upper()} ---")
        print(f"{'angle':>7s} " + " ".join(f"{r:>16s}" for r in args.runs) + f"{'visible':>9s}")
        for a in angles:
            v = [np.mean(out[axis][r][a]) if out[axis][r][a] else np.nan for r in args.runs]
            vf = np.mean(out["visfrac"][axis][a]) if out["visfrac"][axis][a] else np.nan
            print(f"{a:>5d}deg " + " ".join(f"{x:16.4f}" for x in v) + f"{100*vf:8.0f}%")
        print()
    if args.out:
        pickle.dump({"out": out, "yaws": YAWS, "pitches": PITCHES, "n": ok}, open(args.out, "wb"))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
