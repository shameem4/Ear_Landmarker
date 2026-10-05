"""Render real 3D ear geometry at varied yaw as an extra training source.

WHY THIS AND NOT MORE PERSPECTIVE AUGMENTATION. data/dataset.py simulates yaw
with a homography, which cannot produce parallax or self-occlusion. Measured, it
correlates with a true rotation at only 0.47 (30 deg) to 0.15 (65 deg), and a
model trained with +/-65 deg of it is no better under real 3D rotation than one
trained with none (+6.2%, t=+2.53 -- slightly worse). Renders supply what the
warp structurally cannot.

WHERE THE LABELS COME FROM, and why this is not self-distillation. The model is
run face-on and its 55 points are ray-cast onto the mesh to give exact 3D
landmarks. Rotating mesh and landmarks together then yields correct labels at
every other pose: the geometry carries the label, not the model. The model's
competence at its EASIEST pose is transferred to poses where it is weak.

THE EAR IS DEFINED BY ITS LANDMARKS, not by a region of mesh vertices. An earlier
version selected "ear vertices" by projecting the mesh into the detector's 2D box
and keeping a depth window, and used that set for the plane fit, the framing and
the ray-cast target. It was a persistent source of silent failure: a 2D box
selects the whole column through the skull, so the window had to be tight, and on
some subjects a wider box pulled in nearer geometry, raised the frontmost depth
and excluded the ear altogether -- on HUTUBS pp16's second ear the 2.5x region
came out SMALLER than the 1.0x one (8139 vertices against 12226) and 4 of 55 rays
hit. The 55 back-projected landmarks are a better definition of the ear than any
vertex rule: they are on it by construction.

So this runs in two passes:
  1. the detector's own view direction gives a provisional face-on frame. Frame
     the crop from the detector's 2D box, predict, ray-cast onto the WHOLE mesh.
  2. fit the pinna plane to those 55 points, re-render in that frame, and
     re-predict. The second pass is the one that produces the labels.

WHY THE SECOND PASS. The pinna's plane is not the head's lateral plane: over 14
HUTUBS subjects they differ by 8.1 deg on average (sd 3.9, range 1-13), varying
per subject. Labelling along the detector's axis-aligned direction would bake a
subject-dependent tilt into the ground truth.

WHAT THIS CANNOT FIX: a systematic error the model makes face-on is baked into
every pose. This widens pose coverage; it does not improve on the model's own
face-on accuracy. Nor does it solve the scalp problem -- a ray passing just
outside the pinna hits the head behind it, and nothing here measures depth
against anything that knows where the ear ends. See scripts/view_backprojection.py.

Appearance choices, all measured (see RESULTS.md):
  - lighting azimuth is RANDOMISED per render. It is the single largest factor in
    how photo-like a render is, and a fixed light would rake across the ear as it
    turns, confounding appearance with pose. The arc below was re-measured after
    the camera fix; RESULTS.md's -40 deg optimum predates it and is about a
    different axis.
  - skin tone randomised over Fitzpatrick I-VI; second largest factor (0.020).
  - no freckles: they COST confidence (0.426 -> 0.419).
  - occupancy is drawn per sample from N(0.777, 0.093), matching the real
    corpus. A constant occupancy puts 99% of the real test set outside the
    training range -- that bug has been made once already.

Usage:
    python scripts/ingest_render3d.py --ears 150 --renderer blender
    python scripts/ingest_render3d.py --ears 20 --renderer open3d   # fast check
"""

from __future__ import annotations

import argparse
import copy
import csv
import glob
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from render3d_ears import EAR3D_DIR, EYE_Z, VFOV, project, render, rot  # noqa: E402
from eval_pose_consistency import crop_to_full, raycast, visible        # noqa: E402
from eval_test import best_ckpt                                         # noqa: E402
from inference import LandmarkPredictor                                 # noqa: E402
from skin import SKIN_TONES, apply_skin                                 # noqa: E402

OUT = ROOT / "data" / "render3d"
SIZE = 600
MEAN_OCC, SD_OCC = 0.777, 0.093
OCC_CLIP = (0.55, 0.95)
HUTUBS = EAR3D_DIR / "3D head meshes"
SONICOM = EAR3D_DIR / "sonicom_raw_headtorso"

# Key-light azimuth arc, degrees, re-measured in the corrected camera frame over
# two heads: +20 is best (0.407 mean confidence) falling monotonically to 0.387 at
# +80, with -40 -- RESULTS.md's figure, measured about the wrong axis -- at 0.366.
AZIMUTH_ARC = (-10.0, 60.0)

# Change of basis from this project's camera frame into the one Blender renders.
# Blender's camera sits at +Y looking down -Y with +Z up, and bpy.ops.wm.ply_import
# applies no axis conversion, so an untransformed mesh is rendered from a
# different axis entirely: measured, our +Z lands UP in the image and our +X lands
# LEFT -- a 90 deg rotation plus a mirror. This maps our (x, y, z) to Blender's
# (-x, z, y), after which markers land within 0.5 px of where project() puts them.
# Without it every Blender-rendered sample carried labels for a different view.
TO_BLENDER = np.array([[-1.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0],
                       [0.0, 1.0, 0.0]])

# The Blender key light sits at dist*1.1, and the 5 W optimum in RESULTS.md was
# swept with dist=2.2. The camera must be at EYE_Z for project() to be correct,
# so the energy is scaled by the inverse square to keep the swept lighting.
LIGHT_REF_DIST = 2.2
KEY_W, FILL_W = 5.0, 1.5

# World light. The single setting that decides whether the ear reads as a surface
# or a flat blob: it lights from every direction at once, so it fills exactly the
# shadows that carry relief. Calibrated against 300 real crops from data/manual
# rather than against a clay render, which is itself only 0.47x their relief.
# As a fraction of the real median: 0.20 -> 0.39x, 0.12 -> 0.49x, 0.06 -> 0.64x.
# At the original 0.35 the pipeline half failed to find the ear at all (detector
# confidence 0.40, 27.5 of 55 rays), against 0.95 and 55/55 here. Flat renders
# were costing labels, not just looks.
AMBIENT = 0.06


def load_head(path: str):
    """Head mesh, centred and scaled to unit radius. These are already meshes."""
    m = o3d.io.read_triangle_mesh(str(path))
    if len(m.vertices) == 0:
        return None
    V = np.asarray(m.vertices)
    V = (V - V.mean(0)) / np.abs(V - V.mean(0)).max()
    m.vertices = o3d.utility.Vector3dVector(V)
    m.compute_vertex_normals()
    return m


def _shot(mesh, front, up, dist, centre, size):
    r = o3d.visualization.rendering.OffscreenRenderer(size, size)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    r.scene.add_geometry("m", mesh, mat)
    r.scene.set_background([0.5, 0.5, 0.5, 1.0])
    r.scene.scene.set_sun_light([-0.4, -0.3, -0.9], [1, 1, 1], 100000)
    r.scene.scene.enable_sun_light(True)
    eye = np.asarray(centre, float) + np.asarray(front, float) * dist
    r.setup_camera(VFOV, np.asarray(centre, np.float32), eye.astype(np.float32),
                   np.asarray(up, np.float32))
    img = np.asarray(r.render_to_image())[:, :, :3]
    del r
    return img


def find_ears(mesh, det, probe_size=400):
    """Locate ears by sweeping the detector around the head.

    Geometric rules do not transfer between datasets -- HUTUBS and SONICOM store
    heads on different axes -- so the ear is found by what an ear detector sees.
    It works on these renders (0.84 confidence on clay, 0.91 with skin); it fails
    only on SONICOM's *graded* meshes, whose pinna is a featureless blob.
    """
    out = []
    for axis in (0, 1, 2):
        for sgn in (1, -1):
            front = np.zeros(3); front[axis] = sgn
            up = np.array([0.0, 0.0, 1.0]) if axis != 2 else np.array([0.0, 1.0, 0.0])
            img = _shot(mesh, front, up, 2.4, np.zeros(3), probe_size)
            b = det.detect(img)
            if not len(b):
                continue
            h, w = b[0, 2] - b[0, 0], b[0, 3] - b[0, 1]
            if (h * w) / (probe_size ** 2) > 0.25:     # a whole-head box is a false positive
                continue
            out.append((float(b[0, 4]), front, up, b[0, :4].copy()))
    out.sort(key=lambda t: -t[0])
    return out[:2]


def frame_from(normal, up, front=None):
    """Rotation taking `normal` onto +Z (toward the camera), keeping `up` near +Y."""
    n = np.asarray(normal, float)
    if front is not None and n @ np.asarray(front, float) < 0:
        n = -n
    z = n / max(np.linalg.norm(n), 1e-9)
    y = np.asarray(up, float) - z * (np.asarray(up, float) @ z)
    ny = np.linalg.norm(y)
    if ny < 1e-6:
        return None
    y /= ny
    return np.stack([np.cross(y, z), y, z])            # world -> camera frame


def plane_normal(points):
    """Surface normal of the best-fit plane through the landmarks."""
    P = np.asarray(points, float)
    _, _, vt = np.linalg.svd(P - P.mean(0), full_matrices=False)
    return vt[2]


def in_frame(mesh, R, C, scale):
    """A copy of the mesh rotated into frame R about C and scaled by `scale`."""
    g = copy.deepcopy(mesh)
    g.vertices = o3d.utility.Vector3dVector(((np.asarray(mesh.vertices) - C) @ R.T) / scale)
    g.compute_vertex_normals()
    return g


def crop_from_box(img, box, occ, out=192, size=SIZE):
    """Crop a rendered frame so a 2D box fills `occ` of the result.

    Used only for the FIRST pass, where there is nothing in 3D yet to frame on.
    """
    ymin, xmin, ymax, xmax = box
    side = max(ymax - ymin, xmax - xmin) / occ
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    return _paste(img, cx, cy, side, out, size)


def crop_from_points(img, pts3d, occ, out=192, size=SIZE):
    """Crop so the projected 3D points fill `occ` of the result.

    `pts3d` are the landmarks: they define the ear exactly, which is why no mesh
    region is needed. The head is deliberately left in the render for context --
    a floating ear teaches the model that a hard silhouette marks its boundary.
    """
    uv = project(np.asarray(pts3d, float), size)
    side = max(np.ptp(uv[:, 0]), np.ptp(uv[:, 1])) / occ
    cx = (uv[:, 0].min() + uv[:, 0].max()) / 2
    cy = (uv[:, 1].min() + uv[:, 1].max()) / 2
    return _paste(img, cx, cy, side, out, size)


def _paste(img, cx, cy, side, out, size):
    l, t, s = int(round(cx - side / 2)), int(round(cy - side / 2)), int(round(side))
    if s < 48:
        return None, None
    cv = Image.new("RGB", (s, s), (128, 128, 128))     # grey-128 pad, as in training
    src = Image.fromarray(img)
    sx0, sy0 = max(0, l), max(0, t)
    sx1, sy1 = min(size, l + s), min(size, t + s)
    if sx1 <= sx0 or sy1 <= sy0:
        return None, None
    cv.paste(src.crop((sx0, sy0, sx1, sy1)), (sx0 - l, sy0 - t))
    return cv.resize((out, out), Image.BILINEAR), (l, t, s, out)


def sample_occ(rng):
    return float(np.clip(rng.normal(MEAN_OCC, SD_OCC), *OCC_CLIP))


def render_frame(mesh, renderer="open3d", tone=None, light_az=None, script=None,
                 size=SIZE):
    """Face-on render by either renderer. Returns HxWx3 uint8, or None."""
    if renderer == "blender":
        return _blender_render(mesh, tone, light_az, size, script)
    return render(mesh, 0, 0, size=size)[:, :, :3]


def _blender_render(mesh, tone, light_az, size, script):
    """Cycles skin render, in a camera frame that matches project().

    TO_BLENDER and dist=EYE_Z are both load-bearing: without them the render is a
    different view of the mesh than the labels describe, and the error is silent.
    """
    falloff = (EYE_Z / LIGHT_REF_DIST) ** 2
    g = copy.deepcopy(mesh)
    g.vertices = o3d.utility.Vector3dVector(np.asarray(g.vertices) @ TO_BLENDER.T)
    g.compute_vertex_normals()
    with tempfile.TemporaryDirectory() as td:
        mp = os.path.join(td, "m.ply")
        o3d.io.write_triangle_mesh(mp, g)
        args = dict(mesh=mp, ear=[0.0, 0.0, 0.0], dist=float(EYE_Z),
                    tone=[float(v) for v in tone],
                    out=os.path.join(td, "r"), size=int(size), samples=48,
                    key_energy=KEY_W * falloff, fill_energy=FILL_W * falloff,
                    ambient=AMBIENT, key_size=0.5,
                    key_azimuth=float(light_az), key_elevation=30.0)
        ap = os.path.join(td, "a.json")
        json.dump(args, open(ap, "w"))
        try:
            subprocess.run(["blender", "-b", "-P", script, "--", ap],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=900, check=False)
            return np.asarray(Image.open(os.path.join(td, "r.png")).convert("RGB"))
        except Exception:
            return None


def label_ear(mesh, front, up, det, pred):
    """Two-pass face-on labelling. Returns (mesh in pinna frame, P3, hit, conf).

    Pass 1 uses the detector's axis-aligned view just to get landmarks; pass 2
    repeats it in the plane those landmarks define. Ray-casting is against the
    WHOLE mesh in both -- there is no ear region, and no region to get wrong.
    """
    # --- pass 1: provisional frame from the detector's own view direction ----
    R = frame_from(front, up, front=front)
    if R is None:
        return None, None, None, None, "degenerate detector frame"
    V = np.asarray(mesh.vertices)
    C = V.mean(0)
    g = in_frame(mesh, R, C, 1.0)
    img = render(g, 0, 0, size=SIZE)[:, :, :3]
    b = det.detect(img)
    if not len(b):
        return None, None, None, None, "no ear in the face-on render"
    crop, box = crop_from_box(img, b[0, :4], MEAN_OCC)
    if crop is None:
        return None, None, None, None, "pass-1 crop too small"
    lm = np.asarray(pred.predict(np.asarray(crop)), float)
    P1, hit1 = raycast(g, crop_to_full(lm, box))
    if hit1.sum() < 45:
        return None, None, None, None, f"pass 1: only {int(hit1.sum())}/55 rays hit"

    # --- pass 2: the plane those landmarks define ----------------------------
    P1w = (P1[hit1] @ R) + C                           # back to world coordinates
    R2 = frame_from(plane_normal(P1w), up, front=front)
    if R2 is None:
        return None, None, None, None, "degenerate pinna frame"
    C2 = P1w.mean(0)
    scale = float(np.abs((P1w - C2) @ R2.T).max())
    if not np.isfinite(scale) or scale <= 0:
        return None, None, None, None, "degenerate ear scale"
    g2 = in_frame(mesh, R2, C2, scale)
    ear2 = ((P1w - C2) @ R2.T) / scale                 # the pass-1 ear, reframed
    img2 = render(g2, 0, 0, size=SIZE)[:, :, :3]
    crop2, box2 = crop_from_points(img2, ear2, MEAN_OCC)
    if crop2 is None:
        return None, None, None, None, "pass-2 crop too small"
    lm2, conf = pred.predict(np.asarray(crop2), with_confidence=True)
    P3, hit = raycast(g2, crop_to_full(np.asarray(lm2, float), box2))
    return g2, P3, hit, float(np.mean(conf)), None


def main() -> None:
    p = argparse.ArgumentParser(description="Render 3D ears at varied yaw for training")
    p.add_argument("--ears", type=int, default=150)
    p.add_argument("--renderer", choices=["open3d", "blender"], default="blender")
    p.add_argument("--yaws", type=float, nargs="*",
                   default=[-40, -25, -12, 0, 12, 25, 40])
    p.add_argument("--pitches", type=float, nargs="*", default=[0.0])
    p.add_argument("--label-run", default="manual_occ_s42")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quality", type=int, default=95)
    p.add_argument("--min-conf", type=float, default=0.25)
    args = p.parse_args()

    from inference import BLAZEEAR_DIR, DETECTOR_WEIGHTS, EarDetector
    det = EarDetector(BLAZEEAR_DIR / DETECTOR_WEIGHTS, "cpu", 0.5)
    pred = LandmarkPredictor(best_ckpt(args.label_run), "cuda")
    bscript = str(ROOT / "scripts" / "blender_skin_render.py")

    heads = sorted(glob.glob(str(HUTUBS / "*.ply"))) + sorted(glob.glob(str(SONICOM / "*.stl")))
    rng = np.random.default_rng(args.seed)
    (OUT / "images").mkdir(parents=True, exist_ok=True)

    rows, lms = [], []
    n_ear = n_rejected = 0
    for hp in heads:
        if n_ear >= args.ears:
            break
        mesh = load_head(hp)
        if mesh is None:
            continue
        for conf0, front, up, _ in find_ears(mesh, det):
            if n_ear >= args.ears:
                break
            base, P3, hit, conf, why = label_ear(mesh, front, up, det, pred)

            # QUALITY GATE. Every pose inherits this one labelling, so a bad
            # face-on result poisons the whole ear. The checks are on the
            # labelling itself -- ray hits, confidence, and whether the landmarks
            # stay in their own crop -- not on agreement with a mesh region,
            # which is what the old scale/centre checks compared against.
            if why is None:
                if hit.sum() < 45:
                    why = f"pass 2: only {int(hit.sum())}/55 rays hit"
                elif conf < args.min_conf:
                    why = f"confidence {conf:.2f}"
            if why is not None:
                n_rejected += 1
                print(f"    rejected {os.path.basename(hp)}: {why}", flush=True)
                continue
            n_ear += 1

            tone = SKIN_TONES[rng.integers(len(SKIN_TONES))]
            if args.renderer == "open3d":
                base = apply_skin(base, seed=int(rng.integers(1 << 30)))

            for pitch in args.pitches:
                for yaw in args.yaws:
                    Rr = rot(float(yaw), float(pitch))
                    mr = copy.deepcopy(base)
                    mr.rotate(Rr, center=(0, 0, 0))
                    Pr = P3 @ Rr.T
                    az = float(rng.uniform(*AZIMUTH_ARC))   # randomised; see docstring
                    img = render_frame(mr, args.renderer, tone, az, bscript)
                    if img is None:
                        continue
                    occ = sample_occ(rng)
                    out = crop_from_points(img, Pr[hit], occ)
                    if out[0] is None:
                        continue
                    crop, box2 = out
                    vis = hit & visible(mr, Pr)
                    uv = project(Pr, SIZE)
                    lm = np.stack([(uv[:, 0] - box2[0]) / box2[2],
                                   (uv[:, 1] - box2[1]) / box2[2]], axis=1)
                    # Occluded or un-raycast points are written OUT OF FRAME, so
                    # data/dataset.py's visibility mask drops them instead of
                    # supervising a position the model cannot see.
                    lm[~vis] = -1.0
                    if (vis & ((lm >= 0) & (lm <= 1)).all(1)).sum() < 20:
                        continue
                    name = f"r3d_{n_ear:04d}_y{int(yaw):+04d}_p{int(pitch):+03d}.jpg"
                    crop.save(OUT / "images" / name, quality=args.quality)
                    rows.append({"idx": len(rows), "image_file": f"images/{name}",
                                 "source": "render3d", "original_path": os.path.basename(hp),
                                 "width": crop.width, "height": crop.height})
                    lms.append(lm.astype(np.float32))
            print(f"  [{n_ear}/{args.ears}] {os.path.basename(hp)} "
                  f"det={conf0:.2f} label={conf:.2f}", flush=True)

    if not rows:
        sys.exit("nothing ingested")
    arr = np.stack(lms)
    np.save(OUT / "landmarks.npy", arr)
    fields = list(rows[0].keys())
    for fn in ("manifest.csv", "train.csv"):
        with open(OUT / fn, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(rows)
    inb = ((arr >= 0) & (arr <= 1)).all(axis=2)
    span = np.array([max(np.ptp(a[m, 0]), np.ptp(a[m, 1])) if m.sum() > 5 else np.nan
                     for a, m in zip(arr, inb)])
    print(f"\n{n_ear} ears kept, {n_rejected} rejected by the quality gate "
          f"({100*n_rejected/max(n_ear+n_rejected,1):.0f}%) -> {len(rows)} renders")
    print(f"visible landmarks per sample: mean {inb.sum(1).mean():.1f} of 55")
    print(f"occupancy: mean {np.nanmean(span):.3f} sd {np.nanstd(span):.3f} "
          f"(real corpus 0.778 / 0.097)")
    print(f"\nwrote {OUT}")
    print("train with:  python train.py --data-dir data/manual --render3d-ratio 0.5")


if __name__ == "__main__":
    main()
