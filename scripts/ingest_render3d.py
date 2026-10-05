"""Render real 3D ear geometry at varied yaw as an extra training source.

WHY THIS AND NOT MORE PERSPECTIVE AUGMENTATION. data/dataset.py simulates yaw
with a homography, which cannot produce parallax or self-occlusion. Measured, it
correlates with a true rotation at only 0.47 (30 deg) to 0.15 (65 deg), and a
model trained with +/-65 deg of it is no better under real 3D rotation than one
trained with none (+6.2%, t=+2.53 -- slightly worse). Renders supply what the
warp structurally cannot.

WHERE THE LABELS COME FROM, and why this is not self-distillation. The model is
run ONCE per ear, face-on, and its 55 points are ray-cast onto the mesh to give
exact 3D landmarks. Rotating mesh and landmarks together then yields correct
labels at every other pose: the geometry carries the label, not the model. The
model's competence at its EASIEST pose is transferred to poses where it is weak.

WHY FACE-ON, AND NOT A MULTI-VIEW CONSENSUS. The landmark set is only 29% as
deep as it is wide, so a face-on view maximises in-plane separation and minimises
foreshortening. Measured deviation from an all-view consensus is 2.91% of ear
extent at face-on against 5.9% at +/-30 deg, and 82% of the 55 landmarks are
individually best face-on with none better than +/-15. Fusing views was tried
three ways and all lost to face-on alone: uniform +9.3%, per-landmark
inverse-variance weighted +5.8% (both t>2, cross-validated on held-out ears).
Weighting beat uniform fusion, so the per-landmark signal is real -- it just
cannot overcome how much better face-on is.

WHAT THIS CANNOT FIX: a systematic error the model makes face-on is baked into
every pose. This widens pose coverage; it does not improve on the model's own
face-on accuracy.

Appearance choices, all measured (see RESULTS.md):
  - lighting azimuth is RANDOMISED per render. It is the single largest factor in
    how photo-like a render is (0.041 confidence spread, about the same as the
    whole Open3D->Cycles upgrade), and a fixed light would rake across the ear as
    it turns, confounding appearance with pose.
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

from render3d_ears import EAR3D_DIR, project, render, rot          # noqa: E402
from eval_pose_consistency import crop_to_full, raycast, visible    # noqa: E402
from eval_test import best_ckpt                                     # noqa: E402
from inference import LandmarkPredictor                             # noqa: E402
from skin import SKIN_TONES, apply_skin                             # noqa: E402

OUT = ROOT / "data" / "render3d"
SIZE = 600
MEAN_OCC, SD_OCC = 0.777, 0.093
OCC_CLIP = (0.55, 0.95)
HUTUBS = EAR3D_DIR / "3D head meshes"
SONICOM = EAR3D_DIR / "sonicom_raw_headtorso"


def load_head(path: str):
    """Head mesh, centred and scaled to unit radius. These are already meshes."""
    m = o3d.io.read_triangle_mesh(str(path))
    if len(m.vertices) == 0:
        return None
    m.compute_vertex_normals()
    V = np.asarray(m.vertices)
    V = (V - V.mean(0)) / np.abs(V - V.mean(0)).max()
    m.vertices = o3d.utility.Vector3dVector(V)
    m.compute_vertex_normals()
    return m


def find_ears(mesh, det, probe_size=400):
    """Locate ears by sweeping the detector around the head.

    Geometric rules do not transfer between datasets -- HUTUBS and SONICOM store
    heads on different axes -- so the ear is found by what an ear detector sees.
    It works on these renders (0.84 confidence on clay, 0.91 with skin); it fails
    only on SONICOM's *graded* meshes, whose pinna is a featureless blob.
    """
    V = np.asarray(mesh.vertices)
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


def expand_box(box, factor, size=400):
    """Grow a detector box about its centre, clamped to the frame."""
    ymin, xmin, ymax, xmax = box
    cy, cx = (ymin + ymax) / 2.0, (xmin + xmax) / 2.0
    hh, hw = (ymax - ymin) * factor / 2.0, (xmax - xmin) * factor / 2.0
    return np.array([max(0.0, cy - hh), max(0.0, cx - hw),
                     min(float(size), cy + hh), min(float(size), cx + hw)])


def ear_vertices(mesh, front, up, box, dist=2.4, size=400, expand=1.0):
    """Mesh vertices that project inside the detector's ear box, front surface only.

    This replaces picking a ball of vertices near the pinna centroid, which
    scooped up cheek and scalp and skewed both the plane fit and the extent
    estimate -- bad labels then passed the scale and centre checks because those
    checks compared against the same wrong reference.
    """
    # TWO DIFFERENT REGIONS ARE NEEDED, and conflating them breaks things in
    # opposite directions:
    #   expand=2.5 -- the RAYCAST / SNAP TARGET. A detector box bounds the ear
    #     tightly, so using it directly truncates the pinna: rim rays then miss
    #     (hit rate 0.53-0.62) and rim landmarks snap onto the cut edge. 1.5x
    #     still ran the patch edge close to the ear on some subjects; 2.5x gives
    #     clear margin and costs only ~28% more triangles. A larger target can
    #     only reduce spurious misses, so err generous.
    #   expand=1.0 -- the CROP FRAMING. Framing on the expanded region puts ear
    #     plus scalp in the crop, so the model predicts across all of it and the
    #     labels come out larger than the ear (measured landmark extent 1.37
    #     against a region normalised to 1.0).
    # Callers must pass the right one; the default is the tight framing region.
    box = expand_box(box, expand, size) if expand != 1.0 else np.asarray(box, float)
    V = np.asarray(mesh.vertices)
    f = np.asarray(front, float)
    u = np.asarray(up, float)
    z = f / np.linalg.norm(f)
    y = u - z * (u @ z); y /= max(np.linalg.norm(y), 1e-9)
    x = np.cross(y, z)
    cam = np.stack([x, y, z])
    P = V @ cam.T
    t = np.tan(np.radians(50.0) / 2.0)
    zc = dist - P[:, 2]
    uu = (P[:, 0] / np.maximum(zc * t, 1e-9) + 1) / 2 * size
    vv = (1 - P[:, 1] / np.maximum(zc * t, 1e-9)) / 2 * size
    ymin, xmin, ymax, xmax = box
    inside = (uu >= xmin) & (uu <= xmax) & (vv >= ymin) & (vv <= ymax)
    if inside.sum() < 200:
        return None
    # A detector box is a 2D region, so everything in the depth column through
    # the skull projects into it too -- taking the front 65% still left a slab
    # 21% of the whole head, which skewed the plane fit and the scale. The pinna
    # stands off the skull, so keep only what is within a fraction of the
    # frontmost depth in the box.
    zin = P[inside, 2]
    front_z = np.percentile(zin, 99)
    depth_window = 0.22 * max(np.ptp(np.asarray(mesh.vertices), axis=0).max(), 1e-6)
    near = P[:, 2] > (front_z - depth_window)
    sel = inside & near
    # Return the MASK, not the points. Callers need to index the mesh with it;
    # rebuilding it by value-matching rounded coordinates silently mismatches.
    return sel if sel.sum() >= 200 else inside


def _shot(mesh, front, up, dist, centre, size):
    r = o3d.visualization.rendering.OffscreenRenderer(size, size)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    r.scene.add_geometry("m", mesh, mat)
    r.scene.set_background([0.5, 0.5, 0.5, 1.0])
    r.scene.scene.set_sun_light([-0.4, -0.3, -0.9], [1, 1, 1], 100000)
    r.scene.scene.enable_sun_light(True)
    eye = np.asarray(centre, float) + np.asarray(front, float) * dist
    r.setup_camera(50.0, np.asarray(centre, np.float32), eye.astype(np.float32),
                   np.asarray(up, np.float32))
    img = np.asarray(r.render_to_image())[:, :, :3]
    del r
    return img


def pinna_frame_from(sel, front, up):
    """Rotation taking the ear's own surface normal onto +Z (toward the camera).

    LABELLING FRAME. The pinna's plane is not the head's lateral plane: measured
    over 14 HUTUBS subjects they differ by 8.1 deg on average (sd 3.9, range
    1-13), varying per subject. Labelling along the lateral axis would therefore
    bake a subject-dependent 1-13 deg tilt into the ground truth. Deviation grows
    at roughly 0.04% of ear extent per degree off-normal, so this is worth ~0.3%
    -- small, but free to remove and inconsistent if left in.

    `front` points from the head toward this ear, and selects which side's
    vertices to fit.
    """
    if sel is None or len(sel) < 200:
        return None, None
    C = sel.mean(0)
    _, _, vt = np.linalg.svd(sel - C, full_matrices=False)
    n = vt[2]
    if n @ np.asarray(front, float) < 0:
        n = -n
    # Build a frame with n -> +Z, keeping `up` as close to +Y as possible.
    z = n / np.linalg.norm(n)
    y = np.asarray(up, float) - z * (np.asarray(up, float) @ z)
    ny = np.linalg.norm(y)
    if ny < 1e-6:
        return None, None
    y /= ny
    x = np.cross(y, z)
    R = np.stack([x, y, z])              # world -> pinna frame
    return R, C


def occupancy_crop(mesh_rot, rng, extent_pts, out=192, size=SIZE, renderer="open3d",
                   tone=None, light_az=None, blender_script=None):
    """Render and crop so the EAR fills an occupancy drawn from the real corpus.

    `extent_pts` must be the ear, not the mesh. The head is deliberately left in
    the render for context -- a floating ear teaches the model that a hard
    silhouette marks its boundary -- but framing to the whole head gives 0.23
    occupancy against the 0.777 the model was trained at.
    """
    occ = float(np.clip(rng.normal(MEAN_OCC, SD_OCC), *OCC_CLIP))
    uv = project(np.asarray(extent_pts, float), size)
    side = max(np.ptp(uv[:, 0]), np.ptp(uv[:, 1])) / occ
    cx = (uv[:, 0].min() + uv[:, 0].max()) / 2
    cy = (uv[:, 1].min() + uv[:, 1].max()) / 2
    l, t, s = int(round(cx - side / 2)), int(round(cy - side / 2)), int(round(side))
    if s < 48:
        return None, None, None
    if renderer == "blender":
        img = _blender_render(mesh_rot, tone, light_az, size, blender_script)
        if img is None:
            return None, None, None
    else:
        img = render(mesh_rot, 0, 0, size=size)[:, :, :3]
    cv = Image.new("RGB", (s, s), (128, 128, 128))
    src = Image.fromarray(img)
    sx0, sy0 = max(0, l), max(0, t)
    sx1, sy1 = min(size, l + s), min(size, t + s)
    if sx1 <= sx0 or sy1 <= sy0:
        return None, None, None
    cv.paste(src.crop((sx0, sy0, sx1, sy1)), (sx0 - l, sy0 - t))
    return cv.resize((out, out), Image.BILINEAR), (l, t, s, out), occ


def _blender_render(mesh, tone, light_az, size, script):
    with tempfile.TemporaryDirectory() as td:
        mp = os.path.join(td, "m.ply")
        o3d.io.write_triangle_mesh(mp, mesh)
        args = dict(mesh=mp, ear=[0.0, 0.0, 0.0], dist=2.2, tone=list(tone),
                    out=os.path.join(td, "r"), size=size, samples=48,
                    key_energy=5.0, fill_energy=1.5, ambient=0.35,
                    key_azimuth=float(light_az), key_elevation=30.0)
        ap = os.path.join(td, "a.json")
        json.dump(args, open(ap, "w"))
        try:
            subprocess.run(["blender", "-b", "-P", script, "--", ap],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=240, check=False)
            return np.asarray(Image.open(os.path.join(td, "r.png")).convert("RGB"))
        except Exception:
            return None


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
    args = p.parse_args()

    from inference import BLAZEEAR_DIR, DETECTOR_WEIGHTS, EarDetector
    det = EarDetector(BLAZEEAR_DIR / DETECTOR_WEIGHTS, "cpu", 0.5)
    pred = LandmarkPredictor(best_ckpt(args.label_run), "cuda")
    bscript = str(ROOT / "scripts" / "blender_skin_render.py")

    heads = sorted(glob.glob(str(HUTUBS / "*.ply"))) + sorted(glob.glob(str(SONICOM / "*.stl")))
    rng = np.random.default_rng(args.seed)
    (OUT / "images").mkdir(parents=True, exist_ok=True)

    rows, lms = [], []
    n_ear = 0
    n_rejected = 0
    for hp in heads:
        if n_ear >= args.ears:
            break
        mesh = load_head(hp)
        if mesh is None:
            continue
        for conf, front, up, box0 in find_ears(mesh, det):
            if n_ear >= args.ears:
                break
            frame_mask = ear_vertices(mesh, front, up, box0, expand=1.0)
            sel_mask = ear_vertices(mesh, front, up, box0, expand=2.5)
            if frame_mask is None or sel_mask is None:
                continue
            sel = np.asarray(mesh.vertices)[frame_mask]   # plane fit on the EAR
            R, C = pinna_frame_from(sel, front, up)
            if R is None:
                continue
            # Put this ear in its own pinna frame, centred: pose zero is now
            # camera-perpendicular-to-pinna for every subject alike.
            base = copy.deepcopy(mesh)
            Vb = (np.asarray(base.vertices) - C) @ R.T
            sel_b = (sel - C) @ R.T                      # the ear, in the pinna frame
            scale = np.abs(sel_b).max()
            base.vertices = o3d.utility.Vector3dVector(Vb / scale)
            base.compute_vertex_normals()
            tone = SKIN_TONES[rng.integers(len(SKIN_TONES))]
            if args.renderer == "open3d":
                base = apply_skin(base, seed=int(rng.integers(1 << 30)))

            # --- labels: ONE face-on prediction, ray-cast to 3D ---
            ear_pts = sel_b / scale
            crop, box, _ = occupancy_crop(base, np.random.default_rng(0), ear_pts,
                                          renderer="open3d")
            if crop is None:
                continue
            lm0, cf0 = pred.predict(np.asarray(crop), with_confidence=True)
            lm0 = np.asarray(lm0, float)
            P3, hit = raycast(base, crop_to_full(lm0, box))

            # QUALITY GATE. Every pose inherits this one prediction, so a bad
            # face-on label poisons the whole ear. Without a gate roughly 1 ear
            # in 3 came through with landmarks sprawling off the pinna -- either
            # the detector found a poor view, or the pinna-plane fit picked up
            # scalp and skewed the frame.
            lm_n = lm0 / box[3]
            ear_uv = project(ear_pts, SIZE)
            ear_n = np.stack([(ear_uv[:, 0] - box[0]) / box[2],
                              (ear_uv[:, 1] - box[1]) / box[2]], axis=1)
            span_lm = max(np.ptp(lm_n[:, 0]), np.ptp(lm_n[:, 1]))
            span_ear = max(np.ptp(ear_n[:, 0]), np.ptp(ear_n[:, 1]))
            centre_off = np.linalg.norm(lm_n.mean(0) - ear_n.mean(0))
            reasons = []
            if hit.sum() < 45:
                reasons.append(f"only {hit.sum()}/55 rays hit")
            if float(np.mean(cf0)) < 0.25:
                reasons.append(f"confidence {float(np.mean(cf0)):.2f}")
            if not 0.6 < span_lm / max(span_ear, 1e-6) < 1.5:
                reasons.append(f"scale ratio {span_lm/max(span_ear,1e-6):.2f}")
            if centre_off > 0.18:
                reasons.append(f"centre off by {centre_off:.2f}")
            if reasons:
                n_rejected += 1
                print(f"    rejected {os.path.basename(hp)}: {'; '.join(reasons)}", flush=True)
                continue
            n_ear += 1

            for pitch in args.pitches:
                for yaw in args.yaws:
                    Rr = rot(float(yaw), float(pitch))
                    mr = copy.deepcopy(base)
                    mr.rotate(Rr, center=(0, 0, 0))
                    az = float(rng.uniform(-70, 40))     # randomised; see module docstring
                    # frame on the rotated LANDMARKS: they define the ear exactly
                    got = occupancy_crop(mr, rng, (P3 @ Rr.T)[hit],
                                         renderer=args.renderer, tone=tone,
                                         light_az=az, blender_script=bscript)
                    if got[0] is None:
                        continue
                    img, box2, occ = got
                    Pr = P3 @ Rr.T
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
                    img.save(OUT / "images" / name, quality=args.quality)
                    rows.append({"idx": len(rows), "image_file": f"images/{name}",
                                 "source": "render3d", "original_path": os.path.basename(hp),
                                 "width": img.width, "height": img.height})
                    lms.append(lm.astype(np.float32))
            print(f"  [{n_ear}/{args.ears}] {os.path.basename(hp)} conf={conf:.2f}", flush=True)

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
