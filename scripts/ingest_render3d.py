"""Render real 3D ear geometry at varied yaw as an extra training source.

THE 3D PATH LIVES IN ear3d/, NOT HERE. Camera, renderer, back-projection, ear
finding, head pose, the snaps and the two-pass labelling are imported from that
package, the same code scripts/view_backprojection.py drives interactively. This
file used to keep its own copies of load_head, find_ears, in_frame and a private
camera setup, and they had already drifted from the viewer's. What remains here
is what is genuinely ingest's: cropping to a training occupancy, the pose loop,
the quality gate and writing the dataset.

WHY RENDERS AND NOT MORE PERSPECTIVE AUGMENTATION. data/dataset.py simulates yaw
with a homography, which cannot produce parallax or self-occlusion. Measured, it
correlates with a true rotation at only 0.47 (30 deg) to 0.15 (65 deg), and a
model trained with +/-65 deg of it is no better under real 3D rotation than one
trained with none. Renders supply what the warp structurally cannot.

WHERE THE LABELS COME FROM, and why this is not self-distillation. The model is
run face-on and its 55 points are lifted to 3D through the depth buffer of that
render. Rotating mesh and landmarks together then yields correct labels at every
other pose: the geometry carries the label, not the model. The model's competence
at its EASIEST pose is transferred to poses where it is weak.

WHAT THIS CANNOT FIX: a systematic error the model makes face-on is baked into
every pose. This widens pose coverage; it does not improve on the model's own
face-on accuracy.

Occupancy is drawn per sample from N(0.777, 0.093), matching the real corpus. A
constant occupancy puts 99% of the real test set outside the training range --
that bug has been made once already.

Usage:
    python scripts/ingest_render3d.py --ears 150
"""

from __future__ import annotations

import argparse
import copy
import csv
import glob
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ear3d.backproject import reseat, visible                       # noqa: E402
from ear3d.config import DEFAULTS, EAR3D_DIR                       # noqa: E402
from ear3d.camera import cone_angles                               # noqa: E402
from ear3d.frames import find_ears, head_pose, load_head           # noqa: E402
from ear3d.label import label_two_pass, triangulate_landmarks      # noqa: E402
from ear3d.render import render                                    # noqa: E402
from eval_test import best_ckpt                                    # noqa: E402
from skin import LABEL_TONE, apply_skin                            # noqa: E402

OUT = ROOT / "data" / "render3d"
MEAN_OCC, SD_OCC = 0.777, 0.093
OCC_CLIP = (0.55, 0.95)
HUTUBS = EAR3D_DIR / "3D head meshes"
SONICOM = EAR3D_DIR / "sonicom_raw_headtorso"


def rot(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Rotation about +Y (yaw) then +X (pitch). The MESH turns, not the camera,
    so a given yaw means the same thing for every subject."""
    ry, rx = np.radians(yaw_deg), np.radians(pitch_deg)
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    return Rx @ Ry


def sample_occ(rng):
    return float(np.clip(rng.normal(MEAN_OCC, SD_OCC), *OCC_CLIP))


def crop_to_occupancy(img, uv, occ, out=192):
    """Crop so the projected landmarks fill `occ` of the result.

    The landmarks define the ear exactly, which is why no mesh region is needed.
    The head is deliberately left in the render for context -- a floating ear
    teaches the model that a hard silhouette marks its boundary.
    """
    h, w = img.shape[:2]
    side = max(np.ptp(uv[:, 0]), np.ptp(uv[:, 1])) / occ
    cx = (uv[:, 0].min() + uv[:, 0].max()) / 2
    cy = (uv[:, 1].min() + uv[:, 1].max()) / 2
    l, t, s = int(round(cx - side / 2)), int(round(cy - side / 2)), int(round(side))
    if s < 48:
        return None, None
    cv = Image.new("RGB", (s, s), (128, 128, 128))     # grey-128 pad, as in training
    src = Image.fromarray(img)
    sx0, sy0, sx1, sy1 = max(0, l), max(0, t), min(w, l + s), min(h, t + s)
    if sx1 <= sx0 or sy1 <= sy0:
        return None, None
    cv.paste(src.crop((sx0, sy0, sx1, sy1)), (sx0 - l, sy0 - t))
    return cv.resize((out, out), Image.BILINEAR), (l, t, s, out)


def main() -> None:
    p = argparse.ArgumentParser(description="Render 3D ears at varied yaw for training")
    p.add_argument("--ears", type=int, default=150)
    p.add_argument("--yaws", type=float, nargs="*",
                   default=[-40, -25, -12, 0, 12, 25, 40])
    p.add_argument("--pitches", type=float, nargs="*", default=[0.0])
    p.add_argument("--label-run", default="manual_occ_s42")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quality", type=int, default=95)
    p.add_argument("--min-conf", type=float, default=0.25)
    p.add_argument("--reseat-tol", type=float, default=0.005,
                   help="how far off the surface a triangulated label may sit, as "
                        "a fraction of ear extent; small on purpose, see the note "
                        "in main()")
    p.add_argument("--tri-views", type=int, default=0,
                   help="build each label by intersecting rays from this many "
                        "views instead of lifting the face-on render alone")
    args = p.parse_args()

    from inference import BLAZEEAR_DIR, DETECTOR_WEIGHTS, EarDetector, EarLandmarkerPipeline
    cfg = dict(DEFAULTS, run=args.label_run)
    if args.tri_views:
        # Triangulated labels, not a single-view lift. Measured by leave-one-view-
        # out reprojection over 8 heads: 8.9 px for the face-on lift against 6.7
        # px for the ray intersection, ear ~430 px. It costs a render and a
        # pipeline call per view PER EAR, so it is opt-in -- but this is the one
        # place in the project where the labels are the product, and every pose
        # rendered from an ear inherits whatever this produces.
        cfg["triangulate"] = True
        cfg["tri_angles"] = cone_angles(args.tri_views, cfg["tri_cone"])
    if cfg["triangulate"]:
        # TRIANGULATED LABELS MUST BE SEATED ON THE SURFACE HERE, which is not a
        # preference. The pose loop masks each landmark with visible(), a depth
        # test with a 0.012 mesh-unit tolerance -- 0.6% of an ear spanning 1.86 --
        # so a triangulated point scattered off the surface is judged not visible
        # and its label is dropped. Measured on pp12: 55/55 landmarks survive from
        # the face-on lift, 22/55 from raw triangulation, and 55/55 once seated.
        # A partial re-seat does not help, since it moves only the worst handful.
        #
        # It costs some of triangulation's reprojection advantage (6.13 px against
        # 4.23 raw) but still beats the face-on lift's 6.92, and it is the only
        # configuration where every landmark is usable.
        cfg["reseat"] = True
        cfg["reseat_tol"] = args.reseat_tol
    det = EarDetector(BLAZEEAR_DIR / DETECTOR_WEIGHTS, "cpu", 0.5)
    # smooth=False: the tracker is for video; on a still it would smooth a
    # one-frame track against wall-clock time and shift the landmarks.
    pipe = EarLandmarkerPipeline(BLAZEEAR_DIR / DETECTOR_WEIGHTS,
                                 best_ckpt(args.label_run), device="cuda", smooth=False)

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
        # SKIN FIRST, THEN DETECT. Every model in this path reads clay far worse
        # than skin -- measured, 0.835 against 0.94 on the same ear -- and clay
        # fails outright on 3 of 4 heads. Colouring afterwards meant every
        # detection was made on the appearance the models handle worst.
        mesh = apply_skin(mesh, seed=0, tone=LABEL_TONE)

        # The face-on direction comes from the FaceMesh head frame when it can be
        # had, and the detector sweep otherwise. Same choice the viewer makes.
        pose = head_pose(mesh, cfg)
        if pose is not None:
            lat, vert, tr_r, tr_l = pose
            ears = [(1.0, lat, vert, tr_l), (1.0, -lat, vert, tr_r)]
        else:
            ears = [(c, f, u, np.asarray(mesh.vertices).mean(0))
                    for c, f, u, _ in find_ears(mesh, det, cfg)]

        for conf0, front, up, centre in ears:
            if n_ear >= args.ears:
                break
            try:
                base, _, _, _, P3, hit, conf = label_two_pass(
                    mesh, front, up, cfg, pipe, centre)
            except SystemExit as why:
                n_rejected += 1
                print(f"    rejected {os.path.basename(hp)}: {why}", flush=True)
                continue

            if cfg.get("triangulate"):
                Ptri, n_in, n_view = triangulate_landmarks(cfg, base, pipe,
                                                           verbose=False)
                ok = np.isfinite(Ptri).all(axis=1) & (n_view >= 2)
                P3 = np.where(ok[:, None], Ptri, P3)
                hit = hit | ok
                if cfg.get("reseat"):
                    _, depth_f, cam_f = render(base, cfg)
                    ext = float(np.ptp(P3[hit], axis=0).max())
                    P3, _ = reseat(P3, depth_f, cam_f, ext, cfg["reseat_tol"])

            # QUALITY GATE. Every pose inherits this one labelling, so a bad
            # face-on result poisons the whole ear.
            why = None
            if hit.sum() < 45:
                why = f"only {int(hit.sum())}/55 landmarks lifted"
            elif conf < args.min_conf:
                why = f"confidence {conf:.2f}"
            if why:
                n_rejected += 1
                print(f"    rejected {os.path.basename(hp)}: {why}", flush=True)
                continue
            n_ear += 1

            # Re-skin for the OUTPUT images with a random Fitzpatrick I-VI tone.
            # The labelling tone is deeper than real skin and must not reach
            # training data; the labels are unaffected, being 3D points on
            # geometry that did not change.
            base = apply_skin(base, seed=int(rng.integers(1 << 30)), tone=None)

            for pitch in args.pitches:
                for yaw in args.yaws:
                    Rr = rot(float(yaw), float(pitch))
                    mr = copy.deepcopy(base)
                    mr.rotate(Rr, center=(0, 0, 0))
                    Pr = P3 @ Rr.T
                    img, depth, cam = render(mr, cfg)
                    vis = hit & visible(depth, Pr, cam)
                    uv = cam.project(Pr)
                    crop, box = crop_to_occupancy(img, uv[hit], sample_occ(rng))
                    if crop is None:
                        continue
                    lm = np.stack([(uv[:, 0] - box[0]) / box[2],
                                   (uv[:, 1] - box[1]) / box[2]], axis=1)
                    # Occluded or unlifted points are written OUT OF FRAME, so
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
            print(f"  [{n_ear}/{args.ears}] {os.path.basename(hp)} label={conf:.2f}",
                  flush=True)

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
