#!/usr/bin/env python3
"""Score a depth head against a REAL head scan's own geometry.

WHY THIS IS NOT THE BACK-PROJECTION PATH. Back-projection existed because the
2D model gave no depth, so depth had to be recovered from the mesh -- and that
path is not dependable: lateral 2D error becomes amplified depth error through
a narrow view cone, and rays passing just outside the pinna land on the scalp
an ear-depth too deep. A depth head predicts z directly, so the mesh stops
being the SOURCE of depth and becomes an independent CHECK on it.

That check is worth having because this scan is a subject from none of the
training sources, and because the camera is exact: the render's own float32
view-space depth buffer is the rasterisation that produced the pixels, so the
reference needs no camera estimation and no triangulation.

WHAT IT CAN AND CANNOT MEASURE. The scan carries no landmark annotation, so
PLACEMENT cannot be scored -- there is nothing to say where point 29 belongs.
What is measured is whether the predicted depth profile matches the real
surface at the pixels the model itself chose. That conflates placement error
with depth error and the two cannot be separated here; it is still a real
measurement of depth on an unseen head, which nothing else available provides.

THE SILHOUETTE CASE IS REPORTED, NOT HIDDEN. A landmark a pixel or two outside
the pinna samples the scalp behind it, which is one ear-depth too far and would
quietly dominate the error. Those points are counted and excluded from the
headline number, and the count is part of the result.

    python scripts/scan_depth_check.py --mesh supersecret/BJ002/BJ002.OBJ
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ear3d.backproject import backproject, backproject_snapped   # noqa: E402
from ear3d.config import DEFAULTS                                # noqa: E402
from ear3d.frames import (frame_from, head_pose, in_frame,       # noqa: E402
                          load_head, plane_normal)
from ear3d.label import landmark_whole_frame                     # noqa: E402
from ear3d.render import render                                  # noqa: E402
from ear3d.scheme import STRIPS                                  # noqa: E402

TPL = pathlib.Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/wrap_work/templates")


def flag_scalp(P, hit, cam, tol):
    """Which back-projected points sit behind the pinna rather than on it.

    A ray passing a pixel or two outside the pinna hits the SCALP, about one
    ear-depth too far. The point looks right face-on -- the error is purely in
    depth -- so nothing in the 2D overlay reveals it, and it has to be found
    from the depth it came back with. Returns (camera-space z, scalp mask).
    """
    zc = np.full(len(P), np.nan)
    zc[hit] = (P[hit] @ cam.R.T + cam.t)[:, 2]
    if hit.sum() < 3:
        return zc, np.zeros(len(P), bool)
    med = float(np.nanmedian(zc))
    span = float(np.nanmax(zc) - np.nanmin(zc))
    return zc, hit & (zc > med + tol * max(span, 1e-9))


def corpus_z(zc, fx, crop_px):
    """View-space depth -> the corpus's root-relative z, in crop-width units.

    zn = (zc - zbar) * fx / (crop_px * zbar), the definition data/depth_dataset
    documents. It depends on fx and zbar only through their ratio -- pixels per
    unit length at the ear -- which is what a scaled-orthographic fit observes,
    so it is directly comparable to a predicted z.
    """
    zb = float(np.mean(zc))
    return (zc - zb) * fx / (crop_px * zb)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", required=True)
    p.add_argument("--ear", type=int, default=0, help="0 = subject left, 1 = right")
    p.add_argument("--ckpt", default="runs/pash_0.0/ckpt/last.ckpt",
                   help="depth head; default is the Route C recipe that scored "
                        "best on real photographs")
    p.add_argument("--n-points", type=int, default=55)
    p.add_argument("--occupancy", type=float, default=0.677,
                   help="fraction of frame HEIGHT the ear should fill. The "
                        "corpus frames at 0.354 x 0.677 and a depth head is "
                        "sensitive to it -- a crop round-trip alone cost "
                        "z_skill 0.802 -> 0.584 -- so it is matched here")
    p.add_argument("--scalp-tol", type=float, default=0.25,
                   help="a sampled depth this far behind the median, as a "
                        "fraction of the sampled span, is the scalp beside the "
                        "pinna rather than the ear")
    p.add_argument("--no-skin", action="store_true")
    p.add_argument("--out", default="supersecret/out",
                   help="LOCAL ONLY: a gitignored directory")
    a = p.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    import torch
    from PIL import Image
    from model.depth_module import EarDepth3DModule
    from eval_test import best_ckpt
    from inference import (BLAZEEAR_DIR, DETECTOR_WEIGHTS,
                           EarLandmarkerPipeline)
    from scripts.fit_photo_depth import weak_perspective

    cfg = dict(DEFAULTS, mesh=a.mesh, ear=a.ear, skin=not a.no_skin)
    K = a.n_points
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    mesh = load_head(cfg["mesh"])
    if mesh is None:
        sys.exit(f"could not read {cfg['mesh']}")
    if cfg["skin"]:
        from skin import apply_skin
        apply_skin(mesh, seed=cfg["skin_seed"], tone=cfg["tone"],
                   ao_strength=cfg["ao_strength"], n_rays=cfg["ao_rays"])

    ck = best_ckpt(cfg["run"])
    if ck is None:
        sys.exit(f"no checkpoint for run {cfg['run']!r} under runs/checkpoints/")
    # smooth=False: the tracker is for video, and on a single still it would
    # smooth a one-frame track against wall-clock time and shift the landmarks.
    pipe = EarLandmarkerPipeline(BLAZEEAR_DIR / DETECTOR_WEIGHTS, ck,
                                 device=cfg["device"], smooth=False)
    print(f"landmarker {ck.name}")

    pose = head_pose(mesh, cfg)
    if pose is None:
        sys.exit("MediaPipe found no face on this scan")
    lat, vert, tr_r, tr_l = pose
    left = a.ear % 2 == 0
    front, up = (lat if left else -lat), vert
    C1 = np.asarray(tr_l if left else tr_r, float)
    R1 = frame_from(front, up, front=front)
    g1 = in_frame(mesh, R1, C1, 1.0)
    _, depth1, cam1, lm1, conf = landmark_whole_frame(g1, cfg, pipe)
    if lm1 is None:
        sys.exit("the pipeline found no ear face-on; try --ear 1")
    P1, hit1, _ = backproject_snapped(depth1, lm1, cam1, cfg)
    # THE SCALP POINTS MUST NOT REACH THE PLANE FIT. The pinna plane sets the
    # view direction for the render the depth head is scored on, and a point an
    # ear-depth behind the ear tilts that plane. Measured on this scan: leaving
    # them in gave a 0.572-of-width framing and an ear whose landmark box came
    # out 1.17 tall-to-wide against a real ear's ~1.9, i.e. the view was oblique
    # without anything saying so.
    _, scalp1 = flag_scalp(P1, hit1, cam1, a.scalp_tol)
    good1 = hit1 & ~scalp1
    print(f"locate: {'left' if left else 'right'} ear, detector conf "
          f"{conf:.3f}, {int(hit1.sum())}/{len(hit1)} rays hit, "
          f"{int(scalp1.sum())} on the scalp, {int(good1.sum())} used for the "
          f"pinna plane")
    if good1.sum() < 10:
        sys.exit("too few on-pinna rays to fit a pinna plane")
    P1w = (P1[good1] @ R1) + C1

    R2 = frame_from(plane_normal(P1w), up, front=front)
    C2 = P1w.mean(0)
    loc = (P1w - C2) @ R2.T
    ear_h = float(np.ptp(loc[:, 1]))
    half = cfg["eye_z"] * np.tan(np.radians(cfg["vfov"]) / 2)
    # Deliberately LOOSE: this only has to get the ear fully into frame with
    # margin. The framing that matters is set below from the 2D landmarks, not
    # from here -- sizing the crop off P1w would feed the back-projection's
    # errors straight into it, and on this scan 19 of those 55 points are on
    # the scalp, which stretched the box to 0.572 of frame width against the
    # corpus's 0.354.
    scale = ear_h / (2 * half * 0.40)
    g2 = in_frame(mesh, R2, C2, scale)
    img2, depth2, cam2 = render(g2, cfg)

    # ---- crop to the CORPUS framing, from the 2D landmarks ---------------
    # The 2D model is the reliable part of this pipeline (0.0149 NME), so the
    # ear's apparent extent comes from it. A depth head is sensitive to
    # occupancy, so this is matched rather than inherited.
    res2 = pipe(img2, timestamp=0.0)
    if not res2:
        sys.exit("the pipeline found no ear in the pinna-frame render")
    lm2 = np.asarray(max(res2, key=lambda d: float(d["confidence"]))
                     ["landmarks"], float)[:K]
    (bx0, by0), (bx1, by1) = lm2.min(0), lm2.max(0)
    side = max((bx1 - bx0) / 0.354, (by1 - by0) / a.occupancy)
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    x0, y0 = cx - side / 2, cy - side / 2
    H0, W0 = depth2.shape
    if x0 < 0 or y0 < 0 or x0 + side > W0 or y0 + side > H0:
        print(f"note: the corpus crop runs off the {W0}px render "
              f"(x0 {x0:.0f}, side {side:.0f}); it is clamped, so occupancy "
              f"is approximate")
        x0, y0 = max(0.0, min(x0, W0 - 1)), max(0.0, min(y0, H0 - 1))
        side = min(side, W0 - x0, H0 - y0)
    from PIL import Image as _I
    crop_box = (int(round(x0)), int(round(y0)),
                int(round(x0 + side)), int(round(y0 + side)))
    img_c = np.asarray(_I.fromarray(img2).crop(crop_box))
    size = float(side)            # crop width in RENDER pixels: the z unit
    print(f"frame:  ear fills {(bx1 - bx0) / side:.3f} of crop width, "
          f"{(by1 - by0) / side:.3f} of height (corpus 0.354 x 0.677)")

    m = EarDepth3DModule.load_from_checkpoint(
        a.ckpt, map_location="cuda" if torch.cuda.is_available() else "cpu").eval()
    dev = next(m.parameters()).device
    t = np.asarray(Image.fromarray(img_c).resize((192, 192), Image.BILINEAR),
                   np.float32).transpose(2, 0, 1)[None] / 255.0
    with torch.no_grad():
        pr = m(torch.from_numpy((t - 0.5) / 0.5).float().to(dev)
               ).view(-1, 3).cpu().numpy()[:K]
    xy, z_pred = pr[:, :2], pr[:, 2] - pr[:, 2].mean()
    # xy is a fraction OF THE CROP; the depth buffer is the full render
    uv = np.stack([crop_box[0] + xy[:, 0] * side,
                   crop_box[1] + xy[:, 1] * side], axis=1)

    P2, hit2 = backproject(depth2, uv, cam2)
    zc, scalp = flag_scalp(P2, hit2, cam2, a.scalp_tol)
    good = hit2 & ~scalp
    print(f"sample: {int(hit2.sum())}/{K} landmarks hit the surface, "
          f"{int(scalp.sum())} of those behind the pinna (scalp), "
          f"{int(good.sum())} usable")
    if good.sum() < 10:
        sys.exit("too few usable landmarks to report a depth number")

    zt_g = corpus_z(zc[good], cam2.fx, size)
    zt_g = zt_g - zt_g.mean()
    zp = z_pred[good] - z_pred[good].mean()
    mae = float(np.abs(zp - zt_g).mean())
    base = float(np.abs(zt_g).mean())
    zf_all = weak_perspective(
        (lambda mu: mu - mu.mean(0))(
            np.load(TPL / "lm_basis_v6_55_unitheight.npz",
                    allow_pickle=True)["mean"][:K]), xy)[0]
    zf = zf_all[good] - zf_all[good].mean()
    mf = float(np.abs(zf - zt_g).mean())

    print(f"\n{'quantity':<46s} {'value':>9s}")
    print(f"  {'scan depth spread, mean |z| (the signal)':<44s} {base:9.4f}")
    print(f"  {'head 1 error, mean |dz|':<44s} {mae:9.4f}")
    print(f"  {'head 1 z_skill':<44s} {1 - mae / base:9.3f}")
    print(f"  {'head 1 correlation with the scan':<44s} "
          f"{np.corrcoef(zp, zt_g)[0, 1]:9.3f}")
    print(f"  {'mean ear + solved pose, same landmarks':<44s} "
          f"{1 - mf / base:9.3f}")
    print(f"  {'mean ear correlation':<44s} "
          f"{np.corrcoef(zf, zt_g)[0, 1]:9.3f}")
    print(f"  {'predict z = 0':<44s} {0.0:9.3f}")

    np.savez(out / "scan_depth.npz", xy=xy, z_pred=z_pred, z_true_good=zt_g,
             good=good, hit=hit2, scalp=scalp, uv=uv, conf=conf,
             mae=mae, base=base, mean_ear_mae=mf)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    idx = np.arange(K)[good]
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.6))
    ax[0].imshow(img_c)
    cu = xy * side              # crop-local pixels, for drawing
    for lo, hi in STRIPS:
        ax[0].plot(cu[lo:hi, 0], cu[lo:hi, 1], "-", c="#ff2d95", lw=1.4)
    ax[0].plot(cu[:, 0], cu[:, 1], ".", c="#ff2d95", ms=3)
    if scalp.any():
        ax[0].plot(cu[scalp, 0], cu[scalp, 1], "o", mfc="none", mec="w",
                   ms=9, label="sampled the scalp")
        ax[0].legend(fontsize=6, loc="lower right")
    ax[0].set_title(f"render of the scan + head 1 (det conf {conf:.2f})",
                    fontsize=8)
    ax[0].axis("off")
    ax[1].plot(idx, zt_g, "-o", c="#2ca02c", ms=3, label="scan surface")
    ax[1].plot(idx, zp, "-o", c="#ff2d95", ms=3, label="head 1")
    ax[1].plot(idx, zf, "-o", c="#5599ff", ms=3, label="mean ear + pose")
    ax[1].set_xlabel("landmark index")
    ax[1].set_ylabel("z (crop widths)")
    ax[1].set_title("depth profile along the scheme", fontsize=8)
    ax[1].legend(fontsize=6)
    ax[1].grid(alpha=.25)
    ax[2].plot(xy[good, 0], zt_g, "o", c="#2ca02c", ms=4, label="scan")
    ax[2].plot(xy[good, 0], zp, "o", c="#ff2d95", ms=4, label="head 1")
    ax[2].set_xlabel("x")
    ax[2].set_ylabel("z")
    ax[2].invert_yaxis()
    ax[2].set_title("from above (x, z)", fontsize=8)
    ax[2].legend(fontsize=6)
    ax[2].grid(alpha=.25)
    fig.suptitle(
        f"{pathlib.Path(a.mesh).stem}: predicted depth against the scan's own "
        f"geometry.  head 1 z_skill {1 - mae / base:.3f}, mean-ear fit "
        f"{1 - mf / base:.3f}, {int(good.sum())}/{K} usable landmarks."
        "\nPlacement is NOT scored: the scan has no annotation. This asks "
        "whether the depth matches the surface at the pixels the model chose.",
        fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(out / "scan_depth.png", dpi=140)
    print(f"\nwrote {out}/scan_depth.png and scan_depth.npz  (both gitignored)")


if __name__ == "__main__":
    main()
