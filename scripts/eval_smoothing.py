"""Evaluate temporal + bbox smoothing, measuring jitter AND lag together.

Smoothing is trivially "successful" if you only measure jitter: an infinitely
heavy filter outputs a constant and scores zero. So every configuration here is
scored on two opposed metrics, and a configuration only wins if it improves
jitter without inflating error.

Reference signal: for each frame we also run the model on the NOISE-FREE box --
what a perfect detector would have handed us. That reference R_t is the target.

  err_px    mean ||S_t - R_t||            total error, including smoothing lag
  jit_px    mean ||dS_t - dR_t||          frame-to-frame movement that is not
                                          real motion (successive-difference
                                          mismatch against the reference)

Two scenarios, because they trade off differently:

  static    the true box is fixed; only detector wobble. Smoothing should cut
            both jitter and error, since averaging noise moves you toward R.
  moving    the true box pans and scales smoothly, with wobble on top. Here
            smoothing can lag behind real motion, so err_px is the honest cost.

Usage:
    python scripts/eval_smoothing.py
    python scripts/eval_smoothing.py --run v2_fixed --limit 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_jitter import IMG_SIZE, best_checkpoint, load_checkpoint, load_images  # noqa: E402
from model.smoothing import BoxSmoother, LandmarkSmoother  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
SRC = IMG_SIZE * 2
BASE_SIDE = SRC * 0.85
FPS = 30.0


def true_boxes(n: int, scenario: str, motion_hz: float = 0.25) -> list[tuple[float, float, float]]:
    """Ground-truth crop windows (x0, y0, side) with no detector noise.

    The motion must be defined in ABSOLUTE TIME, not per-clip. Phasing a full
    cycle over `n` frames makes the subject move faster the shorter the clip --
    at 20 frames that is a 1.5 Hz oscillation, far quicker than any real head,
    which makes any filter look terrible for reasons that have nothing to do
    with the filter. motion_hz=0.25 is one slow sweep every 4 s, i.e. a person
    turning their head to be photographed.
    """
    out = []
    for i in range(n):
        if scenario == "static":
            cx = cy = SRC / 2
            side = BASE_SIDE
        else:
            ph = 2 * np.pi * motion_hz * (i / FPS)
            cx = SRC / 2 + 0.04 * SRC * np.sin(ph)
            cy = SRC / 2 + 0.02 * SRC * np.cos(ph)
            side = BASE_SIDE * (1.0 + 0.03 * np.sin(ph * 0.5))
        out.append((cx - side / 2, cy - side / 2, side))
    return out


def add_wobble(win, rng, max_shift, max_scale):
    """Detector noise on one window."""
    x0, y0, side = win
    cx, cy = x0 + side / 2, y0 + side / 2
    s = side * (1.0 + rng.uniform(-max_scale, max_scale))
    cx += rng.uniform(-max_shift, max_shift) * SRC
    cy += rng.uniform(-max_shift, max_shift) * SRC
    return (cx - s / 2, cy - s / 2, s)


def crop(img_t: torch.Tensor, win) -> torch.Tensor:
    """Crop (x0, y0, side) and resize to the model input, clamped in-bounds."""
    x0, y0, side = win
    side = float(np.clip(side, 16, SRC))
    x0 = float(np.clip(x0, 0, SRC - side))
    y0 = float(np.clip(y0, 0, SRC - side))
    x0i, y0i, si = int(round(x0)), int(round(y0)), int(round(side))
    si = max(16, min(si, SRC))
    x0i = min(x0i, SRC - si)
    y0i = min(y0i, SRC - si)
    c = img_t[:, :, y0i:y0i + si, x0i:x0i + si]
    return F.interpolate(c, size=(IMG_SIZE, IMG_SIZE), mode="bilinear",
                         align_corners=False), (float(x0i), float(y0i), float(si))


def to_canonical(pred: np.ndarray, win) -> np.ndarray:
    """Crop-normalized coords -> common frame, in nominal-crop units."""
    x0, y0, side = win
    out = np.empty_like(pred)
    out[:, 0] = (x0 + pred[:, 0] * side) / BASE_SIDE
    out[:, 1] = (y0 + pred[:, 1] * side) / BASE_SIDE
    return out


def box_to_win(box: np.ndarray) -> tuple[float, float, float]:
    """[ymin, xmin, ymax, xmax] -> (x0, y0, side), forced square."""
    ymin, xmin, ymax, xmax = box
    side = max((ymax - ymin + xmax - xmin) / 2.0, 16.0)
    cy, cx = (ymin + ymax) / 2.0, (xmin + xmax) / 2.0
    return (cx - side / 2, cy - side / 2, side)


def win_to_box(win) -> np.ndarray:
    x0, y0, side = win
    return np.array([y0, x0, y0 + side, x0 + side], dtype=np.float64)


CONFIGS = [
    ("raw",              dict(box=False, lm=False, conf=False)),
    ("box only",         dict(box=True,  lm=False, conf=False)),
    ("landmark only",    dict(box=False, lm=True,  conf=False)),
    ("box + landmark",   dict(box=True,  lm=True,  conf=False)),
    ("box + lm + conf",  dict(box=True,  lm=True,  conf=True)),
]


@torch.no_grad()
def run_clip(model, has_conf, img, scenario, rng, cfg, n_frames,
             max_shift, max_scale, device, lm_kwargs, box_kwargs, motion_hz):
    img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    wins_true = true_boxes(n_frames, scenario, motion_hz)

    box_f = BoxSmoother(**box_kwargs) if cfg["box"] else None
    lm_f = LandmarkSmoother(**lm_kwargs) if cfg["lm"] else None

    ref, out = [], []
    for i, wt in enumerate(wins_true):
        t = i / FPS

        # Reference: perfect detector, no smoothing.
        c_ref, w_ref = crop(img_t, wt)
        p_ref = model((c_ref.to(device) - 0.5) / 0.5).reshape(55, 2).cpu().numpy()
        ref.append(to_canonical(p_ref, w_ref))

        # Observed: wobbled box, optionally smoothed before cropping.
        wn = add_wobble(wt, rng, max_shift, max_scale)
        if box_f is not None:
            wn = box_to_win(box_f(win_to_box(wn), t))
        c_obs, w_obs = crop(img_t, wn)
        x = (c_obs.to(device) - 0.5) / 0.5
        if has_conf and cfg["conf"]:
            coords, conf = model.predict_with_confidence(x)
            p = coords.reshape(55, 2).cpu().numpy()
            cf = conf.reshape(55).cpu().numpy()
        else:
            p = model(x).reshape(55, 2).cpu().numpy()
            cf = None
        canon = to_canonical(p, w_obs)
        if lm_f is not None:
            canon = lm_f(canon, t, cf)
        out.append(canon)

    R, S = np.stack(ref), np.stack(out)
    err = float(np.linalg.norm(S - R, axis=-1).mean() * IMG_SIZE)
    jit = float(np.linalg.norm(np.diff(S, axis=0) - np.diff(R, axis=0),
                               axis=-1).mean() * IMG_SIZE)
    return err, jit


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate temporal + bbox smoothing")
    p.add_argument("--run", default="v2_heatmap")
    p.add_argument("--limit", type=int, default=120)
    p.add_argument("--frames", type=int, default=30)
    p.add_argument("--max-shift", type=float, default=0.008)
    p.add_argument("--max-scale", type=float, default=0.02)
    p.add_argument("--min-cutoff", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--box-min-cutoff", type=float, default=0.8)
    p.add_argument("--box-beta", type=float, default=0.03)
    p.add_argument("--motion-hz", type=float, default=0.25,
                   help="Head-motion frequency in the moving scenario (0.25 = one sweep per 4s)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = best_checkpoint(PROJECT / "runs" / "checkpoints" / args.run)
    model, arch = load_checkpoint(ck)
    model.to(device)
    has_conf = hasattr(model, "predict_with_confidence")

    lm_kwargs = dict(min_cutoff=args.min_cutoff, beta=args.beta)
    box_kwargs = dict(min_cutoff=args.box_min_cutoff, beta=args.box_beta)

    print(f"run={args.run} arch={arch} device={device}")
    print(f"clips={args.limit} frames={args.frames} @ {FPS:.0f}fps  "
          f"wobble: centre +/-{args.max_shift*SRC:.1f}px scale +/-{args.max_scale*100:.0f}%")
    print(f"landmark filter: min_cutoff={args.min_cutoff} beta={args.beta}   "
          f"box filter: min_cutoff={args.box_min_cutoff} beta={args.box_beta}")
    if not has_conf:
        print("NOTE: this arch has no heatmap confidence; 'conf' rows equal the row above.")
    images = load_images(args.limit)

    for scenario in ("static", "moving"):
        print(f"\n=== {scenario.upper()} "
              f"({'no true motion' if scenario=='static' else 'smooth pan + scale'}) ===")
        print(f"  {'config':18s} {'err_px':>8s} {'jit_px':>8s}   {'vs raw err':>11s} {'vs raw jit':>11s}")
        print("  " + "-" * 64)
        base = None
        for name, cfg in CONFIGS:
            errs, jits = [], []
            for i, img in enumerate(images):
                rng = np.random.default_rng(args.seed + i)   # identical wobble per config
                e, j = run_clip(model, has_conf, img, scenario, rng, cfg,
                                args.frames, args.max_shift, args.max_scale,
                                device, lm_kwargs, box_kwargs, args.motion_hz)
                errs.append(e); jits.append(j)
            e, j = float(np.mean(errs)), float(np.mean(jits))
            if base is None:
                base = (e, j)
                print(f"  {name:18s} {e:8.3f} {j:8.3f}   {'--':>11s} {'--':>11s}")
            else:
                print(f"  {name:18s} {e:8.3f} {j:8.3f}   "
                      f"{(e/base[0]-1)*100:+10.1f}% {(j/base[1]-1)*100:+10.1f}%")
    print("\n(px at 192. err_px includes smoothing lag -- a config that cuts jit_px")
    print(" while raising err_px is trading accuracy for smoothness, not winning.)")


if __name__ == "__main__":
    main()
