"""Measure landmark stability (jitter) without needing video data.

Motivation: the v2 goal was accuracy AND stability, but NME says nothing about
frame-to-frame jitter, and the repo has no video. This builds a "virtual video"
from each test still and measures how much predictions move when they should not.

Two independent perturbations, because they isolate different failure modes:

  geometric  -- wobble the crop window (small centre offset + scale change),
                which is exactly what an upstream detector's bbox does between
                frames. Predictions are mapped back through the known inverse
                transform to canonical image coordinates, so a perfectly
                equivariant model yields identical coordinates on every frame
                and scores zero. Any spread is the model, not the motion.

  photometric -- keep the geometry byte-identical and vary only brightness and
                sensor noise. The correct output is literally unchanged, so this
                needs no inverse mapping and is the cleanest possible probe.

Reported per model:
  jitter_px  -- mean over landmarks of the per-landmark std of canonical
                position across frames (px at 192). Overall wobble magnitude.
  succ_px    -- mean successive-frame displacement (px at 192). This is closest
                to what the eye reads as "jitter" in a live demo.

Usage:
    python scripts/eval_jitter.py                      # all runs found in runs/checkpoints
    python scripts/eval_jitter.py --runs v2_fixed v2_heatmap
    python scripts/eval_jitter.py --frames 16 --limit 300
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.ear_landmarker import EarLandmarker, EarLandmarkerHeatmap  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT / "data" / "preprocessed"
IMG_SIZE = 192


def load_checkpoint(path: Path) -> torch.nn.Module:
    """Build the right architecture from the checkpoint's own hyperparameters."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    hp = ckpt.get("hyper_parameters", {}) or {}
    arch = hp.get("arch", "gap")
    model = (
        EarLandmarkerHeatmap(num_landmarks=55, tau=hp.get("tau", 1.0))
        if arch == "heatmap"
        else EarLandmarker(num_landmarks=55)
    )
    state = {
        k.removeprefix("model."): v
        for k, v in ckpt.get("state_dict", ckpt).items()
        if k.startswith("model.")
    }
    model.load_state_dict(state, strict=True)
    return model.eval(), arch


def best_checkpoint(run_dir: Path) -> Path | None:
    """Lowest-NME checkpoint in a run directory."""
    best, best_nme = None, float("inf")
    for c in run_dir.rglob("*nme=*.ckpt"):
        try:
            nme = float(c.stem.split("nme=")[1])
        except (IndexError, ValueError):
            continue
        if nme < best_nme:
            best, best_nme = c, nme
    return best


def load_images(limit: int | None) -> list[np.ndarray]:
    """Test-split images at 2x working size, so crop wobble resamples cleanly."""
    with open(DATA_DIR / "test.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if limit:
        rows = rows[:limit]
    out = []
    for r in rows:
        img = Image.open(DATA_DIR / r["image_file"]).convert("RGB")
        img = img.resize((IMG_SIZE * 2, IMG_SIZE * 2), Image.BILINEAR)
        out.append(np.asarray(img, dtype=np.float32) / 255.0)
    return out


def make_geometric_frames(
    img: np.ndarray, n_frames: int, rng: np.random.Generator,
    max_shift: float, max_scale: float,
):
    """Crop-window wobble. Returns (frames, windows) where each window is the
    (x0, y0, side) actually used, in source-pixel units, for exact inversion."""
    src = IMG_SIZE * 2
    base_side = src * 0.85                       # leave room to wobble
    frames, windows = [], []
    for _ in range(n_frames):
        s = 1.0 + rng.uniform(-max_scale, max_scale)
        side = base_side * s
        cx = src / 2 + rng.uniform(-max_shift, max_shift) * src
        cy = src / 2 + rng.uniform(-max_shift, max_shift) * src
        x0, y0 = cx - side / 2, cy - side / 2
        # keep the window inside the source
        x0 = float(np.clip(x0, 0, src - side))
        y0 = float(np.clip(y0, 0, src - side))
        # Use the exact integer bounds for both the crop and its inverse, so the
        # mapping back is consistent to the pixel rather than to within rounding.
        x0i, y0i = int(round(x0)), int(round(y0))
        sidei = int(round(side))
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        crop = t[:, :, y0i:y0i + sidei, x0i:x0i + sidei]
        crop = F.interpolate(crop, size=(IMG_SIZE, IMG_SIZE),
                             mode="bilinear", align_corners=False)
        frames.append(crop)
        windows.append((float(x0i), float(y0i), float(sidei)))
    return torch.cat(frames, 0), windows


def make_photometric_frames(
    img: np.ndarray, n_frames: int, rng: np.random.Generator,
    noise_std: float, bright: float,
):
    """Identical geometry; only brightness and pixel noise change."""
    src = IMG_SIZE * 2
    side = src * 0.85
    x0 = y0 = (src - side) / 2
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    crop = t[:, :, int(y0):int(y0 + side), int(x0):int(x0 + side)]
    crop = F.interpolate(crop, size=(IMG_SIZE, IMG_SIZE),
                         mode="bilinear", align_corners=False)
    frames = []
    for _ in range(n_frames):
        f = crop * (1.0 + rng.uniform(-bright, bright))
        f = f + torch.from_numpy(
            rng.normal(0, noise_std, size=tuple(f.shape)).astype(np.float32)
        )
        frames.append(f.clamp(0, 1))
    return torch.cat(frames, 0)


@torch.no_grad()
def predict(model, frames: torch.Tensor, device: torch.device) -> np.ndarray:
    """(T, 3, H, W) in [0,1] -> (T, 55, 2) normalized crop coords."""
    x = (frames.to(device) - 0.5) / 0.5
    return model(x).reshape(len(frames), 55, 2).cpu().numpy()


def to_canonical(pred: np.ndarray, windows, base_side: float) -> np.ndarray:
    """Map per-frame crop coords back to a common frame, in NOMINAL-CROP units.

    A point at normalized u in a crop of side `side` starting at x0 sits at
    source pixel x0 + u*side. Dividing by `base_side` (the unwobbled crop side)
    expresses it in the same units the model's own [0,1] output uses, so the
    photometric path -- which is already in those units -- needs no different
    conversion. Only spreads and differences are reported, so the arbitrary
    origin offset is irrelevant.
    """
    out = np.empty_like(pred)
    for t, (x0, y0, side) in enumerate(windows):
        out[t, :, 0] = (x0 + pred[t, :, 0] * side) / base_side
        out[t, :, 1] = (y0 + pred[t, :, 1] * side) / base_side
    return out


def jitter_stats(coords: np.ndarray) -> tuple[float, float]:
    """(spread, successive) in px at 192, from (T, 55, 2) coords in crop units.

    spread: per-landmark std over frames, averaged over landmarks and axes.
    succ:   mean Euclidean displacement between consecutive frames.
    """
    spread = float(coords.std(axis=0).mean() * IMG_SIZE)
    succ = float(np.linalg.norm(np.diff(coords, axis=0), axis=-1).mean() * IMG_SIZE)
    return spread, succ


def main() -> None:
    p = argparse.ArgumentParser(description="Measure landmark jitter")
    p.add_argument("--runs", nargs="*", default=None)
    p.add_argument("--frames", type=int, default=12)
    p.add_argument("--limit", type=int, default=250,
                   help="Test images to use (None = all 861)")
    p.add_argument("--max-shift", type=float, default=0.008,
                   help="Per-frame crop centre offset, fraction of source (0.008 ~ 3px)")
    p.add_argument("--max-scale", type=float, default=0.02,
                   help="Per-frame crop scale wobble (0.02 = +/-2%%)")
    p.add_argument("--noise-std", type=float, default=0.01)
    p.add_argument("--bright", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_root = PROJECT / "runs" / "checkpoints"
    runs = args.runs or sorted(
        d.name for d in ckpt_root.iterdir()
        if d.is_dir() and best_checkpoint(d) is not None
    )
    if not runs:
        sys.exit("no checkpoints found under runs/checkpoints")

    print(f"device={device}  frames/image={args.frames}  images={args.limit}")
    print(f"geometric wobble: centre +/-{args.max_shift*IMG_SIZE*2:.1f}px of source, "
          f"scale +/-{args.max_scale*100:.0f}%")
    print(f"photometric: brightness +/-{args.bright*100:.0f}%, noise std {args.noise_std}")
    print()
    images = load_images(args.limit)

    print(f"{'run':14s} {'arch':8s} {'GEOMETRIC':>21s}   {'PHOTOMETRIC':>21s}")
    print(f"{'':14s} {'':8s} {'spread':>9s} {'succ':>11s}   {'spread':>9s} {'succ':>11s}")
    print("-" * 74)
    results = {}
    for run in runs:
        ck = best_checkpoint(ckpt_root / run)
        model, arch = load_checkpoint(ck)
        model.to(device)

        g_sp, g_su, p_sp, p_su = [], [], [], []
        for i, img in enumerate(images):
            rng = np.random.default_rng(args.seed + i)   # same wobble for every model
            frames, windows = make_geometric_frames(
                img, args.frames, rng, args.max_shift, args.max_scale)
            canon = to_canonical(predict(model, frames, device), windows, IMG_SIZE * 2 * 0.85)
            s, u = jitter_stats(canon)
            g_sp.append(s); g_su.append(u)

            rng = np.random.default_rng(10_000 + args.seed + i)
            frames = make_photometric_frames(
                img, args.frames, rng, args.noise_std, args.bright)
            pred = predict(model, frames, device)        # geometry fixed -> already canonical
            s, u = jitter_stats(pred)
            p_sp.append(s); p_su.append(u)

        r = dict(geo_spread=np.mean(g_sp), geo_succ=np.mean(g_su),
                 pho_spread=np.mean(p_sp), pho_succ=np.mean(p_su))
        results[run] = r
        print(f"{run:14s} {arch:8s} {r['geo_spread']:8.3f}px {r['geo_succ']:10.3f}px   "
              f"{r['pho_spread']:8.3f}px {r['pho_succ']:10.3f}px")

    print("\n(lower is better; all figures in px at 192x192)")
    if "v2_fixed" in results and "v2_heatmap" in results:
        a, b = results["v2_fixed"], results["v2_heatmap"]
        print("\nsoft-argmax vs GAP head:")
        for k, lbl in (("geo_succ", "geometric successive"),
                       ("pho_succ", "photometric successive")):
            print(f"  {lbl:24s} {a[k]:.3f} -> {b[k]:.3f} px "
                  f"({(b[k]/a[k]-1)*100:+.1f}%)")


if __name__ == "__main__":
    main()
