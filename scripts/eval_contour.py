"""Parameterisation-invariant evaluation: compare models that disagree on what
each landmark index means.

NME cannot compare a model trained on canonicalized labels against one trained on
the originals: they predict genuinely different target positions for the same
index, so the canonicalized model would be penalised for being correct. Any fair
comparison must score the SHAPE, not the indexing.

Three metrics, all index-free:

  chamfer_px    symmetric Chamfer distance between the predicted contour and the
                ground-truth contour, per linestrip, after dense arc-length
                resampling. Pure shape agreement.
  normal_px     mean distance from each predicted point to the GT contour. This
                is the "am I on the ear" error, with tangential drift removed by
                construction.
  meas_err      relative error of the contour-based anthropometric measurements
                (model/measure.py), which is what actually reaches the user.

Usage:
    python scripts/eval_contour.py
    python scripts/eval_contour.py --runs v2_heatmap v3_persp v4_persp_canon
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_jitter import IMG_SIZE, best_checkpoint, load_checkpoint  # noqa: E402
from model.measure import LINESTRIPS, measure_ear, resample_polyline  # noqa: E402

DATA = ROOT / "data" / "preprocessed"
DENSE = 120          # arc-length samples per contour for Chamfer


def point_to_polyline(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest vertex of a densely sampled poly.

    Dense resampling makes nearest-vertex a close approximation of true
    point-to-segment distance, without the segment-projection bookkeeping.
    """
    d = np.linalg.norm(pts[:, None, :] - poly[None, :, :], axis=-1)
    return d.min(axis=1)


def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric mean nearest-neighbour distance between two point sets."""
    return float(0.5 * (point_to_polyline(a, b).mean() + point_to_polyline(b, a).mean()))


def evaluate(run: str, rows, lms_orig, lms_canon, device) -> dict:
    ck = best_checkpoint(ROOT / "runs" / "checkpoints" / run)
    if ck is None:
        return {}
    model, arch = load_checkpoint(ck)
    model.to(device)

    cham, norm, meas = [], [], []
    buf, idxs = [], []

    def flush():
        nonlocal buf, idxs
        if not buf:
            return
        x = ((torch.cat(buf, 0).to(device)) - 0.5) / 0.5
        with torch.no_grad():
            p = model(x).reshape(len(buf), 55, 2).cpu().numpy().astype(np.float64)
        for pred, gi in zip(p, idxs):
            # Ground truth shape is the SAME curve either way; canonicalization
            # only redistributes vertices along it. Use the original vertices as
            # the reference curve so no model is favoured.
            gt = np.asarray(lms_orig[gi], dtype=np.float64)
            cs, ns = [], []
            for name, (a, b) in LINESTRIPS.items():
                gp = resample_polyline(gt[a:b], DENSE)
                pp = resample_polyline(pred[a:b], DENSE)
                cs.append(chamfer(pp, gp))
                ns.append(point_to_polyline(pred[a:b], gp).mean())
            cham.append(np.mean(cs))
            norm.append(np.mean(ns))

            mg, mp = measure_ear(gt), measure_ear(pred)
            rel = [abs(mp[k] - mg[k]) / mg[k] for k in mg if mg[k] > 1e-6]
            meas.append(np.mean(rel))
        buf, idxs = [], []

    for r in rows:
        im = Image.open(DATA / r["image_file"]).convert("RGB").resize(
            (IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        buf.append(torch.from_numpy(
            np.asarray(im, np.float32) / 255.).permute(2, 0, 1).unsqueeze(0))
        idxs.append(int(r["idx"]))
        if len(buf) == 64:
            flush()
    flush()

    return {
        "arch": arch,
        "chamfer_px": float(np.mean(cham)) * IMG_SIZE,
        "normal_px": float(np.mean(norm)) * IMG_SIZE,
        "meas_err_pct": float(np.mean(meas)) * 100,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Parameterisation-invariant evaluation")
    p.add_argument("--runs", nargs="*",
                   default=["v2_fixed", "v2_heatmap", "v3_persp", "v4_persp_canon"])
    p.add_argument("--limit", type=int, default=400)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = list(csv.DictReader(open(DATA / "test.csv", encoding="utf-8")))[:args.limit]
    lms_orig = np.load(DATA / "landmarks.npy", mmap_mode="r")
    canon_path = DATA / "landmarks_canonical.npy"
    lms_canon = np.load(canon_path, mmap_mode="r") if canon_path.exists() else None

    print(f"device={device}  test images={len(rows)}")
    print("All metrics are index-free, so models trained on different label")
    print("parameterisations are directly comparable.\n")
    print(f"{'run':18s} {'arch':9s} {'chamfer':>9s} {'normal':>9s} {'meas err':>10s}")
    print("-" * 60)
    base = None
    for run in args.runs:
        if not (ROOT / "runs" / "checkpoints" / run).exists():
            print(f"{run:18s} (not trained yet)")
            continue
        r = evaluate(run, rows, lms_orig, lms_canon, device)
        if not r:
            print(f"{run:18s} (no checkpoint)")
            continue
        if base is None:
            base = r
        print(f"{run:18s} {r['arch']:9s} {r['chamfer_px']:8.3f}p "
              f"{r['normal_px']:8.3f}p {r['meas_err_pct']:9.2f}%")
    print("\nchamfer/normal in px at 192; meas err is mean relative error over the")
    print("five contour-based measurements (ear length/width, concha, tragus).")


if __name__ == "__main__":
    main()
