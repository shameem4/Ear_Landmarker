#!/usr/bin/env python3
"""Score a depth head on real photographs, against labels from their own scans.

The first real-photo depth measurement in this project. Labels come from
scripts/fit_photo_depth.py --build: each AudioEar3D photograph aligned to the
3D scan of that same ear, 2D taken as the average of the hand annotation and
the shipped model's prediction.

READ THE FLOOR BEFORE THE SCORE. The labels carry about 0.0242 of error
against a signal of 0.0443, so z_skill here is capped near 0.45 and a perfect
model cannot score above it. This set cannot grade a model that is already
good to ~0.014; it can grade one in the 0.05-0.09 range, which is where a
render-trained head lands on photographs. Quote the cap whenever quoting the
score.

z = 0 is reported alongside, as everywhere else in this project, because a
depth number without its baseline can look respectable while carrying no
information at all.

    python scripts/eval_photo_depth.py --ckpt runs/sb_a_prop/ckpt/....ckpt
"""
import argparse
import csv
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", default="data/photo3d/audioear3d_depth.npz")
    p.add_argument("--photo-data", default="data/preprocessed")
    p.add_argument("--models", nargs="+", required=True, help="name=ckpt pairs")
    p.add_argument("--n-points", type=int, default=55)
    a = p.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    import torch
    from PIL import Image
    from model.depth_module import EarDepth3DModule

    B = np.load(a.bench, allow_pickle=True)
    Z, IDX = B["z"], B["idx"]
    floor, base = float(B["label_error"]), float(B["zbase"])
    d = pathlib.Path(a.photo_data)
    rows = {int(r["idx"]): r for r in csv.DictReader(open(d / "manifest.csv"))}
    K = a.n_points
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"{len(IDX)} photographs.  label error {floor:.4f} of a {base:.4f} "
          f"signal -> z_skill capped near {1 - floor / base:.2f}\n")
    print(f"{'model':<22}{'z MAE':>9}{'z skill':>9}{'xy NME':>9}"
          f"{'beats flat':>12}")
    print(f"{'predict z = 0':<22}{base:>9.4f}{0.0:>9.3f}{'-':>9}{'-':>12}")
    for spec in a.models:
        name, ckpt = spec.split("=", 1)
        m = EarDepth3DModule.load_from_checkpoint(
            ckpt, map_location=dev).eval().to(dev)
        zs, nm = [], []
        for gi, ztrue in zip(IDX, Z):
            r = rows[int(gi)]
            im = Image.open(d / r["image_file"]).convert("RGB")
            t = np.asarray(im.resize((192, 192), Image.BILINEAR),
                           np.float32).transpose(2, 0, 1)[None] / 255.0
            with torch.no_grad():
                P = m(torch.from_numpy((t - 0.5) / 0.5).float().to(dev)
                      ).view(-1, 3).cpu().numpy()
            zp = P[:K, 2] - P[:K, 2].mean()
            zs.append(np.abs(zp - ztrue[:K]).mean())
            nm.append(np.linalg.norm(
                P[:K, :2] - 0.0, axis=1).mean() * 0)     # xy needs its own gt
        zs = np.asarray(zs)
        print(f"{name:<22}{zs.mean():>9.4f}{1 - zs.mean() / base:>9.3f}"
              f"{'-':>9}{100 * (zs < base).mean():>11.0f}%")
    print(f"\na score at or above the {1 - floor / base:.2f} cap would mean the "
          f"label noise is being fitted, not the ear")


if __name__ == "__main__":
    main()
