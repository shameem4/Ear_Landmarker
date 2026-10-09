#!/usr/bin/env python3
"""Overlay head 1's predictions on the labels, for samples from all three splits.

TWO VIEWS PER SAMPLE, because the front view cannot show the thing this model
was built for. The top panel is the image with x, y overlaid; the bottom is the
same 55 points plotted as (x, z) -- the ear seen from ABOVE -- which is where a
depth error is visible as geometry rather than as a number.

Train is included deliberately. A model that fits train and misses val/test
looks different here from one that misses everywhere, and that distinction is
invisible in a single test-set metric.

z is root-relative and in crop-width units, so the top view is to the same
scale as the front view and the two are directly comparable.

    python scripts/show_depth_samples.py --out runs/depth_samples.png
"""
import argparse
import csv
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ear3d.scheme import STRIPS                                 # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/depth3d")
    p.add_argument("--ckpt",
                   default="runs/final_z/ckpt/epoch=139-val_z_mae=0.0117.ckpt")
    p.add_argument("--per-split", type=int, default=4)
    p.add_argument("--splits", nargs="+",
                   default=["train.csv", "val.csv", "test.csv"])
    p.add_argument("--n-points", type=int, default=0,
                   help="first N landmarks; must match what the model was "
                        "trained on, and z is re-centred on them exactly as "
                        "data/depth_dataset does it")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/depth_samples.png")
    a = p.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from PIL import Image
    from model.depth_module import EarDepth3DModule

    d = pathlib.Path(a.data)
    L = np.load(d / "landmarks.npy", mmap_mode="r")
    V = np.load(d / "visible.npy", mmap_mode="r")
    m = EarDepth3DModule.load_from_checkpoint(
        a.ckpt, map_location="cuda" if torch.cuda.is_available() else "cpu").eval()
    dev = next(m.parameters()).device
    size = m.hparams.get("image_size", 192)
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    splits = a.splits
    # MUST MATCH THE DATASET, or the picture lies. The corpus stores z centred
    # on all 95 points; a model trained on the first 55 was fed z re-centred
    # on those 55. Plotting the stored z against such a model's output would
    # add a per-view offset of about -0.027 crop widths and draw a depth bias
    # that does not exist. One line, and it has to be the same line.
    k = a.n_points or L.shape[1]
    recentre = k < L.shape[1]
    rng = np.random.default_rng(a.seed)
    cols = a.per_split
    fig, ax = plt.subplots(2 * len(splits), cols,
                           figsize=(3.0 * cols, 3.2 * len(splits) * 2))

    for si, sp in enumerate(splits):
        rows = list(csv.DictReader(open(d / sp)))
        take = [rows[i] for i in rng.choice(len(rows), cols, replace=False)]
        for ci, r in enumerate(take):
            gi = int(r["idx"])
            im = Image.open(d / r["image_file"]).convert("RGB")
            W, H = im.size
            gt = np.asarray(L[gi][:k], float)
            vis = np.asarray(V[gi][:k]).astype(bool)
            if recentre:
                gt[:, 2] -= gt[:, 2].mean()

            x = torch.from_numpy(
                np.asarray(im.resize((size, size), Image.BILINEAR),
                           np.float32).transpose(2, 0, 1) / 255.)
            x = ((x - torch.tensor(mean)[:, None, None])
                 / torch.tensor(std)[:, None, None])[None].to(dev)
            with torch.no_grad():
                pr = m(x).view(-1, 3).cpu().numpy()

            nme = np.linalg.norm(pr[vis, :2] - gt[vis, :2], axis=1).mean()
            zmae = np.abs(pr[vis, 2] - gt[vis, 2]).mean()

            # --- front view -------------------------------------------------
            b = ax[2 * si, ci]
            b.imshow(im)
            for lo, hi in STRIPS:
                b.plot(gt[lo:hi, 0] * W, gt[lo:hi, 1] * H, "-", c="#39ff88",
                       lw=1.4, alpha=.9)
                b.plot(pr[lo:hi, 0] * W, pr[lo:hi, 1] * H, "-", c="#ff2d95",
                       lw=1.4, alpha=.9)
            b.plot(gt[:, 0] * W, gt[:, 1] * H, ".", c="#39ff88", ms=3)
            b.plot(pr[:, 0] * W, pr[:, 1] * H, ".", c="#ff2d95", ms=3)
            b.set_title(f"{sp[:-4]}  {r['ear'][:16]}\nNME {nme:.4f}",
                        fontsize=7)
            b.axis("off")

            # --- top view, x against z --------------------------------------
            t = ax[2 * si + 1, ci]
            for lo, hi in STRIPS:
                t.plot(gt[lo:hi, 0], gt[lo:hi, 2], "-", c="#39ff88", lw=1.3)
                t.plot(pr[lo:hi, 0], pr[lo:hi, 2], "-", c="#ff2d95", lw=1.3)
            t.plot(gt[:, 0], gt[:, 2], ".", c="#39ff88", ms=3)
            t.plot(pr[:, 0], pr[:, 2], ".", c="#ff2d95", ms=3)
            t.set_aspect("equal")
            t.invert_yaxis()
            t.set_title(f"from above (x, z)   z MAE {zmae:.4f}", fontsize=7)
            t.tick_params(labelsize=5)
            t.grid(alpha=.25)

    fig.suptitle(f"head 1, {k} landmarks: labels in GREEN, prediction in "
                 "MAGENTA.  "
                 "Upper row of each pair = image (x, y); lower = seen from "
                 "above (x, z), same scale.", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=135)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
