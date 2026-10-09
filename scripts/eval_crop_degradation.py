#!/usr/bin/env python3
"""Was the real-photo result the domain, or was it my own crop step?

Head 1 scores NME 0.011 on held-out renders and 0.044-0.088 on AudioEar2D
photographs. Before that is read as a domain gap it has to survive a control,
because the photo test did something to the image that the render test did
not. AudioEar2D ships TIGHT EAR CROPS -- the ear fills 0.835 x 0.827 of the
frame against 0.354 x 0.677 in the render corpus -- so matching the framing
meant cropping OUTWARD past the image edge and padding the gap with black.
The model therefore saw an ear at the right scale surrounded by black bars
instead of a head. That is a second off-distribution change, introduced by
the test rather than by the photographs.

So: do the same thing to renders, where ground truth exists.

  1. crop each render tight to the ear at the photo corpus' framing, throwing
     away the surrounding head exactly as AudioEar2D already has
  2. crop back out to the render framing, padding black where there is now
     nothing -- the identical operation --match-framing performs on a photo
  3. score against the UNCHANGED labels

The round trip returns to the original scale by construction (tight side is
ear/0.835, re-expanded side is ear/0.354, which is the render framing), so z
stays in its original crop-width units and the labels need no adjustment.
The only thing lost is the context, which is the thing being tested.

  if renders survive this    -> the crop step is innocent and the photo
                                result is a real domain gap
  if renders collapse too    -> the photo number measures my own pipeline,
                                and the domain gap is unmeasured

    python scripts/eval_crop_degradation.py --n 400 \
        --ckpt runs/sb_a_prop/ckpt/epoch=132-val_z_mae=0.0140.ckpt
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
    p.add_argument("--data", default="data/depth3d_v5")
    p.add_argument("--split", default="common_test.csv")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n-points", type=int, default=55)
    p.add_argument("--n", type=int, default=400, help="views to score")
    p.add_argument("--tight", nargs=2, type=float, default=[0.835, 0.827],
                   metavar=("W", "H"), help="the photo corpus' framing")
    p.add_argument("--render", nargs=2, type=float, default=[0.354, 0.677],
                   metavar=("W", "H"), help="the render corpus' framing")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    import torch
    from PIL import Image
    from model.depth_module import EarDepth3DModule

    d = pathlib.Path(a.data)
    rows = list(csv.DictReader(open(d / a.split)))
    L = np.load(d / "landmarks.npy", mmap_mode="r")
    k = a.n_points
    m = EarDepth3DModule.load_from_checkpoint(
        a.ckpt, map_location="cuda" if torch.cuda.is_available() else "cpu").eval()
    dev = next(m.parameters()).device
    size = m.hparams.get("image_size", 192)
    rng = np.random.default_rng(a.seed)
    take = [rows[i] for i in rng.choice(len(rows), min(a.n, len(rows)), False)]

    def predict(im):
        x = torch.from_numpy(np.asarray(im.resize((size, size), Image.BILINEAR),
                                        np.float32).transpose(2, 0, 1) / 255.)
        x = ((x - 0.5) / 0.5)[None].to(dev)
        with torch.no_grad():
            return m(x).view(-1, 3).cpu().numpy()

    def recrop(im, lab, frac):
        """Square crop on the label box so the ear fills `frac`. Out-of-image
        area comes back black, which is the whole point."""
        fw, fh = frac
        W0, H0 = im.size
        P = lab * [W0, H0]
        (x0, y0), (x1, y1) = P.min(0), P.max(0)
        side = max((x1 - x0) / fw, (y1 - y0) / fh)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        box = (int(round(cx - side / 2)), int(round(cy - side / 2)),
               int(round(cx + side / 2)), int(round(cy + side / 2)))
        out = im.crop(box)
        new = (P - [box[0], box[1]]) / [box[2] - box[0], box[3] - box[1]]
        return out, new

    acc = {kk: [] for kk in ("base_nme", "base_z", "deg_nme", "deg_z",
                             "tight_nme", "tight_z", "zbase")}
    for r in take:
        gi = int(r["idx"])
        gt = np.asarray(L[gi][:k], float)
        if k < L.shape[1]:
            gt[:, 2] -= gt[:, 2].mean()
        im = Image.open(d / r["image_file"]).convert("RGB")

        pr = predict(im)
        acc["base_nme"].append(np.linalg.norm(pr[:, :2] - gt[:, :2], axis=1).mean())
        acc["base_z"].append(np.abs(pr[:, 2] - gt[:, 2]).mean())
        acc["zbase"].append(np.abs(gt[:, 2]).mean())

        # step 1: throw the head away, as AudioEar2D already has
        tight, lab_t = recrop(im, gt[:, :2], a.tight)
        pt = predict(tight)
        acc["tight_nme"].append(np.linalg.norm(pt[:, :2] - lab_t, axis=1).mean())
        acc["tight_z"].append(np.abs(pt[:, 2] - gt[:, 2]).mean())

        # step 2: crop back out to the render framing, padding black
        deg, lab_d = recrop(tight, lab_t, a.render)
        pd = predict(deg)
        acc["deg_nme"].append(np.linalg.norm(pd[:, :2] - lab_d, axis=1).mean())
        acc["deg_z"].append(np.abs(pd[:, 2] - gt[:, 2]).mean())

    n = len(take)
    zb = float(np.mean(acc["zbase"]))
    print(f"\n{n} held-out render views, {k} landmarks.  z baseline {zb:.5f}\n")
    print(f"{'condition':<42}{'NME':>9}{'z MAE':>9}{'z skill':>9}")
    for label, kn, kz in [
            ("as rendered (in domain)", "base_nme", "base_z"),
            (f"tight crop {a.tight[0]}x{a.tight[1]} (photo framing)",
             "tight_nme", "tight_z"),
            (f"  then back out to {a.render[0]}x{a.render[1]}, black padded",
             "deg_nme", "deg_z")]:
        nme, z = float(np.mean(acc[kn])), float(np.mean(acc[kz]))
        print(f"{label:<42}{nme:9.5f}{z:9.5f}{1 - z / zb:9.4f}")
    print(f"\nFor comparison, the same model on AudioEar2D photographs scores "
          f"NME 0.044-0.088.")
    b, g = float(np.mean(acc["base_nme"])), float(np.mean(acc["deg_nme"]))
    print(f"The crop round trip alone moves renders {b:.5f} -> {g:.5f} "
          f"({g / b:.1f}x).")


if __name__ == "__main__":
    main()
