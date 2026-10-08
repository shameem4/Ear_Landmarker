#!/usr/bin/env python3
"""Evaluate a depth head, and measure the ceiling the shape basis imposes.

THE CEILING IS THE POINT OF THIS SCRIPT. Head 2 can only ever produce ears the
morphable model can express, so before comparing trained networks it is worth
knowing what the basis itself can do: fit shape coefficients and pose DIRECTLY
to the ground-truth landmarks, with no image and no network, and whatever
residual remains is a hard floor on that head's error. A trained head 2 beating
this number would mean a bug, and one approaching it has nothing left to gain
from more training -- only from more modes.

Every error is reported beside the flat-predictor baseline, because z is
root-relative and predicting zero already scores mean|z|.

    python scripts/eval_depth.py --data data/depth3d --basis lm_basis_v4.npz
    python scripts/eval_depth.py --data data/depth3d --ckpt runs/depth_z/...ckpt
"""
import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def basis_ceiling(basis, target, iters=400, lr=0.05, device="cpu"):
    """Best the basis can do on these landmarks, fitting shape and pose."""
    import torch
    from model.shape_head import rot6d_to_matrix
    d = np.load(basis)
    L0 = torch.tensor(d["mean"], dtype=torch.float32, device=device)
    B = torch.tensor(d["components"], dtype=torch.float32, device=device)
    T = torch.tensor(target, dtype=torch.float32, device=device)
    n, k, _ = T.shape
    m = len(B)
    c = torch.zeros(n, m, device=device, requires_grad=True)
    r = torch.tensor([[1., 0, 0, 0, 1, 0]], device=device).repeat(n, 1).requires_grad_()
    ls = torch.zeros(n, 1, device=device, requires_grad=True)
    t = torch.full((n, 2), 0.5, device=device, requires_grad=True)
    opt = torch.optim.Adam([c, r, ls, t], lr=lr)
    for i in range(iters):
        R = rot6d_to_matrix(r)
        L = L0 + torch.einsum("bm,mkj->bkj", c, B)
        L = torch.einsum("bij,bkj->bki", R, L)
        s = ls.exp()[:, :, None]
        xy = s * L[:, :, :2] + t[:, None, :]
        z = s[:, :, 0] * L[:, :, 2]
        z = z - z.mean(dim=1, keepdim=True)
        pred = torch.cat([xy, z[:, :, None]], dim=-1)
        loss = (pred - T).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        R = rot6d_to_matrix(r)
        L = L0 + torch.einsum("bm,mkj->bkj", c, B)
        L = torch.einsum("bij,bkj->bki", R, L)
        s = ls.exp()[:, :, None]
        xy = s * L[:, :, :2] + t[:, None, :]
        z = s[:, :, 0] * L[:, :, 2]
        z = z - z.mean(dim=1, keepdim=True)
    return (torch.norm(xy - T[:, :, :2], dim=-1).mean().item(),
            (z - T[:, :, 2]).abs().mean().item(),
            T[:, :, 2].abs().mean().item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/depth3d")
    p.add_argument("--split", default="test.csv")
    p.add_argument("--basis", default="")
    p.add_argument("--ckpt", default="")
    p.add_argument("--head", choices=["z", "shape", "lifter", "feat"], default="z")
    p.add_argument("--samples", type=int, default=600)
    p.add_argument("--iters", type=int, default=400)
    a = p.parse_args()

    import csv
    import torch
    d = pathlib.Path(a.data)
    rows = list(csv.DictReader(open(d / a.split)))
    idx = np.array([int(r["idx"]) for r in rows])
    L = np.load(d / "landmarks.npy", mmap_mode="r")
    rng = np.random.default_rng(0)
    take = idx if len(idx) <= a.samples else rng.choice(idx, a.samples, replace=False)
    target = np.asarray(L[np.sort(take)])
    print(f"{a.split}: {len(rows)} views, evaluating on {len(target)}")

    if a.basis:
        xy_err, z_mae, z_base = basis_ceiling(a.basis, target, iters=a.iters)
        nm = np.load(a.basis)
        print(f"\nCEILING of the {nm['n_modes']}-mode basis, fitted directly to "
              f"the labels (no image, no network):")
        print(f"  x,y error {xy_err:.5f} of crop  ({100*xy_err:.2f}%)")
        print(f"  z   MAE   {z_mae:.5f}   flat-predictor baseline {z_base:.5f}"
              f"   skill {1 - z_mae / z_base:.3f}")
        print("  no trained head 2 can do better than this; one that matches it")
        print("  is limited by the basis, not by optimisation")

    if a.ckpt:
        from torch.utils.data import DataLoader
        from data.depth_dataset import EarDepthDataset
        from model.depth_module import (EarDepth3DModule,
                                        EarFeatureLifter3DModule,
                                        EarLifter3DModule, EarShape3DModule)
        cls = {"z": EarDepth3DModule, "shape": EarShape3DModule,
               "lifter": EarLifter3DModule,
               "feat": EarFeatureLifter3DModule}[a.head]
        model = cls.load_from_checkpoint(a.ckpt, map_location="cpu").eval()
        # the lifter heads were TRAINED on the 2D model's precomputed output;
        # evaluating them on ground-truth coordinates with all-ones confidence
        # is a different input distribution and scores far worse -- it read
        # skill 0.169 against 0.702 on validation purely from this
        ds = EarDepthDataset(d / a.split, d, augmentation=None,
                             with_pred=a.head in ("lifter", "feat"))
        dl = DataLoader(ds, batch_size=64, num_workers=4)
        tot = {"nme": 0.0, "z_mae": 0.0, "z_base": 0.0, "n": 0}
        with torch.no_grad():
            for b in dl:
                r = model._shared_step(b)
                w = len(b["image"])
                for k in ("nme", "z_mae", "z_base"):
                    tot[k] += float(r[k]) * w
                tot["n"] += w
        n = tot["n"]
        z_mae, z_base = tot["z_mae"] / n, tot["z_base"] / n
        print(f"\nhead {a.head!r} on {a.split}:")
        print(f"  nme {tot['nme'] / n:.5f}")
        print(f"  z MAE {z_mae:.5f}   baseline {z_base:.5f}   "
              f"skill {1 - z_mae / z_base:.3f}")


if __name__ == "__main__":
    main()
