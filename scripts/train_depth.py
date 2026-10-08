#!/usr/bin/env python3
"""Train the landmarker with depth on the rendered corpus.

Separate from train.py because the data, the labels and the augmentation all
differ: 55x3 labels instead of 55x2, splits by subject rather than by image,
and the perspective warp disabled because a homography cannot produce the
parallax a depth label encodes.

    python scripts/train_depth.py --data data/depth3d --epochs 40
"""
import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/depth3d")
    p.add_argument("--out", default="runs/depth_h1")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--z-weight", type=float, default=1.0)
    p.add_argument("--z-visible-only", action="store_true")
    p.add_argument("--width-mult", type=float, default=1.0)
    p.add_argument("--limit-batches", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from torch.utils.data import DataLoader

    from data.dataset import AugmentationParams
    from data.depth_dataset import EarDepthDataset
    from model.depth_module import EarDepth3DModule

    pl.seed_everything(a.seed, workers=True)
    d = pathlib.Path(a.data)
    # perspective_deg stays 0: see data/depth_dataset and the note in
    # _geo_augment. Everything else is the 2D recipe unchanged.
    aug = AugmentationParams(perspective_deg=0.0)
    train = EarDepthDataset(d / "train.csv", d, augmentation=aug)
    val = EarDepthDataset(d / "val.csv", d, augmentation=None)
    print(f"{len(train)} train, {len(val)} val views", flush=True)

    dl = dict(batch_size=a.batch, num_workers=a.workers, pin_memory=True,
              persistent_workers=a.workers > 0)
    tl = DataLoader(train, shuffle=True, drop_last=True, **dl)
    vl = DataLoader(val, shuffle=False, **dl)

    model = EarDepth3DModule(lr=a.lr, max_epochs=a.epochs, arch="heatmap",
                             width_mult=a.width_mult, z_weight=a.z_weight,
                             z_visible_only=a.z_visible_only)
    out = pathlib.Path(a.out)
    ck = ModelCheckpoint(dirpath=out / "ckpt", filename="{epoch}-{val_z_mae:.4f}",
                         monitor="val_z_mae", mode="min", save_top_k=2,
                         save_last=True)
    tr = pl.Trainer(
        max_epochs=a.epochs, accelerator="auto", devices=1,
        default_root_dir=str(out), logger=CSVLogger(str(out), name="csv"),
        callbacks=[ck], log_every_n_steps=20,
        limit_train_batches=a.limit_batches, limit_val_batches=a.limit_batches,
        enable_progress_bar=True)
    tr.fit(model, tl, vl)
    print(f"\nbest val_z_mae {ck.best_model_score:.5f} -> {ck.best_model_path}")


if __name__ == "__main__":
    main()
