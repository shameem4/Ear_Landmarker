"""Train the EarLandmarker model.

Usage:
    python train.py
    python train.py --epochs 500 --batch-size 128 --lr 1e-3
    python train.py --resume last              # resume from last checkpoint
    python train.py --resume best              # resume from best checkpoint
    python train.py --resume runs/checkpoints/EarLandmarker_epoch=092_val/nme=0.0351.ckpt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from data.dataset import EarLandmarkDataset, AugmentationParams
from model.lightning_module import EarLandmarkerModule

torch.set_float32_matmul_precision("high")

PROJECT = Path(__file__).resolve().parent
DATA_DIR = PROJECT / "data" / "preprocessed"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EarLandmarker")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50,
                        help="Early stopping patience (epochs without val/nme improvement)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--blazeear-ckpt", type=str, default=None,
                        help="BlazeEar checkpoint for backbone initialization")
    parser.add_argument("--arch", type=str, default="gap", choices=["gap", "heatmap"],
                        help="gap: GAP+FC coordinate regression (v1). "
                             "heatmap: soft-argmax over 24x24 heatmaps.")
    parser.add_argument("--tau", type=float, default=1.0,
                        help="Soft-argmax softmax temperature (--arch heatmap only)")
    parser.add_argument("--wing-w", type=float, default=0.04)
    parser.add_argument("--wing-epsilon", type=float, default=0.01)
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint: 'last', 'best', or path to .ckpt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perspective-deg", type=float, default=0.0,
                        help="Simulated out-of-plane turn in degrees (0 = off)")
    parser.add_argument("--landmarks", type=str, default="landmarks.npy",
                        help="Landmark array under data/preprocessed/")
    parser.add_argument("--run-name", type=str, default="EarLandmarker",
                        help="Names the checkpoint/log subdirectory, so concurrent "
                             "experiments do not overwrite each other")
    args = parser.parse_args()

    ckpt_dir = PROJECT / "runs" / "checkpoints" / args.run_name

    pl.seed_everything(args.seed)

    # Dataset
    train_aug = AugmentationParams(
        horizontal_flip=True,
        flip_prob=0.5,
        translation=0.05,
        rotation_deg=15.0,
        color_jitter={"brightness": 0.3, "contrast": 0.3, "saturation": 0.2, "hue": 0.05},
        bbox_jitter=0.1,
        bbox_jitter_prob=0.5,
        perspective_deg=args.perspective_deg,
    )

    train_ds = EarLandmarkDataset(
        split_csv=DATA_DIR / "train.csv",
        data_dir=DATA_DIR,
        image_size=args.image_size,
        augmentation=train_aug,
        landmarks_file=args.landmarks,
    )
    val_ds = EarLandmarkDataset(
        split_csv=DATA_DIR / "val.csv",
        data_dir=DATA_DIR,
        image_size=args.image_size,
        landmarks_file=args.landmarks,
    )
    test_csv = DATA_DIR / "test.csv"
    test_ds = EarLandmarkDataset(
        split_csv=test_csv,
        data_dir=DATA_DIR,
        image_size=args.image_size,
        landmarks_file=args.landmarks,
    ) if test_csv.exists() else None

    loader_kwargs = dict(
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=3 if args.num_workers > 0 else None,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs,
    ) if test_ds is not None else None

    # Model
    module = EarLandmarkerModule(
        num_landmarks=55,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
        wing_w=args.wing_w,
        wing_epsilon=args.wing_epsilon,
        blazeear_ckpt=args.blazeear_ckpt,
        arch=args.arch,
        tau=args.tau,
    )

    if args.compile:
        module.model = torch.compile(module.model)
        print("Model compiled with torch.compile")

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="EarLandmarker_{epoch:03d}_nme={val_nme:.4f}",
        auto_insert_metric_name=False,
        monitor="val_nme",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    early_stop = EarlyStopping(
        monitor="val_nme",
        mode="min",
        patience=args.patience,
        verbose=True,
    )

    # Logger
    logger = CSVLogger(
        save_dir=PROJECT / "runs" / "logs",
        name=args.run_name,
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        callbacks=[checkpoint_cb, lr_monitor, early_stop],
        logger=logger,
        log_every_n_steps=10,
        val_check_interval=1.0,
    )

    # Resolve resume checkpoint
    ckpt_path = None
    if args.resume:
        if args.resume == "last":
            ckpt_path = ckpt_dir / "last.ckpt"
        elif args.resume == "best":
            candidates = list(ckpt_dir.glob("EarLandmarker_*.ckpt"))
            candidates += list(ckpt_dir.rglob("nme=*.ckpt"))
            best_nme = 999.0
            for c in candidates:
                try:
                    nme = float(c.stem.split("nme=")[1])
                    if nme < best_nme:
                        best_nme = nme
                        ckpt_path = c
                except (IndexError, ValueError):
                    continue
        else:
            ckpt_path = Path(args.resume)
        if ckpt_path and ckpt_path.exists():
            print(f"Resuming from: {ckpt_path}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    trainer.fit(module, train_loader, val_loader, ckpt_path=ckpt_path)

    print(f"\nBest model: {checkpoint_cb.best_model_path}")
    print(f"Best val NME: {checkpoint_cb.best_model_score:.4f}  (model-selection metric)")

    # Held-out test set: evaluated once, on the best checkpoint, after all
    # selection is done. This is the number to report.
    if test_loader is not None:
        print("\nEvaluating best checkpoint on held-out test set...")
        trainer.test(module, test_loader, ckpt_path=checkpoint_cb.best_model_path)
    else:
        print("\nNo test.csv found -- run `python data/split.py` to create one.")


if __name__ == "__main__":
    main()
