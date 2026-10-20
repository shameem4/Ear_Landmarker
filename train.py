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

from checkpoint_utils import best_checkpoint
from data.dataset import EarLandmarkDataset, AugmentationParams
from model.lightning_module import EarLandmarkerModule

torch.set_float32_matmul_precision("high")

PROJECT = Path(__file__).resolve().parent
DATA_DIR = PROJECT / "data" / "preprocessed"


class EMACallback(pl.Callback):
    """Exponential moving average of the weights, evaluated and checkpointed.

    Why, concretely: on this corpus three seeds of one configuration disagree
    with EACH OTHER by 0.943 px tangentially while sitting 1.271 px from the
    label, so roughly a third of the tangential error is independent run-to-run
    variance rather than systematic disagreement. Averaging three seeds removes
    10.2% of it, but costs 3x at inference. An EMA buys variance reduction from
    ONE run and is free at inference.

    Averaging the top-3 checkpoints instead does almost nothing (-0.1% to -2.3%)
    because `save_top_k=3` keeps only the converged plateau, where the weights
    are nearly identical. The diversity worth averaging is spread across
    training, which is what an EMA integrates and a top-k selection discards.

    The swap is deliberate about WHEN: shadow weights go in at the start of
    validation and stay in until the next training epoch begins, so the
    checkpoint written by ModelCheckpoint holds the EMA weights rather than the
    live ones. Restoring at `on_validation_end` instead would checkpoint the
    live weights and silently save the wrong model.
    """

    def __init__(self, decay: float) -> None:
        self.decay = float(decay)
        self.shadow: dict = {}
        self.backup: dict = {}

    def on_train_start(self, trainer, pl_module) -> None:
        self.shadow = {k: v.detach().clone().float()
                       for k, v in pl_module.state_dict().items()
                       if v.is_floating_point()}

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, *args) -> None:
        d = self.decay
        for k, v in pl_module.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1.0 - d)

    def _swap_in(self, pl_module) -> None:
        if not self.shadow or self.backup:
            return
        sd = pl_module.state_dict()
        self.backup = {k: sd[k].detach().clone() for k in self.shadow}
        pl_module.load_state_dict(
            {**sd, **{k: v.to(sd[k].dtype) for k, v in self.shadow.items()}})

    def _swap_out(self, pl_module) -> None:
        if not self.backup:
            return
        pl_module.load_state_dict({**pl_module.state_dict(), **self.backup})
        self.backup = {}

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self._swap_in(pl_module)

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._swap_out(pl_module)


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
    parser.add_argument("--arch", type=str, default="gap",
                        choices=["gap", "heatmap", "facemesh", "facemesh_heatmap"],
                        help="gap: GAP+FC coordinate regression (v1). "
                             "heatmap: soft-argmax over 24x24 heatmaps.")
    parser.add_argument("--mediapipe-ckpt", type=str, default=None,
                        help="MediaPipe blazeface_landmark.pth, for --arch facemesh. "
                             "Transfers 84.6%% of the model's parameters.")
    parser.add_argument("--backbone", type=str, default="default",
                        choices=["default", "blazeear"],
                        help="'default' is the shipped backbone (5x5 depthwise, "
                             "24-48-96-128-192). 'blazeear' mirrors BlazeEar v2's "
                             "backbone1 so --blazeear-ckpt can actually transfer: "
                             "against 'default' only 0.9%% of parameters are "
                             "shape-compatible, so a transfer there measures nothing.")
    parser.add_argument("--width-mult", type=float, default=1.0,
                        help="Scale every backbone channel count. 0.62 matches "
                             "the FaceMesh copy's parameter count, which "
                             "separates this design's merit from its size.")
    parser.add_argument("--tau", type=float, default=1.0,
                        help="Soft-argmax softmax temperature (--arch heatmap only)")
    parser.add_argument("--wing-w", type=float, default=0.04)
    parser.add_argument("--wing-epsilon", type=float, default=0.01)
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint: 'last', 'best', or path to .ckpt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--monitor", type=str, default="val_nme",
                        choices=["val_nme", "val_nme_normal"],
                        help="Metric for early stopping and checkpoint selection. "
                             "Use val_nme_normal with --tangential-weight < 1, or "
                             "selection rewards fitting the noise being discounted.")
    parser.add_argument("--tangential-weight", type=float, default=1.0,
                        help="Weight on residual along the GT contour (1.0 = plain Wing)")
    parser.add_argument("--spacing-weight", type=float, default=0.0,
                        help="Anti-collapse penalty on shrinking predicted segments")
    parser.add_argument("--perspective-deg", type=float, default=0.0,
                        help="Simulated out-of-plane turn in degrees (0 = off)")
    parser.add_argument("--rotation-deg", type=float, default=15.0,
                        help="In-plane rotation augmentation, +/- degrees. The "
                             "default matches every run up to v6_persp65. Raise "
                             "it when the training set's own ear-axis spread is "
                             "narrower than the set being tested on.")
    parser.add_argument("--train-csv", type=str, default="train.csv",
                        help="Training split filename inside the data dir. "
                             "Use this to train on a subset, e.g. a single source.")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Preprocessed data directory, relative to the project "
                             "root (default: data/preprocessed). Must contain "
                             "images/, landmarks.npy and the split CSVs. Use "
                             "data/manual for the commissioned annotation set.")
    parser.add_argument("--landmarks", type=str, default="landmarks.npy",
                        help="Landmark array under data/preprocessed/")
    parser.add_argument("--synthetic-ratio", type=float, default=0.0,
                        help="Synthetic samples per real training sample. 0 (the "
                             "default) is off; 1.0 balances synthetic against real. "
                             "Synthetic NEVER enters val or test, so the held-out "
                             "metrics stay purely real and a regression is visible.")
    parser.add_argument("--point-trajectory-dir", type=str, default=None,
                        help="Write per-point normal/tangential error for every "
                             "training row, once per epoch, into this directory. "
                             "Off by default. Feeds the loss-trajectory label "
                             "screen; costs one small npz per epoch and no "
                             "training time.")
    parser.add_argument("--run-name", type=str, default="EarLandmarker",
                        help="Names the checkpoint/log subdirectory, so concurrent "
                             "experiments do not overwrite each other")
    parser.add_argument("--flip-consistency", type=float, default=0.0,
                        help="Penalise ||m(x) - unmirror(m(mirror(x)))|| during "
                             "training, 0 = off. Flip AUGMENTATION is already on "
                             "and symmetrises the training distribution; it does "
                             "not symmetrise the learned function, which here "
                             "self-disagrees by ~60%% of its own error. Costs a "
                             "second forward pass per step. A constant predictor "
                             "is perfectly equivariant, so keep the weight small.")
    parser.add_argument("--degrade-prob", type=float, default=0.0,
                        help="Probability of image-quality degradation per "
                             "training sample (blur, downscale, noise, JPEG; "
                             "photometric only). 0 = off.")
    parser.add_argument("--stability-consistency", type=float, default=0.0,
                        help="Weight of a penalty on disagreement between the "
                             "prediction and the mapped-back prediction on a "
                             "randomly shifted/scaled/brightened copy (training "
                             "only). Targets frame-to-frame jitter. 0 = off.")
    parser.add_argument("--stab-shift", type=float, default=0.02)
    parser.add_argument("--stab-scale", type=float, default=0.04)
    parser.add_argument("--stab-bright", type=float, default=0.06)
    parser.add_argument("--stab-noise", type=float, default=0.02)
    parser.add_argument("--ema-decay", type=float, default=0.0,
                        help="Exponential moving average of the weights, 0 = off. "
                             "0.999 is a reasonable start. Reduces run-to-run "
                             "variance, which is about a third of the tangential "
                             "error here, and costs nothing at inference.")
    parser.add_argument("--accelerator", type=str, default="auto",
                        choices=["auto", "gpu", "cpu"],
                        help="Where to train. 'cpu' is ~4x slower per epoch "
                             "here (28.1s vs 6.95s on 4,753 rows) because the "
                             "model is only 340K parameters and dataloading "
                             "dominates -- usable when the GPU is busy. NOTE "
                             "that Lightning cannot do fp16 on CPU and "
                             "silently substitutes bf16-mixed, so a CPU run is "
                             "NOT numerically comparable to a GPU one; keep "
                             "both arms of a comparison on the same device.")
    args = parser.parse_args()

    ckpt_dir = PROJECT / "runs" / "checkpoints" / args.run_name

    pl.seed_everything(args.seed)

    # Dataset
    train_aug = AugmentationParams(
        horizontal_flip=True,
        flip_prob=0.5,
        translation=0.05,
        rotation_deg=args.rotation_deg,
        color_jitter={"brightness": 0.3, "contrast": 0.3, "saturation": 0.2, "hue": 0.05},
        bbox_jitter=0.1,
        bbox_jitter_prob=0.5,
        perspective_deg=args.perspective_deg,
        degrade_prob=args.degrade_prob,
    )

    data_dir = (PROJECT / args.data_dir) if args.data_dir else DATA_DIR
    if not (data_dir / "landmarks.npy").exists():
        raise FileNotFoundError(f"{data_dir}/landmarks.npy not found")

    train_ds = EarLandmarkDataset(
        split_csv=data_dir / args.train_csv,
        data_dir=data_dir,
        image_size=args.image_size,
        augmentation=train_aug,
        landmarks_file=args.landmarks,
    )
    val_ds = EarLandmarkDataset(
        split_csv=data_dir / "val.csv",
        data_dir=data_dir,
        image_size=args.image_size,
        landmarks_file=args.landmarks,
    )
    test_csv = data_dir / "test.csv"
    test_ds = EarLandmarkDataset(
        split_csv=test_csv,
        data_dir=data_dir,
        image_size=args.image_size,
        landmarks_file=args.landmarks,
    ) if test_csv.exists() else None

    loader_kwargs = dict(
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=3 if args.num_workers > 0 else None,
    )
    # Optional synthetic source. Kept in its own directory with its own manifest
    # and landmarks, so reverting is either omitting the flag or deleting
    # data/synthetic -- the real splits are never modified.
    if args.synthetic_ratio > 0:
        synth_dir = PROJECT / "data" / "synthetic"
        if not (synth_dir / "train.csv").exists():
            raise FileNotFoundError(
                f"{synth_dir}/train.csv not found -- run scripts/ingest_synthetic.py")
        synth_full = EarLandmarkDataset(
            split_csv=synth_dir / "train.csv",
            data_dir=synth_dir,
            image_size=args.image_size,
            augmentation=train_aug,
        )
        n_want = int(round(len(train_ds) * args.synthetic_ratio))
        if n_want < len(synth_full):
            g = torch.Generator().manual_seed(args.seed)
            keep = torch.randperm(len(synth_full), generator=g)[:n_want].tolist()
            synth_ds = torch.utils.data.Subset(synth_full, keep)
        else:
            synth_ds = synth_full
        real_n, synth_n = len(train_ds), len(synth_ds)
        train_ds = torch.utils.data.ConcatDataset([train_ds, synth_ds])
        print(f"training mix: {real_n} real + {synth_n} synthetic "
              f"= {len(train_ds)} ({synth_n/len(train_ds)*100:.0f}% synthetic)")
        print(f"  val/test remain 100% real, so any regression shows there")
    else:
        print(f"training on {len(train_ds)} real samples (synthetic off)")

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
        mediapipe_ckpt=args.mediapipe_ckpt,
        arch=args.arch,
        backbone=args.backbone,
        width_mult=args.width_mult,
        tau=args.tau,
        tangential_weight=args.tangential_weight,
        spacing_weight=args.spacing_weight,
        flip_consistency=args.flip_consistency,
        stability_consistency=args.stability_consistency,
        stab_shift=args.stab_shift, stab_scale=args.stab_scale,
        stab_bright=args.stab_bright, stab_noise=args.stab_noise,
        point_trajectory_dir=args.point_trajectory_dir,
    )

    if args.compile:
        module.model = torch.compile(module.model)
        print("Model compiled with torch.compile")

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="EarLandmarker_{epoch:03d}_nme={val_nme:.4f}",
        auto_insert_metric_name=False,
        monitor=args.monitor,
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    early_stop = EarlyStopping(
        monitor=args.monitor,
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
        accelerator=args.accelerator,
        devices=1,
        precision="16-mixed",
        callbacks=([checkpoint_cb, lr_monitor, early_stop]
                   + ([EMACallback(args.ema_decay)] if args.ema_decay > 0 else [])),
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
            # By the score inside the file, not the rounded nme= in the name.
            # The name is rounded to four decimals and runs usually end with
            # two or three checkpoints tied there, so the old name-based
            # comparison kept the first in readdir order -- the EARLIEST tied
            # epoch. Resuming fixed_conv_s42 would have rolled back from epoch
            # 433 to 388, discarding 45 epochs. See checkpoint_utils.
            ckpt_path = best_checkpoint(ckpt_dir)
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
