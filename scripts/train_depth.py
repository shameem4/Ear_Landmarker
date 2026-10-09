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
    p.add_argument("--split-prefix", default="",
                   help="read <prefix>train.csv; 'bal_' for the "
                        "source-balanced split from split_depth3d.py")
    p.add_argument("--n-points", type=int, default=0,
                   help="use the first N landmarks only. The corpus carries "
                        "95; 55 is the original iBUG scheme, which is what "
                        "every earlier number on this project was measured on")
    p.add_argument("--photo-aug", type=float, default=0.0,
                   help="strength of the render-to-photograph augmentation "
                        "(data/photo_aug). 1.0 matches the measured photo "
                        "statistics; 0 disables it")
    p.add_argument("--photo-aug-shading", type=float, default=1.0,
                   help="scale the local-shading part of --photo-aug on its "
                        "own. 0 keeps texture, noise, occluders and JPEG "
                        "while leaving the shape-from-shading cue intact")
    p.add_argument("--balance-sampler", action="store_true",
                   help="sample so each source contributes equally per epoch. "
                        "Needs a 'source' column in the train csv")
    p.add_argument("--head", choices=["z", "shape", "lifter", "feat", "shapefeat"],
                   default="z",
                   help="z: per-landmark depth off the heatmap. shape: shape "
                        "coefficients and pose, landmarks follow. lifter: a "
                        "separate network from 55 (x,y) to 55 z. feat: "
                        "the same lifter, plus image features sampled at each "
                        "landmark")
    p.add_argument("--base-ckpt", default="", help="frozen 2D model, for --head feat")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--basis", default="", help="required for --head shape")
    p.add_argument("--n-modes", type=int, default=40)
    p.add_argument("--shape-head", default="spatial", choices=["spatial", "gap"])
    p.add_argument("--coeffs", default="", help="gt_coeffs npz, for --coeff-sup")
    p.add_argument("--refine-steps", type=int, default=2)
    p.add_argument("--coeff-sup", type=float, default=0.0,
                   help="weight on supervising the shape coefficients directly")
    p.add_argument("--out", default=None)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--z-weight", type=float, default=1.0)
    p.add_argument("--z-visible-only", action="store_true")
    p.add_argument("--width-mult", type=float, default=1.0)
    p.add_argument("--limit-batches", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--accelerator", default="auto",
                   help="'cpu' to stay off a GPU someone else is using")
    a = p.parse_args()

    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from torch.utils.data import DataLoader, WeightedRandomSampler

    from data.dataset import AugmentationParams
    from data.depth_dataset import EarDepthDataset
    from model.depth_module import (EarDepth3DModule, EarFeatureLifter3DModule,
                                    EarLifter3DModule, EarShape3DModule,
                                    EarShapeFeat3DModule)

    pl.seed_everything(a.seed, workers=True)
    d = pathlib.Path(a.data)
    # perspective_deg stays 0: see data/depth_dataset and the note in
    # _geo_augment. Everything else is the 2D recipe unchanged.
    aug = AugmentationParams(perspective_deg=0.0)
    # The lifter heads read the 2D model's precomputed output, which was
    # computed on the STORED image, so they must see that image unaugmented.
    lifting = a.head in ("lifter", "feat")
    train = EarDepthDataset(d / f"{a.split_prefix}train.csv", d,
                            augmentation=None if lifting else aug,
                            with_pred=lifting, coeffs=a.coeffs,
                            n_points=a.n_points, photo_aug=a.photo_aug,
                            photo_aug_shading=a.photo_aug_shading)
    val = EarDepthDataset(d / f"{a.split_prefix}val.csv", d, augmentation=None,
                          with_pred=lifting, coeffs=a.coeffs,
                          n_points=a.n_points)
    k_pts = train.n_points
    print(f"{len(train)} train, {len(val)} val views, {k_pts} landmarks",
          flush=True)

    dl = dict(batch_size=a.batch, num_workers=a.workers, pin_memory=True,
              persistent_workers=a.workers > 0)
    if a.balance_sampler:
        # One weight per VIEW, inversely proportional to how many views its
        # source contributes. Every SONICOM ear is still reachable -- nothing
        # is discarded -- it is drawn less often, so neither source dominates
        # the gradient. The cost is that each HUTUBS ear is drawn about four
        # times as often, which is a real overfitting risk and the reason the
        # per-source val numbers are worth reading rather than the pooled one.
        srcs = [r.get("source") for r in train.samples]
        if any(v is None for v in srcs):
            sys.exit(f"--balance-sampler needs a 'source' column in "
                     f"{a.split_prefix}train.csv; rerun scripts/split_depth3d.py")
        import collections
        n_per = collections.Counter(srcs)
        w = [1.0 / n_per[v] for v in srcs]
        print("balanced sampler: " + "  ".join(
            f"{k} {n} views x {1.0 / n:.2e}" for k, n in sorted(n_per.items())),
            flush=True)
        tl = DataLoader(train, sampler=WeightedRandomSampler(
            w, num_samples=len(train), replacement=True), drop_last=True, **dl)
    else:
        tl = DataLoader(train, shuffle=True, drop_last=True, **dl)
    vl = DataLoader(val, shuffle=False, **dl)

    common = dict(lr=a.lr, max_epochs=a.epochs, arch="heatmap",
                  num_landmarks=k_pts,
                  width_mult=a.width_mult, z_weight=a.z_weight,
                  z_visible_only=a.z_visible_only)
    if a.head == "z":
        model = EarDepth3DModule(**common)
    elif a.head == "shape":
        if not a.basis:
            sys.exit("--head shape needs --basis from build_landmark_basis.py")
        model = EarShape3DModule(basis=a.basis, n_modes=a.n_modes,
                                 shape_head=a.shape_head,
                                 coeff_sup=a.coeff_sup, **common)
    elif a.head == "shapefeat":
        if not a.basis:
            sys.exit("--head shapefeat needs --basis")
        model = EarShapeFeat3DModule(basis=a.basis, n_modes=a.n_modes,
                                     coeff_sup=a.coeff_sup,
                                     refine_steps=a.refine_steps,
                                     width=a.width, **common)
    elif a.head == "lifter":
        model = EarLifter3DModule(width=a.width, **common)
    else:
        if not a.base_ckpt:
            sys.exit("--head feat needs --base-ckpt, the frozen 2D model")
        model = EarFeatureLifter3DModule(base_ckpt=a.base_ckpt,
                                         width=a.width, **common)
    n = sum(q.numel() for q in model.model.parameters() if q.requires_grad)
    print(f"head {a.head!r}: {n:,} trainable parameters", flush=True)
    out = pathlib.Path(a.out or f"runs/depth_{a.head}")
    # REFUSE A DIRTY OUTPUT DIRECTORY. save_top_k keeps the best checkpoints,
    # but a run that was interrupted leaves ITS best behind, and an early
    # epoch=0 file sorts first under any lexicographic glob. That is not
    # hypothetical: picking checkpoints with `ls | head -1` after a relaunch
    # into the same directory scored three models at epoch 0 and produced a
    # whole table of plausible-looking nonsense. Cheaper to refuse than to
    # catch by noticing the numbers are wrong.
    if (out / "ckpt").exists() and any((out / "ckpt").glob("*.ckpt")):
        sys.exit(f"{out / 'ckpt'} already holds checkpoints. A relaunch into "
                 f"the same directory mixes runs and the stale ones can win a "
                 f"glob. Remove it, or pass a different --out.")
    ck = ModelCheckpoint(dirpath=out / "ckpt", filename="{epoch}-{val_z_mae:.4f}",
                         monitor="val_z_mae", mode="min", save_top_k=2,
                         save_last=True)
    tr = pl.Trainer(
        max_epochs=a.epochs, accelerator=a.accelerator, devices=1,
        default_root_dir=str(out), logger=CSVLogger(str(out), name="csv"),
        callbacks=[ck], log_every_n_steps=20,
        limit_train_batches=a.limit_batches, limit_val_batches=a.limit_batches,
        enable_progress_bar=True)
    tr.fit(model, tl, vl)
    print(f"\nbest val_z_mae {ck.best_model_score:.5f} -> {ck.best_model_path}")


if __name__ == "__main__":
    main()
