"""PyTorch Lightning module for EarLandmarker training."""

from __future__ import annotations

import pathlib

from typing import Any, Dict, Optional

import torch
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model.ear_landmarker import EarLandmarker, EarLandmarkerHeatmap
from model.losses import WingLoss, TangentialWeightedWingLoss, _contour_frames

# (name, start_idx, end_idx) -- linestrip groups, end exclusive
LANDMARK_REGIONS = [
    ("outer_helix", 0, 20),
    ("inner_helix", 20, 35),
    ("concha_border", 35, 50),
    ("superior_crus", 50, 55),
]


class EarLandmarkerModule(pl.LightningModule):
    """Lightning wrapper for EarLandmarker training.

    Args:
        num_landmarks: Number of landmark points (default 55).
        lr: Peak learning rate.
        weight_decay: AdamW weight decay.
        max_epochs: For cosine annealing schedule.
        wing_w: Wing loss width parameter.
        wing_epsilon: Wing loss curvature parameter.
        blazeear_ckpt: Optional path to BlazeEar checkpoint for backbone init.
        arch: "gap" for the v1 GAP+FC coordinate head, "heatmap" for the
            soft-argmax head over 24x24 heatmaps.
        tau: Soft-argmax softmax temperature (arch="heatmap" only).
    """

    def __init__(
        self,
        num_landmarks: int = 55,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
        wing_w: float = 0.04,
        wing_epsilon: float = 0.01,
        blazeear_ckpt: Optional[str] = None,
        mediapipe_ckpt: Optional[str] = None,
        arch: str = "gap",
        backbone: str = "default",
        width_mult: float = 1.0,
        tau: float = 1.0,
        point_trajectory_dir: str | None = None,
        tangential_weight: float = 1.0,
        spacing_weight: float = 0.0,
        flip_consistency: float = 0.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        if arch == "gap":
            self.model = EarLandmarker(num_landmarks=num_landmarks, backbone=backbone,
                                       width_mult=width_mult)
        elif arch == "heatmap":
            self.model = EarLandmarkerHeatmap(
                num_landmarks=num_landmarks, tau=tau, backbone=backbone,
                width_mult=width_mult)
        elif arch == "facemesh_heatmap":
            # MediaPipe's feature extractor, this project's heatmap decoder.
            from model.facemesh_ear import FaceMeshEarHeatmap
            self.model = FaceMeshEarHeatmap(num_landmarks=num_landmarks, tau=tau)
        elif arch == "facemesh":
            # MediaPipe's face landmark topology, copied exactly bar the output
            # width. See model/facemesh_ear.py for what that trade involves.
            from model.facemesh_ear import FaceMeshEarLandmarker
            self.model = FaceMeshEarLandmarker(num_landmarks=num_landmarks)
        else:
            raise ValueError(
                f"unknown arch {arch!r}, expected 'gap', 'heatmap', "
                f"'facemesh' or 'facemesh_heatmap'")
        # Off unless a directory is given: no behaviour change, and a run that
        # does not ask for the measurement does not pay for it.
        self._traj_dir = (pathlib.Path(point_trajectory_dir)
                          if point_trajectory_dir else None)
        self._traj: list = []

        # tangential_weight < 1 discounts residual along the GT contour, where most
        # of the label noise lives; > 1 up-weights it, which is wanted when the
        # labels are clean enough that position along the contour is signal
        # rather than noise. 1.0 exactly reproduces plain isotropic Wing loss.
        #
        # The test was `>= 1.0`, which silently routed ANY up-weighting to plain
        # WingLoss -- `--tangential-weight 2.0` ran as 1.0 and the flag did
        # nothing, with nothing in the log to say so.
        if tangential_weight == 1.0 and spacing_weight <= 0.0:
            self.criterion = WingLoss(w=wing_w, epsilon=wing_epsilon)
        else:
            self.criterion = TangentialWeightedWingLoss(
                w=wing_w, epsilon=wing_epsilon,
                tangential_weight=tangential_weight,
                spacing_weight=spacing_weight,
            )

        if blazeear_ckpt:
            n = self.model.load_blazeear_backbone(blazeear_ckpt)
            print(f"Transferred {n:,} parameters from BlazeEar backbone")
        if mediapipe_ckpt:
            n = self.model.load_mediapipe_backbone(mediapipe_ckpt)
            total = sum(p.numel() for p in self.model.parameters())
            print(f"Transferred {n:,} of {total:,} parameters "
                  f"({n/total*100:.1f}%) from MediaPipe face landmark weights")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        images = batch["image"]
        targets = batch["landmarks"]
        visible = batch.get("visible")
        preds = self.model(images)

        loss = self.criterion(preds, targets, visible)

        # FLIP-CONSISTENCY PENALTY, off unless flip_consistency > 0.
        #
        # Why it is a different lever from the flip augmentation that is already
        # on: augmentation at p=0.5 symmetrises the training DISTRIBUTION, so
        # both orientations are seen. It does not symmetrise the learned
        # FUNCTION, because nothing in the loss compares m(x) with
        # unmirror(m(mirror(x))). Measured on this corpus, that self-
        # disagreement is ~60% of the model's own total error (0.00691 against
        # 0.01144 for v3_gpu_s42), which is why flip fusion still pays its 2x
        # inference cost.
        #
        # The penalty is on the FUNCTION, per sample, so it constrains exactly
        # what augmentation leaves alone. Mirroring is done on the normalised
        # tensor and undone as 1 - x, matching inference.py's convention
        # exactly -- the decoder's x grid is (i+0.5)/w and the flip carries NO
        # index permutation, so point i is the same anatomical point in both
        # arms.
        #
        # It carries a degenerate direction -- a constant predictor is
        # perfectly equivariant -- so the weight must stay small enough that
        # the supervised term dominates.
        if self.hparams.get("flip_consistency", 0.0) > 0:
            mirrored = self.model(torch.flip(images, dims=[3]))
            m2 = mirrored.view(-1, self.hparams.num_landmarks, 2).clone()
            m2[:, :, 0] = 1.0 - m2[:, :, 0]
            p2 = preds.view(-1, self.hparams.num_landmarks, 2)
            d = torch.norm(p2 - m2, dim=-1)
            if visible is not None:
                d = (d * visible).sum() / visible.sum().clamp_min(1.0)
            else:
                d = d.mean()
            loss = loss + self.hparams["flip_consistency"] * d

        # Per-point NME (Normalized Mean Error) as fraction of [0,1] range.
        # Averaged over visible points only, so it matches what was supervised.
        with torch.no_grad():
            preds_2d = preds.view(-1, self.hparams.num_landmarks, 2)
            targets_2d = targets.view(-1, self.hparams.num_landmarks, 2)
            per_point_err = torch.norm(preds_2d - targets_2d, dim=-1)  # (B, 55)
            if visible is None:
                nme = per_point_err.mean()
            else:
                nme = (per_point_err * visible).sum() / visible.sum().clamp_min(1.0)

            # Split NME along/off the GT contour. 72% of squared error is
            # tangential and largely annotation noise, so nme_normal is the
            # cleaner signal for model selection; plain nme rewards fitting noise.
            tan, nrm = _contour_frames(targets_2d)
            resid = preds_2d - targets_2d
            d_tan = (resid * tan).sum(-1).abs()
            d_nrm = (resid * nrm).sum(-1).abs()
            if visible is None:
                nme_n, nme_t = d_nrm.mean(), d_tan.mean()
            else:
                den = visible.sum().clamp_min(1.0)
                nme_n = (d_nrm * visible).sum() / den
                nme_t = (d_tan * visible).sum() / den

        return {"loss": loss, "nme": nme, "nme_normal": nme_n,
                "nme_tangential": nme_t, "per_point_err": per_point_err,
                "d_normal": d_nrm, "d_tangential": d_tan}

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        result = self._shared_step(batch)
        self.log("train/loss", result["loss"], prog_bar=True)
        self.log("train/nme", result["nme"], prog_bar=True)
        if self._traj_dir is not None and "idx" in batch:
            self._traj.append((
                batch["idx"].detach().cpu().numpy().astype("int32"),
                result["d_normal"].detach().cpu().numpy().astype("float16"),
                result["d_tangential"].detach().cpu().numpy().astype("float16"),
                (batch["visible"] if "visible" in batch
                 else torch.ones_like(result["d_normal"]))
                .detach().cpu().numpy().astype("int8"),
            ))
        return result["loss"]

    def on_train_epoch_end(self) -> None:
        """Write this epoch's per-point error for every training row.

        The screen this feeds is a LOSS TRAJECTORY, not a final residual: a
        correctly annotated point falls steadily and bottoms out late, while an
        inconsistent one barely improves and wanders, and the two are only
        distinguishable across epochs. Recorded per point rather than per image,
        because an image is rarely wrong as a whole.

        Normal and tangential components are kept APART. Tangential error is
        position along the contour, which is where annotators disagree by
        convention; normal error is distance from the contour, which is where
        they are simply wrong. Collapsing them into one number -- as a plain
        per-keypoint loss would -- mixes ambiguity with error, and the two want
        opposite treatments.

        float16 and int8: this is a measurement of a measurement, and a full run
        writes one file per epoch.
        """
        if self._traj_dir is None or not self._traj:
            return
        import numpy as np
        d = self._traj_dir
        d.mkdir(parents=True, exist_ok=True)
        cat = lambda k: np.concatenate([t[k] for t in self._traj])
        np.savez_compressed(d / f"epoch_{self.current_epoch:04d}.npz",
                            idx=cat(0), d_normal=cat(1),
                            d_tangential=cat(2), visible=cat(3))
        self._traj = []

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        result = self._shared_step(batch)
        self.log("val/loss", result["loss"], prog_bar=True, sync_dist=True)
        self.log("val/nme", result["nme"], prog_bar=True, sync_dist=True)
        # Slash-free alias: ModelCheckpoint interpolates the monitored metric into
        # the filename, and a "/" there makes Lightning create nested directories.
        self.log("val_nme", result["nme"], sync_dist=True)
        self.log("val/nme_normal", result["nme_normal"], sync_dist=True)
        self.log("val/nme_tangential", result["nme_tangential"], sync_dist=True)
        self.log("val_nme_normal", result["nme_normal"], sync_dist=True)
        self._log_region_nme("val", result["per_point_err"])

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        result = self._shared_step(batch)
        self.log("test/loss", result["loss"], sync_dist=True)
        self.log("test/nme", result["nme"], prog_bar=True, sync_dist=True)
        self.log("test/nme_normal", result["nme_normal"], sync_dist=True)
        self.log("test/nme_tangential", result["nme_tangential"], sync_dist=True)
        self._log_region_nme("test", result["per_point_err"])

    def _log_region_nme(self, stage: str, per_point_err: torch.Tensor) -> None:
        """Log NME per contour strip (iBUG naming; see model/measure.py)."""
        for name, lo, hi in LANDMARK_REGIONS:
            self.log(
                f"{stage}/nme_{name}",
                per_point_err[:, lo:hi].mean(),
                sync_dist=True,
            )

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=self.hparams.lr * 0.01,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
