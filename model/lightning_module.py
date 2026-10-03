"""PyTorch Lightning module for EarLandmarker training."""

from __future__ import annotations

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
        tau: float = 1.0,
        tangential_weight: float = 1.0,
        spacing_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        if arch == "gap":
            self.model = EarLandmarker(num_landmarks=num_landmarks, backbone=backbone)
        elif arch == "heatmap":
            self.model = EarLandmarkerHeatmap(
                num_landmarks=num_landmarks, tau=tau, backbone=backbone)
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
        # tangential_weight < 1 discounts residual along the GT contour, where most
        # of the label noise lives. 1.0 reproduces plain isotropic Wing loss.
        if tangential_weight >= 1.0 and spacing_weight <= 0.0:
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
                "nme_tangential": nme_t, "per_point_err": per_point_err}

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        result = self._shared_step(batch)
        self.log("train/loss", result["loss"], prog_bar=True)
        self.log("train/nme", result["nme"], prog_bar=True)
        return result["loss"]

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
