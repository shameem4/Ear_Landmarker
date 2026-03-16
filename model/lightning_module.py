"""PyTorch Lightning module for EarLandmarker training."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model.ear_landmarker import EarLandmarker
from model.losses import WingLoss


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
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = EarLandmarker(num_landmarks=num_landmarks)
        self.criterion = WingLoss(w=wing_w, epsilon=wing_epsilon)

        if blazeear_ckpt:
            n = self.model.load_blazeear_backbone(blazeear_ckpt)
            print(f"Transferred {n:,} parameters from BlazeEar backbone")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        images = batch["image"]
        targets = batch["landmarks"]
        preds = self.model(images)

        loss = self.criterion(preds, targets)

        # Per-point NME (Normalized Mean Error) as fraction of [0,1] range
        with torch.no_grad():
            preds_2d = preds.view(-1, self.hparams.num_landmarks, 2)
            targets_2d = targets.view(-1, self.hparams.num_landmarks, 2)
            per_point_err = torch.norm(preds_2d - targets_2d, dim=-1)  # (B, 55)
            nme = per_point_err.mean()

        return {"loss": loss, "nme": nme}

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        result = self._shared_step(batch)
        self.log("train/loss", result["loss"], prog_bar=True)
        self.log("train/nme", result["nme"], prog_bar=True)
        return result["loss"]

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        result = self._shared_step(batch)
        self.log("val/loss", result["loss"], prog_bar=True, sync_dist=True)
        self.log("val/nme", result["nme"], prog_bar=True, sync_dist=True)

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
