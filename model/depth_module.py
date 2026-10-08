"""Training for the 55-point landmarker WITH depth.

Subclasses the 2D module so the backbone, the Wing loss on x and y, the NME
metrics and the optimiser schedule are all literally the same code. What is
added is a depth term and, more importantly, the metrics that make a depth
number mean something.

THE BASELINE IS LOGGED ALONGSIDE THE ERROR, always. z is root-relative, so a
model that predicts zero everywhere already achieves mean|z| -- on this corpus
about 0.07 of crop width. A depth MAE quoted on its own looks respectable at
exactly that value while carrying no information, and this project has been
caught once already by a metric that could not fail. `z_skill` is the fraction
of that baseline error removed: 0 is a flat predictor, 1 is perfect, and
negative means worse than predicting nothing.

OCCLUDED POINTS ARE SUPERVISED BY DEFAULT. A point behind the helix has a real
depth and a 3D model should place it; the renderer's visibility flag is carried
through as `seen` so --z-visible-only can test whether that costs anything.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from model.ear_landmarker import EarLandmarkerHeatmap
from model.lightning_module import EarLandmarkerModule


class EarDepth3DModule(EarLandmarkerModule):
    """x, y and z. Only the heatmap arch has a depth head."""

    def __init__(self, *args, z_weight: float = 1.0, z_beta: float = 0.05,
                 z_visible_only: bool = False, **kwargs) -> None:
        kwargs.setdefault("arch", "heatmap")
        if kwargs["arch"] != "heatmap":
            raise ValueError("depth is predicted off the heatmap decoder; "
                             f"arch={kwargs['arch']!r} has no such head")
        super().__init__(*args, **kwargs)
        self.save_hyperparameters({"z_weight": z_weight, "z_beta": z_beta,
                                   "z_visible_only": z_visible_only})
        self.model = EarLandmarkerHeatmap(
            num_landmarks=self.hparams.num_landmarks, tau=self.hparams.tau,
            backbone=self.hparams.backbone, width_mult=self.hparams.width_mult,
            predict_z=True)

    def _shared_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        k = self.hparams.num_landmarks
        preds = self(batch["image"]).view(-1, k, 3)
        targets = batch["landmarks"].view(-1, k, 3)
        visible = batch.get("visible")

        # x and y go through the existing 2D criterion untouched, so a depth
        # run and a 2D run are comparable on NME rather than only to themselves
        xy_loss = self.criterion(preds[:, :, :2].reshape(len(preds), -1),
                                 targets[:, :, :2].reshape(len(targets), -1),
                                 visible)

        m = visible if visible is not None else torch.ones_like(preds[:, :, 0])
        if self.hparams.z_visible_only and "seen" in batch:
            m = m * batch["seen"]
        den = m.sum().clamp_min(1.0)
        zp, zt = preds[:, :, 2], targets[:, :, 2]
        z_loss = (F.smooth_l1_loss(zp, zt, beta=self.hparams.z_beta,
                                   reduction="none") * m).sum() / den
        loss = xy_loss + self.hparams.z_weight * z_loss

        with torch.no_grad():
            err = torch.norm(preds[:, :, :2] - targets[:, :, :2], dim=-1)
            nme = ((err * m).sum() / den) if visible is not None else err.mean()
            z_mae = ((zp - zt).abs() * m).sum() / den
            z_base = (zt.abs() * m).sum() / den          # predicting z = 0
            z_skill = 1.0 - z_mae / z_base.clamp_min(1e-9)
        return {"loss": loss, "nme": nme, "z_mae": z_mae, "z_base": z_base,
                "z_skill": z_skill, "per_point_err": err,
                "nme_normal": nme, "nme_tangential": nme}

    def training_step(self, batch, batch_idx):
        r = self._shared_step(batch)
        self.log("train/loss", r["loss"], prog_bar=True)
        self.log("train/nme", r["nme"], prog_bar=True)
        self.log("train/z_mae", r["z_mae"], prog_bar=True)
        return r["loss"]

    def validation_step(self, batch, batch_idx):
        r = self._shared_step(batch)
        for k in ("loss", "nme", "z_mae", "z_base", "z_skill"):
            self.log(f"val/{k}", r[k], prog_bar=k in ("nme", "z_mae", "z_skill"),
                     sync_dist=True)
        self.log("val_nme", r["nme"], sync_dist=True)
        self.log("val_z_mae", r["z_mae"], sync_dist=True)

    def test_step(self, batch, batch_idx):
        r = self._shared_step(batch)
        for k in ("loss", "nme", "z_mae", "z_base", "z_skill"):
            self.log(f"test/{k}", r[k], sync_dist=True)
