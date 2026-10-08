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


class EarShape3DModule(EarDepth3DModule):
    """Head 2: shape coefficients and pose; the landmarks follow."""

    def __init__(self, *args, basis: str = "", n_modes: int = 40,
                 coeff_l2: float = 1e-3, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not basis:
            raise ValueError("head 2 needs --basis (scripts/build_landmark_basis.py)")
        from model.shape_head import EarShapeLandmarker
        self.save_hyperparameters({"basis": basis, "n_modes": n_modes,
                                   "coeff_l2": coeff_l2})
        self.model = EarShapeLandmarker(
            basis, num_landmarks=self.hparams.num_landmarks,
            backbone=self.hparams.backbone, width_mult=self.hparams.width_mult,
            n_modes=n_modes)

    def _shared_step(self, batch):
        r = super()._shared_step(batch)
        # the coefficients are in standard deviations, so a unit gaussian is
        # the correct prior and this penalty is calibrated rather than guessed
        if self.hparams.coeff_l2 > 0:
            p = self.model.parameters_from(batch["image"])
            c = p[:, :self.model.n_modes]
            r = dict(r)
            r["loss"] = r["loss"] + self.hparams.coeff_l2 * c.pow(2).mean()
        return r


class EarLifter3DModule(EarDepth3DModule):
    """Head 3: 55 (x, y) -> 55 z, with no image at all.

    Trained on the GROUND-TRUTH coordinates, with noise standing in for the 2D
    model's error, so it measures lifting ability rather than inheriting the
    landmarker's mistakes. Evaluation feeds it real predictions, which is where
    the inherited error belongs.
    """

    def __init__(self, *args, xy_noise: float = 0.0, width: int = 256,
                 use_conf: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        from model.lifter import EarDepthLifter
        self.save_hyperparameters({"xy_noise": xy_noise, "width": width,
                                   "use_conf": use_conf})
        self.model = EarDepthLifter(num_landmarks=self.hparams.num_landmarks,
                                    width=width, use_conf=use_conf)

    def _shared_step(self, batch):
        k = self.hparams.num_landmarks
        t = batch["landmarks"].view(-1, k, 3)
        # the 2D model's real output when it has been precomputed, which is
        # what this head meets at test time; ground truth only as a fallback
        if "pred_xy" in batch:
            xy = batch["pred_xy"].view(-1, k, 2)
            conf = batch.get("pred_conf")
        else:
            xy, conf = t[:, :, :2], torch.ones_like(t[:, :, 0])
            if self.training and self.hparams.xy_noise > 0:
                xy = xy + torch.randn_like(xy) * self.hparams.xy_noise
        zp, zt = self.model(xy, conf if self.model.use_conf else None), t[:, :, 2]
        m = batch.get("visible")
        m = torch.ones_like(zt) if m is None else m
        if self.hparams.z_visible_only and "seen" in batch:
            m = m * batch["seen"]
        den = m.sum().clamp_min(1.0)
        loss = (F.smooth_l1_loss(zp, zt, beta=self.hparams.z_beta,
                                 reduction="none") * m).sum() / den
        with torch.no_grad():
            z_mae = ((zp - zt).abs() * m).sum() / den
            z_base = (zt.abs() * m).sum() / den
            nme = torch.zeros((), device=zp.device)   # this head predicts no x, y
        return {"loss": loss, "nme": nme, "z_mae": z_mae, "z_base": z_base,
                "z_skill": 1.0 - z_mae / z_base.clamp_min(1e-9),
                "per_point_err": torch.zeros_like(zp)}


class EarFeatureLifter3DModule(EarLifter3DModule):
    """Head 3b: the lifter with image features sampled at each landmark.

    The 2D model is frozen and supplies three things: the coordinates, the
    per-landmark confidence, and the decoder feature map the lifter samples.
    Only the lifter trains, so this cannot damage the existing 2D accuracy --
    the same property that makes head 3 attractive in the first place.
    """

    def __init__(self, *args, base_ckpt: str = "", point_width: int = 32,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not base_ckpt:
            raise ValueError("head 3b needs --base-ckpt, the frozen 2D model")
        from model.lifter import FeatureLifter
        base = EarLandmarkerModule.load_from_checkpoint(
            base_ckpt, map_location="cpu").model.eval()
        for q in base.parameters():
            q.requires_grad_(False)
        self.base = base
        self.save_hyperparameters({"base_ckpt": base_ckpt,
                                   "point_width": point_width})
        self.model = FeatureLifter(
            feat_ch=base.stage2_ch, num_landmarks=self.hparams.num_landmarks,
            point_width=point_width, width=self.hparams.width,
            use_conf=self.hparams.use_conf)

    def _shared_step(self, batch):
        k = self.hparams.num_landmarks
        t = batch["landmarks"].view(-1, k, 3)
        if "pred_feat" in batch:
            # already sampled at the predicted landmarks, and fixed because the
            # base is frozen and these heads see no geometric augmentation
            xy = batch["pred_xy"].view(-1, k, 2)
            conf, sampled = batch["pred_conf"], batch["pred_feat"]
            zp = self.model.from_sampled(sampled, xy,
                                         conf if self.model.use_conf else None)
        else:
            with torch.no_grad():
                feat = self.base._decode(batch["image"])
                hm = self.base.to_heatmap(feat)
                xy = self.base.soft_argmax(hm, self.base.tau).view(-1, k, 2)
                conf = self.base.heatmap_confidence(hm, self.base.tau)
            zp = self.model(feat, xy, conf if self.model.use_conf else None)
        zt = t[:, :, 2]
        m = batch.get("visible")
        m = torch.ones_like(zt) if m is None else m
        if self.hparams.z_visible_only and "seen" in batch:
            m = m * batch["seen"]
        den = m.sum().clamp_min(1.0)
        loss = (F.smooth_l1_loss(zp, zt, beta=self.hparams.z_beta,
                                 reduction="none") * m).sum() / den
        with torch.no_grad():
            z_mae = ((zp - zt).abs() * m).sum() / den
            z_base = (zt.abs() * m).sum() / den
            nme = (torch.norm(xy - t[:, :, :2], dim=-1) * m).sum() / den
        return {"loss": loss, "nme": nme, "z_mae": z_mae, "z_base": z_base,
                "z_skill": 1.0 - z_mae / z_base.clamp_min(1e-9),
                "per_point_err": torch.zeros_like(zp)}
