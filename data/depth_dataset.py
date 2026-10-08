"""Dataset for the rendered depth corpus: 55 landmarks with x, y AND z.

Subclasses EarLandmarkDataset so the augmentation is literally the same code,
which is the point -- a second copy would drift, and the one thing that must
not drift here is how a crop scale reaches the depth label.

Three facts about z that the rest of the pipeline depends on:

  SAME UNITS AS x AND y. z is a fraction of the crop width, so a crop scale
  change multiplies it. _bbox_jitter returns that factor and _geo_augment
  applies it; nothing else in the augmentation touches depth.

  ROOT-RELATIVE. z is measured from the mean depth of the 55 points, because
  absolute distance is not recoverable from a crop.

  SUPERVISED EVEN WHERE OCCLUDED. A point behind the helix still has a true
  depth and a model should predict it. `visible` says which points the
  renderer could actually see, so a loss can drop them if that turns out to
  help, but the label is there either way.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from data.dataset import AugmentationParams, EarLandmarkDataset


class EarDepthDataset(EarLandmarkDataset):
    """(image, 55x3 landmarks, visibility) from scripts/ingest_depth3d.py."""

    def __init__(
        self,
        split_csv: Path | str,
        data_dir: Path | str,
        image_size: int = 192,
        augmentation: Optional[AugmentationParams] = None,
        stats: Optional[Dict[str, Sequence[float]]] = None,
        landmarks_file: str = "landmarks.npy",
        visible_file: str = "visible.npy",
        with_pred: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.augmentation = augmentation
        with open(split_csv, encoding="utf-8") as f:
            self.samples = list(csv.DictReader(f))
        self.landmarks = np.load(self.data_dir / landmarks_file, mmap_mode="r")
        if self.landmarks.shape[-1] != 3:
            raise ValueError(f"{landmarks_file} is {self.landmarks.shape}, "
                             f"expected (N, 55, 3) -- this is the depth corpus")
        self.rendered_visible = np.load(self.data_dir / visible_file, mmap_mode="r")
        # The 2D model's OWN output on these images, if it has been precomputed.
        # A lifter is fed this at test time, so it should be trained on it; the
        # alternative, ground truth plus isotropic noise, gets the error
        # distribution wrong -- real landmark error is correlated between
        # neighbours and biased along the contour, not round.
        self.pred_xy = self.pred_conf = self.pred_feat = None
        if with_pred:
            px, pc = self.data_dir / "pred_xy.npy", self.data_dir / "pred_conf.npy"
            if not px.exists():
                raise FileNotFoundError(
                    f"{px} not found -- run scripts/precompute_xy.py first")
            self.pred_xy = np.load(px, mmap_mode="r")
            self.pred_conf = np.load(pc, mmap_mode="r")
            pf = self.data_dir / "pred_feat.npy"
            self.pred_feat = np.load(pf, mmap_mode="r") if pf.exists() else None
        mean = stats["mean"] if stats else [0.5, 0.5, 0.5]
        std = stats["std"] if stats else [0.5, 0.5, 0.5]
        self._norm_mean, self._norm_std = mean, std
        self.color_jitter_fast = None
        self.hue_jitter = 0.0
        if augmentation and augmentation.color_jitter:
            cj = dict(augmentation.color_jitter)
            self.hue_jitter = cj.pop("hue", 0.0)
            if cj:
                from torchvision import transforms
                self.color_jitter_fast = transforms.ColorJitter(**cj)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        gi = int(sample["idx"])
        image = Image.open(self.data_dir / sample["image_file"]).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        lm = torch.tensor(np.asarray(self.landmarks[gi]), dtype=torch.float32)
        xy, z = lm[:, :2].clone(), lm[:, 2].clone()
        seen = torch.tensor(np.asarray(self.rendered_visible[gi]), dtype=torch.bool)
        out_of_crop = ((xy <= 0.0) | (xy >= 1.0)).any(dim=1)

        if self.augmentation:
            image, xy, z = self._geo_augment(image, xy, z)

        tensor_img = TF.to_tensor(image)
        if self.color_jitter_fast:
            tensor_img = self.color_jitter_fast(tensor_img)
        if self.hue_jitter > 0 and torch.rand(()) < 0.3:
            tensor_img = TF.adjust_hue(
                tensor_img, float(torch.empty(()).uniform_(-self.hue_jitter,
                                                           self.hue_jitter)))
        tensor_img = TF.normalize(tensor_img, self._norm_mean, self._norm_std)

        in_frame = ((xy >= 0.0) & (xy <= 1.0)).all(dim=1)
        valid = (in_frame & ~out_of_crop).float()
        out = {
            "image": tensor_img,
            "landmarks": torch.cat([xy, z[:, None]], dim=1).view(-1),  # (55*3,)
            "visible": valid,                       # supervise x, y here
            "seen": seen.float(),                   # the renderer could see it
        }
        if self.pred_xy is not None:
            # NOT augmented: these were computed on the stored image, so they
            # belong with an unaugmented one. A lifter trained on them must
            # therefore run without geometric augmentation, which is why
            # train_depth turns it off for the lifter heads.
            out["pred_xy"] = torch.tensor(np.asarray(self.pred_xy[gi]),
                                          dtype=torch.float32).view(-1)
            out["pred_conf"] = torch.tensor(np.asarray(self.pred_conf[gi]),
                                            dtype=torch.float32)
            if self.pred_feat is not None:
                out["pred_feat"] = torch.tensor(
                    np.asarray(self.pred_feat[gi]), dtype=torch.float32)
        return out
