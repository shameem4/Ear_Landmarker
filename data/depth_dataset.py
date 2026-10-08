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
        return {
            "image": tensor_img,
            "landmarks": torch.cat([xy, z[:, None]], dim=1).view(-1),  # (55*3,)
            "visible": valid,                       # supervise x, y here
            "seen": seen.float(),                   # the renderer could see it
        }
