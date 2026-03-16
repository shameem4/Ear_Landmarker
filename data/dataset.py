"""PyTorch Dataset for ear landmark training.

Loads images from disk and landmarks from a memory-mapped .npy array.
Augmentations are split: geometric ops on PIL (must transform landmarks),
color jitter on tensors (no landmark transform, avoids slow PIL HSV conversion).

Adapted from scratch/old_earlandmarker/EarLandmarks/ear_landmarks/data.py.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

NUM_LANDMARKS = 55


@dataclass
class AugmentationParams:
    """Augmentation configuration for training."""

    horizontal_flip: bool = True
    flip_prob: float = 0.5
    translation: float = 0.05
    rotation_deg: float = 15.0
    color_jitter: Optional[Dict[str, float]] = field(default_factory=lambda: {
        "brightness": 0.3, "contrast": 0.3, "saturation": 0.2, "hue": 0.05,
    })
    bbox_jitter: float = 0.1
    bbox_jitter_prob: float = 0.5


class EarLandmarkDataset(Dataset):
    """Dataset for ear landmark regression.

    Args:
        split_csv: Path to train.csv or val.csv.
        data_dir: Path to preprocessed/ directory containing images/ and landmarks.npy.
        image_size: Target image size (square).
        augmentation: Augmentation params (None for val/test).
        stats: Channel mean/std for normalization. Defaults to [-1, 1] range.
    """

    def __init__(
        self,
        split_csv: Path | str,
        data_dir: Path | str,
        image_size: int = 192,
        augmentation: Optional[AugmentationParams] = None,
        stats: Optional[Dict[str, Sequence[float]]] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.augmentation = augmentation

        # Load split manifest
        with open(split_csv, encoding="utf-8") as f:
            self.samples: List[Dict] = list(csv.DictReader(f))

        # Memory-map landmarks for fast access
        self.landmarks = np.load(
            self.data_dir / "landmarks.npy", mmap_mode="r",
        )  # (N_total, 55, 2) float32

        # Normalization (applied after ToTensor)
        mean = stats["mean"] if stats else [0.5, 0.5, 0.5]
        std = stats["std"] if stats else [0.5, 0.5, 0.5]
        self._norm_mean = mean
        self._norm_std = std

        # Color augmentation: brightness/contrast/saturation as fast tensor ops,
        # hue jitter separated (expensive HSV conversion) and applied less frequently.
        self.color_jitter_fast = None
        self.hue_jitter = 0.0
        if augmentation and augmentation.color_jitter:
            cj = augmentation.color_jitter.copy()
            self.hue_jitter = cj.pop("hue", 0.0)
            if cj:
                self.color_jitter_fast = transforms.ColorJitter(**cj)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        global_idx = int(sample["idx"])

        # Load image and resize
        img_path = self.data_dir / sample["image_file"]
        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        # Load landmarks (already normalized to [0, 1])
        landmarks = torch.tensor(
            self.landmarks[global_idx].copy(), dtype=torch.float32,
        )  # (55, 2)

        # Geometric augmentations (PIL space -- must transform landmarks)
        if self.augmentation:
            image, landmarks = self._geo_augment(image, landmarks)

        # PIL -> tensor [0, 1]
        tensor_img = TF.to_tensor(image)

        # Fast color jitter (brightness/contrast/saturation -- no HSV conversion)
        if self.color_jitter_fast:
            tensor_img = self.color_jitter_fast(tensor_img)
        # Hue jitter only 30% of the time (expensive HSV conversion)
        if self.hue_jitter > 0 and random.random() < 0.3:
            tensor_img = TF.adjust_hue(tensor_img, random.uniform(-self.hue_jitter, self.hue_jitter))

        # Normalize to [-1, 1]
        tensor_img = TF.normalize(tensor_img, self._norm_mean, self._norm_std)

        return {"image": tensor_img, "landmarks": landmarks.view(-1)}

    def _geo_augment(
        self, image: Image.Image, landmarks: torch.Tensor,
    ) -> Tuple[Image.Image, torch.Tensor]:
        """Geometric augmentations that require joint image+landmark transform."""
        aug = self.augmentation

        # Bbox jitter (simulates detector crop variance)
        if aug.bbox_jitter > 0 and random.random() < aug.bbox_jitter_prob:
            image, landmarks = self._bbox_jitter(image, landmarks, aug.bbox_jitter)

        # Horizontal flip
        if aug.horizontal_flip and random.random() < aug.flip_prob:
            image = ImageOps.mirror(image)
            landmarks = landmarks.clone()
            landmarks[:, 0] = 1.0 - landmarks[:, 0]

        # Translation
        if aug.translation > 0:
            tx = random.uniform(-1, 1) * aug.translation
            ty = random.uniform(-1, 1) * aug.translation
            border = int(self.image_size * aug.translation)
            image = ImageOps.expand(image, border=border, fill=(128, 128, 128))
            left = int(border + tx * self.image_size)
            top = int(border + ty * self.image_size)
            image = image.crop((left, top, left + self.image_size, top + self.image_size))
            landmarks = landmarks.clone()
            landmarks[:, 0] = torch.clamp(landmarks[:, 0] - tx, 0.0, 1.0)
            landmarks[:, 1] = torch.clamp(landmarks[:, 1] - ty, 0.0, 1.0)

        # Rotation
        if aug.rotation_deg > 0:
            angle = random.uniform(-aug.rotation_deg, aug.rotation_deg)
            image = image.rotate(angle, resample=Image.BILINEAR, fillcolor=(128, 128, 128))
            rad = math.radians(angle)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            landmarks = landmarks.clone()
            c = landmarks - 0.5
            x_new = c[:, 0] * cos_a - c[:, 1] * sin_a
            y_new = c[:, 0] * sin_a + c[:, 1] * cos_a
            landmarks[:, 0] = torch.clamp(x_new + 0.5, 0.0, 1.0)
            landmarks[:, 1] = torch.clamp(y_new + 0.5, 0.0, 1.0)

        return image, landmarks

    def _bbox_jitter(
        self, image: Image.Image, landmarks: torch.Tensor, jitter: float,
    ) -> Tuple[Image.Image, torch.Tensor]:
        """Simulate detector bbox variance by random crop/scale."""
        w, h = image.size
        scale = random.uniform(1.0 - jitter, 1.0 + jitter)
        tx = random.uniform(-jitter, jitter) * w
        ty = random.uniform(-jitter, jitter) * h

        new_w, new_h = w / scale, h / scale
        cx, cy = w / 2 + tx, h / 2 + ty
        x1 = max(0, cx - new_w / 2)
        y1 = max(0, cy - new_h / 2)
        x2 = min(w, cx + new_w / 2)
        y2 = min(h, cy + new_h / 2)

        cropped = image.crop((x1, y1, x2, y2))
        cropped = cropped.resize((self.image_size, self.image_size), Image.BILINEAR)

        landmarks = landmarks.clone()
        crop_w, crop_h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        landmarks[:, 0] = torch.clamp((landmarks[:, 0] * w - x1) / crop_w, 0.0, 1.0)
        landmarks[:, 1] = torch.clamp((landmarks[:, 1] * h - y1) / crop_h, 0.0, 1.0)

        return cropped, landmarks
