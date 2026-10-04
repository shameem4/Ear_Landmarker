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

# Attribution only. Set to True *solely* by scripts/train_baseline_v1.py, to
# reproduce the v1 rotation-label bug and measure what fixing it was worth.
# Must stay False for any real training run -- it corrupts every rotated label.
LEGACY_ROTATION_SIGN = False


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
    # Simulated out-of-plane head turn. The source datasets are near-frontal, and
    # measured error rises +59% at 40 deg yaw and +89% at 50 deg, which is exactly
    # the range real captures use. No other augmentation covers foreshortening.
    perspective_deg: float = 0.0
    perspective_prob: float = 0.5


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
        landmarks_file: str = "landmarks.npy",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.augmentation = augmentation

        # Load split manifest
        with open(split_csv, encoding="utf-8") as f:
            self.samples: List[Dict] = list(csv.DictReader(f))

        # Memory-map landmarks for fast access
        self.landmarks = np.load(
            self.data_dir / landmarks_file, mmap_mode="r",
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

        # Load landmarks (normalized to the crop; may fall outside [0, 1] when
        # the annotated point lies beyond it)
        landmarks = torch.tensor(
            self.landmarks[global_idx].copy(), dtype=torch.float32,
        )  # (55, 2)

        # Points that were never inside the crop to begin with. Older
        # preprocessed sets clamped these onto the border, so a coordinate
        # sitting exactly on 0.0 or 1.0 is treated as out-of-frame too: a
        # genuinely-located point hits the border exactly with probability ~0,
        # whereas a clamped one hits it by construction. Either way the true
        # position is unknown, so the point must not be supervised.
        out_of_crop = ((landmarks <= 0.0) | (landmarks >= 1.0)).any(dim=1)

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

        # Points pushed outside the frame by augmentation are unreachable by the
        # sigmoid head. Mask them out rather than clamping them onto the border,
        # which would supervise the model toward a position the point isn't at.
        # `out_of_crop` additionally drops points that were outside before any
        # augmentation ran, which augmentation can otherwise carry back inside.
        in_frame = ((landmarks >= 0.0) & (landmarks <= 1.0)).all(dim=1)
        visible = (in_frame & ~out_of_crop).float()  # (55,)

        return {
            "image": tensor_img,
            "landmarks": landmarks.view(-1),
            "visible": visible,
        }

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
            landmarks[:, 0] = landmarks[:, 0] - tx
            landmarks[:, 1] = landmarks[:, 1] - ty

        # Out-of-plane turn (perspective)
        if aug.perspective_deg > 0 and random.random() < aug.perspective_prob:
            yaw = random.uniform(-aug.perspective_deg, aug.perspective_deg)
            pitch = random.uniform(-aug.perspective_deg, aug.perspective_deg)
            image, landmarks = self._perspective(image, landmarks, yaw, pitch)

        # Rotation
        if aug.rotation_deg > 0:
            angle = random.uniform(-aug.rotation_deg, aug.rotation_deg)
            image = image.rotate(angle, resample=Image.BILINEAR, fillcolor=(128, 128, 128))
            rad = math.radians(angle)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            landmarks = landmarks.clone()
            c = landmarks - 0.5
            # PIL rotates the image counter-clockwise; in image coords (y down)
            # that is [[cos, sin], [-sin, cos]]. The transpose (v1) sends the
            # labels the opposite way and mislabels every rotated sample.
            s = -sin_a if LEGACY_ROTATION_SIGN else sin_a
            x_new = c[:, 0] * cos_a + c[:, 1] * s
            y_new = -c[:, 0] * s + c[:, 1] * cos_a
            landmarks[:, 0] = x_new + 0.5
            landmarks[:, 1] = y_new + 0.5

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
        landmarks[:, 0] = (landmarks[:, 0] * w - x1) / crop_w
        landmarks[:, 1] = (landmarks[:, 1] * h - y1) / crop_h

        return cropped, landmarks

    @staticmethod
    def _perspective_matrix(yaw_deg: float, pitch_deg: float, focal: float = 2.0) -> np.ndarray:
        """Homography (in normalized [0,1] coords) simulating out-of-plane turn.

        Treats the ear as a plane, rotates it about the vertical (yaw) and
        horizontal (pitch) axes, and reprojects. This models foreshortening but
        not 3D parallax or self-occlusion, so it is an approximation of a real
        head turn -- a useful one, since foreshortening is the dominant effect
        and it costs nothing to generate.

        Returns:
            (3, 3) forward matrix mapping input -> output in normalized coords.
        """
        ty, tp = math.radians(yaw_deg), math.radians(pitch_deg)
        # Work in centred coords in [-1, 1], then map back to [0, 1].
        to_centred = np.array([[2.0, 0.0, -1.0], [0.0, 2.0, -1.0], [0.0, 0.0, 1.0]])
        from_centred = np.array([[0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]])
        # x' = x cos(yaw), y' = y cos(pitch), with depth z = x sin(yaw) + y sin(pitch)
        # giving the perspective divide (1 + z/f).
        core = np.array([
            [math.cos(ty), 0.0, 0.0],
            [0.0, math.cos(tp), 0.0],
            [math.sin(ty) / focal, math.sin(tp) / focal, 1.0],
        ])
        return from_centred @ core @ to_centred

    @staticmethod
    def _apply_homography(h: np.ndarray, pts: torch.Tensor) -> torch.Tensor:
        """Apply a 3x3 homography to (N, 2) normalized points."""
        p = pts.detach().cpu().numpy().astype(np.float64)
        ones = np.ones((len(p), 1))
        q = np.concatenate([p, ones], axis=1) @ h.T
        w = np.where(np.abs(q[:, 2:3]) < 1e-9, 1e-9, q[:, 2:3])
        return torch.from_numpy((q[:, :2] / w).astype(np.float32))

    def _perspective(
        self, image: Image.Image, landmarks: torch.Tensor,
        yaw_deg: float, pitch_deg: float,
    ) -> Tuple[Image.Image, torch.Tensor]:
        """Warp image and landmarks by the same homography.

        PIL's PERSPECTIVE transform maps OUTPUT pixels back to INPUT pixels, so
        it needs the INVERSE of the matrix used for the landmarks. Getting that
        backwards is the same class of error as the v1 rotation bug, so
        tests/test_augmentation.py asserts image and labels agree.
        """
        h_fwd = self._perspective_matrix(yaw_deg, pitch_deg)
        size = self.image_size

        # Convert the normalized forward matrix to pixel coords, then invert for PIL.
        scale = np.array([[size, 0.0, 0.0], [0.0, size, 0.0], [0.0, 0.0, 1.0]])
        unscale = np.array([[1.0 / size, 0.0, 0.0], [0.0, 1.0 / size, 0.0], [0.0, 0.0, 1.0]])
        h_px = scale @ h_fwd @ unscale
        h_inv = np.linalg.inv(h_px)
        h_inv = h_inv / h_inv[2, 2]

        image = image.transform(
            (size, size), Image.PERSPECTIVE, h_inv.flatten()[:8].tolist(),
            resample=Image.BILINEAR, fillcolor=(128, 128, 128),
        )
        return image, self._apply_homography(h_fwd, landmarks)
