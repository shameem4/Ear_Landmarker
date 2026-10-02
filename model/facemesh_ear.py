"""MediaPipe FaceMesh, copied exactly, with the output head resized for ears.

The question this answers: the EarLandmarker backbone was designed independently
(5x5 depthwise, 24-48-96-128-192, heatmap head) and shares almost nothing with
MediaPipe's shipped models -- 0.9% of parameters are even shape-compatible. Was
that design worth it, or would MediaPipe's face landmark architecture, which
comes with weights trained on millions of faces, do as well or better?

WHAT IS COPIED EXACTLY
  - backbone1: 14 BlazeBlocks, 3x3 depthwise, PReLU, 16-32-64-128, folded-BN
    form, 115,824 params. Byte-identical structure to blazeface_landmark.pth,
    which loads into it with strict=True.
  - backbone2a: the regression head's blocks and 1x1 projection, unchanged.
  - 192x192 input, and MediaPipe's pad-to-193 before the first stride-2 conv.
    The resolution happens to match EarLandmarker's exactly.

THE ONE UNAVOIDABLE CHANGE
  The final conv emits 1404 channels = 468 landmarks x 3. Ears need 55 x 2, so
  that conv becomes Conv2d(32, 110, 3). Nothing else is touched, and the layer
  is necessarily randomly initialised either way -- 468 face points have no
  correspondence with 55 ear points.

  Coordinates are divided by the input resolution rather than passed through a
  sigmoid, because MediaPipe regresses pixel coordinates linearly and adding a
  nonlinearity would stop this being a copy.

WHAT THE RESULT WILL AND WILL NOT MEAN
  This head is DIRECT REGRESSION from a 3x3 conv at 1x1 spatial -- the same
  family as the GAP+FC head this project already measured at 0.0301 against the
  soft-argmax heatmap's 0.0291. So a faithful copy inherits a head worth about
  -4%, which the pretrained features have to win back before it breaks even.
  That is the trade the experiment measures, not a flaw in it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# BlazeBlock_WT is the folded-BatchNorm block MediaPipe's weights are stored in.
# Importing it from BlazeEar rather than copying keeps one definition.
BLAZEEAR_DIR = Path(__file__).resolve().parents[2] / "BlazeEar"

NUM_LANDMARKS = 55
RESOLUTION = 192


def _block(*args, **kwargs):
    sys.path.insert(0, str(BLAZEEAR_DIR))
    from blazebase import BlazeBlock_WT  # type: ignore
    return BlazeBlock_WT(*args, **kwargs)


class FaceMeshEarLandmarker(nn.Module):
    """MediaPipe's face landmark topology, output head resized to 55 x 2."""

    def __init__(self, num_landmarks: int = NUM_LANDMARKS) -> None:
        super().__init__()
        self.num_landmarks = num_landmarks
        B = _block

        self.backbone1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 0, bias=True), nn.PReLU(16),
            B(16, 16, 3, act="prelu"), B(16, 16, 3, act="prelu"),
            B(16, 32, 3, 2, act="prelu"), B(32, 32, 3, act="prelu"),
            B(32, 32, 3, act="prelu"),
            B(32, 64, 3, 2, act="prelu"), B(64, 64, 3, act="prelu"),
            B(64, 64, 3, act="prelu"),
            B(64, 128, 3, 2, act="prelu"), B(128, 128, 3, act="prelu"),
            B(128, 128, 3, act="prelu"),
            B(128, 128, 3, 2, act="prelu"), B(128, 128, 3, act="prelu"),
            B(128, 128, 3, act="prelu"),
        )
        self.backbone2a = nn.Sequential(
            B(128, 128, 3, 2, act="prelu"), B(128, 128, 3, act="prelu"),
            B(128, 128, 3, act="prelu"),
            nn.Conv2d(128, 32, 1, bias=True), nn.PReLU(32),
            B(32, 32, 3, act="prelu"),
            # 1404 -> num_landmarks * 2. The only shape change.
            nn.Conv2d(32, num_landmarks * 2, 3, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MediaPipe pads 192 -> 193 before the first stride-2 conv; without it
        # the depthwise and max-pool skip paths differ by one pixel and the
        # residual add fails.
        x = F.pad(x, (0, 1, 0, 1), "constant", 0)
        feat = self.backbone1(x)
        out = self.backbone2a(feat).view(-1, self.num_landmarks * 2)
        # Pixel coordinates -> [0, 1], matching this project's convention. A
        # plain scale, not a sigmoid, so the head stays linear as MediaPipe's is.
        return out / RESOLUTION

    def load_mediapipe_backbone(self, weights_path: str | Path) -> int:
        """Load the pretrained face-landmark weights.

        backbone1 transfers whole. backbone2a transfers except its final conv,
        which changed shape for 55 points and cannot carry over.

        Returns:
            Number of parameters transferred.
        """
        src = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        dst = self.state_dict()
        moved = 0
        for k, v in src.items():
            if k in dst and dst[k].shape == v.shape:
                dst[k] = v.clone()
                moved += v.numel()
        self.load_state_dict(dst, strict=True)
        return moved
