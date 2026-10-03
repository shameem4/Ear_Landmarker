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


class FaceMeshEarHeatmap(FaceMeshEarLandmarker):
    """MediaPipe's feature extractor under this project's heatmap decoder.

    The third arm. fm_scratch and fm_pre both lost to EarLandmarker, but they
    confounded two things: 207K params against 340K, AND MediaPipe's direct
    regression head against soft-argmax. This project had already measured that
    head family at 0.0301 against the heatmap's 0.0291, so roughly a third of
    the deficit was predicted to be head design rather than the backbone.

    This keeps backbone1 exactly as MediaPipe published it -- so the pretrained
    weights still load -- and replaces backbone2a with the decoder from
    EarLandmarkerHeatmap. Taps are the natural ones: 64ch at 24x24, 128ch at
    12x12, 128ch at 6x6, which is the same shape of pyramid the default backbone
    exposes.

    Reading the result:
      fm_heat_pre vs fm_pre      -- what the head alone was worth
      fm_heat_pre vs the control -- what remains attributable to the backbone
    """

    def __init__(self, num_landmarks: int = NUM_LANDMARKS, tau: float = 1.0) -> None:
        super().__init__(num_landmarks=num_landmarks)
        self.tau = tau

        # backbone2a is MediaPipe's regression head; this arm replaces it. It is
        # dropped so it neither trains nor appears in the parameter count.
        del self.backbone2a

        dec_ch = 64                      # matches the 24x24 tap it is summed with
        self.lat4 = nn.Conv2d(128, dec_ch, 1, bias=False)
        self.lat3 = nn.Conv2d(128, dec_ch, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(dec_ch)
        self.bn3 = nn.BatchNorm2d(dec_ch)
        from model.blocks import BlazeBlock
        self.refine = BlazeBlock(dec_ch, dec_ch)
        self.to_heatmap = nn.Conv2d(dec_ch, num_landmarks, 1, bias=True)
        # Small, not zero: a zero final weight leaves the whole decoder and
        # backbone without gradient on the first step.
        nn.init.normal_(self.to_heatmap.weight, std=0.01)
        nn.init.zeros_(self.to_heatmap.bias)

    def heatmaps(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 1, 0, 1), "constant", 0)
        feats = {}
        h = x
        for i, layer in enumerate(self.backbone1):
            h = layer(h)
            if i in (9, 12, 15):          # 24x24/64, 12x12/128, 6x6/128
                feats[i] = h
        s2, s3, s4 = feats[9], feats[12], feats[15]

        d = F.relu(self.bn4(self.lat4(s4)))
        d = F.interpolate(d, size=s3.shape[-2:], mode="bilinear", align_corners=False)
        d = d + F.relu(self.bn3(self.lat3(s3)))
        d = F.interpolate(d, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        d = d + s2
        return self.to_heatmap(self.refine(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from model.ear_landmarker import EarLandmarkerHeatmap
        return EarLandmarkerHeatmap.soft_argmax(self.heatmaps(x), self.tau)

    def predict_with_confidence(self, x: torch.Tensor, ref_std: float = 0.05):
        from model.ear_landmarker import EarLandmarkerHeatmap
        hm = self.heatmaps(x)
        return (EarLandmarkerHeatmap.soft_argmax(hm, self.tau),
                EarLandmarkerHeatmap.heatmap_confidence(hm, self.tau, ref_std))
