"""EarLandmarker: FaceMesh-style landmark regression using BlazeBlocks.

Architecture mirrors MediaPipe FaceMesh -- a BlazeBlock backbone with
direct coordinate regression, designed for real-time inference on webcam.

Input:  (B, 3, 192, 192) cropped ear ROI, normalized to [-1, 1]
Output: (B, 110) -- 55 landmarks x 2 (x, y) in [0, 1]
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.blocks import BlazeBlock

NUM_LANDMARKS = 55


def _make_stage(in_ch: int, out_ch: int, num_blocks: int) -> nn.Sequential:
    """Build a stage: one stride-2 block + (num_blocks-1) stride-1 blocks."""
    layers = [BlazeBlock(in_ch, out_ch, stride=2)]
    for _ in range(num_blocks - 1):
        layers.append(BlazeBlock(out_ch, out_ch))
    return nn.Sequential(*layers)


class EarLandmarker(nn.Module):
    """BlazeBlock-based ear landmark regressor (FaceMesh architecture pattern).

    ~300K params. Designed for 192x192 input from BlazeEar detector crops.

    Backbone:
        Conv 5x5 s=2     (3 -> 24, 96x96)
        Stage 0: 2 blocks (24 -> 24, 96x96)    -- no stride, refine early features
        Stage 1: 4 blocks (24 -> 48, 48x48)
        Stage 2: 4 blocks (48 -> 96, 24x24)
        Stage 3: 4 blocks (96 -> 128, 12x12)
        Stage 4: 3 blocks (128 -> 192, 6x6)

    Head:
        Global Average Pooling -> FC(192, 110) -> Sigmoid
    """

    def __init__(self, num_landmarks: int = NUM_LANDMARKS) -> None:
        super().__init__()
        self.num_landmarks = num_landmarks

        # Initial convolution (matches BlazeEar/BlazeFace pattern)
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )

        # Stage 0: refine at 96x96 (no downsampling)
        self.stage0 = nn.Sequential(
            BlazeBlock(24, 24),
            BlazeBlock(24, 24),
        )

        # Stages 1-4: progressive downsampling
        self.stage1 = _make_stage(24, 48, num_blocks=4)    # -> 48x48
        self.stage2 = _make_stage(48, 96, num_blocks=4)    # -> 24x24
        self.stage3 = _make_stage(96, 128, num_blocks=4)   # -> 12x12
        self.stage4 = _make_stage(128, 192, num_blocks=3)  # -> 6x6

        # Regression head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(192, num_landmarks * 2),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming init for conv layers, default for BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_features(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the backbone, returning the last three stage outputs.

        Returns:
            (stage2, stage3, stage4) at 24x24, 12x12 and 6x6. Heads that need
            spatial detail can use the earlier maps; the GAP head uses stage4.
        """
        # TFLite-compatible asymmetric padding for first conv
        x = F.pad(x, (1, 2, 1, 2), "constant", 0)
        x = self.conv0(x)      # (B, 24, 96, 96)
        x = self.stage0(x)     # (B, 24, 96, 96)
        x = self.stage1(x)     # (B, 48, 48, 48)
        s2 = self.stage2(x)    # (B, 96, 24, 24)
        s3 = self.stage3(s2)   # (B, 128, 12, 12)
        s4 = self.stage4(s3)   # (B, 192, 6, 6)
        return s2, s3, s4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, 3, 192, 192) input tensor, normalized to [-1, 1].

        Returns:
            (B, num_landmarks * 2) landmark coordinates in [0, 1].
        """
        _, _, s4 = self.forward_features(x)
        return self.head(s4)   # (B, 110)

    def load_blazeear_backbone(self, checkpoint_path: str | Path) -> int:
        """Initialize early layers from a trained BlazeEar detector checkpoint.

        Transfers conv0 and the first few BlazeBlocks from the detector's
        backbone1 where channel dimensions match.

        Returns:
            Number of parameters transferred.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = ckpt if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt else ckpt["model_state_dict"]

        transferred = 0
        my_state = self.state_dict()

        # Map BlazeEar backbone1 layers to our stages
        # BlazeEar backbone1[0] = Conv2d(3, 24, 5, s=2)  -> our conv0[0]
        # BlazeEar backbone1[1] = ReLU                    -> skip
        # BlazeEar backbone1[2] = BlazeBlock_WT(24, 24)   -> our stage0[0]
        # BlazeEar backbone1[3] = BlazeBlock_WT(24, 28)   -> channels diverge, stop

        # Transfer initial conv
        src_key = "backbone1.0.weight"
        dst_key = "conv0.0.weight"
        if src_key in state and dst_key in my_state:
            if state[src_key].shape == my_state[dst_key].shape:
                my_state[dst_key] = state[src_key]
                transferred += state[src_key].numel()

        self.load_state_dict(my_state, strict=True)
        return transferred


class EarLandmarkerHeatmap(EarLandmarker):
    """Same backbone, but predicts landmarks via heatmaps + soft-argmax.

    Motivation: the GAP head collapses the 6x6x192 feature map to 192 numbers
    before regressing coordinates. Global average pooling is permutation-invariant
    over spatial positions, so it discards *where* features are -- the one thing
    landmark localisation needs -- forcing the network to recover position from
    channel statistics alone.

    A heatmap head keeps the spatial map and reduces it with soft-argmax (the
    expectation of a spatial softmax). Unlike hard argmax this is differentiable
    and gives sub-pixel precision, so the usual objection to heatmaps -- that
    argmax quantises to the grid -- does not apply.

    Decoder (lightweight, upsamples 6x6 -> 24x24 fusing stage3 and stage2 so the
    output has both the deep stages' context and stage2's spatial detail):

        stage4 (192, 6x6)  --1x1--> 96, upsample -> 12x12
                                     + stage3 (128 --1x1--> 96)
                           upsample -> 24x24
                                     + stage2 (96)
        --> BlazeBlock(96, 96) --1x1--> num_landmarks heatmaps @ 24x24
        --> soft-argmax --> (B, num_landmarks * 2) in [0, 1]

    Output shape and range match the GAP head exactly, so losses, inference and
    ONNX export are unchanged.

    Args:
        num_landmarks: Number of landmark points.
        tau: Softmax temperature. <1 sharpens the distribution (more peaked,
            closer to argmax), >1 flattens it.
    """

    def __init__(self, num_landmarks: int = NUM_LANDMARKS, tau: float = 1.0) -> None:
        super().__init__(num_landmarks=num_landmarks)
        self.tau = tau

        # The GAP head is unused by this subclass; drop it so it does not appear
        # in the state dict or the parameter count.
        del self.head

        dec_ch = 96
        self.lat4 = nn.Conv2d(192, dec_ch, 1, bias=False)
        self.lat3 = nn.Conv2d(128, dec_ch, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(dec_ch)
        self.bn3 = nn.BatchNorm2d(dec_ch)
        self.refine = BlazeBlock(dec_ch, dec_ch)
        self.to_heatmap = nn.Conv2d(dec_ch, num_landmarks, 1, bias=True)

        self._init_weights()
        # Small (not zero) weights on the final 1x1: the heatmaps start nearly
        # flat, so early predictions sit at the image centre, but gradient still
        # reaches the decoder and backbone. Zero-initialising this weight makes
        # d(loss)/d(decoder) = dL/dheatmap * weight = 0, which leaves the whole
        # network below it without gradient on the first step.
        nn.init.normal_(self.to_heatmap.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.to_heatmap.bias)

    @staticmethod
    def soft_argmax(heatmaps: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        """Expected (x, y) of a spatial softmax over each heatmap.

        Args:
            heatmaps: (B, K, H, W) unnormalized scores.
            tau: Softmax temperature.

        Returns:
            (B, K * 2) coordinates in [0, 1], interleaved as (x0, y0, x1, y1, ...).
        """
        b, k, h, w = heatmaps.shape
        probs = F.softmax(heatmaps.reshape(b, k, h * w) / tau, dim=-1)
        probs = probs.reshape(b, k, h, w)

        # Pixel centres, so a point can reach the full [0, 1] span unbiased.
        xs = (torch.arange(w, dtype=probs.dtype, device=probs.device) + 0.5) / w
        ys = (torch.arange(h, dtype=probs.dtype, device=probs.device) + 0.5) / h

        x = (probs.sum(dim=2) * xs).sum(dim=-1)   # marginalise over y -> (B, K)
        y = (probs.sum(dim=3) * ys).sum(dim=-1)   # marginalise over x -> (B, K)
        return torch.stack((x, y), dim=-1).reshape(b, k * 2)

    def heatmaps(self, x: torch.Tensor) -> torch.Tensor:
        """Return the raw (B, K, 24, 24) heatmaps.

        Exposed because the spatial softmax doubles as a per-point confidence,
        which is the principled quantity to drive temporal smoothing.
        """
        s2, s3, s4 = self.forward_features(x)

        d = self.bn4(self.lat4(s4))                                  # (B, 96, 6, 6)
        d = F.interpolate(d, size=s3.shape[-2:], mode="bilinear", align_corners=False)
        d = d + self.bn3(self.lat3(s3))                              # (B, 96, 12, 12)
        d = F.interpolate(d, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        d = F.relu(d + s2)                                           # (B, 96, 24, 24)
        d = self.refine(d)
        return self.to_heatmap(d)                                    # (B, K, 24, 24)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.soft_argmax(self.heatmaps(x), self.tau)

    @staticmethod
    def heatmap_confidence(
        heatmaps: torch.Tensor, tau: float = 1.0, ref_std: float = 0.05,
    ) -> torch.Tensor:
        """Per-landmark confidence in (0, 1] from heatmap concentration.

        A peaked spatial softmax means the model has localised the point; a
        diffuse one means it is hedging across a region. The spatial standard
        deviation of the distribution is therefore a direct estimate of
        positional uncertainty, which is exactly what temporal smoothing wants:
        smooth uncertain points harder than confident ones.

        Args:
            heatmaps: (B, K, H, W) unnormalized scores.
            tau: Softmax temperature, matching soft_argmax.
            ref_std: Spatial std (in normalized [0,1] units) mapping to
                confidence 1/e. Heuristic scale, not a calibrated probability.

        Returns:
            (B, K) confidence in (0, 1].
        """
        b, k, h, w = heatmaps.shape
        probs = F.softmax(heatmaps.reshape(b, k, h * w) / tau, dim=-1).reshape(b, k, h, w)

        xs = (torch.arange(w, dtype=probs.dtype, device=probs.device) + 0.5) / w
        ys = (torch.arange(h, dtype=probs.dtype, device=probs.device) + 0.5) / h
        px = probs.sum(dim=2)          # marginal over y -> (B, K, W)
        py = probs.sum(dim=3)          # marginal over x -> (B, K, H)

        mx = (px * xs).sum(-1)
        my = (py * ys).sum(-1)
        vx = (px * (xs - mx.unsqueeze(-1)) ** 2).sum(-1)
        vy = (py * (ys - my.unsqueeze(-1)) ** 2).sum(-1)
        std = torch.sqrt((vx + vy).clamp_min(0.0) / 2.0)

        return torch.exp(-std / ref_std)

    def predict_with_confidence(
        self, x: torch.Tensor, ref_std: float = 0.05,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Coordinates plus per-point confidence in one pass.

        Returns:
            ((B, K*2) coords in [0, 1], (B, K) confidence in (0, 1]).
        """
        hm = self.heatmaps(x)
        return (
            self.soft_argmax(hm, self.tau),
            self.heatmap_confidence(hm, self.tau, ref_std),
        )
