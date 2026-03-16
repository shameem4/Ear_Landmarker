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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, 3, 192, 192) input tensor, normalized to [-1, 1].

        Returns:
            (B, num_landmarks * 2) landmark coordinates in [0, 1].
        """
        # TFLite-compatible asymmetric padding for first conv
        x = F.pad(x, (1, 2, 1, 2), "constant", 0)
        x = self.conv0(x)      # (B, 24, 96, 96)
        x = self.stage0(x)     # (B, 24, 96, 96)
        x = self.stage1(x)     # (B, 48, 48, 48)
        x = self.stage2(x)     # (B, 96, 24, 24)
        x = self.stage3(x)     # (B, 128, 12, 12)
        x = self.stage4(x)     # (B, 192, 6, 6)
        return self.head(x)    # (B, 110)

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
