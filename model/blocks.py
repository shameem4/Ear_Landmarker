"""BlazeBlock building blocks for the EarLandmarker.

Adapted from BlazeEar/blazebase.py -- trainable BlazeBlock with explicit
BatchNorm (not the folded-BN BlazeBlock_WT used for MediaPipe weight loading).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BlazeBlock(nn.Module):
    """Depthwise separable conv block with skip connection.

    Architecture: DepthwiseConv -> BN -> PointwiseConv -> BN -> Add -> ReLU
    Matches MediaPipe's BlazeBlock used in BlazeFace/FaceMesh.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.channel_pad = out_channels - in_channels

        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)

        self.dw_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2 if stride == 1 else 0,
            groups=in_channels, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.pw_conv = nn.Conv2d(
            in_channels, out_channels, 1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip_proj = None
        if in_channels != out_channels:
            self.skip_proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 2:
            # TFLite-compatible asymmetric padding (kernel-size aware)
            if self.dw_conv.kernel_size[0] == 3:
                h = F.pad(x, (0, 2, 0, 2), "constant", 0)
            else:
                h = F.pad(x, (1, 2, 1, 2), "constant", 0)
            x_skip = self.max_pool(x)
        else:
            h = x
            x_skip = x

        h = self.bn1(self.dw_conv(h))
        h = self.bn2(self.pw_conv(h))

        if self.skip_proj is not None:
            x_skip = self.skip_proj(x_skip)
        elif self.channel_pad > 0:
            x_skip = F.pad(x_skip, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)

        return self.act(h + x_skip)
