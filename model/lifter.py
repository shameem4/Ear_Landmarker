"""Head 3: a separate network that lifts 55 (x, y) to 55 z.

The 2D landmarker is left exactly as it is and a small MLP learns depth from
its output alone. That makes it the cheapest of the three to try, the easiest
to evaluate on its own terms, and the only one that cannot damage the existing
2D accuracy.

ITS WEAKNESS IS STRUCTURAL, not a tuning problem: it never sees the image. Two
ears photographed at different yaws can project to nearly the same 55 points
while having quite different depth, and nothing in the input distinguishes
them. Whatever it achieves is therefore a lower bound on what the image makes
possible, and the comparison against the other two heads is the measurement
that says how much the pixels were worth.

The input is centred and scaled by the landmarks' own spread, so the lifter
cannot learn "ears low in the crop are tilted away" -- a crop position is an
artefact of the detector, not of the ear. z comes back in the same units by
multiplying the spread back in.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.ear_landmarker import NUM_LANDMARKS


class EarDepthLifter(nn.Module):
    """(B, K, 2) -> (B, K) depth, in the same units as the input coordinates."""

    def __init__(self, num_landmarks: int = NUM_LANDMARKS, width: int = 256,
                 depth: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.k = num_landmarks
        layers = [nn.Linear(num_landmarks * 2, width), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.LayerNorm(width), nn.GELU(),
                       nn.Dropout(dropout)]
        layers += [nn.Linear(width, num_landmarks)]
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)      # start at z = 0, the root depth

    @staticmethod
    def normalise(xy: torch.Tensor):
        """Centre and scale by the point spread. Returns (normalised, scale)."""
        c = xy.mean(dim=1, keepdim=True)
        s = (xy.amax(dim=1) - xy.amin(dim=1)).amax(dim=1).clamp_min(1e-6)
        return (xy - c) / s[:, None, None], s

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        xy = xy.view(len(xy), self.k, 2)
        n, s = self.normalise(xy)
        z = self.net(n.reshape(len(xy), -1))
        z = z - z.mean(dim=1, keepdim=True)     # root-relative, as the label is
        return z * s[:, None]


class LiftedLandmarker(nn.Module):
    """A frozen 2D landmarker with a lifter bolted on; emits (B, K*3)."""

    def __init__(self, base: nn.Module, lifter: EarDepthLifter,
                 freeze_base: bool = True) -> None:
        super().__init__()
        self.base, self.lifter = base, lifter
        self.freeze_base = freeze_base
        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze_base:
            with torch.no_grad():
                xy = self.base(x)
        else:
            xy = self.base(x)
        xy = xy.view(len(x), -1, 2)
        z = self.lifter(xy)
        return torch.cat([xy, z[:, :, None]], dim=-1).reshape(len(x), -1)
