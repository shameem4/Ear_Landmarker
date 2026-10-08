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
    """(B, K, 2) -> (B, K) depth, in the same units as the input coordinates.

    `use_conf` adds the 2D model's per-landmark confidence as a third channel.
    It comes free from the heatmap's concentration, and a point the model is
    unsure of is exactly a point whose depth is ambiguous.
    """

    def __init__(self, num_landmarks: int = NUM_LANDMARKS, width: int = 256,
                 depth: int = 3, dropout: float = 0.1,
                 use_conf: bool = False) -> None:
        super().__init__()
        self.k = num_landmarks
        self.use_conf = use_conf
        in_dim = num_landmarks * (3 if use_conf else 2)
        layers = [nn.Linear(in_dim, width), nn.GELU()]
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

    def forward(self, xy: torch.Tensor,
                conf: torch.Tensor | None = None) -> torch.Tensor:
        xy = xy.view(len(xy), self.k, 2)
        n, s = self.normalise(xy)
        if self.use_conf:
            if conf is None:
                raise ValueError("this lifter was built with use_conf=True")
            n = torch.cat([n, conf.view(len(xy), self.k, 1)], dim=-1)
        z = self.net(n.reshape(len(xy), -1))
        z = z - z.mean(dim=1, keepdim=True)     # root-relative, as the label is
        return z * s[:, None]


class FeatureLifter(nn.Module):
    """Head 3b: the lifter, plus image features sampled AT each landmark.

    Deliberately the same shape as EarDepthLifter -- per-point encoding, then
    one MLP over all the points together -- so the only difference between them
    is what goes in. That makes the pair a clean ablation: whatever 3b gains
    over 3a is what the pixels were worth, and nothing else changed to muddy it.

    Features are read off the decoder's 24x24 map by bilinear sampling at the
    predicted (x, y). That is where the local evidence depth needs actually
    lives: the shading across the helix, the occluding edge of the rim, the
    shadow in the concha. A global pooled vector would blur all of it together.
    """

    def __init__(self, feat_ch: int, num_landmarks: int = NUM_LANDMARKS,
                 point_width: int = 32, width: int = 256, depth: int = 3,
                 dropout: float = 0.1, use_conf: bool = True) -> None:
        super().__init__()
        self.k, self.use_conf = num_landmarks, use_conf
        self.point = nn.Sequential(
            nn.Linear(feat_ch + 2 + (1 if use_conf else 0), point_width),
            nn.GELU(), nn.Linear(point_width, point_width), nn.GELU())
        layers = [nn.Linear(num_landmarks * point_width, width), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.LayerNorm(width), nn.GELU(),
                       nn.Dropout(dropout)]
        layers += [nn.Linear(width, num_landmarks)]
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @staticmethod
    def sample(feat: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) sampled at (B, K, 2) in [0, 1] -> (B, K, C)."""
        g = (xy * 2.0 - 1.0).view(len(xy), -1, 1, 2)      # grid_sample wants [-1,1]
        out = nn.functional.grid_sample(feat, g, mode="bilinear",
                                        align_corners=False)
        return out.squeeze(-1).transpose(1, 2)            # (B, K, C)

    def from_sampled(self, f: torch.Tensor, xy: torch.Tensor,
                     conf: torch.Tensor | None = None) -> torch.Tensor:
        """The second half of forward, given features already sampled."""
        xy = xy.view(len(xy), self.k, 2)
        n, s = EarDepthLifter.normalise(xy)
        parts = [f, n]
        if self.use_conf:
            if conf is None:
                raise ValueError("this lifter was built with use_conf=True")
            parts.append(conf.view(len(xy), self.k, 1))
        t = self.point(torch.cat(parts, dim=-1))
        z = self.net(t.reshape(len(xy), -1))
        z = z - z.mean(dim=1, keepdim=True)
        return z * s[:, None]

    def forward(self, feat: torch.Tensor, xy: torch.Tensor,
                conf: torch.Tensor | None = None) -> torch.Tensor:
        xy = xy.view(len(xy), self.k, 2)
        n, s = EarDepthLifter.normalise(xy)
        f = self.sample(feat, xy.clamp(0.0, 1.0))
        parts = [f, n]
        if self.use_conf:
            if conf is None:
                raise ValueError("this lifter was built with use_conf=True")
            parts.append(conf.view(len(xy), self.k, 1))
        t = self.point(torch.cat(parts, dim=-1))          # (B, K, point_width)
        z = self.net(t.reshape(len(xy), -1))
        z = z - z.mean(dim=1, keepdim=True)
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
