"""Head 2: predict ear SHAPE and POSE, and let the landmarks follow.

Instead of 165 free numbers, the network predicts ~40 shape coefficients of the
morphable model plus a pose, and the 55 landmarks are computed from them. Two
consequences worth being explicit about.

What it buys: the output is always a real ear. A free head can place point 29
on the far side of point 30, or invent a depth profile no ear has; this cannot,
because every output is the mean ear plus a combination of modes measured from
476 real ones. Depth in particular is no longer a separate quantity that has to
be learned -- it is whatever the shape and pose imply, so it comes out
consistent with x and y for free.

What it costs: accuracy is capped by what the basis can express. If 40 modes
cannot reproduce a particular ear, no amount of training will fit it, and the
error floor that imposes is measurable rather than hypothetical -- fit the
basis to the ground-truth landmarks directly and whatever residual remains is
the ceiling for this head. scripts/eval_depth.py reports it.

The coefficients are in standard deviations (see build_landmark_basis), so a
unit gaussian is the right prior and `coeff_l2` is a sensible regulariser
rather than an arbitrary one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.ear_landmarker import NUM_LANDMARKS, EarLandmarker


def rot6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """(B, 6) -> (B, 3, 3). Gram-Schmidt on two vectors.

    A rotation parameterised by Euler angles or a quaternion has
    discontinuities a network has to fight; this one is continuous everywhere,
    which is the whole reason for the representation.
    """
    a1, a2 = x[:, :3], x[:, 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


class EarShapeLandmarker(EarLandmarker):
    """(B, K*3) landmarks, via shape coefficients and a weak-perspective pose."""

    def __init__(self, basis: str | Path, num_landmarks: int = NUM_LANDMARKS,
                 backbone: str = "default", width_mult: float = 1.0,
                 n_modes: int = 0, init_scale: float = 0.65,
                 head: str = "spatial") -> None:
        super().__init__(num_landmarks=num_landmarks, backbone=backbone,
                         width_mult=width_mult)
        del self.head
        d = np.load(basis)
        B = d["components"]
        if n_modes:
            B = B[:n_modes]
        if B.shape[1] != num_landmarks:
            raise ValueError(f"basis has {B.shape[1]} landmarks, model expects "
                             f"{num_landmarks}")
        self.n_modes = len(B)
        self.register_buffer("lm_mean", torch.tensor(d["mean"], dtype=torch.float32))
        self.register_buffer("lm_basis", torch.tensor(B, dtype=torch.float32))

        self.n_out = self.n_modes + 6 + 1 + 2        # shape, rot6d, log s, t
        # A GLOBAL AVERAGE POOL CANNOT SEE WHERE ANYTHING IS, and pose is
        # entirely about where things are -- the translation especially is a
        # position, which pooling destroys by construction. The first version
        # of this head was GAP followed by one Linear, and it underfitted:
        # its TRAINING error, 0.01685, was worse than head 1's validation
        # error, and a learning-rate sweep moved it by 0.00001. Keeping the
        # 6x6 grid is the obvious thing the architecture was missing.
        self.head_kind = head
        if head == "spatial":
            self.pred = nn.Sequential(
                nn.Conv2d(self.stage4_ch, 64, 1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Flatten(), nn.Linear(64 * 6 * 6, 256), nn.ReLU(inplace=True),
                nn.Linear(256, self.n_out))
        else:
            self.pred = nn.Sequential(nn.Flatten(),
                                      nn.Linear(self.stage4_ch, self.n_out))
        last = self.pred[-1]
        nn.init.zeros_(last.weight)
        with torch.no_grad():
            b = torch.zeros(self.n_out)
            b[self.n_modes:self.n_modes + 6] = torch.tensor([1., 0., 0., 0., 1., 0.])
            b[self.n_modes + 6] = float(np.log(init_scale))
            b[self.n_modes + 7:] = 0.5               # centre of the crop
            last.bias.copy_(b)
        # Zero weights so the first prediction is exactly the mean ear at the
        # crop centre. Unlike a heatmap layer this starves nothing: the pose and
        # coefficient gradients w.r.t. the weights are nonzero from step one,
        # because the bias already puts the output somewhere the loss can push.

    def decode(self, p: torch.Tensor) -> torch.Tensor:
        """(B, n_out) parameters -> (B, K, 3) landmarks in crop units."""
        c = p[:, :self.n_modes]
        R = rot6d_to_matrix(p[:, self.n_modes:self.n_modes + 6])
        s = p[:, self.n_modes + 6].exp()[:, None, None]
        t = p[:, self.n_modes + 7:][:, None, :]
        L = self.lm_mean + torch.einsum("bm,mkj->bkj", c, self.lm_basis)
        L = torch.einsum("bij,bkj->bki", R, L)
        xy = s * L[:, :, :2] + t
        # z is ROOT-RELATIVE, matching the label: a crop fixes no absolute
        # depth, so only the profile about the mean is predictable
        z = s[:, :, 0] * L[:, :, 2]
        z = z - z.mean(dim=1, keepdim=True)
        return torch.cat([xy, z[:, :, None]], dim=-1)

    def parameters_from(self, x: torch.Tensor) -> torch.Tensor:
        _, _, s4 = self.forward_features(x)
        if self.head_kind == "spatial":
            return self.pred(s4)                       # keeps the 6x6 layout
        return self.pred(F.adaptive_avg_pool2d(s4, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.parameters_from(x)
        return self.decode(p).reshape(len(x), -1)
