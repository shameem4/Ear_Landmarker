"""Head 4: landmarks, depth AND a mesh from one network.

Head 2 predicts shape coefficients from the backbone's final 6x6 map, pooled or
flattened. That is the wrong place to read shape from: an ear's identity is
local -- how deep the concha is, how tightly the helix curls -- and a 6x6 grid
has pooled that away before the coefficients are formed. The evidence that
localised features matter is already in: sampling them AT the landmarks took
the lifter from skill 0.712 to 0.786.

So this samples the decoder's 24x24 features at the predicted landmark
positions and reads the coefficients from those. The landmarks come from head
1's heatmap branch, unchanged, which means:

  THE LANDMARK BRANCH IS NOT TRAINED THROUGH THE SHAPE MODEL. Head 2 pays for
  its mesh with NME 0.03284 against head 1's 0.01064, because its landmarks
  only exist as a projection of the fitted shape and inherit every limitation
  of the basis. Here the heatmap is supervised directly and the shape branch
  hangs off it, so the mesh cannot drag the landmarks down.

  THE COEFFICIENTS MUST STILL BE SUPERVISED DIRECTLY. Shape and pose are not
  uniquely separable, so a loss on projected points constrains only their
  product -- measured, that gives coefficients with 0.017 correlation to the
  truth and a mesh worse than the mean ear. Localised features do not fix a
  gauge freedom; only supervision does.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.ear_landmarker import NUM_LANDMARKS, EarLandmarkerHeatmap
from model.lifter import FeatureLifter


class EarShapeFeatLandmarker(EarLandmarkerHeatmap):
    """Returns (B, K*3) landmarks; `shape_params` gives coefficients and pose."""

    def __init__(self, basis: str | Path, num_landmarks: int = NUM_LANDMARKS,
                 backbone: str = "default", width_mult: float = 1.0,
                 tau: float = 1.0, n_modes: int = 40, point_width: int = 32,
                 width: int = 256, refine_steps: int = 2) -> None:
        super().__init__(num_landmarks=num_landmarks, tau=tau,
                         backbone=backbone, width_mult=width_mult,
                         predict_z=True)
        d = np.load(basis)
        B = d["components"][:n_modes]
        self.n_modes = len(B)
        self.register_buffer("lm_mean", torch.tensor(d["mean"], dtype=torch.float32))
        self.register_buffer("lm_basis", torch.tensor(B, dtype=torch.float32))
        self.n_out = self.n_modes + 6 + 1 + 2
        c = self.stage2_ch
        self.refine_steps = max(1, refine_steps)
        # PER POINT: the image feature there, where the heatmap put the point,
        # and -- from the second pass on -- where the CURRENT TEMPLATE FIT puts
        # it and the residual between the two. Without those last two the
        # network is regressing coefficients blind: it never sees where the
        # template currently sits, so it cannot reason about the fit, only
        # guess it in one shot through a pose composition. Giving it the
        # current estimate turns that into a correction problem, which is what
        # iterative model fitting has always done and why it works.
        self.point = nn.Sequential(nn.Linear(c + 2 + 4, point_width), nn.GELU(),
                                   nn.Linear(point_width, point_width), nn.GELU())
        self.shape = nn.Sequential(
            nn.Linear(num_landmarks * point_width, width), nn.GELU(),
            nn.Linear(width, width), nn.LayerNorm(width), nn.GELU(),
            nn.Linear(width, self.n_out))
        nn.init.zeros_(self.shape[-1].weight)
        with torch.no_grad():
            b = torch.zeros(self.n_out)
            b[self.n_modes:self.n_modes + 6] = torch.tensor([1., 0, 0, 0, 1, 0])
            b[self.n_modes + 6] = float(np.log(0.65))
            b[self.n_modes + 7:] = 0.5
            self.shape[-1].bias.copy_(b)
        # Steps after the first add a CORRECTION, so they must not re-apply
        # the identity pose. Accumulating raw outputs adds this bias once per
        # step: at three steps the translation lands at 1.5 and the fit leaves
        # the crop entirely. Later passes subtract it, which also makes them
        # exactly zero at initialisation, so adding refinement steps cannot
        # change where an untrained model starts.
        self.register_buffer("init_bias", b.clone())

    def decode_shape(self, p: torch.Tensor) -> torch.Tensor:
        """(B, n_out) -> the template posed and deformed, (B, K, 3)."""
        from model.shape_head import rot6d_to_matrix
        c = p[:, :self.n_modes]
        R = rot6d_to_matrix(p[:, self.n_modes:self.n_modes + 6])
        s = p[:, self.n_modes + 6].exp()[:, None, None]
        t = p[:, self.n_modes + 7:][:, None, :]
        L = self.lm_mean + torch.einsum("bm,mkj->bkj", c, self.lm_basis)
        L = torch.einsum("bij,bkj->bki", R, L)
        z = s[:, :, 0] * L[:, :, 2]
        return torch.cat([s * L[:, :, :2] + t,
                          (z - z.mean(1, keepdim=True))[:, :, None]], dim=-1)

    def forward_all(self, x: torch.Tensor):
        """(landmarks (B,K,3), shape parameters (B, n_out))."""
        d = self._decode(x)
        hm, zm = self.to_heatmap(d), self.to_z(d)
        lm = self.soft_argmax_3d(hm, zm, self.tau).view(len(x), -1, 3)
        xy = lm[:, :, :2]
        # sampled where the heatmap put the points, DETACHED so the shape
        # branch cannot drag the landmarks around to make its own job easier
        f = FeatureLifter.sample(d, xy.detach().clamp(0.0, 1.0))
        n, _ = self._norm(xy.detach())
        cur = torch.zeros_like(n)                      # no fit yet on pass 1
        res = torch.zeros_like(n)
        p = None
        for _ in range(self.refine_steps):
            t = self.point(torch.cat([f, n, cur, res], dim=-1))
            dp = self.shape(t.reshape(len(x), -1))
            p = dp if p is None else p + (dp - self.init_bias)
            fit = self.decode_shape(p)
            cur, _ = self._norm(fit[:, :, :2])
            res = n - cur                              # where the fit is wrong
        return lm, p

    @staticmethod
    def _norm(xy):
        c = xy.mean(dim=1, keepdim=True)
        s = (xy.amax(1) - xy.amin(1)).amax(1).clamp_min(1e-6)
        return (xy - c) / s[:, None, None], s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_all(x)[0].reshape(len(x), -1)
