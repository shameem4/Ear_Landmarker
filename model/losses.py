"""Loss functions for landmark regression.

Wing loss is the standard for facial/ear landmark regression -- it handles
small errors better than L1/L2 by using a log term near zero, giving
higher gradient for small displacements where precision matters most.

Reference: Feng et al., "Wing Loss for Robust Facial Landmark Localisation
with Convolutional Neural Networks", CVPR 2018.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class WingLoss(nn.Module):
    """Wing loss for landmark regression.

    L(x) = w * ln(1 + |x|/epsilon)   if |x| < w
           |x| - C                     otherwise

    where C = w - w * ln(1 + w/epsilon) makes the loss continuous.

    Args:
        w: Width of the non-linear part. Controls where the transition
           from log to linear happens. Typical: 5-10 (in pixel space)
           or 0.01-0.05 (in normalized [0,1] space).
        epsilon: Curvature of the log region. Smaller = sharper near zero.
    """

    def __init__(self, w: float = 0.04, epsilon: float = 0.01) -> None:
        super().__init__()
        self.w = w
        self.epsilon = epsilon
        self.c = w - w * math.log(1.0 + w / epsilon)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute mean Wing loss.

        Args:
            pred: (B, 110) predicted landmarks.
            target: (B, 110) ground truth landmarks.

        Returns:
            Scalar loss.
        """
        diff = torch.abs(pred - target)
        small = diff < self.w
        loss = torch.where(
            small,
            self.w * torch.log1p(diff / self.epsilon),
            diff - self.c,
        )
        return loss.mean()


class AdaptiveWingLoss(nn.Module):
    """Adaptive Wing loss -- extends Wing loss with per-sample adaptation.

    Better handles the varying difficulty of different landmark points.

    Reference: Wang et al., "Adaptive Wing Loss for Robust Face Alignment
    via Heatmap Regression", ICCV 2019 (adapted for direct regression).

    Args:
        omega: Similar role to 'w' in Wing loss.
        theta: Threshold for switching between regions.
        epsilon: Curvature control.
        alpha: Power term (2.1 in paper).
    """

    def __init__(
        self,
        omega: float = 14.0,
        theta: float = 0.5,
        epsilon: float = 1.0,
        alpha: float = 2.1,
    ) -> None:
        super().__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        # For direct regression (not heatmaps), use fixed alpha exponent.
        # The paper's (alpha - y) term is for heatmap targets where y in {0, 1}.
        self.exp = alpha - 1.0
        theta_eps = theta / epsilon
        self.a = omega * (
            1.0 / (1.0 + theta_eps ** self.exp)
        ) * self.exp * (theta_eps ** (self.exp - 1.0)) / epsilon
        self.c = theta * self.a - omega * math.log(1.0 + theta_eps ** self.exp)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(pred - target)
        small = diff < self.theta
        loss = torch.where(
            small,
            self.omega * torch.log1p((diff / self.epsilon) ** self.exp),
            self.a * diff - self.c,
        )
        return loss.mean()
