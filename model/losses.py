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

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        visible: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute mean Wing loss.

        Args:
            pred: (B, 110) predicted landmarks.
            target: (B, 110) ground truth landmarks.
            visible: (B, 55) 1.0 for points inside the frame, 0.0 for points
                pushed outside by augmentation. None means all visible.

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
        if visible is None:
            return loss.mean()

        # (B, 55) -> (B, 55, 2) -> (B, 110) so both coords of a point share its mask
        mask = visible.unsqueeze(-1).expand(-1, -1, 2).reshape(loss.shape)
        denom = mask.sum().clamp_min(1.0)
        return (loss * mask).sum() / denom


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

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        visible: torch.Tensor | None = None,
    ) -> torch.Tensor:
        diff = torch.abs(pred - target)
        small = diff < self.theta
        loss = torch.where(
            small,
            self.omega * torch.log1p((diff / self.epsilon) ** self.exp),
            self.a * diff - self.c,
        )
        if visible is None:
            return loss.mean()

        mask = visible.unsqueeze(-1).expand(-1, -1, 2).reshape(loss.shape)
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)


# (name, start, end) -- end exclusive, matching the 55-point linestrip layout
LINESTRIP_RANGES = [(0, 20), (20, 35), (35, 50), (50, 55)]


def _contour_frames(target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit tangent and normal at each ground-truth landmark.

    The tangent at point i is the direction of the chord between its neighbours
    within the same linestrip (one-sided at the strip ends), which is the standard
    finite-difference estimate for a polyline.

    Args:
        target: (B, K, 2) ground-truth landmarks.

    Returns:
        ((B, K, 2) unit tangent, (B, K, 2) unit normal).
    """
    b, k, _ = target.shape
    prev_idx = torch.arange(k, device=target.device)
    next_idx = torch.arange(k, device=target.device)
    for a, e in LINESTRIP_RANGES:
        seg = torch.arange(a, e, device=target.device)
        prev_idx[a:e] = torch.clamp(seg - 1, min=a)
        next_idx[a:e] = torch.clamp(seg + 1, max=e - 1)

    tangent = target[:, next_idx, :] - target[:, prev_idx, :]
    norm = tangent.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    tangent = tangent / norm
    # Rotate 90 degrees for the normal.
    normal = torch.stack((-tangent[..., 1], tangent[..., 0]), dim=-1)
    return tangent, normal


class TangentialWeightedWingLoss(nn.Module):
    """Wing loss on residuals decomposed against the ground-truth contour.

    Motivation: 72% of this model's squared error is TANGENTIAL -- points sitting
    at the wrong place along a contour rather than off it -- and that component
    is largely irreducible annotation noise. The source datasets space their
    points differently along the same anatomy (helix spacing CV 0.137 to 0.286),
    and the model's per-source tangential error tracks that irregularity
    monotonically. Plain Wing loss penalises this noise as hard as genuine
    localisation error, so the model spends capacity fitting it.

    Splitting the residual lets the normal component -- the genuine "am I on the
    ear" error -- carry full weight while the tangential component is discounted.

    Args:
        w, epsilon: Wing loss parameters, as WingLoss.
        tangential_weight: Multiplier on the tangential component. 1.0 reproduces
            isotropic Wing loss; 0.0 ignores position along the contour entirely.
        spacing_weight: Strength of the anti-collapse term. With the tangential
            constraint relaxed, nothing otherwise stops predicted points bunching
            together along the contour -- which would look good on shape metrics
            while destroying the point distribution. This penalises predicted
            segments that shrink below half their ground-truth length.
    """

    def __init__(
        self,
        w: float = 0.04,
        epsilon: float = 0.01,
        tangential_weight: float = 0.3,
        spacing_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.w = w
        self.epsilon = epsilon
        self.c = w - w * math.log(1.0 + w / epsilon)
        self.tangential_weight = float(tangential_weight)
        self.spacing_weight = float(spacing_weight)

    def _wing(self, d: torch.Tensor) -> torch.Tensor:
        return torch.where(
            d < self.w,
            self.w * torch.log1p(d / self.epsilon),
            d - self.c,
        )

    def _spacing_penalty(self, pred2d: torch.Tensor, tgt2d: torch.Tensor) -> torch.Tensor:
        """Penalise predicted segments collapsing relative to ground truth."""
        total = pred2d.new_zeros(())
        n = 0
        for a, e in LINESTRIP_RANGES:
            if e - a < 2:
                continue
            p = (pred2d[:, a + 1:e, :] - pred2d[:, a:e - 1, :]).norm(dim=-1)
            t = (tgt2d[:, a + 1:e, :] - tgt2d[:, a:e - 1, :]).norm(dim=-1)
            total = total + torch.relu(0.5 * t - p).mean()
            n += 1
        return total / max(n, 1)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        visible: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b = pred.shape[0]
        pred2d = pred.view(b, -1, 2)
        tgt2d = target.view(b, -1, 2)

        tangent, normal = _contour_frames(tgt2d)
        residual = pred2d - tgt2d
        d_tan = (residual * tangent).sum(-1).abs()      # (B, K)
        d_nrm = (residual * normal).sum(-1).abs()       # (B, K)

        per_point = self._wing(d_nrm) + self.tangential_weight * self._wing(d_tan)

        if visible is None:
            loss = per_point.mean()
        else:
            loss = (per_point * visible).sum() / visible.sum().clamp_min(1.0)

        if self.spacing_weight > 0:
            loss = loss + self.spacing_weight * self._spacing_penalty(pred2d, tgt2d)
        return loss
