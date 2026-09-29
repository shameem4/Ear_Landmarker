"""TangentialWeightedWingLoss: the decomposition must be correct, and the
anti-collapse guard must actually fire.

The decomposition tests matter because a sign or axis error would silently
down-weight the NORMAL component instead of the tangential one -- which would
optimise exactly the wrong thing while still training to a plausible-looking
loss curve.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.losses import (  # noqa: E402
    LINESTRIP_RANGES,
    TangentialWeightedWingLoss,
    WingLoss,
    _contour_frames,
)

K = 55


def _straight_line_target() -> torch.Tensor:
    """All strips lying along the x axis, so tangent is +x and normal is +/-y."""
    t = torch.zeros(1, K, 2)
    for a, e in LINESTRIP_RANGES:
        t[0, a:e, 0] = torch.linspace(0.1, 0.9, e - a)
        t[0, a:e, 1] = 0.5
    return t


def test_contour_frames_are_unit_and_orthogonal():
    torch.manual_seed(0)
    tgt = torch.rand(4, K, 2)
    tan, nrm = _contour_frames(tgt)
    assert torch.allclose(tan.norm(dim=-1), torch.ones(4, K), atol=1e-5)
    assert torch.allclose(nrm.norm(dim=-1), torch.ones(4, K), atol=1e-5)
    assert torch.allclose((tan * nrm).sum(-1), torch.zeros(4, K), atol=1e-5)


def test_contour_frames_on_a_known_line():
    tgt = _straight_line_target()
    tan, nrm = _contour_frames(tgt)
    assert torch.allclose(tan[0, :, 0].abs(), torch.ones(K), atol=1e-5)
    assert torch.allclose(tan[0, :, 1], torch.zeros(K), atol=1e-5)
    assert torch.allclose(nrm[0, :, 0], torch.zeros(K), atol=1e-5)


def test_tangential_displacement_is_discounted_but_normal_is_not():
    """The core claim: sliding along the contour must cost less than leaving it."""
    tgt = _straight_line_target()
    f = TangentialWeightedWingLoss(tangential_weight=0.3, spacing_weight=0.0)

    along = tgt.clone()
    along[0, :, 0] += 0.02          # slide along the line (tangential)
    off = tgt.clone()
    off[0, :, 1] += 0.02            # step off the line (normal)

    loss_along = f(along.view(1, -1), tgt.view(1, -1))
    loss_off = f(off.view(1, -1), tgt.view(1, -1))
    assert loss_along < loss_off, "tangential displacement was not discounted"
    assert loss_along == pytest.approx(0.3 * loss_off, rel=1e-4), (
        "discount does not match the configured weight -- axes may be swapped"
    )


def test_weight_one_matches_isotropic_wing_loss():
    """With weight 1 the decomposition is a rotation, so the total must match
    plain Wing loss applied to the same residual magnitudes."""
    torch.manual_seed(1)
    tgt = torch.rand(2, K, 2)
    pred = tgt + torch.randn(2, K, 2) * 0.01

    aniso = TangentialWeightedWingLoss(tangential_weight=1.0, spacing_weight=0.0)
    got = aniso(pred.view(2, -1), tgt.view(2, -1))

    # Reference: Wing applied to the two projected components directly.
    tan, nrm = _contour_frames(tgt)
    r = pred - tgt
    d_t = (r * tan).sum(-1).abs()
    d_n = (r * nrm).sum(-1).abs()
    w = WingLoss()
    expect = (w._wing_ref(d_n) + w._wing_ref(d_t)).mean() if hasattr(w, "_wing_ref") else None
    if expect is not None:
        assert got == pytest.approx(float(expect), rel=1e-5)
    else:
        # WingLoss has no exposed elementwise helper; assert via the class itself.
        ref = aniso._wing(d_n) + aniso._wing(d_t)
        assert got == pytest.approx(float(ref.mean()), rel=1e-5)


def test_zero_weight_ignores_pure_tangential_error():
    tgt = _straight_line_target()
    f = TangentialWeightedWingLoss(tangential_weight=0.0, spacing_weight=0.0)
    slid = tgt.clone()
    slid[0, :, 0] += 0.03
    assert f(slid.view(1, -1), tgt.view(1, -1)).item() == pytest.approx(0.0, abs=1e-6)


def test_perfect_prediction_is_zero_loss():
    tgt = _straight_line_target()
    f = TangentialWeightedWingLoss()
    assert f(tgt.view(1, -1), tgt.view(1, -1)).item() == pytest.approx(0.0, abs=1e-6)


def test_anti_collapse_term_fires_on_bunching():
    """The guard against the known failure mode of relaxing tangential weight."""
    tgt = _straight_line_target()
    collapsed = tgt.clone()
    for a, e in LINESTRIP_RANGES:                       # squash each strip inward
        mid = tgt[0, a:e, 0].mean()
        collapsed[0, a:e, 0] = mid + (tgt[0, a:e, 0] - mid) * 0.1

    with_guard = TangentialWeightedWingLoss(tangential_weight=0.0, spacing_weight=0.5)
    without = TangentialWeightedWingLoss(tangential_weight=0.0, spacing_weight=0.0)

    assert without(collapsed.view(1, -1), tgt.view(1, -1)).item() == pytest.approx(0.0, abs=1e-5), (
        "collapse is invisible without the guard -- which is exactly why it exists"
    )
    assert with_guard(collapsed.view(1, -1), tgt.view(1, -1)).item() > 0.01, (
        "anti-collapse term did not fire on a collapsed prediction"
    )


def test_visibility_mask_excludes_points():
    torch.manual_seed(2)
    tgt = torch.rand(2, K, 2)
    pred = tgt.clone()
    pred[:, 0, :] += 0.5                                # one very wrong point
    vis = torch.ones(2, K)
    vis[:, 0] = 0.0
    f = TangentialWeightedWingLoss(spacing_weight=0.0)
    assert f(pred.view(2, -1), tgt.view(2, -1), vis).item() == pytest.approx(0.0, abs=1e-6)


def test_gradients_flow():
    tgt = _straight_line_target()
    pred = (tgt + 0.01).view(1, -1).clone().requires_grad_(True)
    TangentialWeightedWingLoss()(pred, tgt.view(1, -1)).backward()
    assert pred.grad is not None and pred.grad.abs().sum() > 0
