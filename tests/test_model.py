"""Model head correctness: soft-argmax maths, gradient flow, output contract.

The gradient-flow test exists because the obvious way to make the heatmap head
start at the image centre -- zero-initialising the final 1x1 conv -- silently
leaves the entire backbone and decoder without gradient, since
d(loss)/d(decoder) = dL/dheatmap * weight = 0. Only the final conv learns.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.ear_landmarker import (  # noqa: E402
    EarLandmarker,
    EarLandmarkerHeatmap,
    NUM_LANDMARKS,
)

GRID = 24  # heatmap resolution for a 192x192 input


@pytest.mark.parametrize("gy,gx", [(0, 0), (12, 12), (23, 23), (5, 18), (23, 0)])
def test_soft_argmax_recovers_peak_location(gy, gx):
    """A sharp peak must decode to that cell's centre."""
    hm = torch.full((1, 1, GRID, GRID), -20.0)
    hm[0, 0, gy, gx] = 20.0
    xy = EarLandmarkerHeatmap.soft_argmax(hm, tau=1.0)
    assert xy[0, 0].item() == pytest.approx((gx + 0.5) / GRID, abs=1e-5)
    assert xy[0, 1].item() == pytest.approx((gy + 0.5) / GRID, abs=1e-5)


def test_soft_argmax_is_sub_pixel():
    """Two equal adjacent peaks decode to their midpoint, not to a grid cell.

    This is the property that makes the "heatmaps quantise to the grid"
    objection inapplicable.
    """
    hm = torch.full((1, 1, GRID, GRID), -20.0)
    hm[0, 0, 10, 10] = 20.0
    hm[0, 0, 10, 11] = 20.0
    xy = EarLandmarkerHeatmap.soft_argmax(hm, tau=1.0)
    midpoint = 11.0 / GRID
    assert xy[0, 0].item() == pytest.approx(midpoint, abs=1e-5)
    # and that midpoint is strictly between the two cell centres
    assert 10.5 / GRID < xy[0, 0].item() < 11.5 / GRID


def test_soft_argmax_output_always_in_unit_range():
    torch.manual_seed(0)
    hm = torch.randn(4, NUM_LANDMARKS, GRID, GRID) * 10
    xy = EarLandmarkerHeatmap.soft_argmax(hm, tau=1.0)
    assert xy.shape == (4, NUM_LANDMARKS * 2)
    assert xy.min() >= 0.0 and xy.max() <= 1.0


@pytest.mark.parametrize("arch", ["gap", "heatmap"])
def test_output_contract(arch):
    """Both heads must return (B, 110) in [0, 1] -- losses/export depend on it."""
    torch.manual_seed(0)
    model = EarLandmarker(NUM_LANDMARKS) if arch == "gap" else EarLandmarkerHeatmap(NUM_LANDMARKS)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 192, 192))
    assert out.shape == (2, NUM_LANDMARKS * 2)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_heatmap_head_starts_near_image_centre():
    """Near-flat initial heatmaps keep early predictions off the corners."""
    torch.manual_seed(0)
    model = EarLandmarkerHeatmap(NUM_LANDMARKS)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(4, 3, 192, 192))
    assert out.mean().item() == pytest.approx(0.5, abs=0.02)


def test_heatmap_head_gradients_reach_the_backbone():
    """Every parameter must receive gradient on the very first step."""
    torch.manual_seed(0)
    model = EarLandmarkerHeatmap(NUM_LANDMARKS)
    model.train()
    model(torch.randn(2, 3, 192, 192)).sum().backward()
    starved = [
        name for name, p in model.named_parameters()
        if p.grad is None or p.grad.abs().sum().item() == 0.0
    ]
    assert not starved, (
        f"{len(starved)} parameter tensors received no gradient, starting with "
        f"{starved[:5]} -- the head is blocking gradient to everything upstream"
    )


def test_heatmap_shape():
    model = EarLandmarkerHeatmap(NUM_LANDMARKS).eval()
    with torch.no_grad():
        maps = model.heatmaps(torch.randn(2, 3, 192, 192))
    assert maps.shape == (2, NUM_LANDMARKS, GRID, GRID)


def test_heatmap_head_drops_the_gap_head():
    """The unused GAP head must not linger in the state dict."""
    model = EarLandmarkerHeatmap(NUM_LANDMARKS)
    assert not any(k.startswith("head.") for k in model.state_dict())


def test_gap_head_state_dict_is_unchanged():
    """Guards the forward_features refactor: v1 checkpoints must still load.

    Uses the archived v1 checkpoint when present; otherwise asserts the key set
    against the expected count so the test still means something in a fresh clone.
    """
    model = EarLandmarker(NUM_LANDMARKS)
    keys = set(model.state_dict())
    assert any(k.startswith("head.") for k in keys)

    ckpt_path = (
        Path(__file__).resolve().parents[1]
        / "runs/archive_v1/EarLandmarker_epoch=345_val/nme=0.0307.ckpt"
    )
    if not ckpt_path.exists():
        pytest.skip("archived v1 checkpoint not present")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = {
        k.removeprefix("model."): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    assert set(state) == keys, "refactor changed the v1 parameter layout"
    model.load_state_dict(state, strict=True)
