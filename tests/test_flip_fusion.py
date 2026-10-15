"""Flip fusion must be EXACTLY equivariant, and must not permute point indices.

Fusion averages the prediction on the crop with the un-mirrored prediction on
the mirrored crop. Two things make that exact rather than approximate:

  * the decoder's x grid is (i + 0.5) / w, so the un-mirror is exactly 1 - x.
    MMPose's shift_heatmap convention would inject a half-pixel error here.
  * data/dataset.py trains the flip as landmarks[:, 0] = 1.0 - landmarks[:, 0]
    with NO index permutation, so point i is the same anatomical point in both
    arms and remapping indices would be wrong.

Equivariance holds for any weights, so these run on a fresh random model and
need no checkpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference import FLIP_FUSION_DEFAULT, LandmarkPredictor  # noqa: E402
from model.ear_landmarker import EarLandmarkerHeatmap  # noqa: E402


def predictor(flip_fusion: bool) -> LandmarkPredictor:
    """A LandmarkPredictor around a fresh random model, with no checkpoint."""
    torch.manual_seed(0)
    lp = object.__new__(LandmarkPredictor)
    lp.device = torch.device("cpu")
    lp.model = EarLandmarkerHeatmap(num_landmarks=55, tau=1.0).eval()
    lp.flip_fusion = flip_fusion
    return lp


def crop(size=240, seed=0):
    return np.random.default_rng(seed).integers(0, 255, (size, size, 3), dtype=np.uint8)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fusion_is_exactly_flip_equivariant(seed):
    """pred(crop) must equal mirror(pred(mirror(crop))) to float precision."""
    lp = predictor(flip_fusion=True)
    c = crop(seed=seed)
    a = lp.predict(c)
    b = lp.predict(c[:, ::-1].copy())
    b[:, 0] = c.shape[1] - b[:, 0]
    assert np.abs(a - b).max() < 1e-3, (
        "fusion is not exactly equivariant; check the 1-x un-mirror against "
        f"the decoder grid. max deviation {np.abs(a - b).max():.6f} px")


def test_single_arm_is_not_equivariant():
    """Guards the test above from passing for the wrong reason.

    If the model were flip-symmetric by construction, the equivariance test
    would pass with fusion disabled too and would be proving nothing.
    """
    lp = predictor(flip_fusion=False)
    c = crop()
    a = lp.predict(c)
    b = lp.predict(c[:, ::-1].copy())
    b[:, 0] = c.shape[1] - b[:, 0]
    assert np.abs(a - b).max() > 1e-2, (
        "single-arm prediction is already mirror-symmetric, so the fusion "
        "equivariance test cannot distinguish a correct un-mirror from a no-op")


def test_fusion_averages_rather_than_replaces():
    """The fused point must lie between the two arms, not on one of them."""
    c = crop(seed=3)
    fused = predictor(flip_fusion=True).predict(c)
    lp = predictor(flip_fusion=False)
    arm_a = lp.predict(c)
    arm_b = lp.predict(c[:, ::-1].copy())
    arm_b[:, 0] = c.shape[1] - arm_b[:, 0]
    np.testing.assert_allclose(fused, (arm_a + arm_b) / 2, atol=1e-3)


def test_flip_fusion_is_on_by_default():
    """A regression guard: the measured gain is -1.83% nme_normal (p=0.009)."""
    assert FLIP_FUSION_DEFAULT is True


def test_explicit_argument_overrides_the_instance_default():
    c = crop(seed=4)
    on = predictor(flip_fusion=True)
    assert np.abs(on.predict(c) - on.predict(c, flip_fusion=False)).max() > 1e-2
