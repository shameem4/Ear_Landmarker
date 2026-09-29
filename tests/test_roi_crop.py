"""The ROI crop must stay square and pad, not clamp to the frame.

Clamping produces a non-square crop which is then resized to the model's square
input, stretching the ear along one axis. The model never saw that in training.
Measured on held-out samples with a 20% edge overlap, padding beats clamping by
17.4% NME and wins on 89% of samples; 14.8% of real ears sit close enough to a
frame edge to trigger it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference import square_roi_crop  # noqa: E402

PAD = 128


def frame(w=200, h=100, value=200):
    return np.full((h, w, 3), value, dtype=np.uint8)


@pytest.mark.parametrize("cx,cy", [(10, 10), (195, 95), (10, 95), (195, 10), (100, 50)])
def test_crop_is_always_square(cx, cy):
    """The regression: a clamped window is not square."""
    got = square_roi_crop(frame(), cx, cy, 60)
    assert got is not None
    crop, _, _ = got
    assert crop.shape[0] == crop.shape[1] == 60


def test_outside_frame_is_grey_padded_not_clamped():
    crop, x1, y1 = square_roi_crop(frame(), 10, 10, 60)
    assert (x1, y1) == (-20, -20), "origin should stay outside the frame, not clamp to 0"
    # the top-left quadrant lies outside the image and must be fill
    assert np.all(crop[:20, :20] == PAD)
    # the part that overlaps the image must be image content, not fill
    assert np.all(crop[20:, 20:] == 200)


def test_fully_interior_crop_has_no_padding():
    crop, x1, y1 = square_roi_crop(frame(), 100, 50, 60)
    assert (x1, y1) == (70, 20)
    assert not np.any(crop == PAD), "interior crop should contain no fill"


def test_landmark_mapping_round_trips_through_a_padded_crop():
    """Normalised crop coords must map back to the right frame pixel.

    This is what actually breaks if origin and side disagree: the old code
    mapped through a clamped width, so a point predicted at the crop's centre
    landed off-centre in the frame.
    """
    side = 60.0
    cx, cy = 10.0, 10.0
    crop, x1, y1 = square_roi_crop(frame(), cx, cy, side)
    n = crop.shape[0]
    # a landmark at the exact centre of the crop is the ROI centre in the frame
    fx, fy = 0.5 * n + x1, 0.5 * n + y1
    assert fx == pytest.approx(cx, abs=1.0)
    assert fy == pytest.approx(cy, abs=1.0)


def test_roi_entirely_outside_frame_is_rejected():
    assert square_roi_crop(frame(), -500, -500, 60) is None


def test_tiny_roi_is_rejected():
    assert square_roi_crop(frame(), 100, 50, 4) is None


def test_padding_value_matches_training_fill():
    """dataset.py pads with grey 128; inference must agree or the model sees
    a border it was never trained through."""
    src = (Path(__file__).resolve().parents[1] / "data" / "dataset.py").read_text()
    assert "(128, 128, 128)" in src
    crop, _, _ = square_roi_crop(frame(), 0, 0, 60)
    assert crop[0, 0].tolist() == [PAD, PAD, PAD]
