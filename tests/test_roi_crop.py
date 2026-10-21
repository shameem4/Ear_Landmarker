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

from inference import square_roi_crop, SUBPIXEL_ROI_CROP  # noqa: E402

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


def _expected_origin(c: float, side: float) -> float:
    """Where the ROI origin should land, for whichever crop path is active.

    The two paths differ by exactly half a pixel, and the sub-pixel one is the
    better-centred of the two. `cx - side/2` places the pixel EDGES
    symmetrically about cx, which leaves the pixel CENTRES off by 0.5; the
    sub-pixel path uses `cx - (n-1)/2`, centring the grid of sampled centres on
    cx. That is also what makes it exactly mirror-symmetric, since
    W-1-x_i = x_{n-1-i} only holds for the centred grid.
    """
    n = int(round(side))
    return c - (n - 1) / 2.0 if SUBPIXEL_ROI_CROP else int(round(c - side / 2))


def test_outside_frame_is_grey_padded_not_clamped():
    crop, x1, y1 = square_roi_crop(frame(), 10, 10, 60)
    assert (x1, y1) == (_expected_origin(10, 60), _expected_origin(10, 60)), \
        "origin should stay outside the frame, not clamp to 0"
    # the top-left quadrant lies outside the image and must be fill
    assert np.all(crop[:19, :19] == PAD)
    # the part that overlaps the image must be image content, not fill
    assert np.all(crop[21:, 21:] == 200)


def test_fully_interior_crop_has_no_padding():
    crop, x1, y1 = square_roi_crop(frame(), 100, 50, 60)
    assert (x1, y1) == (_expected_origin(100, 60), _expected_origin(50, 60))
    assert not np.any(crop == PAD), "interior crop should contain no fill"


def test_subpixel_crop_is_exactly_flip_equivariant():
    """The property the sub-pixel path exists for.

    Cropping a frame and cropping its mirror must give mirrored crops. The
    integer path misses this because `round(cx - s/2)` and
    `round(W - cx - s/2)` agree only when the fractional parts cooperate.
    Skipped unless the flag is on, since the integer path is expected to fail
    it -- that failure is the defect, documented in RESULTS.md (part 2, flip review).
    """
    if not SUBPIXEL_ROI_CROP:
        pytest.skip("integer path is knowingly not flip-equivariant")
    rng = np.random.default_rng(0)
    f = rng.integers(0, 255, (100, 200, 3), dtype=np.uint8)
    W = f.shape[1]
    for cx in (37.3, 80.0, 123.6, 150.5):
        a, _, _ = square_roi_crop(f, cx, 50.0, 60)
        b, _, _ = square_roi_crop(np.ascontiguousarray(f[:, ::-1]),
                                  W - 1 - cx, 50.0, 60)
        d = np.abs(a.astype(int) - b[:, ::-1].astype(int))
        # Equal to within one uint8 level: the sampled POSITIONS mirror
        # exactly (W-1-x_i = x_{n-1-i}), so all that remains is bilinear
        # output quantisation, and it is exactly 0 wherever the samples land
        # on integer columns. The integer path is off by whole pixels here,
        # not by one level.
        assert d.max() <= 1, f"not mirror-symmetric at cx={cx}: max {d.max()}"


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
