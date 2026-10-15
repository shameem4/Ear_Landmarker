"""The mirrored-frame un-mirror must swap the x edges, not just negate them.

Mirroring sends a column x to W - x, which reverses the order of a box's two x
edges. Assigning them without the swap produces xmin > xmax, which computes a
negative area and then matches nothing while still surviving NMS -- a silent
failure that looks like "frame TTA does not help" rather than like a bug.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference import DETECTOR_FRAME_TTA_DEFAULT, unmirror_boxes  # noqa: E402

W = 200


def test_round_trip_is_identity():
    """Un-mirroring twice must return the original boxes."""
    boxes = np.array([[10.0, 20.0, 50.0, 80.0, 0.9],
                      [0.0, 0.0, 30.0, 15.0, 0.5]])
    np.testing.assert_allclose(unmirror_boxes(unmirror_boxes(boxes, W), W), boxes)


def test_x_edges_are_swapped_not_just_negated():
    out = unmirror_boxes(np.array([[0.0, 20.0, 10.0, 80.0, 1.0]]), W)
    assert out[0, 1] == W - 80.0, "xmin must come from the mirrored xmax"
    assert out[0, 3] == W - 20.0, "xmax must come from the mirrored xmin"


@pytest.mark.parametrize("box", [[5.0, 0.0, 25.0, 40.0, 1.0],
                                 [5.0, 160.0, 25.0, 200.0, 1.0],
                                 [5.0, 95.0, 25.0, 105.0, 1.0]])
def test_boxes_stay_well_formed_and_keep_width(box):
    """Guards the failure the swap exists to prevent: xmin > xmax."""
    out = unmirror_boxes(np.array([box]), W)[0]
    assert out[1] < out[3], f"un-mirrored box is inverted: {out[:4]}"
    assert out[1] >= 0 and out[3] <= W
    np.testing.assert_allclose(out[3] - out[1], box[3] - box[1])


def test_y_and_score_are_untouched():
    boxes = np.array([[7.0, 20.0, 33.0, 80.0, 0.77]])
    out = unmirror_boxes(boxes, W)
    np.testing.assert_allclose(out[:, [0, 2, 4]], boxes[:, [0, 2, 4]])


def test_empty_input_is_handled():
    assert len(unmirror_boxes(np.zeros((0, 5)), W)) == 0


def test_input_is_not_mutated():
    boxes = np.array([[10.0, 20.0, 50.0, 80.0, 0.9]])
    before = boxes.copy()
    unmirror_boxes(boxes, W)
    np.testing.assert_allclose(boxes, before)


def test_frame_tta_is_off_by_default():
    """It costs a second detector pass; the gain is +3.46 pp of detector recall."""
    assert DETECTOR_FRAME_TTA_DEFAULT is False
