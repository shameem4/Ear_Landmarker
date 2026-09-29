"""NMS must collapse duplicate detections on ONE ear without merging two ears.

The live demo drew two boxes on a single ear, which also draws two landmark sets
over each other. Plain IoU cannot catch that case: a small box inside a larger
one scores IoU = areaSmall / areaLarge, which falls below any sane threshold as
soon as the larger box is roughly 3x the smaller. Intersection-over-minimum does
catch it, and stays near zero for two genuinely different ears.

Detections are [ymin, xmin, ymax, xmax, conf] to match EarDetector's output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference import EarLandmarkerPipeline, NMS_IOMIN_THRESH, NMS_IOU_THRESH  # noqa: E402

nms = EarLandmarkerPipeline._nms


def box(xmin, ymin, xmax, ymax, conf):
    """Build one detection row in the detector's [y, x, y, x, conf] order."""
    return [ymin, xmin, ymax, xmax, conf]


def _iou_iomin(a, b):
    ay1, ax1, ay2, ax2 = a[:4]
    by1, bx1, by2, bx2 = b[:4]
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    aa = (ax2 - ax1) * (ay2 - ay1)
    ab = (bx2 - bx1) * (by2 - by1)
    return inter / (aa + ab - inter), inter / min(aa, ab)


def test_nested_duplicate_is_suppressed():
    """The failure mode seen in the demo: low IoU, high containment."""
    big = box(100, 100, 200, 300, 0.92)
    small = box(120, 150, 170, 240, 0.72)
    iou, iomin = _iou_iomin(big, small)
    assert iou < NMS_IOU_THRESH, "test fixture no longer exercises the low-IoU case"
    assert iomin > NMS_IOMIN_THRESH
    assert len(nms(np.array([big, small], dtype=float))) == 1


def test_near_duplicate_is_suppressed():
    kept = nms(np.array([
        box(100, 100, 180, 260, 0.95),
        box(112, 110, 192, 270, 0.88),
    ], dtype=float))
    assert len(kept) == 1
    assert kept[0][4] == pytest.approx(0.95), "kept the lower-scoring duplicate"


def test_two_separate_ears_both_survive():
    """The regression the fix must not cause."""
    kept = nms(np.array([
        box(100, 100, 180, 260, 0.95),
        box(420, 110, 500, 270, 0.90),
    ], dtype=float))
    assert len(kept) == 2


def test_highest_confidence_is_kept():
    kept = nms(np.array([
        box(120, 150, 170, 240, 0.72),
        box(100, 100, 200, 300, 0.92),
    ], dtype=float))
    assert len(kept) == 1
    assert kept[0][4] == pytest.approx(0.92)


def test_single_and_empty_inputs_pass_through():
    one = np.array([box(100, 100, 180, 260, 0.9)], dtype=float)
    assert len(nms(one)) == 1
    assert len(nms(np.zeros((0, 5)))) == 0


def test_adjacent_but_not_overlapping_ears_survive():
    """Boxes that merely touch must not be merged."""
    kept = nms(np.array([
        box(100, 100, 180, 260, 0.95),
        box(180, 100, 260, 260, 0.90),
    ], dtype=float))
    assert len(kept) == 2


def test_iomin_threshold_actually_matters():
    """With io_min disabled the nested pair survives -- proving the fix is load-bearing."""
    pair = np.array([
        box(100, 100, 200, 300, 0.92),
        box(120, 150, 170, 240, 0.72),
    ], dtype=float)
    assert len(nms(pair, io_min_thresh=1.01)) == 2   # effectively off
    assert len(nms(pair)) == 1                        # default on
