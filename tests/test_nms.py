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


# --- calibration against real webcam duplicates -----------------------------
#
# Box pairs logged from a live run that produced two boxes on ONE ear. They
# measured IoU 0.19-0.30 and IoMin 0.39-0.59, i.e. just under both a 0.30 IoU
# and a 0.60 IoMin threshold, which is why the first attempt at this fix never
# fired. These are the ground truth the thresholds are calibrated against.
REAL_DUPLICATE_PAIRS = [
    # (xmin, ymin, xmax, ymax, conf) x2 -- as logged
    ((372, 271, 418, 349, 0.902), (391, 252, 459, 362, 0.705)),
    ((204, 277, 256, 365, 0.791), (166, 274, 229, 361, 0.704)),
    ((167, 284, 224, 367, 0.735), (198, 278, 248, 361, 0.704)),
    ((201, 282, 255, 368, 0.775), (168, 280, 230, 385, 0.703)),
    ((205, 257, 261, 340, 0.820), (177, 264, 239, 372, 0.721)),
    ((196, 259, 257, 345, 0.926), (173, 269, 231, 378, 0.792)),
    ((202, 257, 259, 340, 0.826), (176, 264, 220, 338, 0.737)),  # lowest IoMin, 0.390
    ((198, 259, 252, 341, 0.899), (178, 259, 223, 330, 0.748)),
    ((197, 260, 251, 346, 0.912), (177, 253, 224, 331, 0.702)),
]


@pytest.mark.parametrize("a,b", REAL_DUPLICATE_PAIRS)
def test_real_webcam_duplicates_are_suppressed(a, b):
    """The regression this fix exists for, using the boxes actually observed."""
    da = box(a[0], a[1], a[2], a[3], a[4])
    db = box(b[0], b[1], b[2], b[3], b[4])
    iou, iomin = _iou_iomin(da, db)
    kept = nms(np.array([da, db], dtype=float))
    assert len(kept) == 1, (
        f"duplicate survived: iou={iou:.3f} iomin={iomin:.3f} "
        f"vs thresholds {NMS_IOU_THRESH}/{NMS_IOMIN_THRESH}")


def test_most_real_duplicates_need_the_iomin_term():
    """IoU alone must be shown insufficient, or the new term is not justified.

    The logged boxes are integer-rounded, so recomputing IoU from them lands a
    little off the logged value -- one pair comes out at 0.304 against a logged
    0.299. The claim is therefore about the bulk, not every single pair.
    """
    by_iou = sum(1 for a, b in REAL_DUPLICATE_PAIRS
                 if _iou_iomin(box(*a), box(*b))[0] > NMS_IOU_THRESH)
    assert by_iou <= 1, (
        f"{by_iou}/{len(REAL_DUPLICATE_PAIRS)} pairs are caught by IoU alone; "
        "the IoMin term would not be carrying its weight")


def test_real_duplicates_all_clear_the_iomin_threshold():
    """Document the calibration margin against the observed minimum."""
    iomins = [_iou_iomin(box(*a), box(*b))[1] for a, b in REAL_DUPLICATE_PAIRS]
    assert min(iomins) > NMS_IOMIN_THRESH, (
        f"tightest real duplicate has IoMin {min(iomins):.3f}, "
        f"threshold {NMS_IOMIN_THRESH} would miss it")


def test_iomin_threshold_has_margin_below_real_duplicates():
    """0.35 was chosen to sit under the observed minimum with room to spare,
    while two genuinely different ears measure ~0.0."""
    iomins = [_iou_iomin(box(*a), box(*b))[1] for a, b in REAL_DUPLICATE_PAIRS]
    assert NMS_IOMIN_THRESH < min(iomins) - 0.02
    two_ears = _iou_iomin(box(100, 100, 180, 260, 0.95),
                          box(420, 110, 500, 270, 0.90))[1]
    assert two_ears < NMS_IOMIN_THRESH / 2
