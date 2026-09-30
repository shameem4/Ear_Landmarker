"""Adaptive ROI refinement: re-derive the crop from the landmarks.

ROI_EXPAND alone cannot frame the ear, because the detector box is not a fixed
fraction of it. Measured on real captures, true ear extent / detector box extent
runs 1.02 to 1.53. At a fixed 1.3x a tight box produces a crop SMALLER than the
ear, so the landmarks jam against the crop border and the predicted ear came out
19% too small on the worst capture.

Refining once from the landmarks brings that to 0.6%, and leaves already-correct
framing untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference import (  # noqa: E402
    ROI_EXPAND_MAX,
    ROI_EXPAND_MIN,
    ROI_MAX_REFINE,
    ROI_OCC_TOL,
    ROI_SATURATED,
    TRAIN_OCCUPANCY,
    refine_roi_side,
)


def occupancy(side, extent):
    return extent / side


def test_correctly_framed_roi_is_left_alone():
    """The ear already fills the training fraction -- do not move the goalposts."""
    side = 100.0
    extent = side * TRAIN_OCCUPANCY
    assert refine_roi_side(side, extent) == pytest.approx(side)


def test_too_tight_roi_grows():
    """The failure this exists for: crop smaller than the ear."""
    side = 87.2          # 1.3x a tight detector box
    extent = 82.8        # measured, clipped by the crop
    assert occupancy(side, extent) > ROI_SATURATED
    new = refine_roi_side(side, extent)
    assert new > side
    # the converged answer for this real capture was ~132
    assert new == pytest.approx(132.0, rel=0.05)


def test_too_wide_roi_shrinks():
    side = 200.0
    extent = 80.0        # ear only fills 0.4 of the crop
    new = refine_roi_side(side, extent)
    assert new < side
    assert occupancy(new, extent) == pytest.approx(TRAIN_OCCUPANCY)


def test_unsaturated_update_lands_exactly_on_target():
    side, extent = 150.0, 100.0
    assert occupancy(side, extent) < ROI_SATURATED
    assert occupancy(refine_roi_side(side, extent), extent) == pytest.approx(TRAIN_OCCUPANCY)


def test_saturated_update_overshoots_deliberately():
    """When clipped, the measured extent under-reads the true ear, so the plain
    update converges slowly. The boost is what makes ONE pass enough."""
    side, extent = 100.0, 95.0
    assert occupancy(side, extent) > ROI_SATURATED
    boosted = refine_roi_side(side, extent)
    plain = extent / TRAIN_OCCUPANCY
    assert boosted > plain


def test_iteration_converges_to_the_true_ear():
    """Simulated model: reports the true ear, clipped to what the crop shows.

    Truth is 102 with a detector box giving an initial side of 87 -- the real
    49-22 capture. One refinement must land within a few percent.
    """
    TRUE = 102.0

    def observe(side):
        return min(TRUE, side * 0.95)      # clipped when the crop is too small

    side = 87.2
    side = refine_roi_side(side, observe(side))
    assert observe(side) == pytest.approx(TRUE, rel=0.05), "one pass should nearly converge"

    side = refine_roi_side(side, observe(side))
    assert occupancy(side, TRUE) == pytest.approx(TRAIN_OCCUPANCY, abs=ROI_OCC_TOL)


def test_zero_side_does_not_divide_by_zero():
    assert refine_roi_side(0.0, 50.0) == pytest.approx(50.0 / TRAIN_OCCUPANCY)


def test_js_and_python_constants_agree():
    """The browser pipeline must refine identically or the demo drifts from
    the reference implementation."""
    js = (Path(__file__).resolve().parents[1] / "docs" / "earlandmarker_inference.js").read_text()
    for name, value in (("TRAIN_OCCUPANCY", TRAIN_OCCUPANCY), ("ROI_OCC_TOL", ROI_OCC_TOL),
                        ("ROI_SATURATED", ROI_SATURATED)):
        assert f"const {name} = {value}" in js, f"{name} missing or different in the JS port"


# --- runaway protection -------------------------------------------------------
#
# The refinement caches its result per track and seeds the next frame from it.
# That is a positive feedback loop whenever the landmarker reports a saturated
# extent -- which is what a confused model does on motion blur, an occluded ear,
# or a false-positive box. Unclamped it measured ~1.53x growth per frame,
# reaching 38x the detector box within eight frames and never recovering, since
# each enlargement makes the ear smaller in the crop.



def _frame(side_seed, observe, box):
    """One frame of the real loop, including the clamp."""
    side = side_seed
    for attempt in range(ROI_MAX_REFINE + 1):
        ext = observe(side)
        if attempt == ROI_MAX_REFINE:
            break
        if abs(ext / side - TRAIN_OCCUPANCY) <= ROI_OCC_TOL:
            break
        side = min(max(refine_roi_side(side, ext), box * ROI_EXPAND_MIN),
                   box * ROI_EXPAND_MAX)
    return side


def test_permanently_saturated_model_does_not_run_away():
    """The regression: eight frames used to reach 38x the detector box."""
    box = 67.0
    side = box * 1.3
    for _ in range(30):
        side = _frame(side, lambda s: 0.95 * s, box)
    assert side <= box * ROI_EXPAND_MAX * 1.001, (
        f"expansion ran away to {side / box:.1f}x the detector box")


def test_healthy_case_still_converges_and_is_not_clipped_by_the_clamp():
    """The clamp must not interfere with the real capture it was built for:
    a 102px ear behind a 67px box needs ~1.99x, inside the bound."""
    box, true_ear = 67.0, 102.0
    side = box * 1.3
    for _ in range(6):
        side = _frame(side, lambda s: min(true_ear, 0.95 * s), box)
    assert side / box == pytest.approx(1.99, abs=0.15)
    assert side < box * ROI_EXPAND_MAX, "the real case must not need the clamp"
    assert true_ear / side == pytest.approx(TRAIN_OCCUPANCY, abs=ROI_OCC_TOL)


def test_clamp_bounds_are_ordered_and_cover_the_observed_range():
    """Observed ear/box ratios span 1.02-1.53, needing 1.31-1.97 expansion."""
    assert ROI_EXPAND_MIN < ROI_EXPAND_MAX
    assert ROI_EXPAND_MIN <= 1.31 and ROI_EXPAND_MAX >= 1.97


def test_js_clamp_matches_python():
    js = (Path(__file__).resolve().parents[1] / "docs" / "earlandmarker_inference.js").read_text()
    assert f"const ROI_EXPAND_MIN = {ROI_EXPAND_MIN}" in js
    assert f"const ROI_EXPAND_MAX = {ROI_EXPAND_MAX}" in js
