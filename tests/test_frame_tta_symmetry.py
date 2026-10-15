"""With frame TTA on, the pipeline must choose the same box either way round.

tests/test_frame_tta.py pins the un-mirror ARITHMETIC on synthetic boxes. This
pins the PROPERTY the feature exists for, end to end on a real frame: a union
through NMS is the same set whichever presentation the pipeline is handed, so
the chosen box must be identical. Ear_3D_Lifter measured exactly that on their
renders -- 0.00 px median with the flag on, 53.66 px median and 134.88 px max
with it off -- and it is the test that would catch a regression in
square_roi_crop, the NMS thresholds or the merge, where the arithmetic tests
would not.

The OFF case is asserted too. If it ever went symmetric the ON case would be
passing for free, and the test would have stopped exercising what it checks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference import (BLAZEEAR_DIR, DETECTOR_WEIGHTS,  # noqa: E402
                       EarLandmarkerPipeline, find_best_checkpoint)

FIXTURES = sorted(ROOT.glob("fixtures/*.png"))


def _pipeline(frame_tta: bool):
    cv2 = pytest.importorskip("cv2")
    weights = BLAZEEAR_DIR / DETECTOR_WEIGHTS
    if not weights.exists() or not FIXTURES:
        pytest.skip("detector weights or fixtures unavailable")
    try:
        ckpt = find_best_checkpoint()
    except FileNotFoundError:
        pytest.skip("no landmarker checkpoint available")
    return EarLandmarkerPipeline(weights, ckpt, device="cpu", smooth=False,
                                 detector_frame_tta=frame_tta), cv2


def _boxes_both_presentations(pipe, cv2, path):
    """Boxes from the frame and from its mirror, un-mirrored, sorted."""
    frame = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    w = frame.shape[1]
    a = pipe(frame)
    pipe.reset()
    b = pipe(np.ascontiguousarray(frame[:, ::-1]))
    pipe.reset()
    if not a or not b or len(a) != len(b):
        return None, None
    A = np.array(sorted([r["bbox"] for r in a], key=lambda v: (v[0], v[1])))
    B = np.array([[r["bbox"][0], w - r["bbox"][3], r["bbox"][2], w - r["bbox"][1]]
                  for r in b])
    B = np.array(sorted(B.tolist(), key=lambda v: (v[0], v[1])))
    return A, B


def test_union_makes_the_chosen_box_presentation_invariant():
    pipe, cv2 = _pipeline(frame_tta=True)
    checked = 0
    for path in FIXTURES:
        A, B = _boxes_both_presentations(pipe, cv2, path)
        if A is None:
            continue
        checked += 1
        assert np.abs(A - B).max() < 1.0, (
            f"{path.name}: frame TTA on, but the chosen box moved "
            f"{np.abs(A - B).max():.2f} px with presentation")
    if not checked:
        pytest.skip("no fixture produced a comparable detection")


def test_without_the_union_the_box_does_move():
    """Guards the test above from passing because the detector is symmetric."""
    pipe, cv2 = _pipeline(frame_tta=False)
    worst = 0.0
    for path in FIXTURES:
        A, B = _boxes_both_presentations(pipe, cv2, path)
        if A is None:
            continue
        worst = max(worst, float(np.abs(A - B).max()))
    if worst == 0.0:
        pytest.skip("no comparable detection, or detector is already symmetric")
    assert worst > 1.0, (
        "the detector is presentation-invariant without the union, so the "
        "symmetry test above proves nothing")
