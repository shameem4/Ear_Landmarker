"""Smoothing correctness: it must reduce noise without introducing drift,
and the tracker must never cross filter state between two ears.

The track-association tests matter most. If two ears swap filter identities
between frames, smoothing actively corrupts the output -- each ear gets pulled
toward the other's history -- and that failure looks like "the model is bad"
rather than "the tracker is wrong".
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.smoothing import (  # noqa: E402
    BoxSmoother,
    EarTracker,
    LandmarkSmoother,
    OneEuroFilter,
)

FPS = 30.0


def test_first_observation_passes_through():
    f = OneEuroFilter()
    x = np.array([1.0, 2.0])
    assert np.allclose(f(x, 0.0), x)


def test_constant_signal_converges_without_drift():
    """A steady input must settle exactly on it, not creep past or short of it."""
    f = OneEuroFilter(min_cutoff=1.0, beta=0.05)
    target = np.array([0.4, 0.6])
    out = None
    for i in range(200):
        out = f(target, i / FPS)
    assert np.allclose(out, target, atol=1e-6)


def test_noise_is_reduced():
    """The whole point: output must vary less than a noisy constant input."""
    rng = np.random.default_rng(0)
    truth = np.array([0.5, 0.5])
    f = OneEuroFilter(min_cutoff=1.0, beta=0.05)
    noisy, smoothed = [], []
    for i in range(300):
        obs = truth + rng.normal(0, 0.01, 2)
        noisy.append(obs)
        smoothed.append(f(obs, i / FPS))
    n_std = np.std(np.stack(noisy[50:]), axis=0).mean()
    s_std = np.std(np.stack(smoothed[50:]), axis=0).mean()
    assert s_std < n_std * 0.5, f"smoothing barely helped: {s_std:.5f} vs {n_std:.5f}"


def test_tracks_a_ramp_with_bounded_lag():
    """Smoothing must not lag without bound on sustained motion."""
    f = OneEuroFilter(min_cutoff=3.0, beta=0.4)
    speed = 0.2                       # units per second
    last_err = None
    for i in range(200):
        t = i / FPS
        x = np.array([speed * t, 0.0])
        out = f(x, t)
        last_err = abs(out[0] - x[0])
    assert last_err < 0.02, f"steady-state lag {last_err:.4f} is too large"


def test_zero_and_negative_dt_are_safe():
    """Duplicate or out-of-order frames must not blow up or move the estimate."""
    f = OneEuroFilter()
    f(np.array([0.5, 0.5]), 1.0)
    out_dup = f(np.array([0.9, 0.9]), 1.0)     # same timestamp
    out_back = f(np.array([0.9, 0.9]), 0.5)    # earlier timestamp
    assert np.all(np.isfinite(out_dup)) and np.all(np.isfinite(out_back))
    assert np.allclose(out_dup, out_back)


def test_confidence_smooths_uncertain_points_harder():
    """Low-confidence points must move less than high-confidence ones."""
    rng = np.random.default_rng(1)
    s = LandmarkSmoother(min_cutoff=3.0, beta=0.0, conf_strength=1.0, conf_floor=0.1)
    conf = np.array([1.0, 0.0])               # point 0 certain, point 1 not
    lm0 = np.array([[0.5, 0.5], [0.5, 0.5]])
    s(lm0, 0.0, conf)
    target = np.array([[0.7, 0.5], [0.7, 0.5]])
    out = None
    for i in range(1, 6):
        out = s(target, i / FPS, conf)
    moved_confident = abs(out[0, 0] - 0.5)
    moved_uncertain = abs(out[1, 0] - 0.5)
    assert moved_confident > moved_uncertain, (
        "confidence weighting had no effect or the wrong sign"
    )


def test_box_smoother_preserves_shape_under_noise():
    """Centre/size parameterisation must not let the box distort."""
    rng = np.random.default_rng(2)
    b = BoxSmoother()
    true_box = np.array([100.0, 200.0, 180.0, 280.0])   # 80x80
    out = None
    for i in range(120):
        noisy = true_box + rng.normal(0, 2.0, 4)
        out = b(noisy, i / FPS)
    h, w = out[2] - out[0], out[3] - out[1]
    assert abs(h - 80.0) < 4.0 and abs(w - 80.0) < 4.0
    assert abs(h - w) < 3.0, "box drifted away from square"


def test_tracker_keeps_two_ears_separate():
    """Two well-separated ears must keep distinct, non-swapping track ids."""
    tr = EarTracker()
    left = np.array([100.0, 100.0, 180.0, 180.0])
    right = np.array([100.0, 400.0, 180.0, 480.0])
    ids0 = tr.assign([left, right])
    for i in range(1, 20):
        t = i / FPS
        for tid, box in zip(ids0, (left, right)):
            tr.smooth_box(tid, box, t)
        ids = tr.assign([left + 1.0, right + 1.0])
        assert ids == ids0, f"track ids changed at frame {i}: {ids} != {ids0}"
    assert len(set(ids0)) == 2


def test_tracker_does_not_swap_when_order_changes():
    """Detection order is not identity -- swapping the list must not swap tracks."""
    tr = EarTracker()
    left = np.array([100.0, 100.0, 180.0, 180.0])
    right = np.array([100.0, 400.0, 180.0, 480.0])
    ids = tr.assign([left, right])
    mapping = {"left": ids[0], "right": ids[1]}
    for tid, box in zip(ids, (left, right)):
        tr.smooth_box(tid, box, 1 / FPS)
    swapped = tr.assign([right, left])          # same ears, reversed order
    assert swapped[0] == mapping["right"] and swapped[1] == mapping["left"]


def test_tracker_expires_and_reuses_nothing_stale():
    tr = EarTracker(max_missed=2)
    box = np.array([100.0, 100.0, 180.0, 180.0])
    first = tr.assign([box])[0]
    tr.smooth_box(first, box, 0.0)
    for _ in range(4):
        tr.assign([])                            # no detections
    later = tr.assign([box])[0]
    assert later != first, "expired track was resurrected with stale filter state"


def test_new_track_starts_at_the_observation():
    """A newly acquired ear must not ease in from the previous ear's position."""
    tr = EarTracker()
    box = np.array([100.0, 100.0, 180.0, 180.0])
    tid = tr.assign([box])[0]
    out = tr.smooth_box(tid, box, 0.0)
    assert np.allclose(out, box, atol=1e-6)
