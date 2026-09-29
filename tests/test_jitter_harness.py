"""Validate the jitter benchmark's coordinate inversion.

The geometric jitter metric only means anything if a perfectly equivariant model
scores zero. If `to_canonical` were wrong, every model would show large spurious
jitter and the benchmark would be measuring the harness instead of the models --
silently, since there is no ground truth to contradict it.

These tests substitute an ideal model (one that always reports the true landmark
position, transformed into whatever crop it is given) and assert it scores ~0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_jitter import (  # noqa: E402
    IMG_SIZE,
    jitter_stats,
    make_geometric_frames,
    to_canonical,
)

SRC = IMG_SIZE * 2
BASE_SIDE = SRC * 0.85


def _ideal_predictions(windows, truth: np.ndarray) -> np.ndarray:
    """What a perfectly equivariant model outputs for each crop window."""
    pred = np.empty((len(windows), truth.shape[0], 2), np.float32)
    for t, (x0, y0, side) in enumerate(windows):
        pred[t, :, 0] = (truth[:, 0] * SRC - x0) / side
        pred[t, :, 1] = (truth[:, 1] * SRC - y0) / side
    return pred


@pytest.mark.parametrize("trial", range(4))
def test_ideal_model_scores_zero_geometric_jitter(trial):
    truth = np.random.default_rng(7).uniform(0.25, 0.75, size=(55, 2))
    rng = np.random.default_rng(trial)
    _, windows = make_geometric_frames(
        np.zeros((SRC, SRC, 3), np.float32), 12, rng,
        max_shift=0.008, max_scale=0.02,
    )
    canon = to_canonical(_ideal_predictions(windows, truth), windows, BASE_SIDE)
    spread, succ = jitter_stats(canon)
    assert spread < 1e-3, f"ideal model shows {spread:.4f}px spread; inversion is wrong"
    assert succ < 1e-3, f"ideal model shows {succ:.4f}px succ; inversion is wrong"


def test_inversion_recovers_the_true_position():
    """Beyond scoring zero, canonical coords must be the actual truth."""
    truth = np.random.default_rng(3).uniform(0.25, 0.75, size=(55, 2))
    rng = np.random.default_rng(0)
    _, windows = make_geometric_frames(
        np.zeros((SRC, SRC, 3), np.float32), 8, rng,
        max_shift=0.008, max_scale=0.02,
    )
    canon = to_canonical(_ideal_predictions(windows, truth), windows, BASE_SIDE)
    recovered = canon * BASE_SIDE / SRC          # back to source-normalized
    assert np.abs(recovered - truth[None]).max() < 1e-5


def test_jitter_stats_zero_on_constant_input():
    coords = np.tile(np.random.default_rng(1).uniform(0, 1, (55, 2))[None], (12, 1, 1))
    spread, succ = jitter_stats(coords)
    assert spread < 1e-9 and succ == 0.0


def test_jitter_stats_scales_as_expected():
    """A known constant per-frame displacement must read back at that size."""
    n, shift = 10, 0.01                      # 0.01 of the crop = 1.92px at 192
    coords = np.zeros((n, 55, 2), np.float64)
    coords[:, :, 0] = np.arange(n)[:, None] * shift   # steady drift along x
    _, succ = jitter_stats(coords)
    assert succ == pytest.approx(shift * IMG_SIZE, rel=1e-6)
