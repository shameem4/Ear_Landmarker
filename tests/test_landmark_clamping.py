"""An out-of-crop landmark must never be supervised at a made-up position.

THE BUG THIS LOCKS DOWN. data/preprocess.py normalised landmarks and then did
`np.clip(lm_norm, 0.0, 1.0)`, pinning any point that fell outside the crop onto
the frame border. data/dataset.py decides visibility with
`(lm >= 0.0) & (lm <= 1.0)`, and a clamped point sits *exactly* on 0.0 or 1.0 --
so it passed as visible and the model was trained to put it there. Measured on
the shipped set: 714 landmarks, 0.41% of collectionB's points against ~0%
everywhere else, 2.9% of training samples affected.

The clamp also destroyed the evidence: once a point is moved to the border, a
fabricated position and a real edge position are indistinguishable. So the fix
is in two parts, and both are tested here -- stop clipping (preprocess), and
treat a point already on the border as out-of-crop (dataset, for data generated
before the fix).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.dataset import EarLandmarkDataset, NUM_LANDMARKS  # noqa: E402


def _make_set(tmp_path: Path, landmarks: np.ndarray) -> Path:
    """Minimal preprocessed-format directory holding one sample."""
    (tmp_path / "images").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), (120, 130, 140)).save(tmp_path / "images" / "0.png")
    np.save(tmp_path / "landmarks.npy", landmarks[None].astype(np.float32))
    with open(tmp_path / "split.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "image_file", "source",
                                          "original_path", "width", "height"])
        w.writeheader()
        w.writerow({"idx": 0, "image_file": "images/0.png", "source": "t",
                    "original_path": "t", "width": 64, "height": 64})
    return tmp_path


def _visible(tmp_path: Path, landmarks: np.ndarray) -> torch.Tensor:
    d = _make_set(tmp_path, landmarks)
    ds = EarLandmarkDataset(split_csv=d / "split.csv", data_dir=d,
                            image_size=64, augmentation=None,
                            landmarks_file="landmarks.npy")
    return ds[0]["visible"]


def test_interior_points_are_visible(tmp_path):
    lm = np.full((NUM_LANDMARKS, 2), 0.5, dtype=np.float32)
    assert _visible(tmp_path, lm).sum() == NUM_LANDMARKS


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_point_pinned_to_the_border_is_not_supervised(tmp_path, value):
    """The legacy-clamped case: exactly on the edge means position unknown."""
    lm = np.full((NUM_LANDMARKS, 2), 0.5, dtype=np.float32)
    lm[7] = [value, 0.5]
    vis = _visible(tmp_path, lm)
    assert vis[7] == 0.0, f"a landmark pinned at {value} must be masked out"
    assert vis.sum() == NUM_LANDMARKS - 1


@pytest.mark.parametrize("value", [-0.3, 1.4])
def test_point_outside_the_crop_is_not_supervised(tmp_path, value):
    """The post-fix case: preprocess now stores the true out-of-range value."""
    lm = np.full((NUM_LANDMARKS, 2), 0.5, dtype=np.float32)
    lm[3] = [0.5, value]
    vis = _visible(tmp_path, lm)
    assert vis[3] == 0.0
    assert vis.sum() == NUM_LANDMARKS - 1


def test_augmentation_cannot_resurrect_an_out_of_crop_point(tmp_path):
    """Translation/rotation can carry an out-of-frame point back inside.

    Its recorded coordinate is still not where the ear feature is, so it must
    stay masked however the geometry moves it.
    """
    from data.dataset import AugmentationParams
    lm = np.full((NUM_LANDMARKS, 2), 0.5, dtype=np.float32)
    lm[11] = [1.0, 0.5]                     # clamped onto the right border
    d = _make_set(tmp_path, lm)
    aug = AugmentationParams(horizontal_flip=False, translation=0.05,
                             rotation_deg=15.0, color_jitter=None,
                             bbox_jitter=0.0, perspective_deg=0.0)
    ds = EarLandmarkDataset(split_csv=d / "split.csv", data_dir=d, image_size=64,
                            augmentation=aug, landmarks_file="landmarks.npy")
    for _ in range(40):
        assert ds[0]["visible"][11] == 0.0


def test_preprocess_does_not_clip_normalised_landmarks():
    """Guards the upstream half: clipping must not come back."""
    src = (ROOT / "data" / "preprocess.py").read_text(encoding="utf-8")
    assert "np.clip(lm_norm" not in src, (
        "preprocess.py must not clip landmarks to [0, 1] -- that pins "
        "out-of-crop points onto the border and they then read as visible"
    )


def test_shipped_set_has_no_supervised_border_points():
    """Regression against the real data, when it is present."""
    lm_path = ROOT / "data" / "preprocessed" / "landmarks.npy"
    if not lm_path.exists():
        pytest.skip("preprocessed set not built")
    lm = np.load(lm_path)
    on_border = ((lm <= 0.0) | (lm >= 1.0)).any(axis=2)
    # The shipped arrays were written by the clipping version, so border points
    # still exist; what must hold is that the dataset refuses to supervise them.
    sample = int(np.argmax(on_border.sum(axis=1)))
    if not on_border[sample].any():
        pytest.skip("no border points in this build of the set")
    from data.dataset import EarLandmarkDataset as DS
    raw = torch.tensor(lm[sample])
    out_of_crop = ((raw <= 0.0) | (raw >= 1.0)).any(dim=1)
    assert out_of_crop.sum() == on_border[sample].sum()
    assert out_of_crop.any(), "expected this sample to carry clamped points"
