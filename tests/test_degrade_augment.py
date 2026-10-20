"""Image-quality degradation augmentation: labels untouched, never a no-op,
and genuinely off by default.

The failure modes worth pinning are silent ones. A degradation that moved the
landmarks would train toward wrong labels; one that was a no-op would make a
"degradation didn't help" result meaningless; one that leaked into the
default path would change every existing recipe's numbers.
"""
import random
import numpy as np
import pytest
import cv2
from PIL import Image

from data.dataset import EarLandmarkDataset, AugmentationParams


def _img(seed=0):
    rng = np.random.default_rng(seed)
    a = np.zeros((192, 192, 3), np.uint8)
    cv2.circle(a, (96, 96), 50, (220, 180, 160), -1)
    cv2.circle(a, (96, 96), 30, (80, 60, 50), 3)
    a = np.clip(a + rng.normal(0, 4, a.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(a)


def _sharp(im):
    return cv2.Laplacian(cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()


def test_degrade_is_never_a_no_op():
    im = _img()
    random.seed(0)
    for _ in range(40):
        out = EarLandmarkDataset._degrade(im)
        assert out.size == im.size and out.mode == "RGB"
        assert np.abs(np.asarray(out, float) - np.asarray(im, float)).mean() > 0.2


def test_degrade_lowers_sharpness_on_average():
    """Non-vacuity at the level that matters: the image gets softer."""
    im = _img(); random.seed(1)
    ratios = [_sharp(EarLandmarkDataset._degrade(im)) / _sharp(im) for _ in range(60)]
    assert np.median(ratios) < 0.8, f"median sharpness ratio {np.median(ratios):.2f}"


def test_degrade_prob_defaults_to_off():
    assert AugmentationParams().degrade_prob == 0.0


@pytest.fixture(scope="module")
def ds_pair():
    from pathlib import Path
    D = Path(__file__).resolve().parents[1] / "data" / "v3_manual"
    if not (D / "train.csv").exists():
        pytest.skip("data/v3_manual not built")
    base = dict(translation=0.0, rotation_deg=0.0, color_jitter=None,
                bbox_jitter=0.0, horizontal_flip=False, perspective_deg=0.0)
    off = EarLandmarkDataset(split_csv=D / "train.csv", data_dir=D, image_size=192,
                             augmentation=AugmentationParams(**base))
    on = EarLandmarkDataset(split_csv=D / "train.csv", data_dir=D, image_size=192,
                            augmentation=AugmentationParams(**base, degrade_prob=1.0))
    return off, on


def test_degrade_leaves_landmarks_and_visibility_untouched(ds_pair):
    off, on = ds_pair
    for i in range(10):
        random.seed(i); a = off[i]
        random.seed(i); b = on[i]
        assert (a["landmarks"] == b["landmarks"]).all()
        assert (a["visible"] == b["visible"]).all()
        assert not (a["image"] == b["image"]).all(), "degrade_prob=1 changed nothing"
