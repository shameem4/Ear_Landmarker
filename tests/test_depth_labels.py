"""The depth label's invariants under augmentation.

z is a fraction of the crop width, so anything that rescales the crop must
rescale z by the same factor. Nothing in a loss can detect a violation -- a
wrong-scale z is a perfectly plausible depth -- so it is pinned here.
"""
import math
import random

import numpy as np
import pytest
import torch
from PIL import Image

from data.dataset import AugmentationParams, EarLandmarkDataset


def _bare(aug):
    ds = EarLandmarkDataset.__new__(EarLandmarkDataset)
    ds.image_size = 192
    ds.augmentation = aug
    return ds


def _sample():
    img = Image.new("RGB", (192, 192), (128, 128, 128))
    xy = torch.tensor([[0.3, 0.3], [0.7, 0.3], [0.5, 0.7]])
    z = torch.tensor([-0.10, 0.05, 0.20])
    return img, xy, z


@pytest.mark.parametrize("seed", range(12))
def test_bbox_jitter_scales_depth_like_the_coordinates(seed):
    """z must gain exactly the factor the in-plane coordinates gain."""
    random.seed(seed)
    aug = AugmentationParams(bbox_jitter=0.2, bbox_jitter_prob=1.0,
                             horizontal_flip=False, translation=0.0,
                             rotation_deg=0.0, color_jitter=None,
                             perspective_deg=0.0)
    img, xy, z = _sample()
    out_img, out_xy, out_z = _bare(aug)._geo_augment(img, xy.clone(), z.clone())
    # the factor the coordinates actually gained, from a distance between two
    # points rather than from the jitter parameter, so this tests the code
    d0 = float(torch.linalg.norm(xy[1] - xy[0]))
    d1 = float(torch.linalg.norm(out_xy[1] - out_xy[0]))
    k_xy = d1 / d0
    k_z = float(out_z[0] / z[0])
    assert math.isclose(k_xy, k_z, rel_tol=1e-5), (
        f"coordinates scaled by {k_xy:.6f} but depth by {k_z:.6f}")


@pytest.mark.parametrize("seed", range(8))
def test_flip_translate_and_rotate_leave_depth_alone(seed):
    """A mirror, a shift and an in-plane turn do not change depth."""
    random.seed(seed)
    aug = AugmentationParams(bbox_jitter=0.0, horizontal_flip=True, flip_prob=1.0,
                             translation=0.05, rotation_deg=15.0,
                             color_jitter=None, perspective_deg=0.0)
    img, xy, z = _sample()
    _, out_xy, out_z = _bare(aug)._geo_augment(img, xy.clone(), z.clone())
    assert torch.allclose(out_z, z), "depth changed under an in-plane transform"
    assert not torch.allclose(out_xy, xy), "the test did not actually augment"


def test_perspective_is_refused_with_depth():
    """A homography has no parallax; it would leave z frontal on a turned image."""
    aug = AugmentationParams(perspective_deg=20.0, perspective_prob=1.0)
    img, xy, z = _sample()
    with pytest.raises(ValueError, match="parallax"):
        _bare(aug)._geo_augment(img, xy, z)


def test_perspective_still_allowed_without_depth():
    """The 2D path is unaffected -- the warp is only wrong for depth."""
    random.seed(0)
    aug = AugmentationParams(perspective_deg=20.0, perspective_prob=1.0,
                             color_jitter=None)
    img, xy, _ = _sample()
    out_img, out_xy, out_z = _bare(aug)._geo_augment(img, xy.clone())
    assert out_z is None
    assert out_xy.shape == xy.shape
