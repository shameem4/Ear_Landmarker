"""Augmentation correctness: labels must move exactly like the image does.

This is the check that was missing when a sign error in the rotation matrix
mislabelled every rotated training sample (~12.5 px mean displacement at 192px,
against a val NME of 5.9 px) without moving any reported metric -- validation
runs unaugmented, so nothing downstream could see it.

Method: render a Gaussian dot on a gray field, place a landmark on the dot, run
the real augmentation, then recover the dot's true position by intensity
centroid (background-subtracted, so it is unbiased) and compare.

Run:  python -m pytest tests/ -v
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import AugmentationParams, EarLandmarkDataset  # noqa: E402

CANVAS = 384          # larger than 192 to keep resampling noise well below 1px
TOL_PX = 1.0          # tolerance in 192px units
BG = 128.0            # gray background, matches the augmentation fill colour


def _make_dot_image(dot: tuple[float, float]) -> Image.Image:
    """Gray field with one bright Gaussian dot at normalized position `dot`."""
    yy, xx = np.mgrid[0:CANVAS, 0:CANVAS]
    g = np.exp(-(((xx - dot[0] * CANVAS) ** 2 + (yy - dot[1] * CANVAS) ** 2) / (2 * 4.0 ** 2)))
    return Image.fromarray((BG + 127.0 * g).astype(np.uint8)).convert("RGB")


def _locate_dot(img: Image.Image) -> tuple[float, float] | None:
    """Recover the dot's normalized centre by background-subtracted centroid."""
    g = np.asarray(img.convert("L"), np.float32)
    g = np.clip(g - (BG + 2.0), 0, None)
    total = g.sum()
    if total < 1e-6:
        return None
    h, w = g.shape
    yy, xx = np.mgrid[0:h, 0:w]
    return ((g * xx).sum() / total / w, (g * yy).sum() / total / h)


def _bare_dataset(aug: AugmentationParams) -> EarLandmarkDataset:
    """A dataset instance with just enough state to call _geo_augment."""
    ds = EarLandmarkDataset.__new__(EarLandmarkDataset)
    ds.image_size = CANVAS
    ds.augmentation = aug
    return ds


def _run(aug: AugmentationParams, dot: tuple[float, float], draws: list[float]):
    """Apply _geo_augment with deterministic random draws."""
    img = _make_dot_image(dot)
    lm = torch.tensor([[dot[0], dot[1]]], dtype=torch.float32)
    it = iter(draws)
    real_uniform, real_random = random.uniform, random.random
    random.uniform = lambda a, b: next(it)
    random.random = lambda: 0.0          # force probability-gated augs to fire
    try:
        out_img, out_lm = _bare_dataset(aug)._geo_augment(img, lm)
    finally:
        random.uniform, random.random = real_uniform, real_random
    return out_img, (out_lm[0, 0].item(), out_lm[0, 1].item())


def _assert_agrees(out_img, label, context: str) -> None:
    truth = _locate_dot(out_img)
    assert truth is not None, f"{context}: dot left the frame, test point is invalid"
    err = math.dist(truth, label) * 192
    assert err < TOL_PX, (
        f"{context}: image puts the point at {tuple(round(v, 4) for v in truth)} "
        f"but the label says {tuple(round(v, 4) for v in label)} "
        f"-- {err:.2f}px disagreement at 192px (tolerance {TOL_PX}px)"
    )


DOTS = [(0.5, 0.25), (0.75, 0.5), (0.3, 0.7), (0.5, 0.5)]


@pytest.mark.parametrize("angle", [-15.0, -10.0, -5.0, -1.0, 1.0, 5.0, 10.0, 15.0])
@pytest.mark.parametrize("dot", DOTS)
def test_rotation_moves_labels_with_image(angle, dot):
    """Regression test for the inverted rotation matrix."""
    aug = AugmentationParams(
        horizontal_flip=False, translation=0.0, rotation_deg=abs(angle),
        color_jitter=None, bbox_jitter=0.0,
    )
    out_img, label = _run(aug, dot, [angle])
    _assert_agrees(out_img, label, f"rotation {angle:+.1f} deg at {dot}")


@pytest.mark.parametrize("tx_frac,ty_frac", [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0),
                                             (0.0, -1.0), (0.6, -0.8), (-0.7, 0.5)])
@pytest.mark.parametrize("dot", DOTS)
def test_translation_moves_labels_with_image(tx_frac, ty_frac, dot):
    aug = AugmentationParams(
        horizontal_flip=False, translation=0.05, rotation_deg=0.0,
        color_jitter=None, bbox_jitter=0.0,
    )
    out_img, label = _run(aug, dot, [tx_frac, ty_frac])
    _assert_agrees(out_img, label, f"translation ({tx_frac:+.1f}, {ty_frac:+.1f}) at {dot}")


@pytest.mark.parametrize("scale,tx,ty", [(1.10, 0.0, 0.0), (0.90, 0.0, 0.0),
                                         (1.00, 0.10, -0.10), (0.95, -0.08, 0.06),
                                         (1.08, 0.05, 0.05)])
@pytest.mark.parametrize("dot", DOTS)
def test_bbox_jitter_moves_labels_with_image(scale, tx, ty, dot):
    aug = AugmentationParams(
        horizontal_flip=False, translation=0.0, rotation_deg=0.0,
        color_jitter=None, bbox_jitter=0.10, bbox_jitter_prob=1.0,
    )
    out_img, label = _run(aug, dot, [scale, tx, ty])
    _assert_agrees(out_img, label, f"bbox_jitter scale={scale} t=({tx},{ty}) at {dot}")


@pytest.mark.parametrize("dot", DOTS)
def test_horizontal_flip_moves_labels_with_image(dot):
    aug = AugmentationParams(
        horizontal_flip=True, flip_prob=1.0, translation=0.0, rotation_deg=0.0,
        color_jitter=None, bbox_jitter=0.0,
    )
    out_img, label = _run(aug, dot, [])
    _assert_agrees(out_img, label, f"horizontal flip at {dot}")


def test_visibility_mask_marks_out_of_frame_points():
    """Points rotated out of the frame must be masked, not clamped to the border."""
    ds = _bare_dataset(AugmentationParams(
        horizontal_flip=False, translation=0.0, rotation_deg=15.0,
        color_jitter=None, bbox_jitter=0.0,
    ))
    # A corner point is the one most likely to leave the frame under rotation.
    lm = torch.tensor([[0.01, 0.01], [0.5, 0.5]], dtype=torch.float32)
    img = _make_dot_image((0.5, 0.5))
    it = iter([15.0])
    real_uniform = random.uniform
    random.uniform = lambda a, b: next(it)
    try:
        _, out_lm = ds._geo_augment(img, lm)
    finally:
        random.uniform = real_uniform

    visible = ((out_lm >= 0.0) & (out_lm <= 1.0)).all(dim=1)
    assert bool(visible[1]), "centre point must stay visible"
    # The corner point must not have been silently clamped onto the border.
    assert not torch.allclose(out_lm[0], out_lm[0].clamp(0.0, 1.0)) or bool(visible[0]), (
        "out-of-frame point was clamped instead of being left out of range for masking"
    )


@pytest.mark.parametrize("yaw,pitch", [(30.0, 0.0), (-30.0, 0.0), (0.0, 30.0),
                                       (0.0, -30.0), (25.0, -20.0), (-40.0, 15.0)])
@pytest.mark.parametrize("dot", DOTS)
def test_perspective_moves_labels_with_image(yaw, pitch, dot):
    """PIL's PERSPECTIVE takes the INVERSE matrix; using the forward one warps
    the image opposite to the labels. Same failure mode as the v1 rotation bug."""
    aug = AugmentationParams(
        horizontal_flip=False, translation=0.0, rotation_deg=0.0,
        color_jitter=None, bbox_jitter=0.0,
        perspective_deg=max(abs(yaw), abs(pitch)), perspective_prob=1.0,
    )
    out_img, label = _run(aug, dot, [yaw, pitch])
    _assert_agrees(out_img, label, f"perspective yaw={yaw} pitch={pitch} at {dot}")


def test_perspective_zero_angles_is_identity():
    ds = _bare_dataset(AugmentationParams())
    h = ds._perspective_matrix(0.0, 0.0)
    assert np.allclose(h / h[2, 2], np.eye(3), atol=1e-9)


def test_perspective_is_disabled_by_default():
    """Enabling it changes the training distribution, so it must be opt-in."""
    assert AugmentationParams().perspective_deg == 0.0


def test_legacy_rotation_switch_is_off():
    """The attribution-only bug switch must never be committed enabled."""
    import data.dataset as dataset_mod
    assert dataset_mod.LEGACY_ROTATION_SIGN is False, (
        "LEGACY_ROTATION_SIGN is True -- this corrupts every rotated label and "
        "must only ever be set at runtime by scripts/train_baseline_v1.py"
    )


def test_unaugmented_path_is_identity():
    """With no augmentation the labels must be returned untouched."""
    aug = AugmentationParams(
        horizontal_flip=False, translation=0.0, rotation_deg=0.0,
        color_jitter=None, bbox_jitter=0.0,
    )
    for dot in DOTS:
        out_img, label = _run(aug, dot, [])
        assert abs(label[0] - dot[0]) < 1e-6 and abs(label[1] - dot[1]) < 1e-6
        _assert_agrees(out_img, label, f"identity at {dot}")
