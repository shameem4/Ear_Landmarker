"""Occupancy re-framing: labels stay on their pixels, the drawn occupancy is
the one achieved, and the default path is untouched.

A label error here would not crash: it would train every re-framed sample
toward a shifted point and show up only as a small accuracy loss -- the class
of bug (wrong rotation sign, W-1-x un-mirror) this repo has shipped before.
"""
import random
import numpy as np
import torch
import pytest
from pathlib import Path
from PIL import Image

from data.dataset import EarLandmarkDataset, AugmentationParams

D = Path(__file__).resolve().parents[1] / "data" / "v3_manual"


class _Bare(EarLandmarkDataset):
    """Just enough object to call _reframe without a split on disk."""
    def __init__(self, size=192):
        self.image_size = size


def _marked(w, h, pts):
    """Grey image with a single bright pixel at each (continuous) point."""
    a = np.full((h, w, 3), 60, np.uint8)
    for x, y in pts:
        a[int(y), int(x)] = 255
    return Image.fromarray(a)


@pytest.mark.parametrize("t", [0.55, 0.70, 0.85, 0.98])
def test_labels_land_on_their_pixels(t):
    random.seed(0)
    w = h = 600
    rng = np.random.default_rng(1)
    pts = np.c_[rng.uniform(200, 400, 12), rng.uniform(180, 420, 12)].round() + 0.5   # pixel centres
    img = _marked(w, h, pts)
    lm = torch.tensor(pts / [w, h], dtype=torch.float32)
    ds = _Bare(size=600)                     # no downscale, so a pixel stays a pixel
    out, lm2 = ds._reframe(img, lm, (t, t))
    a = np.asarray(out)[..., 0].astype(float)
    for (u, v) in lm2.numpy() * 600:
        y, x = int(v), int(u)
        win = a[max(0, y-2):y+3, max(0, x-2):x+3]
        assert win.max() > 150, f"no mark near label ({u:.1f},{v:.1f}) at T={t}"


@pytest.mark.parametrize("t", [0.55, 0.75, 0.95])
def test_achieved_occupancy_matches_draw(t):
    random.seed(0)
    pts = np.array([[100, 120], [300, 140], [210, 380], [150, 300]], float)
    lm = torch.tensor(pts / 500, dtype=torch.float32)
    _, lm2 = _Bare()._reframe(Image.new("RGB", (500, 500)), lm, (t, t))
    occ = np.ptp(lm2.numpy(), 0).max()
    assert abs(occ - t) < 0.01, f"drew {t}, achieved {occ:.3f}"


def test_default_is_off():
    p = AugmentationParams()
    assert p.occupancy_range is None


@pytest.fixture(scope="module")
def v3():
    if not (D / "train.csv").exists():
        pytest.skip("data/v3_manual not built")
    base = dict(translation=0.0, rotation_deg=0.0, color_jitter=None, bbox_jitter=0.0,
                horizontal_flip=False, perspective_deg=0.0)
    return base


def test_off_path_is_plain_resize(v3):
    """With re-framing off the image is exactly the stored crop resized."""
    ds = EarLandmarkDataset(split_csv=D/"train.csv", data_dir=D, image_size=192,
                            augmentation=AugmentationParams(**v3))
    r = ds.samples[0]
    want = Image.open(D / r["image_file"]).convert("RGB").resize((192, 192), Image.BILINEAR)
    got = ds[0]["image"]
    ref = (torch.from_numpy(np.asarray(want, np.float32) / 255).permute(2, 0, 1) - 0.5) / 0.5
    assert torch.allclose(got, ref, atol=1e-5)


def test_real_samples_reach_the_drawn_occupancy(v3):
    ds = EarLandmarkDataset(split_csv=D/"train.csv", data_dir=D, image_size=192,
                            augmentation=AugmentationParams(**v3, occupancy_range=(0.9, 0.9),
                                                            occupancy_prob=1.0))
    occ = []
    for i in range(20):
        s = ds[i]; lm = s["landmarks"].view(55, 2); v = s["visible"].bool()
        occ.append(np.ptp(lm[v].numpy(), 0).max())
    assert abs(np.median(occ) - 0.9) < 0.02, f"median achieved {np.median(occ):.3f}"
