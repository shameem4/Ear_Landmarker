"""The pretraining experiment has to be able to measure something.

Transferring a donor backbone only means anything if the weights actually land.
Against the shipped backbone (5x5 depthwise, 24-48-96-128-192) just 0.9% of
parameters are shape-compatible with BlazeEar's detector -- essentially the
first conv -- so an A/B there would be a null by construction and would read as
"pretraining does not help" when it only shows the weights never arrived.

The `blazeear` backbone mirrors the donor block for block so the transfer is
real. These tests pin that, and pin that the shipped architecture is untouched.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.ear_landmarker import EarLandmarkerHeatmap  # noqa: E402

BLAZE = Path(os.environ.get("BLAZEEAR_DIR",
                            Path(__file__).resolve().parents[2] / "BlazeEar"))
DONOR = BLAZE / "runs" / "checkpoints_crop" / "BlazeEar_best.pth"
needs_donor = pytest.mark.skipif(not DONOR.exists(), reason="BlazeEar donor not on disk")


def backbone_params(model) -> int:
    return sum(p.numel() for n, p in model.named_parameters()
               if n.startswith(("conv0", "stage")))


def test_shipped_backbone_is_unchanged():
    """The experiment must not perturb what ships."""
    m = EarLandmarkerHeatmap()
    assert m.backbone_kind == "default"
    assert sum(p.numel() for p in m.parameters()) == 340_167


def test_both_backbones_produce_the_same_output_contract():
    for kind in ("default", "blazeear"):
        m = EarLandmarkerHeatmap(backbone=kind).eval()
        with torch.no_grad():
            out = m(torch.randn(2, 3, 192, 192))
        assert out.shape == (2, 110), f"{kind} changed the output contract"


def test_unknown_backbone_is_rejected():
    with pytest.raises(ValueError):
        EarLandmarkerHeatmap(backbone="facemesh")


@needs_donor
def test_mirrored_backbone_actually_receives_the_donor_weights():
    """The point of the whole exercise."""
    m = EarLandmarkerHeatmap(backbone="blazeear")
    moved = m.load_blazeear_backbone(str(DONOR))
    assert moved / backbone_params(m) > 0.5, (
        f"only {moved} of {backbone_params(m)} backbone params transferred; "
        "the experiment would measure nothing")


@needs_donor
def test_default_backbone_barely_receives_anything():
    """Documents WHY the experiment does not use the shipped backbone."""
    m = EarLandmarkerHeatmap(backbone="default")
    moved = m.load_blazeear_backbone(str(DONOR))
    assert moved / backbone_params(m) < 0.05


@needs_donor
def test_transfer_changes_the_weights_it_claims_to_change():
    """A transfer that reports a count but leaves the tensors alone would make
    the two arms identical and the experiment unfalsifiable."""
    m = EarLandmarkerHeatmap(backbone="blazeear")
    before = m.stage1[0].pw_conv.weight.detach().clone()
    m.load_blazeear_backbone(str(DONOR))
    after = m.stage1[0].pw_conv.weight.detach()
    assert not torch.allclose(before, after), "weights unchanged after transfer"


@needs_donor
def test_transferred_model_still_runs_forward():
    m = EarLandmarkerHeatmap(backbone="blazeear")
    m.load_blazeear_backbone(str(DONOR))
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(1, 3, 192, 192))
    assert out.shape == (1, 110)
    assert torch.isfinite(out).all()
