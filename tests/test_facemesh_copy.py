"""The FaceMesh copy must actually be a copy, or the experiment means nothing.

If the reconstruction drifts from MediaPipe's topology, the pretrained weights
stop loading cleanly and the result stops being "can their architecture match
ours" -- it becomes "can some architecture match ours", which is a different and
far less interesting question.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.facemesh_ear import FaceMeshEarLandmarker, RESOLUTION  # noqa: E402

WEIGHTS = ROOT.parent / "trainable_blazeface" / "model_weights" / "blazeface_landmark.pth"
needs_weights = pytest.mark.skipif(not WEIGHTS.exists(),
                                   reason="MediaPipe face landmark weights not on disk")


def test_input_resolution_matches_earlandmarker():
    """They happen to agree at 192, which is why the comparison is fair."""
    assert RESOLUTION == 192


def test_forward_contract_matches_the_other_architectures():
    m = FaceMeshEarLandmarker().eval()
    with torch.no_grad():
        out = m(torch.rand(2, 3, 192, 192))
    assert out.shape == (2, 110)
    assert torch.isfinite(out).all()


@needs_weights
def test_pretrained_weights_load_into_the_shared_layers():
    """84.6% is the figure the experiment is premised on."""
    m = FaceMeshEarLandmarker()
    total = sum(p.numel() for p in m.parameters())
    moved = m.load_mediapipe_backbone(str(WEIGHTS))
    assert moved / total > 0.8, (
        f"only {moved}/{total} transferred; the copy has drifted from MediaPipe")


@needs_weights
def test_feature_extractor_transfers_completely():
    """backbone1 is the part that is copied exactly, so it must go across whole."""
    m = FaceMeshEarLandmarker()
    b1 = sum(p.numel() for p in m.backbone1.parameters())
    before = {k: v.clone() for k, v in m.backbone1.state_dict().items()}
    m.load_mediapipe_backbone(str(WEIGHTS))
    after = m.backbone1.state_dict()
    changed = sum(v.numel() for k, v in after.items()
                  if not torch.allclose(before[k], v))
    assert changed / b1 > 0.95, f"only {changed}/{b1} of backbone1 was replaced"


@needs_weights
def test_the_only_reshaped_layer_is_the_output_conv():
    """468x3 -> 55x2 is unavoidable; nothing else may change shape."""
    src = torch.load(str(WEIGHTS), map_location="cpu", weights_only=True)
    dst = FaceMeshEarLandmarker().state_dict()
    mismatched = [k for k in src
                  if k in dst and dst[k].shape != src[k].shape]
    # backbone2b (face-presence head) is dropped, so it is absent, not reshaped.
    assert all("backbone2a.6" in k for k in mismatched), (
        f"unexpected shape changes beyond the output conv: {mismatched}")


@needs_weights
def test_transfer_is_not_silently_a_no_op():
    m = FaceMeshEarLandmarker()
    w = m.backbone1[2].convs[0].weight.detach().clone()
    m.load_mediapipe_backbone(str(WEIGHTS))
    assert not torch.allclose(w, m.backbone1[2].convs[0].weight.detach())
