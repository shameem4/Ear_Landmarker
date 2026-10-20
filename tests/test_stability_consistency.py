"""The stability-consistency penalty's geometry, and that it is wired at all.

`unwobble` must invert `wobble` EXACTLY in continuous [0,1] coordinates. An
off-by-half-pixel or sign error would not crash; it would train every model
toward a shifted target and read as a small accuracy loss. This repo has done
that once already, with a W-1-x un-mirror that inflated equivariance 3x.

So a blob is planted at a known continuous position, the image is wobbled,
the blob is re-located by intensity centroid, and mapped back.
"""
import torch
import pytest

from model.lightning_module import EarLandmarkerModule, wobble, unwobble

W = 192


def _blob(u, v, sigma=3.0):
    """(1,3,W,W) normalised image with a Gaussian at continuous (u, v)."""
    c = (torch.arange(W, dtype=torch.float64) + 0.5) / W        # pixel centres
    g = torch.exp(-(((c[None, :] - u) ** 2) + ((c[:, None] - v) ** 2))
                  / (2 * (sigma / W) ** 2))
    return (g * 2 - 1).expand(1, 3, W, W).clone()


def _centroid(img):
    c = (torch.arange(W, dtype=torch.float64) + 0.5) / W
    # Only the blob's peak region. Zero padding reads 0 (mid-grey) in the
    # normalised tensor, so a weight like (img+1) gives the padded border
    # strip enough mass to swamp the blob -- the first version of this test
    # failed exactly that way, recovering ~(0.63, 0.47) for every blob.
    w = (img[0, 0] - 0.5).clamp_min(0)
    return (w * c[None, :]).sum() / w.sum(), (w * c[:, None]).sum() / w.sum()


@pytest.mark.parametrize("u,v", [(0.5, 0.5), (0.31, 0.62), (0.7, 0.28)])
def test_unwobble_inverts_wobble(u, v):
    img = _blob(u, v)
    gen = torch.Generator().manual_seed(0)
    for _ in range(8):
        out, prm = wobble(img, 0.05, 0.10, 0.0, 0.0, generator=gen)
        cu, cv = _centroid(out)
        back = unwobble(torch.tensor([[[cu, cv]]], dtype=torch.float64), prm)[0, 0]
        assert abs(back[0] - u) < 2e-3 and abs(back[1] - v) < 2e-3, (
            f"planted ({u},{v}) recovered ({back[0]:.4f},{back[1]:.4f})")


def test_wobble_actually_moves_the_blob():
    """Non-vacuity: if wobble were the identity, the test above passes too."""
    img = _blob(0.5, 0.5)
    gen = torch.Generator().manual_seed(1)
    out, prm = wobble(img, 0.05, 0.10, 0.0, 0.0, generator=gen)
    cu, cv = _centroid(out)
    assert abs(cu - 0.5) + abs(cv - 0.5) > 5e-3


def test_penalty_changes_training_loss_only():
    """Wiring: the flag must change the TRAINING loss and leave eval alone.

    `--flip-consistency` once reached train.py without reaching the module;
    this catches the same failure for this flag.
    """
    torch.manual_seed(0)
    batch = {"image": torch.rand(4, 3, W, W) * 2 - 1,
             "landmarks": torch.rand(4, 110) * 0.6 + 0.2,
             "visible": torch.ones(4, 55)}
    base = EarLandmarkerModule(arch="heatmap")
    on = EarLandmarkerModule(arch="heatmap", stability_consistency=1.0)
    on.load_state_dict(base.state_dict())
    base.train(); on.train()
    torch.manual_seed(1); lb = base._shared_step(batch)["loss"]
    torch.manual_seed(1); lo = on._shared_step(batch)["loss"]
    assert lo > lb + 1e-6, "penalty did not reach the training loss"
    # Eval on FRESH copies. After a training step the two models already
    # differ through BatchNorm running statistics: the penalty's extra forward
    # on the wobbled batch updates them (as the flip penalty's mirrored forward
    # does). That is a property of training with the penalty, not the penalty
    # term itself appearing in the eval loss, which is what this checks.
    fb = EarLandmarkerModule(arch="heatmap")
    fo = EarLandmarkerModule(arch="heatmap", stability_consistency=1.0)
    fo.load_state_dict(fb.state_dict())
    fb.eval(); fo.eval()
    with torch.no_grad():
        assert torch.allclose(fb._shared_step(batch)["loss"],
                              fo._shared_step(batch)["loss"]), "penalty leaked into eval"
