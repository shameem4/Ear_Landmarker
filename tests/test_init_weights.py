"""--init-weights must load the network exactly, or refuse loudly."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.lightning_module import EarLandmarkerModule   # noqa: E402
from train import load_init_weights                      # noqa: E402


def _ckpt(tmp_path, module):
    f = tmp_path / "src.ckpt"
    torch.save({"state_dict": module.state_dict(), "epoch": 433}, f)
    return f


def test_loaded_model_predicts_identically(tmp_path):
    torch.manual_seed(0)
    src = EarLandmarkerModule(arch="heatmap").eval()
    torch.manual_seed(1)
    dst = EarLandmarkerModule(arch="heatmap").eval()
    x = torch.randn(2, 3, 192, 192)
    with torch.no_grad():
        assert not torch.allclose(src.model(x), dst.model(x))
        f = load_init_weights(dst, _ckpt(tmp_path, src))
        assert torch.equal(src.model(x), dst.model(x))
    assert dst.hparams["init_weights"] == str(f)


def test_architecture_mismatch_refuses(tmp_path):
    src = EarLandmarkerModule(arch="heatmap")
    dst = EarLandmarkerModule(arch="heatmap", width_mult=0.5)
    with pytest.raises(RuntimeError):
        load_init_weights(dst, _ckpt(tmp_path, src))


def test_missing_file_refuses(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_init_weights(EarLandmarkerModule(arch="heatmap"), tmp_path / "nope.ckpt")
