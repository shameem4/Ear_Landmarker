"""The export verification must fail loudly rather than pass vacuously.

BlazeEar shipped an ONNX post-process graph whose confidence threshold was
never applied. It survived because the self-check probed with a random image,
nothing scored above the gate, and it reported success having compared two
empty results -- a check that passes loudest when it has verified nothing.

These tests pin the two properties that stop this one doing the same: a missing
dependency is a FAILURE, and agreement is rejected when the graph ignores its
input.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import export_onnx  # noqa: E402


def test_missing_onnxruntime_is_a_failure_not_a_pass(monkeypatch):
    """The whole point: an unrunnable check must not report success."""
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def no_ort(name, *a, **k):
        if name == "onnxruntime":
            raise ImportError("simulated missing onnxruntime")
        return real_import(name, *a, **k)

    monkeypatch.setattr("builtins.__import__", no_ort)
    assert export_onnx.verify_export(
        torch.nn.Identity(), Path("/nonexistent.onnx"), ["landmarks"]) is False


def test_export_raises_when_verification_fails(monkeypatch, tmp_path):
    """A failed check must stop the export being treated as good."""
    monkeypatch.setattr(export_onnx, "verify_export", lambda *a, **k: False)
    monkeypatch.setattr(export_onnx, "load_model", lambda p: torch.nn.Identity())

    class _Wrapper(torch.nn.Module):
        def forward(self, x):
            return x.reshape(1, -1)[:, :110].reshape(1, 55, 2)

    monkeypatch.setattr(export_onnx, "EarLandmarkerWeb", lambda m: _Wrapper())
    with pytest.raises(SystemExit):
        export_onnx.export(Path("unused.ckpt"), tmp_path / "out.onnx")


def test_probe_inputs_are_in_the_trained_range_and_differ():
    probes = export_onnx._probe_inputs()
    assert len(probes) >= 2
    for p in probes:
        assert p.shape == (1, 3, 192, 192)
        assert p.min() >= -1.01 and p.max() <= 1.01, "probes must be in [-1, 1]"
    spread = max(float((p - probes[0]).abs().max()) for p in probes[1:])
    assert spread > 0.1, "probes must differ, or the non-vacuity guard is untestable"


def test_no_check_flag_exists():
    """Skipping deliberately is fine; skipping silently is what we are stopping."""
    src = Path(export_onnx.__file__).read_text()
    assert "--no-check" in src
    assert "check: bool = True" in src, "verification must be on by default"
