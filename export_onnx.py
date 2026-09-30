"""Export EarLandmarker to ONNX for web deployment.

Creates a web-optimized ONNX model that:
- Takes (1, 3, 192, 192) input in [-1, 1]
- Outputs (1, 55, 2) landmark coordinates in [0, 1],
  plus (1, 55) per-landmark confidence for the heatmap head

Usage:
    python export_onnx.py
    python export_onnx.py --checkpoint path/to/model.ckpt
    python export_onnx.py --output docs/EarLandmarker_web.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import onnx

try:                                    # optional: shrinks the graph, not required
    from onnxsim import simplify
except ImportError:                     # pragma: no cover - depends on the env
    simplify = None

from model.ear_landmarker import EarLandmarker, EarLandmarkerHeatmap
from inference import find_best_checkpoint

PROJECT = Path(__file__).resolve().parent


def load_model(checkpoint_path: Path) -> EarLandmarker:
    """Load EarLandmarker weights from a Lightning checkpoint.

    The architecture is read from the checkpoint's saved hyperparameters, so a
    heatmap-head checkpoint exports correctly without passing a matching flag.
    """
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    hparams = ckpt.get("hyper_parameters", {}) or {}
    arch = hparams.get("arch", "gap")

    if arch == "heatmap":
        model = EarLandmarkerHeatmap(num_landmarks=55, tau=hparams.get("tau", 1.0))
    else:
        model = EarLandmarker(num_landmarks=55)
    print(f"Checkpoint architecture: {arch}")

    state = ckpt.get("state_dict", ckpt)
    state = {k.removeprefix("model."): v for k, v in state.items()
             if k.startswith("model.")} or state
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


class EarLandmarkerWeb(torch.nn.Module):
    """Reshapes to (1, 55, 2), and passes through per-landmark confidence.

    The heatmap head returns (landmarks, confidence); the GAP head returns
    landmarks alone. The shipped graph has BOTH outputs and the browser reads
    the second to weight temporal smoothing, so exporting only "landmarks" --
    which this wrapper used to do -- produces a model the demo cannot use.
    """

    def __init__(self, model: EarLandmarker) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor):
        """Forward pass.

        Args:
            image: (1, 3, 192, 192) in [-1, 1] -- data/dataset.py does
                to_tensor() -> [0,1] then normalize(0.5, 0.5).

        Returns:
            (1, 55, 2) landmarks in [0, 1], and for the heatmap head a
            (1, 55) confidence.
        """
        # The heatmap head exposes confidence through predict_with_confidence,
        # not through forward() -- forward returns coordinates alone. Checking
        # forward()'s return type therefore always saw a bare tensor and
        # exported a single-output graph the browser cannot use.
        if hasattr(self.model, "predict_with_confidence"):
            landmarks, confidence = self.model.predict_with_confidence(image)
            return landmarks.view(1, 55, 2), confidence.view(1, 55)
        return self.model(image).view(1, 55, 2)


def export(checkpoint_path: Path, output_path: Path) -> None:
    """Export model to ONNX with simplification."""
    model = load_model(checkpoint_path)
    wrapper = EarLandmarkerWeb(model)
    wrapper.eval()

    # Trace on an input in the range the model actually sees, [-1, 1].
    dummy = (torch.rand(1, 3, 192, 192) - 0.5) / 0.5

    with torch.no_grad():
        traced = wrapper(dummy)
    output_names = (["landmarks", "confidence"]
                    if isinstance(traced, tuple) else ["landmarks"])

    torch.onnx.export(
        wrapper,
        dummy,
        str(output_path),
        input_names=["image"],
        output_names=output_names,
        dynamic_axes=None,  # fixed batch size 1
        opset_version=14,
        do_constant_folding=True,
    )
    print(f"Exported raw ONNX: {output_path}")
    print(f"Outputs: {', '.join(output_names)}")

    # Torch's current exporter writes weights to a sidecar "<name>.onnx.data"
    # by default. The browser fetches one URL, so a split graph loads as a model
    # with no weights -- fold it back into a single self-contained file.
    sidecar = output_path.with_suffix(output_path.suffix + ".data")
    if sidecar.exists():
        model_with_weights = onnx.load(str(output_path), load_external_data=True)
        onnx.save(model_with_weights, str(output_path), save_as_external_data=False)
        sidecar.unlink()
        print(f"Folded {sidecar.name} back into a single file")

    # Simplify, when onnxsim is available. It shrinks the graph and is not
    # required for correctness, so a missing install is a note, not a failure.
    if simplify is None:
        print("onnxsim not installed, keeping the unsimplified graph")
    else:
        onnx_model = onnx.load(str(output_path))
        simplified, check = simplify(onnx_model)
        if check:
            onnx.save(simplified, str(output_path))
            print(f"Simplified ONNX saved: {output_path}")
        else:
            print("WARNING: simplification check failed, keeping unsimplified model")

    # Print model info
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Model size: {size_mb:.2f} MB")
    print(f"Input:  image (1, 3, 192, 192) float32, [-1, 1]")
    print(f"Output: {', '.join(output_names)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export EarLandmarker to ONNX")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .ckpt file (default: best by NME)")
    parser.add_argument("--output", type=str,
                        default=str(PROJECT / "docs" / "EarLandmarker_web.onnx"),
                        help="Output ONNX path")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint) if args.checkpoint else find_best_checkpoint()
    print(f"Checkpoint: {ckpt}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    export(ckpt, output)


if __name__ == "__main__":
    main()
