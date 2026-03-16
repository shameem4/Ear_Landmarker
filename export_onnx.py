"""Export EarLandmarker to ONNX for web deployment.

Creates a web-optimized ONNX model that:
- Takes (1, 3, 192, 192) input in [-1, 1]
- Outputs (1, 55, 2) landmark coordinates in [0, 1]

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
from onnxsim import simplify

from model.ear_landmarker import EarLandmarker
from inference import find_best_checkpoint

PROJECT = Path("C:/Users/shame/OneDrive/Desktop/ear_stuff/Ear Landmarker")


def load_model(checkpoint_path: Path) -> EarLandmarker:
    """Load EarLandmarker weights from a Lightning checkpoint."""
    model = EarLandmarker(num_landmarks=55)
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    state = ckpt.get("state_dict", ckpt)
    state = {k.removeprefix("model."): v for k, v in state.items()
             if k.startswith("model.")} or state
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


class EarLandmarkerWeb(torch.nn.Module):
    """Wrapper that reshapes output to (1, 55, 2) for cleaner ONNX graph."""

    def __init__(self, model: EarLandmarker) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            image: (1, 3, 192, 192) normalized to [-1, 1].

        Returns:
            (1, 55, 2) landmarks in [0, 1].
        """
        out = self.model(image)  # (1, 110)
        return out.view(1, 55, 2)


def export(checkpoint_path: Path, output_path: Path) -> None:
    """Export model to ONNX with simplification."""
    model = load_model(checkpoint_path)
    wrapper = EarLandmarkerWeb(model)
    wrapper.eval()

    dummy = torch.randn(1, 3, 192, 192)
    # Normalize like real input
    dummy = (dummy - 0.5) / 0.5

    torch.onnx.export(
        wrapper,
        dummy,
        str(output_path),
        input_names=["image"],
        output_names=["landmarks"],
        dynamic_axes=None,  # fixed batch size 1
        opset_version=14,
        do_constant_folding=True,
    )
    print(f"Exported raw ONNX: {output_path}")

    # Simplify
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
    print(f"Input:  image (1, 3, 192, 192) float32, normalized [-1, 1]")
    print(f"Output: landmarks (1, 55, 2) float32, [0, 1]")


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
