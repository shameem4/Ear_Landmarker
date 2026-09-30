"""All three paths must feed the model the range it was trained on.

data/dataset.py does `to_tensor()` -> [0, 1] then `normalize(mean=0.5, std=0.5)`,
so the model sees **[-1, 1]**. Both inference paths already did this; the risk is
that someone "simplifies" one to /255 and stops. Nothing errors if they do, the
landmarks still look broadly right, and it costs 15.6% test NME:

    input [0, 1]    test NME 0.03413
    input [-1, 1]   test NME 0.02883      (400 held-out samples, v6_persp65)

I briefly made exactly that change during a review, having convinced myself the
range was [0,1] on the strength of a vacuous check -- comparing the ONNX graph
against the torch module with the SAME array fed to both, which cannot say
anything about which range training used. These tests exist so the next such
edit fails loudly instead.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

NORMALISE = re.compile(r"-\s*0\.5\s*\)\s*/\s*0\.5")


def test_training_is_the_reference_and_maps_to_minus_one_to_one():
    src = (ROOT / "data" / "dataset.py").read_text()
    assert "to_tensor" in src
    assert "normalize" in src and "_norm_mean" in src, (
        "training normalisation changed shape; the inference paths must follow")


def test_python_inference_normalises():
    src = (ROOT / "inference.py").read_text()
    assert "/ 255.0" in src
    assert NORMALISE.search(src), (
        "inference.py stops at /255 and feeds [0,1]; training feeds [-1,1]")


def test_browser_inference_normalises():
    js = (ROOT / "docs" / "earlandmarker_inference.js").read_text()
    block = js[js.index("tensorData[i] ="):][:400]
    assert "/ 255.0" in block
    assert NORMALISE.search(block), (
        "browser stops at /255 and feeds [0,1]; training feeds [-1,1]")


def test_python_normalises_exactly_once():
    """A duplicated normalize maps [-1,1] to [-3,1] and is easy to add twice."""
    src = (ROOT / "inference.py").read_text()
    start = src.index("class LandmarkPredictor")
    body = src[start:start + 4000]
    assert len(NORMALISE.findall(body)) == 1, (
        "LandmarkPredictor normalises more than once")


def test_export_documents_the_range_it_traces():
    src = (ROOT / "export_onnx.py").read_text()
    assert "[-1, 1]" in src
    assert "input in [0, 1]" not in src


def test_export_emits_confidence_for_the_heatmap_head():
    """The browser weights smoothing by per-landmark confidence, so a
    single-output graph silently disables it."""
    src = (ROOT / "export_onnx.py").read_text()
    assert "predict_with_confidence" in src, (
        "forward() returns coordinates only; confidence comes from "
        "predict_with_confidence, so checking forward()'s type exports one output")
    assert '"confidence"' in src


def test_export_produces_a_single_self_contained_file():
    """Torch's exporter writes weights to a sidecar .onnx.data by default; the
    browser fetches one URL, so a split graph loads with no weights."""
    src = (ROOT / "export_onnx.py").read_text()
    assert "save_as_external_data=False" in src
