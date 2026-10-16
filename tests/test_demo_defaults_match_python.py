"""The browser demo must ship the same gate the Python pipeline ships.

BlazeEar shipped a live page that confirmed tracks at 0.70 while its constants
and its tests said 0.85, because the slider carried `value="0.70"` in HTML and
the loader fed that attribute straight into the pipeline. Their test checked
the constants and never the value that overrides them at load.

This repo has the identical shape -- docs/index.html builds the pipeline with
`confidenceThreshold: parseFloat(confSlider.value)`, so the HTML attribute is
authoritative and the JS default is unreachable on that path. The values agree
today; these tests are what keeps them agreeing, and they became more necessary
when DETECTOR_PROFILES made the Python side re-pointable.
"""
import re
from pathlib import Path

import pytest

from inference import DETECTOR_CONFIDENCE

DOCS = Path(__file__).resolve().parents[1] / "docs"
HTML = (DOCS / "index.html").read_text()
JS = (DOCS / "earlandmarker_inference.js").read_text()


def test_slider_value_attribute_matches_the_shipped_gate():
    m = re.search(r'id="confSlider"[^>]*\bvalue="([0-9.]+)"', HTML)
    assert m, "confSlider lost its value attribute, or the id changed"
    assert float(m.group(1)) == pytest.approx(DETECTOR_CONFIDENCE), (
        "the demo loads the pipeline from this attribute, so it IS the "
        "shipped browser gate regardless of what any constant says"
    )


def test_displayed_label_matches_the_slider():
    m = re.search(r'id="confValue"[^>]*>([0-9.]+)<', HTML)
    assert m, "confValue span lost its initial text, or the id changed"
    assert float(m.group(1)) == pytest.approx(DETECTOR_CONFIDENCE), (
        "the label is what a user reads as the current setting"
    )


def test_js_default_matches_the_shipped_gate():
    m = re.search(r"this\.confidenceThreshold\s*=\s*options\.confidenceThreshold\s*\?\?\s*([0-9.]+)", JS)
    assert m, "the JS confidenceThreshold default moved or was renamed"
    assert float(m.group(1)) == pytest.approx(DETECTOR_CONFIDENCE), (
        "unreachable on the demo path, but it is the default for anyone "
        "embedding the module directly"
    )


def test_slider_range_admits_the_shipped_gate():
    lo = float(re.search(r'id="confSlider"[^>]*\bmin="([0-9.]+)"', HTML).group(1))
    hi = float(re.search(r'id="confSlider"[^>]*\bmax="([0-9.]+)"', HTML).group(1))
    assert lo <= DETECTOR_CONFIDENCE <= hi


def test_the_demo_still_builds_the_pipeline_from_the_slider():
    # If this stops being true the tests above stop being load-bearing, and
    # whoever changes it should decide deliberately which source wins.
    assert "confidenceThreshold: parseFloat(confSlider.value)" in HTML, (
        "the demo no longer loads its gate from the slider attribute -- "
        "re-point these tests at whatever is authoritative now"
    )
