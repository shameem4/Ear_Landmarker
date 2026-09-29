"""Guard against NumPy 2 removals that only fail at runtime.

This environment runs NumPy 2.4. The removed spellings below still *parse* and
still pass py_compile and import, so they slip through everything except
actually executing the line -- which for inference code means the webcam path,
not the test suite. One of them (`arr.ptp()`) shipped in the ROI refinement loop
and was only caught by running the real pipeline on a fixture.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

# (pattern, what to use instead) -- method spellings removed in NumPy 2.0
REMOVED = [
    (r"\.ptp\(\)", "np.ptp(arr)"),
    (r"\bnp\.float_\b", "np.float64"),
    (r"\bnp\.complex_\b", "np.complex128"),
    (r"\bnp\.unicode_\b", "np.str_"),
    (r"\bnp\.alltrue\b", "np.all"),
    (r"\bnp\.sometrue\b", "np.any"),
    (r"\bnp\.cumproduct\b", "np.cumprod"),
    (r"\bnp\.product\b", "np.prod"),
    (r"\.newbyteorder\(\)", "arr.dtype.newbyteorder()"),
]

SOURCES = sorted(
    p for p in ROOT.rglob("*.py")
    if "/tests/" not in str(p) and ".git" not in p.parts and "__pycache__" not in p.parts
)


def test_numpy_is_version_2_or_later():
    """If this ever fails the guard below is moot -- revisit it."""
    assert int(np.__version__.split(".")[0]) >= 2


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_removed_numpy_spellings(path):
    text = path.read_text()
    for pattern, replacement in REMOVED:
        hits = [
            i + 1 for i, line in enumerate(text.splitlines())
            if re.search(pattern, line) and not line.lstrip().startswith("#")
        ]
        assert not hits, (
            f"{path.relative_to(ROOT)}:{hits} uses {pattern!r}, removed in NumPy 2 "
            f"-- use {replacement}")


def test_the_guard_actually_matches_the_bug_that_shipped():
    """Without this the parametrised test above could pass by matching nothing."""
    assert re.search(REMOVED[0][0], "extent = arr[:, 0].ptp()")
    assert not re.search(REMOVED[0][0], "extent = np.ptp(arr[:, 0])")
