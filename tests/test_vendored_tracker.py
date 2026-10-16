"""The vendored tracker must stay byte-identical to BlazeEar's.

Both files are maintained there, pinned to each other by a behaviour fixture,
and their constants were measured there. A local edit would silently unpin the
two languages, so this fails loudly instead.
"""
import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BLAZEEAR = ROOT.parent / "BlazeEar"

# EVERY file vendored from BlazeEar, not just the tracker. Add a new one here
# when vendoring it: this list is the pin, so a file missing from it is a file
# free to drift -- which is exactly how `blazeear_inference.js` fell a commit
# behind unnoticed while the two pinned files held.
#
# Deliberately an explicit list rather than a matcher over shared basenames.
# That was tried and over-reported: this repo's own index.html and README.md
# share a name with BlazeEar's without being copies of them.
PAIRS = [
    (ROOT / "vendor/box_tracker.py", BLAZEEAR / "utils/box_tracker.py"),
    (ROOT / "docs/ear_tracker.js", BLAZEEAR / "docs/ear_tracker.js"),
    (ROOT / "docs/blazeear_inference.js", BLAZEEAR / "docs/blazeear_inference.js"),
    (ROOT / "docs/BlazeEar_web.onnx", BLAZEEAR / "docs/BlazeEar_web.onnx"),
    (ROOT / "docs/BlazeFace_web.onnx", BLAZEEAR / "docs/BlazeFace_web.onnx"),
]


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.mark.parametrize("ours,theirs", PAIRS, ids=lambda p: p.name)
def test_vendored_copy_is_byte_identical(ours, theirs):
    assert ours.exists(), f"{ours} is missing"
    if not theirs.exists():
        pytest.skip(f"{theirs} not checked out beside this repo")
    assert _sha(ours) == _sha(theirs), (
        f"{ours.name} has diverged from BlazeEar's copy. Fix it THERE and "
        "re-copy both files together -- copying one alone breaks the "
        "cross-language parity the fixture exists to protect."
    )


def test_vendored_js_is_not_an_es_module():
    # A bare `export` would break BlazeEar's require()-based parity test, which
    # fails silently rather than loudly -- so it is pinned from this side too.
    src = (ROOT / "docs/ear_tracker.js").read_text()
    assert not any(l.startswith("export") for l in src.splitlines()), (
        "ear_tracker.js must stay a classic script; import it for side effects "
        "and read BoxTracker off globalThis"
    )
