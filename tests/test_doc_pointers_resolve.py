"""Every file path the docs cite must resolve.

A pointer whose destination moved is the quiet failure mode: nothing in either
file is wrong, only the relation between them, so no review of either one finds
it. This repo has produced the defect twice -- three scripts that moved to
Ear_3D_Lifter in the 2D/3D split were still cited as `scripts/...` three lines
below a note saying where they had gone, and a memory note pointed at a data
directory that no longer existed.

Scoped deliberately to refs containing a `/`, which unambiguously claim to be
a path. A bare `dataset.py` in prose is a mention, not a pointer, and treating
it as one made the first two versions of this test report 20+ false positives
between them -- the lesson being to scope an audit to the unambiguous case and
widen only if it is shown to miss something real, because an audit that cries
wolf on its first run is one nobody runs twice.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ["README.md", "RESULTS.md", "vendor/README.md"]
PAT = re.compile(r"`((?:\.\./)?[A-Za-z0-9_./-]+\.(?:py|js|md|onnx|csv|json|html))`")


def _candidates(ref, doc):
    """Where a cited path may legitimately live.

    Resolved relative to the DOCUMENT's own directory first -- vendor/README.md
    cites `../docs/...` and bare upstream names, and a first version of this
    test resolved everything against the repo root and reported all eight as
    dangling. An audit that cries wolf on its first run is one nobody runs
    twice.
    """
    here = (ROOT / doc).parent
    return [here / ref, ROOT / ref, ROOT.parent / ref,
            ROOT / "scripts" / Path(ref).name,
            ROOT.parent / "BlazeEar" / ref,
            ROOT.parent / "BlazeEar" / "docs" / Path(ref).name,
            ROOT.parent / "BlazeEar" / "utils" / Path(ref).name]


@pytest.mark.parametrize("doc", DOCS)
def test_cited_paths_resolve(doc):
    text = (ROOT / doc).read_text()
    missing = sorted({
        ref for ref in PAT.findall(text)
        if "/" in ref and "*" not in ref
        and not any(c.exists() for c in _candidates(ref, doc))
    })
    assert not missing, (
        f"{doc} cites paths that do not resolve: {missing}. If the file moved, "
        "repoint every mention -- not just the one nearest the sentence that "
        "explains the move."
    )
