"""Landmark group names must match the iBUG ear annotation scheme.

These sources (collectionA/B .pts, AudioEar LabelMe) carry the 55-point iBUG
layout from Zhou & Zaferiou, "Deformable Models of Ears in-the-wild" (FG 2017):

    ascending helix 0-3, descending helix 4-7, helix 8-13, ear lobe 14-19,
    ascending inner helix 20-24, descending inner helix 25-28,
    inner helix 29-34, tragus 35-38, canal 39, antitragus 40-42,
    concha 43-46, inferior crus 47-49, superior crus 50-54

v1 invented its own names for the four drawing strips and three of the four were
wrong -- most consequentially, "tragus" was applied to 50-54, which is the
superior crus, while the real tragus sits inside the strip that was called
"concha". Measurements taken by those names measured the wrong structure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.measure import IBUG_REGIONS, LINESTRIPS, measure_ear  # noqa: E402


def test_ibug_regions_tile_all_55_points_without_gaps_or_overlap():
    spans = sorted(IBUG_REGIONS.values())
    assert spans[0][0] == 0 and spans[-1][1] == 55
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert end == start, f"gap or overlap at {end} vs {start}"


def test_tragus_is_35_to_38_not_50_to_54():
    """The specific error this file exists to prevent."""
    assert IBUG_REGIONS["tragus"] == (35, 39)
    assert IBUG_REGIONS["superior_crus"] == (50, 55)


def test_no_strip_is_named_for_a_structure_it_does_not_span():
    """A strip may span several regions, but its name must not claim a
    structure that lies outside it."""
    for name, (a, b) in LINESTRIPS.items():
        if name in IBUG_REGIONS:
            ra, rb = IBUG_REGIONS[name]
            assert a <= ra and rb <= b, (
                f"strip {name!r} spans {a}-{b-1} but the iBUG region of that "
                f"name is {ra}-{rb-1}")


def test_old_wrong_names_are_gone():
    assert "tragus" not in LINESTRIPS, "50-54 is the superior crus, not the tragus"
    assert "antihelix" not in LINESTRIPS, "20-34 is the inner helix in iBUG terms"
    assert "concha" not in LINESTRIPS, (
        "35-49 spans tragus, canal, antitragus, concha and inferior crus")


def test_tragus_to_antitragus_uses_the_real_tragus_and_antitragus():
    """Built so the answer is known: tragus and antitragus points are placed a
    known distance apart, and the superior crus is put somewhere absurd. A
    measurement reading the superior crus would return the absurd number.
    """
    lm = np.zeros((55, 2), dtype=np.float64)
    lm[:, 0] = np.linspace(0, 1, 55)      # generic filler
    lm[:, 1] = 0.5

    ta, tb = IBUG_REGIONS["tragus"]
    aa, ab = IBUG_REGIONS["antitragus"]
    lm[ta:tb] = [0.50, 0.50]
    lm[aa:ab] = [0.53, 0.50]              # 0.03 away

    sa, sb = IBUG_REGIONS["superior_crus"]
    lm[sa:sb] = [[0.0, 0.0], [9.0, 9.0], [0.0, 9.0], [9.0, 0.0], [4.5, 4.5]]

    m = measure_ear(lm)
    assert m["tragus_to_antitragus"] == pytest.approx(0.03, abs=1e-6)
    assert m["tragus_to_antitragus"] < 1.0, "measurement is reading the superior crus"


def test_concha_measurement_uses_concha_points_only():
    """The 35-49 strip is five structures wide; measuring the concha from all of
    it overstates it. Blow up a non-concha part of the strip and the concha
    numbers must not move."""
    rng = np.random.default_rng(0)
    lm = rng.random((55, 2)) * 0.1 + 0.45
    base = measure_ear(lm)

    moved = lm.copy()
    ca, cb = IBUG_REGIONS["canal"]
    moved[ca:cb] = [5.0, 5.0]             # far outside, but not a concha point
    after = measure_ear(moved)

    assert after["concha_height"] == pytest.approx(base["concha_height"])
    assert after["concha_width"] == pytest.approx(base["concha_width"])


def test_tragus_span_is_invariant_to_vertex_sliding():
    """measure.py exists to be index-free, so this measurement must not change
    when points slide ALONG their own contour.

    The first version of this fix took the max over the raw point sets, which
    reintroduced exactly the tangential sensitivity the module removes. The
    earlier naming test used degenerate input (all tragus points identical) and
    so could not have caught it.
    """
    lm = np.zeros((55, 2), dtype=np.float64)
    lm[:, 0] = np.linspace(0, 1, 55)
    lm[:, 1] = 0.5

    ta, tb = IBUG_REGIONS["tragus"]
    aa, ab = IBUG_REGIONS["antitragus"]
    # tragus as a slanted segment, antitragus as another -- genuine extent
    lm[ta:tb] = np.stack([np.linspace(0.50, 0.54, tb - ta),
                          np.linspace(0.50, 0.58, tb - ta)], 1)
    lm[aa:ab] = np.stack([np.linspace(0.60, 0.64, ab - aa),
                          np.linspace(0.50, 0.56, ab - aa)], 1)
    base = measure_ear(lm)["tragus_to_antitragus"]

    # same two contours, vertices redistributed along them (endpoints fixed)
    slid = lm.copy()
    slid[ta:tb] = np.stack([np.linspace(0.50, 0.54, tb - ta) ** 1.0,
                            np.linspace(0.50, 0.58, tb - ta)], 1)
    t0, t1 = lm[ta], lm[tb - 1]
    frac = np.array([0.0, 0.7, 0.85, 1.0])[: tb - ta]
    slid[ta:tb] = t0 + frac[:, None] * (t1 - t0)
    after = measure_ear(slid)["tragus_to_antitragus"]

    assert after == pytest.approx(base, rel=0.02), (
        "measurement moved when vertices slid along an unchanged contour")
