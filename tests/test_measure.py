"""Contour-based measurement correctness and its robustness claim.

The robustness test is the point of the module: contour measurements must be
less sensitive to points sliding along the contour than index-based ones, since
72% of the model's squared error is exactly that sliding.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.canonicalize import resample_uniform  # noqa: E402
from model.measure import (  # noqa: E402
    LINESTRIPS,
    caliper_length,
    measure_ear,
    measure_ear_indexed,
    resample_polyline,
    width_perpendicular_to,
)


def _synthetic_ear(rng) -> np.ndarray:
    """55 points laid out as four closed-ish arcs, roughly ear shaped."""
    lm = np.zeros((55, 2))
    for name, (a, b) in LINESTRIPS.items():
        n = b - a
        t = np.linspace(0.2, 0.8, n) * 2 * np.pi
        rx, ry = rng.uniform(0.15, 0.3), rng.uniform(0.25, 0.4)
        cx, cy = rng.uniform(0.4, 0.6), rng.uniform(0.4, 0.6)
        lm[a:b, 0] = cx + rx * np.cos(t)
        lm[a:b, 1] = cy + ry * np.sin(t)
    return lm


# --- resampling -------------------------------------------------------------

def test_resample_preserves_endpoints():
    pts = np.array([[0.0, 0.0], [0.3, 0.1], [0.7, 0.6], [1.0, 1.0]])
    out = resample_polyline(pts, 50)
    assert np.allclose(out[0], pts[0]) and np.allclose(out[-1], pts[-1])


def test_resample_gives_uniform_spacing():
    pts = np.array([[0.0, 0.0], [0.05, 0.0], [0.1, 0.0], [1.0, 0.0]])  # very uneven
    out = resample_polyline(pts, 40)
    seg = np.linalg.norm(np.diff(out, axis=0), axis=1)
    assert seg.std() / seg.mean() < 1e-6


def test_resample_handles_degenerate_input():
    assert resample_polyline(np.zeros((1, 2)), 10).shape == (10, 2)
    assert resample_polyline(np.zeros((5, 2)), 10).shape == (10, 2)   # all coincident


def test_resample_is_near_idempotent_and_only_cuts_corners():
    """Resampling twice is NOT exactly idempotent, and the reason matters.

    Uniform samples on a polyline generally miss its corner vertices, so the
    resampled path cuts those corners slightly. Re-resampling that path gives
    slightly different points. The drift is bounded and shrinks with sample
    count -- which is why data/canonicalize.py interpolates with a spline rather
    than resampling the raw polyline.
    """
    pts = np.array([[0.0, 0.0], [0.2, 0.5], [0.9, 0.3], [1.0, 1.0]])
    coarse = np.abs(resample_polyline(pts, 20) - resample_polyline(resample_polyline(pts, 20), 20)).max()
    fine = np.abs(resample_polyline(pts, 400) - resample_polyline(resample_polyline(pts, 400), 400)).max()
    assert coarse < 0.05, "corner cutting is larger than expected"
    assert fine < coarse, "denser sampling should reduce the corner-cutting drift"


# --- geometry ---------------------------------------------------------------

def test_caliper_length_on_known_shape():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    length, axis = caliper_length(pts)
    assert length == pytest.approx(np.sqrt(2.0), rel=1e-9)
    assert np.allclose(np.abs(axis), np.array([1, 1]) / np.sqrt(2), atol=1e-9)


def test_width_perpendicular_to_axis():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.4]])
    assert width_perpendicular_to(pts, np.array([1.0, 0.0])) == pytest.approx(0.4)


def test_measurements_are_rotation_invariant():
    """An ear measured at any in-plane angle must give the same numbers."""
    rng = np.random.default_rng(0)
    lm = _synthetic_ear(rng)
    base = measure_ear(lm)
    for deg in (15, 45, 90, 180):
        th = np.radians(deg)
        r = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        rot = (lm - 0.5) @ r.T + 0.5
        got = measure_ear(rot)
        for k in base:
            assert got[k] == pytest.approx(base[k], rel=1e-6), f"{k} changed under {deg} deg"


def test_measurements_scale_linearly():
    rng = np.random.default_rng(1)
    lm = _synthetic_ear(rng)
    base = measure_ear(lm)
    scaled = measure_ear((lm - 0.5) * 2.0 + 0.5)
    for k in base:
        assert scaled[k] == pytest.approx(base[k] * 2.0, rel=1e-6)


# --- the robustness claim ---------------------------------------------------

def _slide_along_contour(lm, px, rng, size=192):
    """Move each point along its local contour tangent -- pure tangential error."""
    out = lm.copy()
    for a, b in LINESTRIPS.values():
        for i in range(a, b):
            j0, j1 = max(a, i - 1), min(b - 1, i + 1)
            t = lm[j1] - lm[j0]
            n = np.linalg.norm(t)
            if n < 1e-9:
                continue
            out[i] = lm[i] + (t / n) * rng.normal(0, px / size)
    return out


def test_contour_measurement_beats_index_under_sliding():
    """The module's reason for existing, asserted numerically."""
    rng = np.random.default_rng(3)
    c_err, i_err = [], []
    for k in range(60):
        lm = _synthetic_ear(np.random.default_rng(k))
        pert = _slide_along_contour(lm, 4.4, rng)      # the model's measured drift
        bc, bi = measure_ear(lm), measure_ear_indexed(lm)
        pc, pi = measure_ear(pert), measure_ear_indexed(pert)
        if bc["ear_length"] > 1e-6:
            c_err.append(abs(pc["ear_length"] - bc["ear_length"]) / bc["ear_length"])
        if bi["ear_length"] > 1e-6:
            i_err.append(abs(pi["ear_length"] - bi["ear_length"]) / bi["ear_length"])
    assert np.mean(c_err) < np.mean(i_err), (
        f"contour ({np.mean(c_err)*100:.2f}%) did not beat index "
        f"({np.mean(i_err)*100:.2f}%) under tangential sliding"
    )


def test_canonicalization_makes_spacing_uniform():
    rng = np.random.default_rng(5)
    lm = _synthetic_ear(rng)
    for a, b in LINESTRIPS.values():
        canon = resample_uniform(lm[a:b])
        assert np.allclose(canon[0], lm[a]) and np.allclose(canon[-1], lm[b - 1])
        seg = np.linalg.norm(np.diff(canon, axis=0), axis=1)
        assert seg.std() / seg.mean() < 0.12, "spacing is still uneven after canonicalization"


def test_canonicalization_keeps_vertices_on_the_original_curve():
    """The honest invariant.

    Canonicalization cannot leave the *polyline* identical -- vertices move along
    it, and a polyline through different vertices of a curved path is a slightly
    different piecewise-linear approximation. What it must not do is move points
    OFF the original curve. Measured on real data this is ~1.8px at 192, mostly
    representation artefact rather than true distortion; anything much larger
    would mean canonicalization is inventing shape.
    """
    rng = np.random.default_rng(5)
    lm = _synthetic_ear(rng)
    for a, b in LINESTRIPS.values():
        strip = lm[a:b]
        canon = resample_uniform(strip)
        dense = resample_polyline(strip, 2000)
        d = np.linalg.norm(canon[:, None, :] - dense[None, :, :], axis=-1).min(axis=1)
        assert d.max() < 0.02, (
            f"canonical vertices sit {d.max():.4f} off the original curve"
        )


def test_canonicalization_handles_duplicate_vertices():
    """Coincident vertices make the arc-length parameter non-monotonic, which
    the spline interpolator rejects outright. Four real strips hit this."""
    strip = np.array([[0.1, 0.1], [0.2, 0.2], [0.2, 0.2], [0.4, 0.5], [0.6, 0.6]])
    out = resample_uniform(strip, spline=True)
    assert out.shape == strip.shape and np.isfinite(out).all()
