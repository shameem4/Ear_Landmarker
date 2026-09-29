"""Anthropometric ear measurements taken from fitted contours, not point indices.

Why this exists: 72% of the model's squared landmark error is TANGENTIAL -- points
sitting at the wrong place *along* a contour rather than off it. An index-based
measurement such as distance(point_3, point_16) inherits that error directly,
because if point 3 slides along the helix the measured length changes even though
the model traced the ear correctly.

Contour-based measurements are invariant to that sliding. The caliper length of a
polyline does not care how its vertices are distributed along it, so the dominant
error term drops out. The trade-off is that these measure the *shape*, so they
cannot reproduce a landmark-specific clinical definition that depends on an exact
anatomical point.

All functions take landmarks as (55, 2) in any consistent unit (normalized or
pixels) and return measurements in that same unit. Multiply by a mm-per-pixel
scale afterwards.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "LINESTRIPS",
    "resample_polyline",
    "caliper_length",
    "width_perpendicular_to",
    "measure_ear",
]

# (name, start, end) -- end exclusive, matching the 55-point layout
LINESTRIPS = {
    "helix": (0, 20),
    "antihelix": (20, 35),
    "concha": (35, 50),
    "tragus": (50, 55),
}


def resample_polyline(points: np.ndarray, n: int = 200) -> np.ndarray:
    """Resample a polyline to `n` points spaced uniformly by arc length.

    This is what makes the downstream measurements independent of how the input
    vertices happened to be distributed.

    Args:
        points: (k, 2) polyline vertices in order.
        n: Number of output samples.

    Returns:
        (n, 2) resampled polyline.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return np.repeat(pts, n, axis=0) if len(pts) else np.zeros((n, 2))

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total < 1e-12:                      # degenerate: all vertices coincident
        return np.repeat(pts[:1], n, axis=0)

    target = np.linspace(0.0, total, n)
    return np.stack([
        np.interp(target, cum, pts[:, 0]),
        np.interp(target, cum, pts[:, 1]),
    ], axis=1)


def caliper_length(points: np.ndarray) -> tuple[float, np.ndarray]:
    """Maximum distance between any two points (the caliper diameter).

    Returns:
        (length, unit_axis) where unit_axis is the direction of that diameter.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return 0.0, np.array([1.0, 0.0])

    # n is small after resampling, so the exact O(n^2) diameter is fine and
    # avoids the edge cases of a convex-hull rotating-calipers implementation.
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    i, j = np.unravel_index(np.argmax(d), d.shape)
    axis = pts[j] - pts[i]
    n = np.linalg.norm(axis)
    return float(d[i, j]), (axis / n if n > 1e-12 else np.array([1.0, 0.0]))


def width_perpendicular_to(points: np.ndarray, axis: np.ndarray) -> float:
    """Extent of `points` measured perpendicular to `axis`."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    perp = np.array([-axis[1], axis[0]], dtype=np.float64)
    proj = pts @ perp
    return float(proj.max() - proj.min())


def measure_ear(landmarks: np.ndarray, n_resample: int = 200) -> dict[str, float]:
    """Contour-based ear measurements.

    Args:
        landmarks: (55, 2) landmark coordinates.
        n_resample: Arc-length samples per contour.

    Returns:
        Dict with ear_length, ear_width, concha_height, concha_width,
        tragus_to_antitragus -- in the same units as the input.
    """
    lm = np.asarray(landmarks, dtype=np.float64).reshape(-1, 2)
    strips = {
        name: resample_polyline(lm[a:b], n_resample)
        for name, (a, b) in LINESTRIPS.items()
    }

    # Ear length/width from the outer rim, along its own principal axis, so the
    # result does not depend on how the head was rotated in frame.
    ear_length, axis = caliper_length(strips["helix"])
    ear_width = width_perpendicular_to(strips["helix"], axis)

    # Concha measured on its own axis rather than the ear's, since the concha
    # bowl is not generally aligned with the ear's long axis.
    concha_height, c_axis = caliper_length(strips["concha"])
    concha_width = width_perpendicular_to(strips["concha"], c_axis)

    # Tragus-to-antitragus: the tragus strip spans the intertragic notch, so its
    # caliper diameter is the span between the two tragal prominences.
    tragus_span, _ = caliper_length(strips["tragus"])

    return {
        "ear_length": ear_length,
        "ear_width": ear_width,
        "concha_height": concha_height,
        "concha_width": concha_width,
        "tragus_to_antitragus": tragus_span,
    }


def measure_ear_indexed(landmarks: np.ndarray) -> dict[str, float]:
    """The index-based measurements, kept for comparison only.

    These are what the original pipeline used (length from points 3-16, width
    from points 0-7). They are directly exposed to tangential drift; see
    tests/test_measure.py for the quantified difference.
    """
    lm = np.asarray(landmarks, dtype=np.float64).reshape(-1, 2)
    return {
        "ear_length": float(np.linalg.norm(lm[3] - lm[16])),
        "ear_width": float(np.linalg.norm(lm[0] - lm[7])),
    }
