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
    "IBUG_REGIONS",
    "LINESTRIPS",
    "resample_polyline",
    "caliper_length",
    "width_perpendicular_to",
    "measure_ear",
]

# The authoritative semantics of the 55 points, from the iBUG ear annotation
# scheme these sources are labelled with (Zhou & Zaferiou, "Deformable Models of
# Ears in-the-wild", FG 2017). End exclusive.
#
# Earlier versions of this file named the four drawing strips helix / antihelix /
# concha / tragus. Three of those four were wrong: "tragus" was applied to the
# superior crus, and the real tragus sits inside the strip that was called
# "concha". Anything measuring by those names measured the wrong structure.
IBUG_REGIONS = {
    "ascending_helix": (0, 4),
    "descending_helix": (4, 8),
    "helix": (8, 14),
    "lobe": (14, 20),
    "ascending_inner_helix": (20, 25),
    "descending_inner_helix": (25, 29),
    "inner_helix": (29, 35),
    "tragus": (35, 39),
    "canal": (39, 40),
    "antitragus": (40, 43),
    "concha": (43, 47),
    "inferior_crus": (47, 50),
    "superior_crus": (50, 55),
}

# The four connected polylines, used for drawing, contour losses and smoothness
# checks. These groupings are correct as *contours* -- each is a continuous
# chain -- so only their names needed fixing. Each spans several iBUG regions,
# which is why they are named for what they trace rather than for one structure.
LINESTRIPS = {
    "outer_helix": (0, 20),    # ascending + descending helix, helix, lobe
    "inner_helix": (20, 35),   # ascending + descending inner helix, inner helix
    "concha_border": (35, 50), # tragus, canal, antitragus, concha, inferior crus
    "superior_crus": (50, 55),
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

    def region(name: str) -> np.ndarray:
        a, b = IBUG_REGIONS[name]
        return lm[a:b]

    # Ear length/width from the outer rim, along its own principal axis, so the
    # result does not depend on how the head was rotated in frame.
    ear_length, axis = caliper_length(strips["outer_helix"])
    ear_width = width_perpendicular_to(strips["outer_helix"], axis)

    # Concha measured on its own axis rather than the ear's, since the concha
    # bowl is not generally aligned with the ear's long axis. This uses the
    # actual concha points (43-46), not the whole 35-49 strip, which also spans
    # the tragus, canal, antitragus and inferior crus.
    concha = resample_polyline(region("concha"), n_resample)
    concha_height, c_axis = caliper_length(concha)
    concha_width = width_perpendicular_to(concha, c_axis)

    # Tragus to antitragus: the span across the intertragic notch, measured
    # between the two structures themselves rather than along a strip that
    # happens to contain them. Taken as the maximum separation between the two
    # point sets, which is the intertragic width in the usual sense.
    tr, at = region("tragus"), region("antitragus")
    tragus_span = float(np.linalg.norm(tr[:, None, :] - at[None, :, :], axis=-1).max())

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
