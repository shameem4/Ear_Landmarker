"""Label which vertices are PINNA and which are the scalp behind it.

WHY A LABEL AND NOT A CROPPED TARGET. Two earlier attempts both failed:
  - ray-cast the tight detector box only -> the box truncates the pinna, so rim
    rays miss entirely (direct-hit rate 0.53-0.62) and snap onto the cut edge;
  - ray-cast a 1.5x expanded box -> the pinna is whole, but the region now
    contains scalp, so a ray passing just outside the ear hits the head behind
    and the landmark is pinned ~1 ear-depth too deep.
First-hit ray-casting is CORRECT wherever the pinna exists -- it returns the
frontmost surface. The problem is only knowing whether what was hit IS the
pinna. So keep the generous target, and label each vertex instead.

HOW. In the pinna frame (+Z toward the camera) the scalp is a smooth, slowly
varying surface and the ear stands proud of it. A quadratic is fitted to the
OUTER ANNULUS of the region -- which is scalp, since the ear sits in the middle
-- and extrapolated inward to predict where the head would be with no ear on it.
Anything standing far enough in front of that prediction is pinna.
"""

from __future__ import annotations

import numpy as np


def fit_scalp(P, inner_frac=0.55):
    """Quadratic z = f(x, y) fitted to the outer ring, where there is no ear."""
    xy = P[:, :2]
    r = np.linalg.norm(xy - np.median(xy, axis=0), axis=1)
    outer = r > np.quantile(r, inner_frac)
    if outer.sum() < 80:
        return None
    x, y, z = P[outer, 0], P[outer, 1], P[outer, 2]
    A = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=1)
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    return coef


def pinna_mask(P, standoff_frac=0.03, inner_frac=0.55):
    """Boolean mask over P: True where the surface stands proud of the scalp fit.

    `standoff_frac` is in units of the region's lateral extent. The first value
    tried, 0.12, was about 4x too high and labelled 0-8% of the region as pinna
    across subjects where the true share is 20-35%. The measured rise/extent
    distribution puts the scalp median near -0.01 and the p90 near +0.08, so the
    boundary between "on the head" and "standing out of it" sits near 0.03.
    """
    coef = fit_scalp(P, inner_frac)
    if coef is None:
        return None, None
    x, y = P[:, 0], P[:, 1]
    A = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=1)
    z_scalp = A @ coef
    rise = P[:, 2] - z_scalp
    extent = max(np.ptp(P[:, 0]), np.ptp(P[:, 1]))
    return rise > standoff_frac * extent, rise
