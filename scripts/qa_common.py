"""Shared pieces of the 2D QA checks.

The padding mask is here because getting it wrong silently corrupts every
ridge number: 45.6% of data/manual images carry more than 5% letterbox
padding, and a flat pad meets the photo at a perfectly straight, perfectly
sharp edge, which is the strongest ridge in the frame. Left in, the filters
trace the LETTERBOX, that line counts as "structure", and any landmark near
the frame scores well for being near nothing at all.
"""
import numpy as np
from scipy import ndimage


def padding_mask(gray, tol=2.0, flat=1.0):
    """Constant-valued letterbox bands that touch the frame.

    `gray` is a float image in [0,1]. A border row or column whose standard
    deviation is below `flat` (in 0-255 units) is treated as pad, and the
    connected band of pixels within `tol` of its value, reaching the frame,
    is returned. Dilated by one pixel so the pad/photo seam itself -- the
    edge the filters actually fire on -- is covered too.
    """
    g = np.asarray(gray, float) * 255.0
    h, w = g.shape
    m = np.zeros((h, w), bool)
    for line in (g[0], g[-1], g[:, 0], g[:, -1]):
        if line.std() < flat:
            m |= np.abs(g - line.mean()) < tol
    if not m.any():
        return m
    lab, _ = ndimage.label(m)
    touching = set(lab[0].tolist() + lab[-1].tolist()
                   + lab[:, 0].tolist() + lab[:, -1].tolist()) - {0}
    if not touching:
        return np.zeros((h, w), bool)
    m = np.isin(lab, list(touching))
    return ndimage.binary_dilation(m, iterations=2)


def structure_mask(val, valid, q):
    """Top (100-q)% of the response, over `valid` pixels only."""
    pool = val[valid]
    if pool.size < 50:
        return None
    hit = (val >= np.percentile(pool, q)) & valid
    return hit if hit.any() else None
