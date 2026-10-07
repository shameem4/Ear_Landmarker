"""The 55-point ear scheme, and the MediaPipe indices used to find the head.

Constants only. Everything that consumes them takes them from here, so a change
to the point convention cannot land in one module and not another."""

from __future__ import annotations



# 55-point iBUG ear scheme, four ordered linestrips.
STRIPS = [(0, 20), (20, 35), (35, 50), (50, 55)]


STRIP_NAMES = ["outer_helix", "inner_helix", "concha_border", "superior_crus"]


STRIP_COLOURS = [(0.95, 0.25, 0.25), (0.25, 0.65, 0.95),
                 (0.30, 0.85, 0.35), (0.98, 0.80, 0.15)]


# The scheme's OWN 13 groups, per ibug.doc.ic.ac.uk/resources/ibug-ears. The
# four STRIPS above are a coarser grouping used for drawing and for chain
# statistics; these are what the points actually mean, and they are what a
# report should name. Two consequences worth knowing:
#   - a large turn at a group BOUNDARY (3->4, 13->14, 38->39, ...) is the chain
#     jumping between features, not a misplaced point
#   - point 39 is the EAR CANAL, a single isolated point. It sits off the
#     tragus-to-antitragus curve, so any statistic treating 35..49 as one
#     continuous chain passes through it. That stays self-consistent only
#     because the same statistic is computed the same way on the corpus.
GROUPS = [(0, 4), (4, 8), (8, 14), (14, 20), (20, 25), (25, 29), (29, 35),
          (35, 39), (39, 40), (40, 43), (43, 47), (47, 50), (50, 55)]

GROUP_NAMES = ["ascending_helix", "descending_helix", "helix", "ear_lobe",
               "ascending_inner_helix", "descending_inner_helix", "inner_helix",
               "tragus", "canal", "antitragus", "concha", "inferior_crus",
               "superior_crus"]


def group_of(j):
    """The scheme's own name for point j."""
    return next(GROUP_NAMES[i] for i, (lo, hi) in enumerate(GROUPS) if lo <= j < hi)


# MediaPipe FaceMesh canonical indices.
TRAGION_R, TRAGION_L, FOREHEAD, CHIN = 234, 454, 10, 152


NOSE_TIP, NASION = 1, 168
