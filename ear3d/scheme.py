"""The 55-point ear scheme, and the MediaPipe indices used to find the head.

Constants only. Everything that consumes them takes them from here, so a change
to the point convention cannot land in one module and not another."""

from __future__ import annotations



# 55-point iBUG ear scheme, four ordered linestrips.
STRIPS = [(0, 20), (20, 35), (35, 50), (50, 55)]


STRIP_NAMES = ["outer_helix", "inner_helix", "concha_border", "superior_crus"]


STRIP_COLOURS = [(0.95, 0.25, 0.25), (0.25, 0.65, 0.95),
                 (0.30, 0.85, 0.35), (0.98, 0.80, 0.15)]


# MediaPipe FaceMesh canonical indices.
TRAGION_R, TRAGION_L, FOREHEAD, CHIN = 234, 454, 10, 152


NOSE_TIP, NASION = 1, 168
