"""Shared defaults for the 3D path.

Every knob that affects geometry, appearance or the snaps lives here, so the
viewer and ingest cannot be running at different settings while appearing to
share code. Callers copy it -- dict(DEFAULTS, **overrides) -- and pass the
result down; no module reads it directly.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Third-party research data; not in any repo. Override with EAR3D_DIR.
EAR3D_DIR = Path(os.environ.get("EAR3D_DIR", ROOT.parent / "clean_3d_data"))

DEFAULTS = dict(
    # --- what to load -------------------------------------------------------
    mesh=str(EAR3D_DIR / "3D head meshes" / "pp12_3DheadMesh.ply"),
    ear=0,                  # which detected ear: 0 = highest confidence, 1 = other side

    # --- model --------------------------------------------------------------
    run="manual_occ_s42",   # checkpoint under runs/checkpoints/<run>/
    device="cuda",

    # --- head pose ----------------------------------------------------------
    use_mediapipe=True,     # derive the face-on direction from a FaceMesh head
                            # frame instead of the ear detector's 6-view sweep
    face_task=os.environ.get(
        "FACE_LANDMARKER_TASK",
        "/mnt/14BE47C2BE479ADE/Code/landmarking_stuff/landmarker/face_landmarker.task"),

    # --- appearance ---------------------------------------------------------
    skin=True,              # colour the mesh with scripts/skin.py before rendering
    tone=10,                # tone index; None draws a random Fitzpatrick I-VI.
                            # 10 is deeper than real skin and is there because the
                            # landmarker reads the ear most cleanly on it -- see
                            # EXTRA_TONES in scripts/skin.py. Fine here, where the
                            # point is to inspect landmarks; ingest renders its
                            # training images with a realistic tone instead.
    skin_seed=0,            # also drives the blotching and grain
    ao_strength=0.75,       # how hard ambient occlusion darkens cavities
    ao_rays=24,             # rays per vertex. The cost of apply_skin is all here.

    # --- camera / framing (these mirror the training pipeline) --------------
    size=600,               # full render resolution before cropping
    eye_z=3.0,              # camera at (0, 0, eye_z) looking at the origin
    vfov=50.0,              # vertical field of view, degrees

    # NOTE there is no crop setting here. The WHOLE render goes to
    # EarLandmarkerPipeline, which runs the shipped detect -> ROI -> refine ->
    # landmark path and hands back landmarks in full-frame pixels. That path
    # already drives the ROI toward TRAIN_OCCUPANCY (ROI_OCC_TOL/ROI_SATURATED),
    # so framing and occupancy are ITS business, not this script's -- which also
    # means this viewer now shows what inference actually does, rather than what
    # a hand-rolled crop here happened to do.

    # --- back-projection ----------------------------------------------------
    snap=True,              # place landmarks that sit behind a depth cliff onto
                            # the near lip of that cliff. See snap_to_cliff().
    snap_radius=0.08,       # JITTER RANGE: the search window, as a fraction of
                            # ear extent -- ~35 px at the usual framing, a 71x71
                            # window. This is the whole safety bound: the further
                            # it reaches, the more a "snap" becomes a relocation
                            # of a landmark the model put in the wrong place.
                            # 8% is the knee. Over four heads, every one of the 13
                            # landmarks flagged bad is already fixed there, and
                            # going further fixes none and moves steadily more
                            # points that were never flagged:
                            #   radius   fixed   disturbed   moved >0.15
                            #      5%      11        13           10
                            #      8%      13        16           13
                            #     10%      13        20           17
                            #     15%      13        24           21
                            #     20%      13        31           28
                            # Note the flag behind "fixed" only catches
                            # z < median - 0.30, so it is blind to a landmark on
                            # the wrong anatomy at roughly the right depth. It
                            # cannot separate these radii on placement; nothing
                            # here can yet.
    # --- triangulation (--triangulate) --------------------------------------
    triangulate=False,      # build each landmark by intersecting the rays from
                            # several views instead of lifting one view through
                            # its depth buffer. Off by default only because it
                            # costs a render and a pipeline call per view.
                            #
                            # VIEWS ARE SYNTHETIC, SO THEY ARE CHEAP, AND MORE OF
                            # THEM HELP. Reprojection into a FIXED held-out set of
                            # 8 views, so only the fit changes with N (px, ear
                            # ~430 px, 4 heads):
                            #     views      4     8    15    30    60   100   190
                            #     ray fit 18.69 12.72 11.34 10.68  9.97  9.52  8.89
                            #     face-on 13.46 (constant, by construction)
                            # Below ~8 views the fit is WORSE than the single
                            # face-on lift -- too few rays across a narrow
                            # baseline. It passes face-on at 8 and keeps improving
                            # with diminishing returns, 34% better at 190.
    tri_angles=((0, 0), (-18, 0), (18, 0), (-10, -10), (10, 10), (0, -15), (0, 15)),
                            # 7 views inside the 20 deg optimum below.
    tri_cone=20.0,          # HALF-ANGLE OF THE VIEW CONE, and it has an optimum.
                            # Reprojection into a FIXED held-out set, so only the
                            # fit changes (24 views, 3 heads; face-on is 7.24 px
                            # throughout, as it must be):
                            #   cone    triangulated   gain over face-on
                            #    10d        5.00            +31.0%
                            #    15d        4.58            +36.7%
                            #    20d        4.58            +36.8%
                            #    25d        4.72            +34.8%
                            #    30d        5.21            +28.1%
                            #    40d        6.13            +15.3%
                            # Narrower than 15 and there is too little baseline to
                            # pin depth; wider than 25 and the detector's 2D output
                            # degrades faster than the geometry improves. 30 gives
                            # up a third of the available gain, 40 gives up half.
    tri_thresh=0.03,        # RANSAC inlier distance, as a fraction of ear extent
    tri_method="lsq",       # "lsq" over all rays, or "ransac".
                            #
                            # RANSAC NEVER WINS HERE, and it was expected to. It
                            # ties least squares at every view count from 4 to 190
                            # (within 0.1 px) and at both a 30 and a 60 degree
                            # cone. Widening the cone was the test of the obvious
                            # explanation -- that a narrow cone has no outliers to
                            # reject -- and it fails: at 60 deg the inlier
                            # fraction drops from 91% to 69% on pp12, so RANSAC is
                            # rejecting plenty, and still does not win.
                            #
                            # The likeliest reason is that these are not gross
                            # outliers. A landmark at an extreme angle drifts
                            # consistently rather than landing somewhere random, so
                            # the rays are BIASED, not wild. A vote cannot fix a
                            # bias: the rays it keeps are biased too, and it has
                            # thrown away information. Untested.
                            #
                            # Keep the cone near 30 deg regardless. At 60 deg every
                            # method is 3x worse and detection falls to 53-62 of 70
                            # views, because the ear starts occluding itself.

    # --- partial re-seat (--reseat) -----------------------------------------
    reseat=False,           # pull triangulated points that sit far off the drawn
                            # surface back onto it, along the viewing axis.
                            #
                            # A triangulated landmark is a free 3D point -- the
                            # rays place it and nothing requires it to lie on the
                            # ear. Measured over four heads the median lands on
                            # the surface (+0.33% of ear extent, 47% inside /
                            # 53% outside, so no systematic push) but the p90 is
                            # 22%, and those outliers cluster on the helix rim.
                            #
                            # THERE IS NO FREE LUNCH, which is why this is off:
                            #   tol    reproj px   off-surf p90   points moved
                            #   none      4.23        22.06%          0
                            #   0.20      5.32        10.53%          4.8
                            #   0.12      5.97         7.24%         10.0
                            #   0.08      6.21         6.40%         13.5
                            #   0.05      6.52         3.91%         21.5
                            #   0.03      6.76         2.35%         32.0
                            # Every threshold costs reprojection, monotonically,
                            # and even moving 5 points of 55 costs 26%. The two
                            # measures genuinely disagree: reprojection cannot see
                            # the surface prior, and the surface prior cannot see
                            # which pixel a label has to land in. Pick by use --
                            # training labels for rendered poses want reprojection
                            # (leave it off); 3D anatomy wants the surface.
    reseat_tol=0.15,        # move a point only if it is this far off, as a
                            # fraction of ear extent. "Only where large."

    # --- multi-view agreement (--multiview) ---------------------------------
    mv_angles=((0, 0), (-25, 0), (25, 0), (-12, -12), (12, 12), (0, -20), (0, 20)),
                            # yaw/pitch to re-landmark from. Kept inside +/-25
                            # because beyond that the ear starts occluding itself
                            # and spread stops measuring placement and starts
                            # measuring visibility.

    # CHAIN PASS OFF. It works -- it cuts link-length CV by a further 20%, from
    # 0.220 to 0.176 -- but link CV is its own objective, so that only says the
    # optimiser runs. Against multi-view agreement, which it does not optimise,
    # it buys nothing: median 3.96% -> 3.89%, >5% count 68 -> 65 of 220 (inside
    # noise), and p90 gets WORSE, 7.33% -> 8.54%. Per head it is inconsistent --
    # pp10 improves 33 -> 16 while pp12 19 -> 25, pp11 7 -> 13 and pp16 9 -> 11
    # all worsen; one win carries the average.
    #
    # The cliff snap, by contrast, earns its place on that same independent
    # metric: >5% count 90 -> 68 and p90 8.68% -> 7.33%, better on 3 of 4 heads.
    #
    # Left in and reachable with chain=True. n=4 heads, and agreement is
    # precision not accuracy, so this is no evidence of benefit rather than
    # proof of harm -- but nothing measurable says the moves it makes are right.
    chain=False,            # second pass: fix landmarks that break link spacing
    chain_tol=1.0,          # only reached when chain=True. Flag a point whose two
                            # links deviate from the strip median by this much in
                            # total, as a fraction of that median. A clean point
                            # scores ~0.2; displaced ones measured 1.5-1.7.
    chain_radius=0.08,      # search window for the replacement, as a fraction of
                            # ear extent -- same safety bound as the cliff snap.
    chain_passes=4,         # one point re-placed per pass, worst first, so a
                            # corrected neighbour informs the next decision.
    snap_step=0.30,         # how big a depth step counts as a cliff, in mesh
                            # units with the ear spanning ~1. Scalp-behind-pinna
                            # measures 0.85-1.08; concha bowl structure varies by
                            # ~0.3, so this sits between them. Lower (0.15) drags
                            # concha points forward; higher (0.50) fixes nothing.

    # --- display ------------------------------------------------------------
    sphere_frac=0.022,      # landmark sphere radius, in mesh units (ear spans ~1)
    show_2d=True,           # pop the landmark overlay in its own window
    show_rays=False,        # draw the camera ray to each landmark
    # these derive from third-party 3D data, so they default OUTSIDE the repo
    snapshot_png=str(Path(tempfile.gettempdir()) / "backprojection_snapshot.png"),
    overlay_png=str(Path(tempfile.gettempdir()) / "backprojection_overlay.png"),
    interactive=True,
)
