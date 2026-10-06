"""Landmarking a rendered ear and lifting the result to 3D.

The whole frame goes to EarLandmarkerPipeline -- the shipped detect -> ROI ->
refine -> landmark path -- which returns points in full-frame pixels, the space
the Camera works in, so nothing here crops or maps coordinates itself."""

from __future__ import annotations

import numpy as np

from .backproject import backproject_snapped
from .camera import orbit_extrinsic, pose_angles
from .frames import frame_from, in_frame, plane_normal
from .render import render
from .scheme import STRIPS, STRIP_NAMES

def landmark_whole_frame(mesh, cfg, pipe):
    """Render the whole head and landmark it with the SHIPPED pipeline.

    No crop is made here. EarLandmarkerPipeline detects the ear, builds and
    refines its own ROI, runs the landmarker and maps the 55 points back into
    full-frame pixels -- exactly the space the returned Camera works in, so the
    result feeds the ray-cast directly with no mapping of ours in between.

    Returns (image, depth, Camera, landmarks_px, confidence); landmarks None if no ear.
    """
    img, depth, cam = render(mesh, cfg)
    res = pipe(img, timestamp=0.0)
    if not res:
        return img, depth, cam, None, 0.0
    # Several ears can be detected on a head render; take the most confident.
    best = max(res, key=lambda d: float(d["confidence"]))
    # NOTE this is the DETECTOR's box confidence, which is what the pipeline
    # returns. It is not the landmarker's per-point confidence and must not be
    # compared against it -- the pipeline does not expose that, because it owns
    # the crop. Reach into pipe.landmarker.predict(..., with_confidence=True) if
    # you need the per-point values back.
    return img, depth, cam, np.asarray(best["landmarks"], float), float(best["confidence"])


def label_two_pass(mesh, front, up, cfg, pipe, centre=None):
    """Face-on landmarks in 3D.

    Returns (mesh in the pinna frame, image, Camera, landmarks, P3, hit, conf).

    PASS 1 uses the detector's axis-aligned view direction purely to get a usable
    face-on render, and back-projects its landmarks.
    PASS 2 fits a plane to those 55 points and repeats in that frame. Both passes
    ray-cast against the WHOLE mesh.

    Why the second pass: the pinna's plane is not the head's lateral plane. Over
    14 HUTUBS subjects they differ by 8.1 deg on average (sd 3.9, range 1-13),
    varying per subject, so labelling along the detector's axis would bake a
    subject-dependent tilt into pose zero.
    """
    R = frame_from(front, up, front=front)
    if R is None:
        sys.exit("degenerate detector frame")
    V = np.asarray(mesh.vertices)
    # Centre on the tragion when MediaPipe gave one: the head centroid puts the
    # ear near the frame edge, where the ROI has least room to grow.
    C1 = V.mean(0) if centre is None else np.asarray(centre, float)
    g = in_frame(mesh, R, C1, 1.0)
    img, depth1, cam1, lm, conf = landmark_whole_frame(g, cfg, pipe)
    if lm is None:
        sys.exit("pass 1: the pipeline found no ear in the face-on render.\n"
                 "  With the MediaPipe frame the SIDE is chosen deterministically, so a\n"
                 "  head whose other ear is easier will still fail here. Try --ear 1,\n"
                 "  or --no-mediapipe to let the detector sweep pick whichever it likes.")
    P1, hit1, _ = backproject_snapped(depth1, lm, cam1, cfg)
    print(f"pass 1: detector conf {conf:.3f}, {int(hit1.sum())}/55 rays hit")
    if hit1.sum() < 10:
        sys.exit("pass 1: too few rays hit to fit a pinna plane")

    P1w = (P1[hit1] @ R) + C1                      # back to world coordinates
    R2 = frame_from(plane_normal(P1w), up, front=front)
    if R2 is None:
        sys.exit("degenerate pinna frame")
    C2 = P1w.mean(0)
    scale = float(np.abs((P1w - C2) @ R2.T).max())
    if not np.isfinite(scale) or scale <= 0:
        sys.exit("degenerate ear scale")
    tilt = np.degrees(np.arccos(np.clip(abs(R[2] @ R2[2]), -1, 1)))
    print(f"pinna plane is {tilt:.1f} deg off the detector's view direction")

    g2 = in_frame(mesh, R2, C2, scale)
    img2, depth2, cam2, lm2, conf2 = landmark_whole_frame(g2, cfg, pipe)
    if lm2 is None:
        sys.exit("pass 2: the pipeline found no ear in the pinna-frame render")
    P3, hit, snapped = backproject_snapped(depth2, lm2, cam2, cfg)
    if snapped.any():
        print(f"snap: {int(snapped.sum())} landmarks moved onto a depth cliff")
    print(f"pass 2: detector conf {conf2:.3f}, {int(hit.sum())}/55 rays hit")
    return g2, img2, cam2, lm2, P3, hit, conf2


def multiview(mesh, cfg, pipe):
    """Landmark the ear from several viewpoints and back-project each.

    Returns (P [V, 55, 3], ok [V, 55], angles). Every view is landmarked
    independently and back-projected through ITS OWN depth buffer, so the points
    from different views are separate measurements of the same anatomy, in the
    mesh's own frame.

    WHAT THIS IS FOR. Nothing else here measures whether a landmark is in the
    right PLACE. The depth-outlier flag catches only grossly deep points, and
    detector confidence has been shown to stay flat while landmarks drift. But a
    correctly placed landmark should land in the same spot whichever direction it
    was seen from, while one that slid onto the scalp depends on the ray that
    produced it and moves with the view.

    WHAT IT CANNOT DO: this is precision, not accuracy. A model that puts a point
    in the same wrong place from every angle scores perfectly. It needs human
    annotation to become a measure of correctness -- what it gives for free is a
    way to find the points worth annotating.
    """
    out, ok = [], []
    for yaw, pitch in cfg["mv_angles"]:
        E = orbit_extrinsic(yaw, pitch, cfg)
        img, depth, cam = render(mesh, cfg, E=E)
        res = pipe(img, timestamp=0.0)
        if not res:
            # A view where the detector finds nothing contributes no measurement.
            # Reported rather than silently dropped: if most views fail, the
            # agreement figure is averaging two opinions, not seven.
            print(f"  view ({yaw:+d},{pitch:+d}): no ear detected")
            out.append(np.full((55, 3), np.nan))
            ok.append(np.zeros(55, bool))
            continue
        best = max(res, key=lambda d: float(d["confidence"]))
        lm = np.asarray(best["landmarks"], float)
        P, hit, _ = backproject_snapped(depth, lm, cam, cfg)
        print(f"  view ({yaw:+d},{pitch:+d}): det {float(best['confidence']):.2f}, "
              f"{int(hit.sum())}/55")
        out.append(P)
        ok.append(hit & np.isfinite(P).all(axis=1))
    return np.stack(out), np.stack(ok), list(cfg["mv_angles"])


def agreement(P, ok):
    """Per-landmark spread across views: median distance to that point's median.

    Median rather than mean throughout, so one bad view does not set the score
    for a landmark the other views agree on.
    """
    spread = np.full(P.shape[1], np.nan)
    centre = np.full((P.shape[1], 3), np.nan)
    nview = ok.sum(axis=0)
    for k in range(P.shape[1]):
        pts = P[ok[:, k], k]
        if len(pts) < 2:
            continue
        c = np.median(pts, axis=0)
        centre[k] = c
        spread[k] = np.median(np.linalg.norm(pts - c, axis=1))
    return spread, centre, nview


def report_agreement(spread, nview, ear_extent=1.0):
    """Print per-strip and worst-landmark agreement."""
    print(f"\nmulti-view agreement (spread as % of ear extent; "
          f"lower = the views concur)")
    print(f"{'strip':>15s}{'median':>9s}{'p90':>8s}{'worst':>8s}{'views':>8s}")
    for i, (a, b) in enumerate(STRIPS):
        sp = spread[a:b]
        fin = np.isfinite(sp)
        if not fin.any():
            continue
        print(f"{STRIP_NAMES[i]:>15s}{100*np.median(sp[fin])/ear_extent:>8.1f}%"
              f"{100*np.percentile(sp[fin], 90)/ear_extent:>7.1f}%"
              f"{100*np.nanmax(sp)/ear_extent:>7.1f}%{np.mean(nview[a:b]):>8.1f}")
    order = np.argsort(-np.nan_to_num(spread, nan=-1))
    worst = [k for k in order if np.isfinite(spread[k])][:8]
    print("worst landmarks: " + ", ".join(
        f"{k}({100*spread[k]/ear_extent:.0f}%)" for k in worst))
