"""Finding the ear, and building the frames to look at it from.

load_head, find_ears, the frame helpers and the MediaPipe head pose. These were
duplicated in ingest_render3d.py and had already diverged -- find_ears there
rendered through a private camera setup of its own."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import open3d as o3d

from .backproject import backproject
from .render import render
from .scheme import CHIN, FOREHEAD, NASION, NOSE_TIP, TRAGION_L, TRAGION_R

def load_head(path):
    """Head mesh, centred and scaled to unit radius."""
    m = o3d.io.read_triangle_mesh(str(path))
    if len(m.vertices) == 0:
        return None
    V = np.asarray(m.vertices)
    V = (V - V.mean(0)) / np.abs(V - V.mean(0)).max()
    m.vertices = o3d.utility.Vector3dVector(V)
    m.compute_vertex_normals()
    return m


def frame_from(normal, up, front=None):
    """Rotation taking `normal` onto +Z (toward the camera), keeping `up` near +Y."""
    n = np.asarray(normal, float)
    if front is not None and n @ np.asarray(front, float) < 0:
        n = -n
    z = n / max(np.linalg.norm(n), 1e-9)
    y = np.asarray(up, float) - z * (np.asarray(up, float) @ z)
    ny = np.linalg.norm(y)
    if ny < 1e-6:
        return None
    y /= ny
    return np.stack([np.cross(y, z), y, z])            # world -> camera frame


def plane_normal(points):
    """Surface normal of the best-fit plane through the landmarks."""
    P = np.asarray(points, float)
    _, _, vt = np.linalg.svd(P - P.mean(0), full_matrices=False)
    return vt[2]


def in_frame(mesh, R, C, scale):
    """A copy of the mesh rotated into frame R about C and scaled by `scale`."""
    g = copy.deepcopy(mesh)
    g.vertices = o3d.utility.Vector3dVector(
        ((np.asarray(mesh.vertices) - C) @ R.T) / scale)
    g.compute_vertex_normals()
    return g


def find_ears(mesh, det, cfg, probe_size=400, dist=2.4):
    """Locate ears by sweeping the detector around the head.

    Geometric rules do not transfer between datasets -- HUTUBS and SONICOM store
    heads on different axes -- so the ear is found by what an ear detector sees.
    """
    out = []
    for axis in (0, 1, 2):
        for sgn in (1, -1):
            front = np.zeros(3); front[axis] = sgn
            up = np.array([0.0, 0.0, 1.0]) if axis != 2 else np.array([0.0, 1.0, 0.0])
            R = frame_from(front, up, front=front)
            if R is None:
                continue
            g = in_frame(mesh, R, np.asarray(mesh.vertices).mean(0), 1.0)
            b = det.detect(render(g, cfg, size=probe_size)[0])
            if not len(b):
                continue
            h, w = b[0, 2] - b[0, 0], b[0, 3] - b[0, 1]
            if (h * w) / (probe_size ** 2) > 0.25:   # a whole-head box is a false positive
                continue
            out.append((float(b[0, 4]), front, up, b[0, :4].copy()))
    out.sort(key=lambda t: -t[0])
    return out[:2]


def head_pose(mesh, cfg):
    """Head frame from MediaPipe FaceMesh, in the mesh's own coordinates.

    Returns (lateral, vertical, tragion_R, tragion_L) or None.

    WHY THIS BEATS SWEEPING THE EAR DETECTOR. The sweep renders six axis-aligned
    views and keeps the two best ear boxes, which on a head gives two nearly equal
    confidences and no way to tell which side is which. MediaPipe finds a face on
    exactly ONE of those six views -- on every head tried -- so the frontal
    direction is unambiguous, and the face landmarks then give a real anatomical
    frame rather than whichever axis the dataset stored the head on. The ear
    direction is the sagittal-plane normal -- 90 degrees from the nose -- built
    from forehead, chin, nasion and nose tip. The tragion landmarks are used only
    to pick which side is which and to centre the view, never for the direction.

    IT DOES NOT REMOVE PASS 2, AND IT IS NOT MEASURABLY MORE ACCURATE. Head to
    head over four heads, pass-2 confidence against the detector sweep: 0.931 vs
    0.955, 0.942 vs 0.927, fails vs 0.881, 0.980 vs 0.981. The remaining offset
    from the pinna plane is the same either way (8-35 deg here, 16-37 for the
    sweep). What it buys is that the SIDE is known -- the sweep returns two ear
    boxes at near-equal confidence with no way to tell left from right, so "ear 0"
    is arbitrary and changes between heads. The pinna's own tilt is per-subject
    anatomy that no face-derived frame can supply, which is why pass 2 stays.

    Measured per side (pass-2 confidence / degrees off the pinna plane), the RIGHT
    ear is consistently better: pp16 fails/0.839@12, pp12 0.931@33/0.990@28,
    pp11 0.942@35/0.986@13, pp10 0.980@19/0.982@8. Left and right ears are mirror
    images, so this is most likely an asymmetry in the landmarker rather than in
    this frame -- it is not explained, and worth a look before trusting left-ear
    labels as much as right.
    """
    task = cfg.get("face_task")
    if not task or not Path(task).exists():
        print(f"mediapipe: no face_landmarker.task at {task} -- using the ear sweep")
        return None
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    fl = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(task)), num_faces=1))
    C0 = np.asarray(mesh.vertices).mean(0)
    for axis in (0, 1, 2):
        for sgn in (1, -1):
            front = np.zeros(3); front[axis] = sgn
            up = np.array([0.0, 0.0, 1.0]) if axis != 2 else np.array([0.0, 1.0, 0.0])
            R = frame_from(front, up, front=front)
            if R is None:
                continue
            g = in_frame(mesh, R, C0, 1.0)
            img, depth, cam = render(g, cfg)
            res = fl.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                     data=np.ascontiguousarray(img)))
            if not res.face_landmarks:
                continue
            # MediaPipe normalises to the IMAGE it was handed, so scale by that
            # image's own shape, never by the size we requested.
            h, w = img.shape[:2]
            uv = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]])
            P, hit = backproject(depth, uv, cam)
            if not hit[[TRAGION_R, TRAGION_L, FOREHEAD, CHIN,
                        NOSE_TIP, NASION]].all():
                continue
            # The landmarks come back in THIS frame's coordinates; everything
            # downstream works in the mesh's own, so convert before using them.
            # Skipping this put the ear view 62-70 deg off instead of 10-24.
            P = (P @ R) + C0
            # THE EAR DIRECTION IS 90 DEG FROM THE NOSE: the normal of the
            # sagittal plane, from the head's own up and forward axes.
            #
            # The obvious alternative is the tragion-to-tragion line, and over 12
            # ears it is no better -- mean 25.1 deg from the pinna plane against
            # 25.7 here, median 26.9 against 24.2. What decides it is that the
            # tragion landmarks are the weak ones: the reference implementation in
            # ../landmarking_stuff/landmarker marks exactly these indices "most
            # likely not possible" and does not use them. Forehead, chin, nasion
            # and nose tip are points FaceMesh is actually good at, so the same
            # answer rests on firmer ground.
            #
            # A third option, the head's local surface normal at the tragion, is
            # better typically (mean 16.4, median 14.4) but has a worse tail
            # (48.2) and failed outright on 1 of 12. Not taken.
            vert = P[FOREHEAD] - P[CHIN]
            vert /= np.linalg.norm(vert)
            fwd = P[NOSE_TIP] - P[NASION]
            fwd -= vert * (fwd @ vert)
            if np.linalg.norm(fwd) < 1e-6:
                continue
            fwd /= np.linalg.norm(fwd)
            lat = np.cross(vert, fwd)
            lat /= np.linalg.norm(lat)
            # Orient toward the subject's left so "ear 0" keeps its meaning.
            if lat @ (P[TRAGION_L] - P[TRAGION_R]) < 0:
                lat = -lat
            vert -= lat * (vert @ lat)
            vert /= np.linalg.norm(vert)
            print(f"mediapipe: face on axis {axis}{'+' if sgn > 0 else '-'}, "
                  f"tragion separation {np.linalg.norm(P[TRAGION_L] - P[TRAGION_R]):.3f}")
            return lat, vert, P[TRAGION_R], P[TRAGION_L]
    print("mediapipe: no face found on any of the six views -- using the ear sweep")
    return None
