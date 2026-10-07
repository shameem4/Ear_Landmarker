#!/usr/bin/env python3
"""Split a head into left and right halves for hand-wrapping in Faceform Wrap.

Cutting a patch around the ear exists for the AUTOMATIC fit, where something
has to stand in for hand-placed control points. It is the wrong thing to hand a
person: a box tight enough to localise is tight enough to clip, and a clipped
ear cannot be wrapped at all.

For manual work the whole head is fine -- Wrap handles it, and Shameem's first
pp1 wrap was done that way with three point pairs. Bisecting only removes the
ambiguity about which ear is meant, and takes half the geometry out of the
viewport. No ear is truncated: the plane passes through the midline, nowhere
near a pinna.

    python scripts/bisect_heads.py --subjects pp31 pp22 --out halves/
"""
import argparse
import pathlib
import sys

import numpy as np
import open3d as o3d

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cut_ears import DATA, _largest_piece, pose_by_transfer   # noqa: E402
from ear3d.config import DEFAULTS                             # noqa: E402
from ear3d.frames import head_pose, load_head                 # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subjects", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--reference", default=None)
    p.add_argument("--tragion-mm", type=float, default=140.0,
                   help="rescale each half so tragion separation is this, in mm. "
                        "Heads arrive in metres, head-radius units and "
                        "millimetres; normalising by an ANATOMICAL length rather "
                        "than by mesh extent makes every ear the same real size, "
                        "so one template fits them all without rescaling per frame")
    p.add_argument("--keep", type=float, default=0.12,
                   help="fraction of head width to keep PAST the midline, so the "
                        "half is not sliced flush and has something to sit on")
    a = p.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(DEFAULTS)
    cfg["skin"] = False

    for subj in a.subjects:
        src = DATA / "3D head meshes" / f"{subj}_3DheadMesh.ply"
        if not src.exists():
            print(f"  {subj}: no mesh at {src}")
            continue
        mesh = load_head(src)
        cfg["mesh"] = str(src)
        pose = head_pose(mesh, cfg)
        if pose is None:
            if not a.reference:
                print(f"  {subj}: no face found and no --reference given")
                continue
            pose, _ = pose_by_transfer(mesh, a.reference, cfg)
        lat, vert, tr_r, tr_l = pose
        # The sagittal normal is the line BETWEEN THE TRAGIONS, not head_pose's
        # lat -- whose convention split heads horizontally when trusted here.
        # Tragion to tragion cannot be misread: it is lateral by definition.
        lat = np.asarray(tr_l, float) - np.asarray(tr_r, float)
        lat /= np.linalg.norm(lat)
        V = np.asarray(mesh.vertices)
        mid = 0.5 * (np.asarray(tr_l) + np.asarray(tr_r))
        t = (V - mid) @ lat                     # signed distance from the midline
        pad = a.keep * float(t.max() - t.min())
        k = a.tragion_mm / float(np.linalg.norm(np.asarray(tr_l) - np.asarray(tr_r)))
        for side, keep in (("left", t > -pad), ("right", t < pad)):
            g = _largest_piece(mesh, keep)
            W = np.asarray(g.vertices) * k
            g.vertices = o3d.utility.Vector3dVector(W)
            g.compute_vertex_normals()
            o3d.io.write_triangle_mesh(str(out / f"{subj}_{side}half.obj"), g)
            print(f"  {subj}_{side}half: {len(W)} verts, extent "
                  f"{np.ptp(W, axis=0).round(1)} mm (scaled x{k:.4g})")


if __name__ == "__main__":
    main()
