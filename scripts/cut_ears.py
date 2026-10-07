#!/usr/bin/env python3
"""Cut every ear out of the 3D corpus, into one canonical frame, unattended.

This replaces per-scan anchor placement. Non-rigid fitting needs the template
put roughly on the ear before it starts, and the usual way to supply that is to
click a few point pairs on every scan -- 229 ears of it. Neither dataset
actually needs that, for different reasons:

  SONICOM  the graded meshes preserve resolution around the pinna and decimate
           everywhere else, so the DENSEST region of the mesh is the ear. They
           are Frankfurt-aligned, so one fixed rotation per side puts every
           subject in the same frame. Nothing is detected at all.

  HUTUBS   whole heads in arbitrary pose, so the frame comes from the tragions
           (MediaPipe on the head) and the crop is sized by tragion separation.
           The pinna is then found geometrically, as what protrudes. MediaPipe
           fails on a minority of subjects and those are reported, not guessed.

Nothing here uses the 2D landmark lift: it places points on the scalp depending
on viewpoint, and an ear cut around one would be wrong without being obviously
wrong. Every cut is written as an OBJ and should be eyeballed as a contact
sheet (--sheet) before a corpus is trusted.

    python scripts/cut_ears.py --dataset sonicom --out ears/ --sheet s.png
    python scripts/cut_ears.py --dataset hutubs  --out ears/ --limit 8
"""
import argparse
import glob
import pathlib
import sys

import numpy as np
import open3d as o3d

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ear3d.config import DEFAULTS                       # noqa: E402
from ear3d.frames import frame_from, head_pose, in_frame, load_head  # noqa: E402

DATA = pathlib.Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/clean_3d_data")

# Frankfurt (x anterior, y left-lateral, z superior) -> the ear frame the
# templates use (x protrusion, y height, z anteroposterior). Both have
# determinant +1: a rotation, never a reflection. A reflection here would turn
# every left ear into a right one, which looks perfectly plausible and would
# only surface as nonsense once a shape model averaged the two together.
R_LEFT = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
R_RIGHT = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
assert np.linalg.det(R_LEFT) > 0 and np.linalg.det(R_RIGHT) > 0


def _largest_piece(mesh, keep_idx):
    """Keep the chosen vertices, then the biggest connected shell of what is left.

    The box can clip a corner of jaw or hair that is not attached to the ear,
    and a stray island would give the fit something irrelevant to latch onto.
    """
    g = mesh.select_by_index(np.flatnonzero(keep_idx))
    g.remove_unreferenced_vertices()
    if len(g.triangles) == 0:
        raise RuntimeError("the crop box kept nothing")
    lab, cnt, _ = g.cluster_connected_triangles()
    lab, cnt = np.asarray(lab), np.asarray(cnt)
    g.remove_triangles_by_mask(lab != int(np.argmax(cnt)))
    g.remove_unreferenced_vertices()
    g.compute_vertex_normals()
    return g


def ear_centre_by_density(V, cell=20.0, top=6):
    """The densest cells of a SONICOM graded mesh are the pinna, by construction."""
    g = np.floor((V - V.min(0)) / cell).astype(int)
    _, inv, cnt = np.unique(g, axis=0, return_inverse=True, return_counts=True)
    return V[np.isin(inv, np.argsort(-cnt)[:top])].mean(0)


def cut_sonicom(path, side, half=(26.0, 38.0, 30.0)):
    import trimesh
    m = trimesh.load(path)          # process=True welds the STL triangle soup
    V, F = np.asarray(m.vertices), np.asarray(m.faces)
    R = R_LEFT if side == "left" else R_RIGHT
    W = (V - ear_centre_by_density(V)) @ R.T
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(W),
                                     o3d.utility.Vector3iVector(F))
    return _largest_piece(mesh, np.all(np.abs(W) < np.array(half), axis=1))


def cut_hutubs(mesh, C, front, vert, sep):
    R = frame_from(front, vert, front=front)
    g = in_frame(mesh, R, np.asarray(C), 1.0)
    V = np.asarray(g.vertices)
    wide = ((np.abs(V[:, 0]) < 0.42 * sep) & (np.abs(V[:, 1]) < 0.55 * sep)
            & (V[:, 2] > -0.22 * sep))
    if wide.sum() < 500:
        raise RuntimeError("the coarse box about the tragion caught almost nothing")
    # the pinna is what protrudes, so let the protruding vertices vote on a centre
    prot = wide & (V[:, 2] > np.percentile(V[wide][:, 2], 75))
    c = V[prot].mean(0)
    half = np.array([0.20, 0.26, 0.40]) * sep
    keep = (np.all(np.abs(V[:, :2] - c[:2]) < half[:2], axis=1)
            & (V[:, 2] > c[2] - half[2]))
    e = _largest_piece(g, keep)
    W = np.asarray(e.vertices) - np.asarray(e.vertices).mean(0)
    W = np.stack([-W[:, 2], W[:, 1], W[:, 0]], axis=1)      # into the ear frame
    e.vertices = o3d.utility.Vector3dVector(W)
    e.compute_vertex_normals()
    return e


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["hutubs", "sonicom"], required=True)
    p.add_argument("--out", default=None, help="directory for the cut ears")
    p.add_argument("--sheet", default=None, help="contact sheet PNG to eyeball")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--tile", type=int, default=300)
    a = p.parse_args()
    cfg = dict(DEFAULTS)
    cfg["skin"] = False                 # geometry only; shape is what gets fitted
    out = pathlib.Path(a.out) if a.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    jobs = []
    if a.dataset == "sonicom":
        for f in sorted(glob.glob(str(DATA / "sonicom_graded" / "*_graded_*.stl"))):
            side = "left" if f.endswith("_left.stl") else "right"
            # the stem already names the side, so strip it -- the writer adds it
            subj = pathlib.Path(f).stem.replace("_graded", "").replace(f"_{side}", "")
            jobs.append((subj, f, side))
    else:
        for f in sorted(glob.glob(str(DATA / "3D head meshes" / "*.ply"))):
            jobs.append((pathlib.Path(f).stem.replace("_3DheadMesh", ""), f, None))
    if a.limit:
        jobs = jobs[:a.limit]

    done, failed, tiles = 0, [], []
    for name, path, side in jobs:
        try:
            if a.dataset == "sonicom":
                cuts = [(side, cut_sonicom(path, side))]
            else:
                mesh = load_head(path)
                cfg["mesh"] = path
                pose = head_pose(mesh, cfg)
                if pose is None:
                    raise RuntimeError("MediaPipe found no face on any view")
                lat, vert, tr_r, tr_l = pose
                sep = float(np.linalg.norm(np.asarray(tr_l) - np.asarray(tr_r)))
                cuts = [(s, cut_hutubs(mesh, C, fr, vert, sep)) for s, C, fr in
                        (("left", tr_l, lat), ("right", tr_r, -lat))]
            for s, g in cuts:
                V = np.asarray(g.vertices)
                if len(V) < 1000:
                    raise RuntimeError(f"only {len(V)} vertices survived the cut")
                if out:
                    o3d.io.write_triangle_mesh(str(out / f"{name}_{s}.obj"), g)
                if a.sheet:
                    tiles.append(_tile(g, cfg, a.tile))
            done += 1
            print(f"  {name:<22s} ok")
        except Exception as e:
            failed.append((name, str(e)))
            print(f"  {name:<22s} FAILED: {e}")

    print(f"\n{done}/{len(jobs)} scans cut" + (f", written to {out}" if out else ""))
    if failed:
        print(f"{len(failed)} failed -- these need attention, not a guess:")
        for n, why in failed:
            print(f"  {n}: {why}")
    if a.sheet and tiles:
        from PIL import Image
        w = 8
        rows = [np.hstack(tiles[i:i + w]) for i in range(0, len(tiles), w)]
        wid = min(r.shape[1] for r in rows)
        Image.fromarray(np.vstack([r[:, :wid] for r in rows])).save(a.sheet)
        print(f"wrote {a.sheet} -- look at it before trusting the corpus")


def _tile(g, cfg, size):
    from ear3d.camera import camera_ke, orbit_extrinsic
    from ear3d.render import MATERIAL, light_scene
    v = np.asarray(g.vertices)
    h = o3d.geometry.TriangleMesh(g.vertices, g.triangles)
    h.vertices = o3d.utility.Vector3dVector(
        (v - v.mean(0)) / (0.55 * float(np.ptp(v, axis=0).max())))
    h.compute_vertex_normals()
    r = o3d.visualization.rendering.OffscreenRenderer(size, size)
    r.scene.add_geometry("m", h, MATERIAL)
    light_scene(r.scene, (0.5, 0.5, 0.5))
    K, E = camera_ke(size, cfg, orbit_extrinsic(-90.0, 0.0, cfg))
    r.setup_camera(K, E, size, size)
    img = np.asarray(r.render_to_image())[:, :, :3]
    del r
    return img


if __name__ == "__main__":
    main()
