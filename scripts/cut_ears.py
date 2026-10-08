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
from scipy.spatial import cKDTree
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
# The HUTUBS cut puts the protrusion along -x (see cut_hutubs), and a local ICP
# cannot recover a 180 degree error, so SONICOM has to agree with it. Composing
# with a half turn about y is what makes the two datasets land in ONE frame;
# without it every SONICOM ear starts back-to-front and the fit grinds the
# template onto the wrong side of the ear while reporting a plausible surface
# gap. A half turn is a rotation, so handedness survives it.
_Y180 = np.diag([-1.0, 1.0, -1.0])
R_LEFT = _Y180 @ np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
R_RIGHT = _Y180 @ np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
assert np.linalg.det(_Y180) > 0
assert np.linalg.det(R_LEFT) > 0 and np.linalg.det(R_RIGHT) > 0


def _affine(A, b):
    """The 4x4 taking cut coordinates back to the source mesh's frame: Ax + b.

    Saved beside every cut, because nothing downstream can recover it -- see the
    note in cut_hutubs -- and verified against the source on the way out, since
    a mis-composed transform still produces a tidy-looking matrix.
    """
    M = np.eye(4)
    M[:3, :3] = A
    M[:3, 3] = np.asarray(b)
    return M


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


def _sonicom_vertices(path):
    import trimesh
    return np.asarray(trimesh.load(path).vertices)


def ear_centre_by_density(V, cell=20.0, top=6):
    """The densest cells of a SONICOM graded mesh are the pinna, by construction."""
    g = np.floor((V - V.min(0)) / cell).astype(int)
    _, inv, cnt = np.unique(g, axis=0, return_inverse=True, return_counts=True)
    return V[np.isin(inv, np.argsort(-cnt)[:top])].mean(0)


# The crop boxes are DELIBERATELY generous. The first sizes clipped every
# single cut -- 100% of them reached the box edge in height and
# anteroposterior, so the box was deciding the ear's size rather than the
# anatomy, and the tops, bottoms and backs of ears were being cut off. A
# truncated scan is worse than a loose one: the template has nothing to fit to
# where the ear is missing, and the border is free of the data term anyway so
# the surrounding skin costs little. They cannot grow without limit, though --
# the ear is attached to the head, so a big enough box just swallows the skull.
# An adult pinna is about 60-65 mm tall, and these are roughly 1.7x that.
def cut_sonicom(path, side, half=(30.0, 52.0, 42.0)):
    import trimesh
    m = trimesh.load(path)          # process=True welds the STL triangle soup
    V, F = np.asarray(m.vertices), np.asarray(m.faces)
    R = R_LEFT if side == "left" else R_RIGHT
    c = ear_centre_by_density(V)
    W = (V - c) @ R.T
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(W),
                                     o3d.utility.Vector3iVector(F))
    g = _largest_piece(mesh, np.all(np.abs(W) < np.array(half), axis=1))
    return g, _affine(R.T, c)          # cut -> the scan's own coordinates


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
    half = np.array([0.30, 0.42, 0.50]) * sep
    keep = (np.all(np.abs(V[:, :2] - c[:2]) < half[:2], axis=1)
            & (V[:, 2] > c[2] - half[2]))
    e = _largest_piece(g, keep)
    m = np.asarray(e.vertices).mean(0)
    W = np.asarray(e.vertices) - m
    W = np.stack([-W[:, 2], W[:, 1], W[:, 0]], axis=1)      # into the ear frame
    e.vertices = o3d.utility.Vector3dVector(W)
    e.compute_vertex_normals()
    # cut -> the mesh load_head handed us. The ear frame is an axis PERMUTATION,
    # a 90 degree rotation, which is why trying to recover this afterwards by
    # ICP fails: every sensible initialisation starts a quarter turn out and ICP
    # cannot cross that.
    # x_cut = P (R (x_norm - C) - m), so the way back is
    #   x_norm = R.T (P.T x_cut + m) + C = (R.T P.T) x_cut + (R.T m + C).
    # The recentre belongs AFTER the permutation is undone, not before it; the
    # other order looks just as reasonable and lands 15% of the scan away.
    P = np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])
    return e, _affine(R.T @ P.T, np.asarray(C) + R.T @ m)


def pose_by_transfer(mesh, ref_path, cfg, iters=60):
    """Borrow a head frame from a subject MediaPipe could pose.

    When no face is found there is still a head, and heads are alike enough to
    register. A reference head whose tragions ARE known is aligned to this one
    by trimmed rigid ICP, and its tragions ride along. The result is coarse --
    it is one person's landmarks pushed onto another's skull -- but the crop
    only needs to find the ear, and the pinna is then located geometrically
    inside it exactly as in the normal path.
    """
    from scipy.spatial import cKDTree
    from ear3d.wrap import rigid_fit
    ref = load_head(ref_path)
    cfg2 = dict(cfg)
    cfg2["mesh"] = str(ref_path)
    pose = head_pose(ref, cfg2)
    if pose is None:
        raise RuntimeError(f"the reference head {pathlib.Path(ref_path).name} "
                           f"cannot be posed either")
    lat, vert, tr_r, tr_l = pose
    A = np.asarray(ref.vertices)
    B = np.asarray(mesh.vertices)
    # coarse: match centroid and size, then trimmed ICP on a subsample
    ia = np.random.default_rng(0).choice(len(A), min(20000, len(A)), replace=False)
    X = A[ia]
    k = float(np.ptp(B, axis=0).max() / np.ptp(A, axis=0).max())
    X = (X - A.mean(0)) * k + B.mean(0)
    carry = np.array([lat, vert, np.asarray(tr_r), np.asarray(tr_l)])
    carry[:2] = carry[:2] * k + 0.0                     # directions scale only
    carry[2:] = (carry[2:] - A.mean(0)) * k + B.mean(0)
    tree = cKDTree(B)
    for _ in range(iters):
        d, idx = tree.query(X)
        keep = d <= np.quantile(d, 0.7)
        R, t, sc = rigid_fit(X[keep], B[idx][keep])
        X = sc * X @ R.T + t
        carry[:2] = sc * carry[:2] @ R.T                # directions: no offset
        carry[2:] = sc * carry[2:] @ R.T + t
    gap = float(np.median(tree.query(X)[0])) / float(np.ptp(B, axis=0).max())
    lat2 = carry[0] / np.linalg.norm(carry[0])
    vert2 = carry[1] / np.linalg.norm(carry[1])
    return (lat2, vert2, carry[2], carry[3]), gap


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["hutubs", "sonicom"], required=True)
    p.add_argument("--out", default=None, help="directory for the cut ears")
    p.add_argument("--sheet", default=None, help="contact sheet PNG to eyeball")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--only", nargs="*", default=None,
                   help="restrict to these subject names")
    p.add_argument("--tile", type=int, default=300)
    p.add_argument("--transforms", default=None,
                   help="directory for the cut -> scan transform of every ear. "
                        "Without it the cut is unmoored: the ear frame involves "
                        "a 90 degree axis permutation and a per-ear recentring, "
                        "and no amount of registration afterwards recovers them "
                        "reliably -- measured, ICP settles 5 to 8 times the mesh "
                        "resolution away and a free scale collapses the ear "
                        "outright")
    p.add_argument("--force-transfer", action="store_true",
                   help="transfer the frame even when MediaPipe succeeds -- it "
                        "can find a face and still pose it badly, which shows up "
                        "as a cut with no ear relief in it")
    p.add_argument("--reference", default=None,
                   help="a HUTUBS head that DOES pose; when MediaPipe fails on a "
                        "subject, its frame is transferred from this one by ICP "
                        "instead of the subject being dropped")
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
    if a.only:
        jobs = [j for j in jobs if j[0] in set(a.only)]
    if a.limit:
        jobs = jobs[:a.limit]

    done, failed, tiles = 0, [], []
    for name, path, side in jobs:
        try:
            if a.dataset == "sonicom":
                cuts = [(side, *cut_sonicom(path, side))]
            else:
                mesh = load_head(path)
                cfg["mesh"] = path
                pose = None if a.force_transfer else head_pose(mesh, cfg)
                if pose is None:
                    if not a.reference:
                        raise RuntimeError("MediaPipe found no face on any view")
                    pose, gap = pose_by_transfer(mesh, a.reference, cfg)
                    print(f"    (no face; frame transferred from "
                          f"{pathlib.Path(a.reference).stem}, surface gap "
                          f"{100*gap:.2f}% of head size)")
                lat, vert, tr_r, tr_l = pose
                sep = float(np.linalg.norm(np.asarray(tr_l) - np.asarray(tr_r)))
                cuts = [(s, *cut_hutubs(mesh, C, fr, vert, sep)) for s, C, fr in
                        (("left", tr_l, lat), ("right", tr_r, -lat))]
            for s, g, M in cuts:
                V = np.asarray(g.vertices)
                if len(V) < 1000:
                    raise RuntimeError(f"only {len(V)} vertices survived the cut")
                # VERIFY THE TRANSFORM AGAINST THE SOURCE, here, every time. The
                # cut is a subset of the source's vertices, so mapping it back
                # must land exactly on them -- a wrong composition shows up as a
                # gap rather than as a plausible-looking fit nobody rechecks.
                if a.transforms:
                    src = (np.asarray(load_head(path).vertices)
                           if a.dataset == "hutubs" else _sonicom_vertices(path))
                    back = V @ M[:3, :3].T + M[:3, 3]
                    gap = float(np.median(cKDTree(src).query(back)[0]))
                    scale = float(np.ptp(src, axis=0).max())
                    if gap > 1e-6 * scale:
                        raise RuntimeError(
                            f"the saved transform does not invert the cut: "
                            f"{gap:.2e} = {100 * gap / scale:.4f}% of the scan")
                    td = pathlib.Path(a.transforms)
                    td.mkdir(parents=True, exist_ok=True)
                    np.savez(td / f"{name}_{s}.npz", matrix=M, source=str(path),
                             dataset=a.dataset, verified_gap=gap)
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
