#!/usr/bin/env python3
"""Build a self-contained Faceform Wrap project for an ear that folds.

Everything the project needs is written next to it with absolute paths, so it
opens and computes without the batch pipeline: the template, the scan at the
same working scale the pipeline uses, our fitted result for comparison, and
the fold locations as loadable geometry so the problem area is visible rather
than hunted for.

The SelectPointPairs node carries the 14 automatic control points the pipeline
placed. Adding more by hand in the UI is the point of the exercise -- a fold
is vertices bunching where nothing pins them, and a control point is the one
thing that pins a specific vertex to a specific place on the scan.

    python scripts/wrap_fold_debug.py --ears P0057_left pp16_right P0020_left
"""
import argparse
import collections
import json
import pathlib
import shutil
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
WRAP = ROOT.parent / "wrap_work"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ear3d.wrap import boundary_vertices, load_mesh, rigid_fit   # noqa: E402
from ear3d.scheme import GROUPS, GROUP_NAMES                     # noqa: E402
from wrap_batch_faceform import corner_barycentric, spread_vertices  # noqa: E402
from refit_variants import nodes_by_type, one, rewire            # noqa: E402

LADDER = {"default": {}, "p2p_2": {"globalPoint2PointFittingWeight": 2.0},
          "p2p_4": {"globalPoint2PointFittingWeight": 4.0},
          "p2p_32": {"globalPoint2PointFittingWeight": 32.0},
          "p2p_64": {"globalPoint2PointFittingWeight": 64.0},
          "p2p_128": {"globalPoint2PointFittingWeight": 128.0},
          "maxDp_003": {"maxDp": 0.003}}


def trim1(F):
    drop = set(boundary_vertices(F).tolist())
    keep = np.array([not (int(a) in drop or int(b) in drop or int(c) in drop)
                     for a, b, c in F])
    return F[keep]


def edge_pairs(F):
    ea = collections.defaultdict(list)
    for t, f in enumerate(F):
        for i in range(3):
            ea[tuple(sorted((int(f[i]), int(f[(i+1) % 3]))))].append(t)
    return (np.array([e for e, ts in ea.items() if len(ts) == 2]),
            np.array([ts for ts in ea.values() if len(ts) == 2]))


def blob(centre, r):
    """A small octahedron, so a fold location is visible as geometry."""
    d = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], float)*r
    f = np.array([[0,2,4],[2,1,4],[1,3,4],[3,0,4],
                  [2,0,5],[1,2,5],[3,1,5],[0,3,5]])
    return centre + d, f


def write_obj(path, V, F):
    with open(path, "w") as fh:
        for p in V:
            fh.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        for t in F:
            fh.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ears", nargs="+", required=True)
    p.add_argument("--template", default=str(WRAP/"templates/ear_L_template.obj"))
    p.add_argument("--base", default=str(WRAP/"ear_wrap_pp12_left.wrap"))
    # corpus/adaptive_v1 held the ladder's PER-RUNG outputs (1,428 objs for
    # 478 ears) and was deleted in the Oct 2026 cleanup; the picked winners
    # live in corpus/escalated_v1, but this tool reads adaptive.json and
    # needs the per-rung set, so re-run wrap_refit_adaptive.py to use it.
    p.add_argument("--fits", default=str(WRAP/"corpus/adaptive_v1"))
    p.add_argument("--out", default=str(WRAP/"debug_folds"))
    p.add_argument("--points", type=int, default=14)
    p.add_argument("--work-size", type=float, default=100.0)
    a = p.parse_args()

    import trimesh
    SCANS = {"pp": WRAP/"corpus/ears4_hutubs", "P": WRAP/"corpus/ears3_sonicom"}
    OURS  = {"pp": WRAP/"corpus/wrapped3_hutubs", "P": WRAP/"corpus/wrapped3_sonicom"}
    Vt, Ft = load_mesh(a.template)
    Fk = trim1(Ft)
    E, A = edge_pairs(Fk)
    idx = spread_vertices(Vt, a.points)
    cn = corner_barycentric(Ft)
    uv = [(1.0, 0.0), (0.0, 1.0), (0.0, 0.0)]
    left = [[int(cn[v][0]), *map(float, uv[cn[v][1]])] for v in idx]
    base = json.load(open(a.base))
    rec = {r["ear"]: r for r in
           json.load(open(pathlib.Path(a.fits)/"adaptive.json")) if r.get("ok")}
    Vm, Fm = load_mesh(WRAP/"templates/ear_mean_v4.obj")
    from gt_landmark_check import read_marks, DEFAULT_MARKS
    tri_m, bary_m, _ = read_marks(DEFAULT_MARKS)
    L = np.einsum("ij,ijk->ik", bary_m[:55], Vm[Fm[tri_m[:55]]])
    R0, t0, s0 = rigid_fit(Vt, Vm, with_scale=True)

    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for ear in a.ears:
        k = "pp" if ear.startswith("pp") else "P"
        d = out / ear; d.mkdir(exist_ok=True)
        Vs, Fs = load_mesh(SCANS[k]/f"{ear}.obj")
        if ear.endswith("_right"):
            Vs, Fs = Vs*np.array([1., 1., -1.]), Fs[:, ::-1]
        w = a.work_size / float(np.ptp(Vs, axis=0).max())
        V, _ = load_mesh(pathlib.Path(a.fits)/f"{ear}.obj")
        Vb = V[:len(Vt)]

        # folds, in the SAME working scale the project computes in
        p_ = Vb[Fk]
        n = np.cross(p_[:, 1]-p_[:, 0], p_[:, 2]-p_[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True)+1e-30
        ang = np.degrees(np.arccos(np.clip(
            np.einsum("ij,ij->i", n[A[:, 0]], n[A[:, 1]]), -1, 1)))
        fold = ang > 90
        mid = np.array([0.5*(Vb[e[0]]+Vb[e[1]]) for e in E[fold]])

        write_obj(d/"scan.obj", Vs*w, Fs)
        write_obj(d/"fit_ours.obj", V*w, load_mesh(pathlib.Path(a.fits)/f"{ear}.obj")[1])
        shutil.copy(a.template, d/"template.obj")
        BV, BF, off = [], [], 0
        r = 0.6
        for c in mid:
            v, f = blob(c*w, r); BV.append(v); BF.append(f+off); off += len(v)
        if BV:
            write_obj(d/"fold_markers.obj", np.vstack(BV), np.vstack(BF))

        # control points on this scan, exactly as the pipeline computes them
        sm = trimesh.Trimesh(vertices=Vs, faces=Fs, process=False)
        Vf, _ = load_mesh(OURS[k]/f"{ear}.obj")
        cl, _, st = trimesh.proximity.closest_point(sm, Vf[idx])
        bb = trimesh.triangles.points_to_barycentric(Vs[Fs[st]], cl)
        right = [[int(st[j]), float(bb[j][0]), float(bb[j][1])]
                 for j in range(len(idx))]

        rung = rec.get(ear, {}).get("rung", "default")
        proj = json.loads(json.dumps(base))
        byid = {v["nodeId"]: v for v in proj["nodes"].values()}
        rig = one(proj, "RigidAlignment")
        byid[rig["params"]["geomFloating"]["connectedNodeId"]]["params"]["fileName"]["value"] = str((d/"template.obj").resolve())
        byid[rig["params"]["geomFloating"]["connectedNodeId"]]["params"]["scale"]["value"] = 1
        byid[rig["params"]["geomFixed"]["connectedNodeId"]]["params"]["fileName"]["value"] = str((d/"scan.obj").resolve())
        byid[rig["params"]["geomFixed"]["connectedNodeId"]]["params"]["scale"]["value"] = 1
        for nn in nodes_by_type(proj, "SelectPointPairs"):
            nn["params"]["pointsLeft"]["value"] = left
            nn["params"]["pointsRight"]["value"] = right
        one(proj, "SaveGeom")["params"]["fileName"]["value"] = str((d/"fit_new.obj").resolve())
        rewire(proj, project=False, wrap_params=LADDER.get(rung, {}))
        proj["timeline"] = {"current": 0, "min": 0, "max": 0}
        json.dump(proj, open(d/f"{ear}.wrap", "w"), indent=4)

        # which anatomy, so the README says where to look
        Vin = (Vt @ R0.T)*s0 + t0
        where = collections.Counter()
        for e in E[fold]:
            q = 0.5*(Vin[e[0]]+Vin[e[1]])
            j = int(np.argmin(np.linalg.norm(L-q, axis=1)))
            where[next(GROUP_NAMES[z] for z, (lo, hi) in enumerate(GROUPS)
                       if lo <= j < hi)] += 1
        with open(d/"README.txt", "w") as fh:
            fh.write(f"{ear}: {int(fold.sum())} folded edges after trim 1, "
                     f"max dihedral {ang.max():.1f} deg\n")
            fh.write(f"rung the pipeline chose: {rung}"
                     f" ({rec.get(ear,{}).get('tries','?')} fits)\n\n")
            fh.write("where the folds are:\n")
            for nm, c in where.most_common():
                fh.write(f"  {nm:24s} {c}\n")
            fh.write(f"\nfiles (all in Wrap's working scale, ear extent = "
                     f"{a.work_size:.0f}):\n")
            fh.write("  template.obj      the fitting source (1188 v)\n")
            fh.write("  scan.obj          the target\n")
            fh.write("  fit_ours.obj      what the pipeline produced\n")
            fh.write("  fold_markers.obj  a blob at every folded edge\n")
            fh.write(f"  {ear}.wrap        open this; SaveGeom writes fit_new.obj\n")
            fh.write("\nThe 14 control points are already in SelectPointPairs.\n"
                     "Add more near the markers and recompute to compare.\n")
        print(f"{ear}: {int(fold.sum())} folds -> {d}  "
              f"({', '.join(f'{n_} {c}' for n_, c in where.most_common(3))})")


if __name__ == "__main__":
    main()
