#!/usr/bin/env python3
"""Re-fit the AudioEar3D clouds, escalating until the mesh does not fold.

The cloud path is our own NICP, not Faceform -- Wrap needs meshes and these
are point clouds. It folds WORSE than the mesh path: 0 of 112 fits are
fold-free, median 57 folded edges against the Faceform path's 5 at the same
1188-vertex resolution.

ear3d.wrap names the cause in its own comment:

    DEFAULT_SCHEDULE = (50, 20, 10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05)
    # The tail matters: the last few values are what lets the
    # template enter the concha and follow the helix fold.

What lets the template FOLLOW the helix fold is what lets it BUCKLE there, and
0.05 is the same final stiffness as the Wrapping node's globalSmoothWeightMin.

The two knobs are not interchangeable, which sets the ladder's order:

  min_cos  rejects correspondences whose normals disagree. The default 0.5
           accepts matches up to 60 degrees apart, which is licence to match
           the far face of a thin rim. Raising it removes bad matches without
           taking any freedom away from the fit, so it is tried FIRST.
  schedule truncating the tail removes freedom directly, so it costs exactly
           the concha and helix detail the comment is describing. Last resort,
           and the accuracy columns are there to price it.

Unlike the Faceform ladder, where scan->fit was flat across every rung, here
accuracy is expected to move. It is reported per rung rather than assumed.

    python scripts/wrap_cloud_adaptive.py --root AudioEar3D --out fits/ --jobs 8
"""
import argparse
import collections
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
WRAP = ROOT.parent / "wrap_work"
sys.path.insert(0, str(ROOT))

from ear3d.wrap import DEFAULT_SCHEDULE                           # noqa: E402

# (name, min_cos, schedule). The first rung is the shipped default, so a fit
# that does not fold today is reproduced exactly and stops there.
def _tail(stop):
    """The default schedule truncated so it ends at `stop`."""
    return tuple(s for s in DEFAULT_SCHEDULE if s >= stop)


LADDER = [
    ("default",      0.50, DEFAULT_SCHEDULE),
    ("cos80",        0.80, DEFAULT_SCHEDULE),
    ("cos90",        0.90, DEFAULT_SCHEDULE),
    ("cos80_t0.2",   0.80, _tail(0.2)),
    ("cos90_t0.5",   0.90, _tail(0.5)),
    ("cos90_t1.0",   0.90, _tail(1.0)),
]


def folds(V, F, A):
    p = V[F]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-30
    a = np.degrees(np.arccos(
        np.clip(np.einsum("ij,ij->i", n[A[:, 0]], n[A[:, 1]]), -1, 1)))
    return int((a > 90).sum()), float(a.max())


def edge_pairs(F):
    adj = collections.defaultdict(list)
    for t, f in enumerate(F):
        for i in range(3):
            adj[tuple(sorted((int(f[i]), int(f[(i + 1) % 3]))))].append(t)
    return np.array([ts for ts in adj.values() if len(ts) == 2])


def _one(job):
    cloud, template, out, max_dist, max_rungs = job
    import trimesh
    from scipy.spatial import cKDTree

    from ear3d.wrap import (boundary_points, boundary_vertices, load_mesh,
                            nicp, rigid_icp, vertex_normals)

    def load_cloud(path):
        c = trimesh.load(path, process=False)
        raw = c.metadata["_ply_raw"]["vertex"]["data"]
        return (np.asarray(c.vertices, float),
                np.asarray(raw[["nx", "ny", "nz"]].tolist(), float))

    cloud = pathlib.Path(cloud)
    name = f"{cloud.parent.name}_{cloud.stem}"
    try:
        Vc, Nc = load_cloud(cloud)
        if cloud.stem == "right":
            Vc, Nc = Vc * [1.0, 1.0, -1.0], Nc * [1.0, 1.0, -1.0]
        Vt, Ft = load_mesh(template)
        A = edge_pairs(Ft)
        size = float(np.ptp(Vt, axis=0).max())
        Ncn = Nc / np.maximum(np.linalg.norm(Nc, axis=1, keepdims=True), 1e-12)
        k0 = float(np.ptp(Vc, axis=0).max()) / size
        Vp = (Vt - Vt.mean(0)) * k0 + Vc.mean(0)
        # the rigid stage does not depend on the ladder, so it runs once
        V0, _ = rigid_icp(Vp, Ft, Vc, Ns=Nc, max_dist=0.3 * size,
                          with_scale=False, verbose=False)
        edge = boundary_points(Vc, Nc)
        cut = np.zeros(len(Vt), bool)
        cut[boundary_vertices(Ft)] = True
        tree = cKDTree(Vc)
        step = max(1, len(Vc) // 40000)

        trace, best, bestV = [], None, None
        for rung, min_cos, sched in LADDER[:max_rungs]:
            t0 = time.perf_counter()
            V = nicp(V0, Ft, Vc, Ns=Nc, max_dist=max_dist * size,
                     min_cos=min_cos, schedule=sched, verbose=False)
            n90, mx = folds(V, Ft, A)
            d, idx = tree.query(V)
            Nv = vertex_normals(V, Ft)
            sup = ((d < max_dist * size) & ~edge[idx] & ~cut
                   & (np.einsum("ij,ij->i", Nv, Ncn[idx]) > min_cos))
            mesh = trimesh.Trimesh(vertices=V, faces=Ft, process=False)
            _, cd, _ = trimesh.proximity.closest_point(mesh, Vc[::step])
            rec = dict(rung=rung, folds=n90, maxdih=mx,
                       support=float(sup.mean()),
                       cloud_med=float(np.median(cd)),
                       cloud_p90=float(np.percentile(cd, 90)),
                       sec=time.perf_counter() - t0)
            trace.append(rec)
            # fewest folds wins; accuracy breaks ties, so stiffening that
            # buys nothing cannot displace a better-fitting equal
            key = (n90, rec["cloud_med"])
            if best is None or key < (best["folds"], best["cloud_med"]):
                best, bestV = dict(rec), V.copy()
            if n90 == 0:
                break
        trimesh.Trimesh(vertices=bestV, faces=Ft, process=False).export(
            pathlib.Path(out) / f"{name}.obj")
        return dict(name=name, ok=1, tries=len(trace), **best, trace=trace)
    except Exception as e:
        return dict(name=name, ok=0, err=str(e)[:200], trace=[])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="AudioEar3D root")
    p.add_argument("--template", default=str(WRAP / "templates/ear_mean_v4.obj"))
    p.add_argument("--out", required=True)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--max-dist", type=float, default=0.06)
    p.add_argument("--max-rungs", type=int, default=len(LADDER))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--only", nargs="*", default=None)
    a = p.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    clouds = sorted(pathlib.Path(a.root).glob("*/*.ply"))
    if a.only:
        want = set(a.only)
        clouds = [c for c in clouds
                  if f"{c.parent.name}_{c.stem}" in want]
    if a.limit:
        clouds = clouds[:a.limit]
    print(f"{len(clouds)} clouds, ladder "
          f"{[r for r, _, _ in LADDER[:a.max_rungs]]}", flush=True)

    jobs = [(str(c), a.template, str(out), a.max_dist, a.max_rungs)
            for c in clouds]
    res = []
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(_one, j): j for j in jobs}
        for f in as_completed(futs):
            r = f.result()
            res.append(r)
            if r["ok"]:
                print(f"  {r['name']:14s} {r['rung']:12s} {r['tries']} fit(s) "
                      f" folds {r['folds']:3d}  cloud>surf {r['cloud_med']:.4f}"
                      f"  support {r['support']:.3f}", flush=True)
            else:
                print(f"  {r['name']:14s} FAILED {r['err']}", flush=True)

    ok = [r for r in res if r["ok"]]
    if ok:
        print(f"\n{sum(1 for r in ok if r['folds'] == 0)}/{len(ok)} fold-free; "
              f"mean folds {np.mean([r['folds'] for r in ok]):.2f}; "
              f"mean fits/ear {np.mean([r['tries'] for r in ok]):.1f}")
        print("rung used:", dict(collections.Counter(r["rung"] for r in ok)))
        print(f"cloud>surface median {np.mean([r['cloud_med'] for r in ok]):.4f}"
              f"  support {np.mean([r['support'] for r in ok]):.3f}")
    json.dump(res, open(out / "adaptive.json", "w"), indent=1)


if __name__ == "__main__":
    main()
