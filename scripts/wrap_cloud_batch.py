#!/usr/bin/env python3
"""Wrap the template onto every AudioEar3D point cloud, in parallel.

Scores each fit two ways, and the directions mean different things:

  TEMPLATE -> CLOUD   how much of the template sat on measured points. This is
                      a coverage figure: it says how much of the result was
                      observed rather than invented.

  CLOUD -> SURFACE    for every scanned point, the distance to the fitted
                      surface. This is the accuracy figure, and it is the
                      honest one, because it cannot be satisfied by ignoring
                      data. A fit that explains only half the ear and floats
                      over the rest is near-zero in the first direction and bad
                      in this one.

Both are reported, because neither alone is enough: a template that collapsed
onto a single ridge scores well on cloud->surface over that ridge and terribly
on coverage.

    python scripts/wrap_cloud_batch.py --root AudioEar3D --template ear_mean_v3.obj \\
        --out fits/ --workers 8
"""
import argparse
import csv
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _one(args):
    cloud, template, out, max_dist, min_cos = args
    import trimesh
    from scipy.spatial import cKDTree

    from ear3d.wrap import (boundary_points, boundary_vertices, load_mesh,
                            nicp, rigid_icp, vertex_normals)

    def load_cloud(path):
        # inlined rather than imported from wrap_cloud, which would make
        # scripts/ a package just so the worker processes can reach it
        c = trimesh.load(path, process=False)
        raw = c.metadata["_ply_raw"]["vertex"]["data"]
        return (np.asarray(c.vertices, float),
                np.asarray(raw[["nx", "ny", "nz"]].tolist(), float))

    name = f"{cloud.parent.name}_{cloud.stem}"
    try:
        Vc, Nc = load_cloud(cloud)
        if cloud.stem == "right":
            Vc, Nc = Vc * [1.0, 1.0, -1.0], Nc * [1.0, 1.0, -1.0]
        Vt, Ft = load_mesh(template)
        size = float(np.ptp(Vt, axis=0).max())
        Ncn = Nc / np.maximum(np.linalg.norm(Nc, axis=1, keepdims=True), 1e-12)

        k0 = float(np.ptp(Vc, axis=0).max()) / size
        Vp = (Vt - Vt.mean(0)) * k0 + Vc.mean(0)
        V0, _ = rigid_icp(Vp, Ft, Vc, Ns=Nc, max_dist=0.3 * size,
                          with_scale=False, verbose=False)
        V = nicp(V0, Ft, Vc, Ns=Nc, max_dist=max_dist * size, min_cos=min_cos,
                 verbose=False)

        edge = boundary_points(Vc, Nc)
        d, idx = cKDTree(Vc).query(V)
        Nv = vertex_normals(V, Ft)
        cut = np.zeros(len(V), bool)
        cut[boundary_vertices(Ft)] = True
        sup = ((d < max_dist * size) & ~edge[idx] & ~cut
               & (np.einsum("ij,ij->i", Nv, Ncn[idx]) > min_cos))

        # cloud -> surface, the accuracy direction
        mesh = trimesh.Trimesh(vertices=V, faces=Ft, process=False)
        step = max(1, len(Vc) // 40000)          # 40k points is plenty
        _, cd, _ = trimesh.proximity.closest_point(mesh, Vc[::step])
        trimesh.Trimesh(vertices=V, faces=Ft, process=False).export(
            pathlib.Path(out) / f"{name}.obj")
        return dict(name=name, ok=1, support=float(sup.mean()),
                    cloud_med=float(np.median(cd)),
                    cloud_p90=float(np.percentile(cd, 90)),
                    cloud_p99=float(np.percentile(cd, 99)),
                    cloud_max=float(cd.max()), size=float(np.ptp(V, axis=0).max()),
                    err="")
    except Exception as e:
        return dict(name=name, ok=0, support=0.0, cloud_med=0.0, cloud_p90=0.0,
                    cloud_p99=0.0, cloud_max=0.0, size=0.0, err=str(e)[:200])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--template", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-dist", type=float, default=0.06)
    p.add_argument("--min-cos", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # exactly left.ply and right.ply: 011 carries a "right (copy).ply" that a
    # glob would wrap twice under a name that sorts next to the real one
    clouds = sorted(c for d in sorted(pathlib.Path(a.root).iterdir()) if d.is_dir()
                    for c in (d / "left.ply", d / "right.ply") if c.exists())
    if a.limit:
        clouds = clouds[:a.limit]
    print(f"{len(clouds)} clouds, {a.workers} workers", flush=True)

    jobs = [(c, a.template, str(out), a.max_dist, a.min_cos) for c in clouds]
    rows, t0 = [], time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_one, j): j[0] for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            rows.append(r)
            if not r["ok"]:
                print(f"  [{i}/{len(jobs)}] {r['name']} FAILED: {r['err']}",
                      flush=True)
            elif i % 10 == 0 or i == len(jobs):
                print(f"  [{i}/{len(jobs)}] {(time.perf_counter()-t0)/i:.1f}s "
                      f"each", flush=True)

    rows.sort(key=lambda r: r["name"])
    with open(out / "fits.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if r["ok"]]
    print(f"\n{len(ok)}/{len(rows)} fitted in "
          f"{(time.perf_counter()-t0)/60:.1f} min")
    if ok:
        s = np.array([r["support"] for r in ok])
        m = np.array([r["cloud_med"] for r in ok])
        n = np.array([r["cloud_p90"] for r in ok])
        print(f"  coverage (template on measured points): median {np.median(s):.1%}"
              f", worst {s.min():.1%}")
        print(f"  cloud -> surface: median of medians {np.median(m):.3f} mm, "
              f"median p90 {np.median(n):.3f} mm, worst p90 {n.max():.3f} mm")
        bad = sorted(ok, key=lambda r: -r["cloud_p90"])[:8]
        print("  worst by cloud->surface p90 -- look at these:")
        for r in bad:
            print(f"    {r['name']:<14s} p90 {r['cloud_p90']:6.3f} mm  "
                  f"coverage {r['support']:5.1%}")
    print(f"wrote {out / 'fits.csv'}")


if __name__ == "__main__":
    main()
