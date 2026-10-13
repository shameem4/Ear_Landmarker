#!/usr/bin/env python3
"""Run Faceform Wrap over the corpus, unattended, using our fits as the anchors.

Wrap does the non-rigid fit better than this project's NICP -- measured, 1.7 to
3.8% apart on the ears where both were run. What it cannot do is decide WHERE
the ear is: RigidAlignment refuses to run without at least three point pairs,
which is why batching it has meant placing points on every frame by hand.

Our pipeline solves exactly that part. It has no anchors and finds the ear
anyway, so it can hand Wrap the control points and let Wrap do what it is good
at. Nobody clicks anything.

The points are taken by farthest-point sampling over the TEMPLATE, so they pin
the whole ear rather than clustering; each one's position on this particular
scan is read off our fitted mesh and snapped to the surface, which is where the
automatic fit's knowledge actually enters.

WrapCmd computes under the trial licence -- the "License state: Inactive" line
refers to maintenance, not to whether it runs.

    python scripts/wrap_batch_faceform.py --fits corpus/wrapped3_hutubs \\
        --scans corpus/ears4_hutubs --out batch/ --base ear_wrap_pp12_left.wrap
"""
import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ear3d.wrap import load_mesh                       # noqa: E402

WRAPCMD = pathlib.Path("/home/shameem/snap/Faceform_Wrap/"
                       "Faceform_Wrap_2025.11.14_Linux/WrapCmd.sh")


def spread_vertices(V, k):
    """Farthest-point sampling: k vertices spread over the whole mesh."""
    idx = [0]
    d = np.linalg.norm(V - V[0], axis=1)
    for _ in range(k - 1):
        idx.append(int(np.argmax(d)))
        d = np.minimum(d, np.linalg.norm(V - V[idx[-1]], axis=1))
    return idx


def corner_barycentric(F):
    """For each vertex, one triangle containing it and which corner it is."""
    out = {}
    for t, f in enumerate(F):
        for c, v in enumerate(f):
            out.setdefault(int(v), (t, c))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fits", required=True, help="our automatic fits")
    p.add_argument("--scans", required=True, help="the cut ears they were fitted to")
    p.add_argument("--template", required=True)
    p.add_argument("--base", required=True, help="a working .wrap to clone")
    p.add_argument("--out", required=True)
    p.add_argument("--points", type=int, default=14)
    p.add_argument("--work-size", type=float, default=100.0,
                   help="scale every scan to this before wrapping. ProjectGeom's "
                        "dist is an ABSOLUTE distance, so a project tuned on one "
                        "dataset tears meshes on another: at dist=1 a HUTUBS cut "
                        "(extent 0.84) lets vertices project 1.2x the whole ear "
                        "and the mesh tears, while a SONICOM cut (extent 104 mm) "
                        "limits them to 1% and the step does nothing")
    p.add_argument("--project-dist", type=float, default=0.02,
                   help="ProjectGeom limit as a fraction of --work-size")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--only", nargs="*", default=None)
    a = p.parse_args()

    import trimesh
    out = pathlib.Path(a.out)
    (out / "work").mkdir(parents=True, exist_ok=True)
    Vt, Ft = load_mesh(a.template)
    idx = spread_vertices(Vt, a.points)
    corner = corner_barycentric(Ft)
    # Wrap reads P = u*a + v*b + (1-u-v)*c, so corner a is (1,0), b is (0,1)
    # and c is (0,0). The obvious-looking [(0,0),(1,0),(0,1)] puts every
    # control point on the WRONG CORNER of its triangle; see wrap_barycentric.
    uv = [(1.0, 0.0), (0.0, 1.0), (0.0, 0.0)]
    left = [[int(corner[v][0]), *map(float, uv[corner[v][1]])] for v in idx]

    fits = sorted(pathlib.Path(a.fits).glob("*.obj"))
    if a.only:
        fits = [f for f in fits if f.stem in set(a.only)]
    if a.limit:
        fits = fits[:a.limit]
    print(f"{len(fits)} ears, {a.points} control points each", flush=True)

    base = json.load(open(a.base))
    done, failed = 0, []
    t0 = time.perf_counter()
    for i, f in enumerate(fits, 1):
        name = f.stem
        try:
            Vf, _ = load_mesh(f)
            Vs, Fs = load_mesh(pathlib.Path(a.scans) / f"{name}.obj")
            if name.endswith("_right"):
                # our fits are stored in left form, so the scan must match
                Vs, Fs = Vs * np.array([1.0, 1.0, -1.0]), Fs[:, ::-1]
            mesh = trimesh.Trimesh(vertices=Vs, faces=Fs, process=False)
            close, dist, stri = trimesh.proximity.closest_point(mesh, Vf[idx])
            sb = trimesh.triangles.points_to_barycentric(Vs[Fs[stri]], close)
            right = [[int(stri[j]), float(sb[j][0]), float(sb[j][1])]
                     for j in range(len(idx))]

            # one working scale for every ear, so the absolute limits inside
            # the project mean the same thing on every frame
            w = a.work_size / float(np.ptp(Vs, axis=0).max())
            scan_p = out / "work" / "scan.obj"
            trimesh.Trimesh(vertices=Vs * w, faces=Fs, process=False).export(scan_p)
            right = [[r[0], r[1], r[2]] for r in right]   # barycentric: unchanged
            proj = json.loads(json.dumps(base))
            byid = {v["nodeId"]: v for v in proj["nodes"].values()}
            tpl = next(n for n in proj["nodes"].values()
                       if n["nodeType"] == "RigidAlignment")
            t_id = tpl["params"]["geomFloating"]["connectedNodeId"]
            s_id = tpl["params"]["geomFixed"]["connectedNodeId"]
            byid[t_id]["params"]["fileName"]["value"] = str(pathlib.Path(a.template).resolve())
            byid[t_id]["params"]["scale"]["value"] = 1
            byid[s_id]["params"]["fileName"]["value"] = str(scan_p)
            byid[s_id]["params"]["scale"]["value"] = 1
            for n in proj["nodes"].values():
                if n["nodeType"] == "ProjectGeom":
                    n["params"]["limitDist"]["value"] = True
                    n["params"]["dist"]["value"] = a.project_dist * a.work_size
            for n in proj["nodes"].values():
                if n["nodeType"] == "SelectPointPairs":
                    n["params"]["pointsLeft"]["value"] = left
                    n["params"]["pointsRight"]["value"] = right
                if n["nodeType"] == "SaveGeom":
                    n["params"]["fileName"]["value"] = str(out / f"{name}.obj")
            proj["timeline"] = {"current": 0, "min": 0, "max": 0}
            pj = out / "work" / "job.wrap"
            json.dump(proj, open(pj, "w"), indent=4)

            r = subprocess.run([str(WRAPCMD), "compute", str(pj), "-s", "0", "-e", "0"],
                               capture_output=True, text=True, timeout=600)
            bad = [l for l in (r.stdout + r.stderr).splitlines()
                   if l.startswith("ERROR")]
            if bad or not (out / f"{name}.obj").exists():
                raise RuntimeError(bad[0] if bad else "no output written")
            # back to the scan's own units, and refuse a torn result: a few
            # edges tens of times the median length are a stretched mesh, which
            # a distance-to-surface check cannot see
            wm = trimesh.load(out / f"{name}.obj", process=False, force="mesh")
            Vo = np.asarray(wm.vertices) / w
            Fo = np.asarray(wm.faces)
            e = np.vstack([Fo[:, [0, 1]], Fo[:, [1, 2]], Fo[:, [2, 0]]])
            el = np.linalg.norm(Vo[e[:, 0]] - Vo[e[:, 1]], axis=1)
            torn = int((el > 10 * np.median(el)).sum())
            if torn:
                raise RuntimeError(f"torn: {torn} edges over 10x the median")
            trimesh.Trimesh(vertices=Vo, faces=Fo, process=False).export(
                out / f"{name}.obj")
            done += 1
            if i % 25 == 0 or i == len(fits):
                print(f"  [{i}/{len(fits)}] {done} ok, {len(failed)} failed, "
                      f"{(time.perf_counter()-t0)/i:.1f}s each", flush=True)
        except Exception as e:
            failed.append((name, str(e)[:160]))
            print(f"  [{i}/{len(fits)}] {name} FAILED: {str(e)[:160]}", flush=True)

    dt = time.perf_counter() - t0
    print(f"\n{done}/{len(fits)} wrapped by Faceform in {dt/60:.1f} min")
    if failed:
        print(f"{len(failed)} failed:")
        for n, why in failed[:20]:
            print(f"  {n}: {why}")


if __name__ == "__main__":
    main()
