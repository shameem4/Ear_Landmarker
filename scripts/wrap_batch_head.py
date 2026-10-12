#!/usr/bin/env python3
"""Wrap the ear template onto the WHOLE HEAD, not onto a cut-out ear.

WHY THE CUT WAS A MISTAKE. Wrap's Wrapping node takes any target mesh; it
needs control points and a sane initial alignment, not a cropped target. The
crop was a convenience, and it cost something real: the template's own open
boundary could only ever land on the cut edge, and that edge is a BOX, sized
differently per dataset -- SONICOM a fixed (30, 52, 42) mm half-extent,
HUTUBS (0.30, 0.42, 0.50) x tragion separation. So the fitted template's rim
sat wherever a box happened to slice the head, varying per subject and per
source. Measured on the old fits, surface variance correlated -0.507 with
distance to that edge, and the ten hand-placed outer_helix_ext marks landed
0.7-1.9 mm from it, anchored to nothing anatomical.

Wrapped to an uncropped head the template's rim settles where the TEMPLATE
says, which is the same anatomy on every subject by construction.

THE TRAP, AND IT IS SILENT. The old driver set its working scale from the
TARGET's extent. A cut ear is ~104 mm, a whole head ~260 mm, so reusing that
rule would leave ProjectGeom's absolute distance limit about 2.5x too
permissive and let the rim crawl onto the scalp -- with no error, just worse
fits. Here the scale comes from the EAR, taken as the extent of our own fit,
so "2% of work size" means the same millimetres whatever surrounds it.

EVERYTHING IS IN MILLIMETRES. HUTUBS heads are read through load_head, which
divides by that subject's head radius, so they are scaled back by the factor
recorded alongside; SONICOM STLs are already mm. Each output records the units,
the frame, the scale and the provenance of every input.

    python scripts/wrap_batch_head.py --fits corpus/faceform2_sonicom \\
        --transforms corpus/tf_sonicom --template templates/ear_mean_v4.obj \\
        --base ear_wrap_pp12_left.wrap --out corpus/head_sonicom
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

from ear3d.frames import load_head                     # noqa: E402
from ear3d.wrap import base_topology, load_mesh        # noqa: E402

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
    out = {}
    for t, f in enumerate(F):
        for c, v in enumerate(f):
            out.setdefault(int(v), (t, c))
    return out


def head_mm(src, dataset, k):
    """The source head in MILLIMETRES.

    HUTUBS must still be read through load_head, because its cut transform was
    built against that normalised copy and the two have to agree about where
    the origin is; `k` then restores millimetres. SONICOM STLs are already mm.
    The factor is not recomputed here -- it is carried in the mm transform, so
    there is exactly one place it is derived.
    """
    import open3d as o3d
    if dataset == "hutubs":
        m = load_head(src)
        return np.asarray(m.vertices) * k, np.asarray(m.triangles)
    m = o3d.io.read_triangle_mesh(src)
    return np.asarray(m.vertices), np.asarray(m.triangles)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fits", required=True, help="our automatic fits, cut frame")
    p.add_argument("--transforms", required=True,
                   help="tf_mm_* dir from make_mm_transforms.py")
    p.add_argument("--template", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--points", type=int, default=14)
    p.add_argument("--work-size", type=float, default=100.0,
                   help="scale so the EAR is this many units; not the head")
    p.add_argument("--project-dist", type=float, default=0.02)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--keep-projects", action="store_true")
    a = p.parse_args()

    import trimesh
    out = pathlib.Path(a.out)
    (out / "work").mkdir(parents=True, exist_ok=True)
    (out / "audit").mkdir(parents=True, exist_ok=True)
    Vt, Ft = load_mesh(a.template)
    idx = spread_vertices(Vt, a.points)
    corner = corner_barycentric(Ft)
    # Wrap reads P = u*a + v*b + (1-u-v)*c, so corner a is (1,0), b is (0,1),
    # c is (0,0). The obvious [(0,0),(1,0),(0,1)] puts every control point on
    # the WRONG corner; see wrap_barycentric.
    uv = [(1.0, 0.0), (0.0, 1.0), (0.0, 0.0)]
    left = [[int(corner[v][0]), *map(float, uv[corner[v][1]])] for v in idx]

    fits = sorted(pathlib.Path(a.fits).glob("*.obj"))
    if a.only:
        fits = [f for f in fits if f.stem in set(a.only)]
    if a.limit:
        fits = fits[:a.limit]
    base = json.load(open(a.base))
    print(f"{len(fits)} ears, {a.points} control points, target = WHOLE HEAD",
          flush=True)

    done, failed, rows = 0, [], []
    t0 = time.perf_counter()
    for i, f in enumerate(fits, 1):
        name = f.stem
        try:
            tf = np.load(pathlib.Path(a.transforms) / f"{name}.npz",
                         allow_pickle=True)
            if str(tf.get("units", "")) != "mm":
                raise RuntimeError("expects the mm transforms from "
                                   "make_mm_transforms.py")
            M, src, ds = tf["matrix"], str(tf["source"]), str(tf["dataset"])
            k_mm = float(tf["to_mm"])
            Vf, Ff = load_mesh(f)
            Vf, Ff = base_topology(Vf, Ff, Vt, Ft, name)
            Vh_mm, Fh = head_mm(src, ds, k_mm)

            # THE HEAD IS BROUGHT INTO THE CUT FRAME, not the fit into head
            # coordinates. The fits are stored in LEFT form -- right ears are
            # mirrored about z -- so a right fit is not in the frame its own
            # cut transform expects, and applying M to it directly puts the
            # anchors off the surface. Measured before this was fixed:
            # pp10_left snapped at 4e-07 of ear size and pp10_right at 0.153.
            # Inverting M instead keeps every existing convention, because the
            # mirror is a plain z flip in the cut frame and is not a plain
            # flip in head coordinates.
            Vh = (Vh_mm - M[:3, 3]) @ M[:3, :3]        # head, cut frame, mm
            Vf_mm = Vf * k_mm                          # fit, cut frame, mm
            if name.endswith("_right"):
                Vh = Vh * np.array([1.0, 1.0, -1.0])
                Fh = Fh[:, ::-1]
            ear_extent = float(np.ptp(Vf_mm, axis=0).max())

            # THE EAR sets the working scale, not the head
            w = a.work_size / ear_extent
            Vh_w, Vf_w = Vh * w, Vf_mm * w

            mesh = trimesh.Trimesh(vertices=Vh_w, faces=Fh, process=False)
            close, dist, stri = trimesh.proximity.closest_point(
                mesh, Vf_w[idx])
            sb = trimesh.triangles.points_to_barycentric(Vh_w[Fh[stri]], close)
            right = [[int(stri[j]), float(sb[j][0]), float(sb[j][1])]
                     for j in range(len(idx))]
            anchor_err = float(np.max(dist)) / a.work_size

            scan_p = out / "work" / "scan.obj"
            trimesh.Trimesh(vertices=Vh_w, faces=Fh,
                            process=False).export(scan_p)

            proj = json.loads(json.dumps(base))
            byid = {v["nodeId"]: v for v in proj["nodes"].values()}
            ra = next(n for n in proj["nodes"].values()
                      if n["nodeType"] == "RigidAlignment")
            t_id = ra["params"]["geomFloating"]["connectedNodeId"]
            s_id = ra["params"]["geomFixed"]["connectedNodeId"]
            byid[t_id]["params"]["fileName"]["value"] = str(
                pathlib.Path(a.template).resolve())
            byid[t_id]["params"]["scale"]["value"] = w
            byid[s_id]["params"]["fileName"]["value"] = str(scan_p)
            byid[s_id]["params"]["scale"]["value"] = 1
            for n in proj["nodes"].values():
                if n["nodeType"] == "ProjectGeom":
                    n["params"]["limitDist"]["value"] = True
                    n["params"]["dist"]["value"] = a.project_dist * a.work_size
                if n["nodeType"] == "SelectPointPairs":
                    n["params"]["pointsLeft"]["value"] = left
                    n["params"]["pointsRight"]["value"] = right
                if n["nodeType"] == "SaveGeom":
                    n["params"]["fileName"]["value"] = str(out / f"{name}.obj")
            proj["timeline"] = {"current": 0, "min": 0, "max": 0}
            pj = out / "work" / f"{name}.wrap"
            json.dump(proj, open(pj, "w"), indent=4)

            r = subprocess.run([str(WRAPCMD), "compute", str(pj),
                                "-s", "0", "-e", "0"],
                               capture_output=True, text=True, timeout=900)
            bad = [l for l in (r.stdout + r.stderr).splitlines()
                   if l.startswith("ERROR")]
            if bad or not (out / f"{name}.obj").exists():
                raise RuntimeError(bad[0] if bad else "no output written")
            wm = trimesh.load(out / f"{name}.obj", process=False, force="mesh")
            Vo = np.asarray(wm.vertices) / w        # back to head millimetres
            Fo = np.asarray(wm.faces)
            # refuse a torn result: a few edges tens of times the median are a
            # stretched mesh, which a distance-to-surface check cannot see
            e = np.vstack([Fo[:, [0, 1]], Fo[:, [1, 2]], Fo[:, [2, 0]]])
            el = np.linalg.norm(Vo[e[:, 0]] - Vo[e[:, 1]], axis=1)
            torn = int((el > 10 * np.median(el)).sum())
            if torn:
                raise RuntimeError(f"torn: {torn} edges over 10x the median")
            trimesh.Trimesh(vertices=Vo, faces=Fo, process=False).export(
                out / f"{name}.obj")
            np.savez(out / "audit" / f"{name}.npz",
                     source=src, dataset=ds, to_mm=k_mm, work_scale=w,
                     ear_extent_mm=ear_extent, cut_to_scan=M,
                     anchor_err_frac=anchor_err,
                     head_faces=len(Fh), out_verts=len(Vo),
                     frame="cut frame, left form, millimetres",
                     template=str(pathlib.Path(a.template).resolve()),
                     project_dist=a.project_dist * a.work_size)
            rows.append(dict(name=name, dataset=ds, ear_mm=ear_extent,
                             anchor_err=anchor_err, head_faces=len(Fh),
                             out_verts=len(Vo)))
            done += 1
            if not a.keep_projects:
                pj.unlink(missing_ok=True)
        except Exception as e:
            failed.append((name, f"{type(e).__name__}: {e}"))
        if i % 10 == 0 or i == len(fits):
            el = time.perf_counter() - t0
            print(f"  [{i}/{len(fits)}] {done} ok, {len(failed)} failed, "
                  f"{el/i:.1f}s each", flush=True)

    import csv
    if rows:
        with open(out / "audit" / "manifest.csv", "w", newline="") as fh:
            wri = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wri.writeheader()
            wri.writerows(rows)
        ae = np.array([r["anchor_err"] for r in rows])
        print(f"\n{done} wrapped, {len(failed)} failed")
        print(f"  control-point snap error: median {100*np.median(ae):.2f}% "
              f"of ear, worst {100*ae.max():.2f}%")
    for n, why in failed[:8]:
        print(f"  ! {n}: {why}")


if __name__ == "__main__":
    main()
