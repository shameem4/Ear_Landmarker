#!/usr/bin/env python3
"""Wrap an ear template onto a scan, and score it against Faceform Wrap.

Run it with a .wrap project and it takes the SAME inputs the tool was given --
the same template, the same scan, the same hand-placed point pairs, read
straight out of the project's JSON -- so the comparison is of the algorithm,
not of the setup.

    python scripts/wrap_ear.py --project wrap_work/ear_wrap_pp12_left.wrap \
                               --reference wrap_work/ear_left_wrapped.obj

Two numbers come out, and they answer different questions:

  vs Wrap        how close this gets to the tool. The development target.
  vs the scan    how close the result sits to the surface. NOT a measure of
                 quality -- Wrap's own ProjectGeom drives it to zero by
                 construction, and so does --project-to-surface here. A fit
                 that landed on the wrong fold scores just as well.

The question neither answers is whether the correspondence is right, which is
what a shape model actually needs. For that, wrap two subjects and render the
checkerboard-by-vertex-index check (scripts/wrap_correspondence.py).
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ear3d.wrap import (DEFAULT_SCHEDULE, bary_to_xyz, load_mesh,  # noqa: E402
                        nicp, project_to_surface, rigid_fit, rigid_icp)


def read_project(path):
    """Pull the template, the scan and the point pairs out of a .wrap project.

    Wrap stores a pick as (triangle index, u, v) into the TRIANGULATED mesh, so
    the meshes have to be read the same way here -- trimesh splits each quad
    into two triangles in face order, which is the convention that reproduces
    the tool's own anchor positions.
    """
    path = pathlib.Path(path)
    d = json.load(open(path))
    nodes = list(d["nodes"].values())

    def resolve(f):
        """Projects outlive the paths in them -- assets get moved and the file
        still says where they used to be. Fall back to the basename near the
        project before giving up, so a tidy-up does not break the comparison."""
        f = pathlib.Path(f)
        if f.exists():
            return str(f)
        for cand in (path.parent / f.name, path.parent / "templates" / f.name,
                     path.parent.parent / f.name,
                     path.parent.parent / "templates" / f.name):
            if cand.exists():
                print(f"  (found {f.name} at {cand.parent})")
                return str(cand)
        sys.exit(f"cannot find {f} referenced by {path.name}")

    # LoadGeom carries a scale/rotation/translation, and it is not decoration:
    # the pp1 head is stored in metres and loaded at scale 1000, so ignoring it
    # puts the result in a space 1000x from the one the project worked in.
    loads = {}
    for n in nodes:
        if n["nodeType"] != "LoadGeom":
            continue
        q = n["params"]
        rot = q.get("rotation", {}).get("value", {"x": 0, "y": 0, "z": 0})
        if any(abs(float(rot.get(k, 0))) > 1e-9 for k in "xyz"):
            sys.exit(f"{path.name}: LoadGeom {n['nodeId']} has a rotation "
                     f"({rot}); this reader does not apply one, and guessing "
                     f"the Euler convention would silently misplace the mesh")
        tr = q.get("translation", {}).get("value", {"x": 0, "y": 0, "z": 0})
        loads[n["nodeId"]] = (resolve(q["fileName"]["value"]),
                              float(q.get("scale", {}).get("value", 1.0)),
                              np.array([float(tr.get(k, 0)) for k in "xyz"]))
    pairs = [n for n in nodes if n["nodeType"] == "SelectPointPairs"
             and n["params"]["pointsLeft"]["value"]]
    if not pairs:
        sys.exit(f"{path}: no point pairs in the project")
    # A project can hold two sets: a sparse one to orient the template, and a
    # denser one feeding the Wrapping node as control points. Use the sparse set
    # for the rigid step and the DENSEST for the fit, which is what the tool was
    # given -- scoring against Wrap while withholding half its input would
    # flatter or damn this solver for the wrong reason.
    node = min(pairs, key=lambda n: len(n["params"]["pointsLeft"]["value"]))
    ctrl = max(pairs, key=lambda n: len(n["params"]["pointsLeft"]["value"]))
    rig = next((n for n in nodes if n["nodeType"] == "RigidAlignment"), None)
    tpl_id = rig["params"]["geomFloating"]["connectedNodeId"] if rig else min(loads)
    scan_id = rig["params"]["geomFixed"]["connectedNodeId"] if rig else max(loads)
    return (loads[tpl_id], loads[scan_id],
            node["params"]["pointsLeft"]["value"],
            node["params"]["pointsRight"]["value"],
            ctrl["params"]["pointsLeft"]["value"],
            ctrl["params"]["pointsRight"]["value"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True, help="a .wrap project to copy inputs from")
    p.add_argument("--reference", default=None, help="Wrap's output, to score against")
    p.add_argument("--out", default=None, help="where to write the wrapped template")
    p.add_argument("--no-scale", action="store_true",
                   help="rigid alignment without scale, as Wrap's default")
    p.add_argument("--project-to-surface", action="store_true",
                   help="snap onto the scan at the end, like Wrap's ProjectGeom")
    p.add_argument("--template", default=None,
                   help="use this mesh as the template instead of the project's. "
                        "It must have the project template's topology -- then the "
                        "project's barycentric anchors still name the same "
                        "anatomy and transfer unchanged.")
    p.add_argument("--no-icp", action="store_true",
                   help="skip the surface rigid ICP and trust the anchors alone")
    p.add_argument("--stages", type=int, default=len(DEFAULT_SCHEDULE))
    a = p.parse_args()

    (tpl_path, tpl_s, tpl_t), (scan_path, scan_s, scan_t), L, R, CL, CR = \
        read_project(a.project)
    print(f"template {pathlib.Path(tpl_path).name}\nscan     {pathlib.Path(scan_path).name}")
    Vt, Ft = load_mesh(tpl_path)
    if a.template:
        Va, Fa = load_mesh(a.template)
        if len(Va) != len(Vt) or not np.array_equal(Fa, Ft):
            sys.exit(f"{a.template} has {len(Va)} verts / {len(Fa)} faces but the "
                     f"project's template has {len(Vt)} / {len(Ft)}. The anchors "
                     f"are barycentric on the project template's faces, so a "
                     f"different topology would scatter them.")
        print(f"  template overridden with {pathlib.Path(a.template).name} "
              f"(same topology, so the anchors carry over)")
        Vt, tpl_s, tpl_t = Va, 1.0, np.zeros(3)
    Vs, Fs = load_mesh(scan_path)
    Vt = Vt * tpl_s + tpl_t
    Vs = Vs * scan_s + scan_t
    if scan_s != 1 or tpl_s != 1:
        print(f"  project scales: template x{tpl_s:g}, scan x{scan_s:g}")
    print(f"  {len(Vt)} template verts, {len(Vs)} scan verts, "
          f"{len(L)} rigid anchors, {len(CL)} control points")

    at = bary_to_xyz(Vt, Ft, L)            # anchors on the template
    asc = bary_to_xyz(Vs, Fs, R)           # the same anatomy on the scan

    # Put the template on the scan first. NICP is a local method: it refines an
    # alignment, it does not find one.
    Rm, t, s = rigid_fit(at, asc, with_scale=not a.no_scale)
    V0 = s * Vt @ Rm.T + t
    at0 = s * at @ Rm.T + t
    print(f"  rigid: scale {s:.4f}, anchor residual "
          f"{np.linalg.norm(at0 - asc, axis=1).mean():.4f}")

    if not a.no_icp:
        # the anchors said roughly where; the surface says exactly where
        V0, (Ri, ti, si) = rigid_icp(V0, Ft, Vs, Fs, verbose=True)
        at0 = si * at0 @ Ri.T + ti

    # the control points steer the fit; they are given as positions on template
    # triangles, so pin the nearest template VERTEX to each -- the solver
    # constrains vertices, not surfaces
    ct = s * bary_to_xyz(Vt, Ft, CL) @ Rm.T + t
    if not a.no_icp:
        ct = si * ct @ Ri.T + ti
    ca = bary_to_xyz(Vs, Fs, CR)
    li = [int(np.argmin(np.linalg.norm(V0 - p, axis=1))) for p in ct]

    t0 = time.perf_counter()
    V = nicp(V0, Ft, Vs, Fs, anchors=(li, ca),
             schedule=DEFAULT_SCHEDULE[:a.stages])
    print(f"  nicp: {time.perf_counter() - t0:.1f}s")

    if a.project_to_surface:
        V, d = project_to_surface(V, Vs, Fs)
        print(f"  projected onto the scan (moved median {np.median(d):.4f})")

    # Scale errors by the EAR, not by the scan. One project crops to the ear and
    # another wraps onto a whole head, so a percentage of the scan's extent
    # means something different in each and the two cannot be compared.
    size = float(np.ptp(V, axis=0).max())
    if a.reference:
        Vr, _ = load_mesh(a.reference)
        Vr = Vr[:len(V)]                   # subdivision keeps the originals first
        e = np.linalg.norm(V - Vr, axis=1)
        print(f"\nvs Wrap  ({pathlib.Path(a.reference).name}, first {len(Vr)} verts):")
        print(f"  median {100*np.median(e)/size:5.2f}%  p90 {100*np.percentile(e,90)/size:5.2f}%"
              f"  max {100*e.max()/size:5.2f}%   of the ear's {size:.1f}-unit extent")

    _, d = project_to_surface(V, Vs, Fs)
    print(f"vs the scan surface: median {100*np.median(d)/size:.2f}%  "
          f"p90 {100*np.percentile(d,90)/size:.2f}%  (0 by construction if "
          f"--project-to-surface)")

    if a.out:
        import trimesh
        trimesh.Trimesh(vertices=V, faces=Ft, process=False).export(a.out)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
