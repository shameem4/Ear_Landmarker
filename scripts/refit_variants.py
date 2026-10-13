#!/usr/bin/env python3
"""Compare Wrap graph variants on a few ears, before re-fitting the corpus.

Every fit in the corpus is folded: 588 of 588 carry edges whose two faces meet
at more than 90 degrees, a median of 42 of them, and the same holds at full
subdivided resolution (about 290 of 20600 edges). The folds are not anatomy --
only 2 edges fold in 95% of fits while 250 fold in under 10%, so they move
around per ear, and the generic template they all start from is clean.

The shipped graph is

    template(1188v) -> RigidAlignment -> Wrapping -> SubdivideGeom -> ProjectGeom

so the non-rigid fit has only the 1188-vertex control mesh to work with, and
the dense detail is added afterwards by ProjectGeom, which snaps every vertex
to its nearest scan point with NO smoothness term. On a thin ridge adjacent
vertices snap to opposite faces of it, which is a 180 degree fold on a
normal-length edge -- exactly what the corpus shows.

So the fix under test is to subdivide BEFORE the fit and let Wrapping's own
smoothness regularisation carry the detail. Wrap's subdivision keeps the
original vertices first, so the control-point indices stay valid and the
output keeps the vertex count it has today.

    python scripts/refit_variants.py --ears pp10_left P0001_left --variants all
"""
import argparse
import collections
import json
import pathlib
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
WRAP = ROOT.parent / "wrap_work"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ear3d.wrap import load_mesh                                  # noqa: E402
from wrap_batch_faceform import (WRAPCMD, corner_barycentric,     # noqa: E402
                                 spread_vertices)

# name -> what it changes about the shipped graph
VARIANTS = {
    "baseline":     dict(),
    "pre":          dict(subdivide_first=True),
    "pre_noproj":   dict(subdivide_first=True, project=False),
    "pre_stiff":    dict(subdivide_first=True, smooth_min=0.25),
    "noproj":       dict(project=False),
    # a tightened projection rather than none: does a short snap buy the
    # surface accuracy back without reintroducing the folds?
    "proj_tight":   dict(project_dist=0.002),
    "proj_tiny":    dict(project_dist=0.0005),
    # detail without mesh subdivision: Wrapping's OWN deformation-graph
    # resolution, which is the lever that was actually meant for this
    "noproj_fine":  dict(project=False, wrap_nsub=6),
    "noproj_finer": dict(project=False, wrap_nsub=7),
    # SUBDIVIDE BEFORE THE FIT, projection off. Subdividing halves the edge
    # length, so a smoothness weight calibrated for the 1188-vertex control
    # mesh is far floppier per vertex on the dense one; the earlier pre_stiff
    # test raised it with PROJECTION STILL ON, which confounds it -- a stiffer
    # wrap leaves a bigger residual for ProjectGeom to snap across.
    "pre_np_s05":   dict(subdivide_first=True, project=False, smooth_min=0.05),
    "pre_np_s15":   dict(subdivide_first=True, project=False, smooth_min=0.15),
    "pre_np_s30":   dict(subdivide_first=True, project=False, smooth_min=0.30),
    "pre_np_s50":   dict(subdivide_first=True, project=False, smooth_min=0.50),
    "pre_np_s80":   dict(subdivide_first=True, project=False, smooth_min=0.80),
    "pre_np_hi":    dict(subdivide_first=True, project=False, smooth_min=0.50,
                         smooth_max=3.0),
    # THE CORRECT subdivide-before-fit: the template file itself is already
    # subdivided, so the control points are computed on the fitted mesh and
    # SubdivideGeom is bypassed entirely.
    "sub_np":       dict(sub_template=True, project=False),
    "sub_np_s15":   dict(sub_template=True, project=False, smooth_min=0.15),
    "sub_np_s30":   dict(sub_template=True, project=False, smooth_min=0.30),
    "sub_proj":     dict(sub_template=True, project_dist=0.002),
}


def nodes_by_type(proj, t):
    return [n for n in proj["nodes"].values() if n["nodeType"] == t]


def one(proj, t):
    return nodes_by_type(proj, t)[0]


def connect(node, param, target):
    node["params"][param]["connectedNodeId"] = target["nodeId"]


def rewire(proj, subdivide_first=False, project=True, smooth_min=None,
           n_subdiv=None, wrap_nsub=None, project_dist=None,
           smooth_max=None, sub_template=False, wrap_params=None):
    """Apply a variant to a cloned project, in place."""
    rig = one(proj, "RigidAlignment")
    wrp = one(proj, "Wrapping")
    sub = one(proj, "SubdivideGeom")
    prj = one(proj, "ProjectGeom")
    sav = one(proj, "SaveGeom")
    pp = next(n for n in nodes_by_type(proj, "SelectPointPairs")
              if n["nodeId"] == wrp["params"]["points"]["connectedNodeId"])

    if n_subdiv is not None:
        sub["params"]["nSubdivisions"]["value"] = n_subdiv
    if smooth_min is not None:
        wrp["params"]["globalSmoothWeightMin"]["value"] = smooth_min
    for k, v in (wrap_params or {}).items():
        if k not in wrp["params"]:
            raise KeyError(f"Wrapping has no parameter {k!r}; it has "
                           f"{sorted(wrp['params'])}")
        wrp["params"][k]["value"] = v
    if smooth_max is not None:
        wrp["params"]["globalSmoothWeightMax"]["value"] = smooth_max
    if wrap_nsub is not None:
        wrp["params"]["nSubdivisions"]["value"] = wrap_nsub

    if sub_template:
        # already subdivided on disk: SubdivideGeom is skipped, not moved
        connect(wrp, "geomFloating", rig)
        connect(pp, "geomLeft", rig)
        last = wrp
    elif subdivide_first:
        connect(sub, "geom", rig)        # subdivide the aligned template
        connect(wrp, "geomFloating", sub)
        connect(pp, "geomLeft", sub)
        last = wrp
    else:
        connect(sub, "geom", wrp)
        connect(wrp, "geomFloating", rig)
        connect(pp, "geomLeft", rig)
        last = sub

    if project:
        connect(prj, "geomSrc", last)
        connect(sav, "geom", prj)
    else:
        connect(sav, "geom", last)
    return proj


def make_sub_template(base, template, dst, scan_p):
    """Write Wrap's own subdivision of the template, to use AS the template.

    Rewiring SubdivideGeom to sit before Wrapping does NOT work: a control
    point is stored as (triangleIndex, u, v), and subdivision renumbers every
    triangle while keeping only the vertices. The points then land somewhere
    arbitrary and the fit silently degrades. Subdividing offline and treating
    the result as the template keeps the control points consistent, because
    they are recomputed on the mesh that is actually fitted.
    """
    proj = json.loads(json.dumps(base))
    byid = {v["nodeId"]: v for v in proj["nodes"].values()}
    rig = one(proj, "RigidAlignment")
    tn = byid[rig["params"]["geomFloating"]["connectedNodeId"]]
    sn = byid[rig["params"]["geomFixed"]["connectedNodeId"]]
    tn["params"]["fileName"]["value"] = str(pathlib.Path(template).resolve())
    tn["params"]["scale"]["value"] = 1
    sn["params"]["fileName"]["value"] = str(scan_p)
    sn["params"]["scale"]["value"] = 1
    sub = one(proj, "SubdivideGeom")
    connect(sub, "geom", tn)                     # straight off the loaded file
    connect(one(proj, "SaveGeom"), "geom", sub)
    one(proj, "SaveGeom")["params"]["fileName"]["value"] = str(dst)
    proj["timeline"] = {"current": 0, "min": 0, "max": 0}
    pj = pathlib.Path(dst).parent / "make_sub.wrap"
    json.dump(proj, open(pj, "w"), indent=4)
    r = subprocess.run([str(WRAPCMD), "compute", str(pj), "-s", "0", "-e", "0"],
                       capture_output=True, text=True, timeout=1800)
    if not pathlib.Path(dst).exists():
        raise RuntimeError((r.stdout + r.stderr)[-400:])
    return dst


def fold_stats(V, F):
    adj = collections.defaultdict(list)
    for t, f in enumerate(F):
        for i in range(3):
            adj[tuple(sorted((int(f[i]), int(f[(i + 1) % 3]))))].append(t)
    A = np.array([ts for ts in adj.values() if len(ts) == 2])
    p = V[F]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-30
    return np.degrees(np.arccos(
        np.clip(np.einsum("ij,ij->i", n[A[:, 0]], n[A[:, 1]]), -1, 1)))


def accuracy(V, F, Vs, Fs):
    """scan->fit where the template has support, as % of ear extent.

    Unrestricted scan->fit is dominated by the uncut head around the ear,
    which no variant covers and which therefore moves identically for all of
    them. Restricting to the supported region is what makes the number read
    on the fit rather than on the crop. Percent of extent, because HUTUBS
    cuts are in normalised head units and SONICOM is in millimetres.
    """
    import trimesh
    # THE FIT AND THE SCAN MUST BE IN THE SAME UNITS. Fits written at the
    # 100-unit working scale have to be divided by w before they get here.
    # Forgetting it does not look like an error: on SONICOM (native ~104)
    # it is a 4% discrepancy that passes, while on HUTUBS (native ~0.85) it
    # is 118x, and the proximity query below then searches a fit that
    # dwarfs its scan -- 21.9 GB per worker, and an empty `cov` mask.
    ext = float(np.ptp(Vs, axis=0).max())
    ratio = max(ext, (ef := float(np.ptp(V, axis=0).max()))) / max(min(ext, ef), 1e-12)
    if ratio > 3.0:
        raise ValueError(
            f"accuracy(): fit and scan are {ratio:.0f}x apart in extent "
            f"({ef:.4g} vs {ext:.4g}). A fit saved at work scale must be "
            f"divided by w first.")
    fit = trimesh.Trimesh(vertices=V, faces=F, process=False)
    scan = trimesh.Trimesh(vertices=Vs, faces=Fs, process=False)
    d1 = trimesh.proximity.closest_point(scan, V)[1]          # fit -> scan
    d2 = trimesh.proximity.closest_point(fit, Vs)[1]          # scan -> fit
    cov = d2 < 0.03 * ext
    return (100 * float(np.median(d2[cov])) / ext,
            100 * float(np.percentile(d2[cov], 95)) / ext,
            100 * float(np.median(d1)) / ext,
            100 * float(cov.mean()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fits", default=str(WRAP / "corpus/wrapped3_hutubs"))
    p.add_argument("--scans", default=str(WRAP / "corpus/ears4_hutubs"))
    p.add_argument("--fits2", default=str(WRAP / "corpus/wrapped3_sonicom"))
    p.add_argument("--scans2", default=str(WRAP / "corpus/ears3_sonicom"))
    p.add_argument("--template", default=str(WRAP / "templates/ear_L_template.obj"))
    p.add_argument("--sub-template",
                   default=str(WRAP / "templates/ear_L_template_sub1.obj"))
    p.add_argument("--base", default=str(WRAP / "ear_wrap_pp12_left.wrap"))
    p.add_argument("--out", default=str(WRAP / "refit_variants"))
    p.add_argument("--ears", nargs="+", required=True)
    p.add_argument("--variants", nargs="+", default=["all"])
    p.add_argument("--combo", nargs="*", default=[],
                   help="NAME:p1=v1,p2=v2 -- one variant combining several "
                        "Wrapping parameters, projection off")
    p.add_argument("--grid", nargs="*", default=[],
                   help="NAME=v1,v2,... one variant per value, projection "
                        "off. Isolates Wrapping's own contribution")
    p.add_argument("--points", type=int, default=14)
    p.add_argument("--front-only", action="store_true",
                   help="sample control points only from front-facing template "
                        "vertices. 5 of the 14 farthest-point anchors are not "
                        "clearly front-facing and one points backwards; the "
                        "back of the ear is not of interest and its anchors "
                        "spend budget away from the region that folds")
    p.add_argument("--extra-lm", nargs="*", type=int, default=[],
                   help="landmark indices to add as EXTRA control points. "
                        "Farthest-point sampling spreads over the surface and "
                        "puts nothing in the inner helix (none of the 14 land "
                        "in landmarks 20-35), which is the least-constrained "
                        "region and where the folds are. Hand-fixing the three "
                        "worst ears needed 21 of 23 extra points there")
    p.add_argument("--work-size", type=float, default=100.0)
    p.add_argument("--project-dist", type=float, default=0.02)
    a = p.parse_args()

    import trimesh
    if a.combo:
        for spec in a.combo:
            nm, kv = spec.split(":", 1)
            VARIANTS[nm] = dict(project=False, wrap_params={
                k: float(v) for k, v in
                (x.split("=", 1) for x in kv.split(","))})
        names = ["control"] + [spec.split(":", 1)[0] for spec in a.combo]
        VARIANTS.setdefault("control", dict(project=False))
    elif a.grid:
        for spec in a.grid:
            k, vals = spec.split("=", 1)
            for v in vals.split(","):
                VARIANTS[f"{k[:14]}@{v}"] = dict(
                    project=False, wrap_params={k: float(v)})
        names = [n for n in VARIANTS if "@" in n]
        if "control" not in VARIANTS:
            VARIANTS["control"] = dict(project=False)
        names = ["control"] + names
    else:
        names = list(VARIANTS) if a.variants == ["all"] else a.variants
    out = pathlib.Path(a.out)
    (out / "work").mkdir(parents=True, exist_ok=True)
    base = json.load(open(a.base))
    uv = [(1.0, 0.0), (0.0, 1.0), (0.0, 0.0)]

    def front_mask(V, F):
        from ear3d.wrap import rigid_fit, vertex_normals
        from gt_landmark_check import read_marks, DEFAULT_MARKS
        Vm_, Fm_ = load_mesh(WRAP / "templates/ear_mean_v4.obj")
        tm, bm, _ = read_marks(DEFAULT_MARKS)
        Lm_ = np.einsum("ij,ijk->ik", bm[:55], Vm_[Fm_[tm[:55]]])
        R_, t_, s_ = rigid_fit(V, Vm_, with_scale=True)
        Vi = (V @ R_.T) * s_ + t_
        N = vertex_normals(V, F)
        nr = np.array([int(np.argmin(np.linalg.norm(Vi - Lm_[j], axis=1)))
                       for j in range(55)])
        f = N[nr].mean(0); f /= np.linalg.norm(f)
        return (N @ f) > 0.3

    def control_points(tpl, limit=None):
        """Spread control points over the template.

        limit caps the sample to the first `limit` vertices. That matters for
        a SUBDIVIDED template: its anchors must have counterparts on our
        coarse automatic fit, which is where each point's position on this
        particular scan is read from. Sampling all 4676 vertices put most
        anchors on subdivision midpoints with no counterpart, they were
        dropped, and the fit ran on the 3 or 4 that survived -- which is how
        SONICOM ended up 30x worse while HUTUBS looked merely poor.
        """
        Vt, Ft = load_mesh(tpl)
        if a.front_only:
            m = front_mask(Vt, Ft)
            if limit:
                m[limit:] = False
            sub = np.flatnonzero(m)
            ix = sub[spread_vertices(Vt[sub], a.points)]
            cn = corner_barycentric(Ft)
            return Vt, [[int(cn[v][0]), *map(float, uv[cn[v][1]])] for v in ix], list(ix)
        ix = spread_vertices(Vt[:limit] if limit else Vt, a.points)
        cn = corner_barycentric(Ft)
        return Vt, [[int(cn[v][0]), *map(float, uv[cn[v][1]])] for v in ix], ix

    Vt, left, idx = control_points(a.template)
    Ft = load_mesh(a.template)[1]
    if a.extra_lm:
        # the marks address ear_mean_v3; carry them onto the template frame and
        # take the nearest template VERTEX, so the anchor has a counterpart on
        # our coarse automatic fit (which is where its scan position is read)
        from ear3d.wrap import rigid_fit
        from gt_landmark_check import read_marks, DEFAULT_MARKS
        Vm, Fm = load_mesh(WRAP / "templates/ear_mean_v4.obj")
        tri_m, bary_m, _ = read_marks(DEFAULT_MARKS)
        Lm = np.einsum("ij,ijk->ik", bary_m[:55], Vm[Fm[tri_m[:55]]])
        R0, t0, s0 = rigid_fit(Vt, Vm, with_scale=True)
        Vin = (Vt @ R0.T) * s0 + t0
        cn = corner_barycentric(Ft)
        uvv = [(1.0, 0.0), (0.0, 1.0), (0.0, 0.0)]
        add = []
        for j in a.extra_lm:
            v = int(np.argmin(np.linalg.norm(Vin - Lm[j], axis=1)))
            if v in set(idx) or v in add:
                continue
            add.append(v)
        left = left + [[int(cn[v][0]), *map(float, uvv[cn[v][1]])] for v in add]
        idx = list(idx) + add
        print(f"added {len(add)} control points at landmarks {a.extra_lm} "
              f"-> {len(idx)} total", flush=True)      # base topology, for base-resolution folds
    sub_tpl, sub_left, sub_idx, sub_Vt = None, None, None, None
    if any(VARIANTS[v].get("sub_template") for v in names):
        sub_tpl = pathlib.Path(a.sub_template)
        if not sub_tpl.exists():
            print(f"building {sub_tpl} with Wrap's own subdivision", flush=True)
            sub_tpl.parent.mkdir(parents=True, exist_ok=True)
            seed = sorted(pathlib.Path(a.scans).glob("*.obj"))[0]
            make_sub_template(base, a.template, str(sub_tpl), str(seed))
        sub_Vt, sub_left, sub_idx = control_points(sub_tpl, limit=len(Vt))
        print(f"subdivided template: {len(sub_Vt)} vertices "
              f"(coarse {len(Vt)})", flush=True)
    print(f"template {len(Vt)} v; {len(names)} variants x {len(a.ears)} ears",
          flush=True)

    rows = []
    for ear in a.ears:
        for fd, sd in ((a.fits, a.scans), (a.fits2, a.scans2)):
            if (pathlib.Path(fd) / f"{ear}.obj").exists():
                break
        else:
            print(f"{ear}: not found in either corpus")
            continue
        Vfit_src, _ = load_mesh(pathlib.Path(fd) / f"{ear}.obj")
        Vs, Fs = load_mesh(pathlib.Path(sd) / f"{ear}.obj")
        if ear.endswith("_right"):
            Vs, Fs = Vs * np.array([1.0, 1.0, -1.0]), Fs[:, ::-1]
        mesh = trimesh.Trimesh(vertices=Vs, faces=Fs, process=False)

        def scan_points(ix):
            # our automatic fit shares the COARSE topology, so a subdivided
            # template's extra vertices have no counterpart on it; the spread
            # sample is taken over the first len(Vfit_src) either way
            ixc = [v for v in ix if v < len(Vfit_src)]
            cl, _, st = trimesh.proximity.closest_point(mesh, Vfit_src[ixc])
            bb = trimesh.triangles.points_to_barycentric(Vs[Fs[st]], cl)
            return ixc, [[int(st[j]), float(bb[j][0]), float(bb[j][1])]
                         for j in range(len(ixc))]

        _, right = scan_points(idx)
        w = a.work_size / float(np.ptp(Vs, axis=0).max())
        scan_p = out / "work" / f"{ear}_scan.obj"
        trimesh.Trimesh(vertices=Vs * w, faces=Fs, process=False).export(scan_p)

        for vn in names:
            proj = json.loads(json.dumps(base))
            byid = {v["nodeId"]: v for v in proj["nodes"].values()}
            rig = one(proj, "RigidAlignment")
            tn = byid[rig["params"]["geomFloating"]["connectedNodeId"]]
            sn = byid[rig["params"]["geomFixed"]["connectedNodeId"]]
            use_sub = bool(VARIANTS[vn].get("sub_template"))
            tpl_path = sub_tpl if use_sub else pathlib.Path(a.template)
            vleft = sub_left if use_sub else left
            vidx = sub_idx if use_sub else idx
            keep, vright = scan_points(vidx)
            if len(keep) != len(vidx):
                raise RuntimeError(f"only {len(keep)} of {len(vidx)} control "
                                   f"points have a counterpart on the fit")
            tn["params"]["fileName"]["value"] = str(pathlib.Path(tpl_path).resolve())
            tn["params"]["scale"]["value"] = 1
            sn["params"]["fileName"]["value"] = str(scan_p)
            sn["params"]["scale"]["value"] = 1
            pdist = VARIANTS[vn].get("project_dist", a.project_dist)
            for n in nodes_by_type(proj, "ProjectGeom"):
                n["params"]["limitDist"]["value"] = True
                n["params"]["dist"]["value"] = pdist * a.work_size
            for n in nodes_by_type(proj, "SelectPointPairs"):
                n["params"]["pointsLeft"]["value"] = vleft
                n["params"]["pointsRight"]["value"] = vright
            dst = out / f"{ear}__{vn}.obj"
            one(proj, "SaveGeom")["params"]["fileName"]["value"] = str(dst)
            rewire(proj, **{k: v for k, v in VARIANTS[vn].items()
                            if k != "project_dist"})
            proj["timeline"] = {"current": 0, "min": 0, "max": 0}
            pj = out / "work" / f"{ear}_{vn}.wrap"
            json.dump(proj, open(pj, "w"), indent=4)

            t0 = time.perf_counter()
            try:
                r = subprocess.run(
                    [str(WRAPCMD), "compute", str(pj), "-s", "0", "-e", "0"],
                    capture_output=True, text=True, timeout=3600)
                dt = time.perf_counter() - t0
                bad = [l for l in (r.stdout + r.stderr).splitlines()
                       if l.startswith("ERROR")]
                if bad or not dst.exists():
                    raise RuntimeError(bad[0] if bad else "no output")
                wm = trimesh.load(dst, process=False, force="mesh")
                Vo = np.asarray(wm.vertices) / w
                Fo = np.asarray(wm.faces)
                trimesh.Trimesh(vertices=Vo, faces=Fo, process=False).export(dst)
                ang = fold_stats(Vo, Fo)
                angb = fold_stats(Vo[:len(Vt)], Ft)
                m1, p1, m2, p2 = accuracy(Vo, Fo, Vs, Fs)
                rows.append([ear, vn, len(Vo), len(Fo), dt, float(ang.max()),
                             int((ang > 90).sum()), int((ang > 60).sum()),
                             float(np.median(ang)), m1, p1, m2, p2,
                             int((angb > 90).sum()), float(angb.max())])
                print(f"  {ear:14s} {vn:20s} {dt:5.1f}s  base>90 "
                      f"{int((angb>90).sum()):3d}  full>90 {int((ang>90).sum()):3d}"
                      f"  scan>fit {m1:.4f}%  p95 {p1:.4f}%", flush=True)
            except Exception as e:
                print(f"  {ear:14s} {vn:12s} FAILED: {str(e)[:140]}", flush=True)

    if not rows:
        return
    print(f"\n{'variant':20s} {'sec':>5s} {'base>90':>8s} {'clean':>7s} "
          f"{'full>90':>8s} {'scan>fit':>9s} {'p95':>8s}   "
          f"(mean over ears, % of ear extent)")
    for vn in names:
        r = [x for x in rows if x[1] == vn]
        if not r:
            continue
        def g(i):
            return float(np.mean([x[i] for x in r]))
        nclean = sum(1 for x in r if x[13] == 0)
        print(f"{vn:20s} {g(4):5.1f} {g(13):8.2f} {nclean:4d}/{len(r):<2d} "
              f"{g(6):8.2f} {g(9):8.4f}% {g(10):7.4f}%")
    json.dump(rows, open(out / "rows.json", "w"), indent=1)


if __name__ == "__main__":
    main()
