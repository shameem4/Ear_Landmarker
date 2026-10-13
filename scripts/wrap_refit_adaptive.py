#!/usr/bin/env python3
"""Re-fit an ear with Wrap, escalating parameters until the mesh does not fold.

No single Wrapping parameter fixes the folding. globalPoint2PointFittingWeight
looked decisive on four ears -- 10.8 folds to 0.8 -- and did not replicate: on
twelve fresh ears the best weight gives 2x, weight 2.0 is worse than control,
and the per-ear response is chaotic (pp31_left goes 2, 14, 7, 8, 3, 19 as the
weight rises). The weight is not suppressing a mechanism; it perturbs a
non-convex optimisation into a different local minimum and whether that
minimum folds is luck.

But the luck is per ear and cheap to buy. No single weight clears more than 7
of 12 ears while 10 of 12 are clean at SOME weight, and scan->fit is flat
across the whole ladder (0.122-0.130% of ear extent), so choosing per ear
costs nothing on the other axis. So: fit, count folds, escalate, stop clean.

FOLDS ARE COUNTED AT BASE RESOLUTION, which is what the exit condition tests.
Wrap returns a subdivided mesh, but everything downstream reads the first
len(template) vertices with the template's faces; the two counts disagree and
sometimes anti-correlate, so exiting on the subdivided one can stop on a fit
that is still folded where it matters. `folds_full` keeps the subdivided
count for comparison.

THE LADDER STARTS AT THE SHIPPED DEFAULT. An ear that does not fold today is
refitted with today's parameters and stops there, so its output is unchanged
rather than merely re-checked. Integrity for the ears that are already fine is
a property of the ordering, not something measured afterwards. The rungs after
it are ordered by how often they came out clean in the twelve-ear sweep.

    python scripts/wrap_refit_adaptive.py --ears pp31_left --out refit/
    python scripts/wrap_refit_adaptive.py --all --out refit/ --jobs 4
"""
import argparse
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

from ear3d.wrap import load_mesh                                   # noqa: E402
from wrap_batch_faceform import (WRAPCMD, corner_barycentric,      # noqa: E402
                                 spread_vertices)
from refit_variants import (accuracy, fold_stats, nodes_by_type,   # noqa: E402
                            one, rewire)

# (name, Wrapping parameter overrides). First rung is the shipped default.
# The rest are ordered by clean rate over the twelve-ear sweep: 128 cleared
# 7/12, 64 cleared 6/12, 4 cleared 5/12, 2 cleared 4/12, 32 cleared 3/12.
# Each rung is (name, Wrapping params, extra landmark anchors).
#
# EXTRA ANCHORS ARE A RESCUE, NEVER A DEFAULT. Added to every ear they are
# harmful -- on 20 fresh ears they broke 5 of the 8 that were already
# fold-free, introducing 1.75 folds each, which is why a flat A/B reads them
# as a regression. Added only to ears that are still folded, they rescue a
# class the parameter rungs cannot touch: on the 15 worst of 80 ears, all 5
# that no p2p rung improved were rescued, 3 of them to zero. The two
# mechanisms are complementary -- params alone reach 0 on 7/15, extras alone
# on 7/15, both available on 10/15 -- so the parameter rungs come first and
# the anchor rungs follow.
#
# They cannot fix coverage. The lobe and helix rim are short of real ears by
# template geometry, not by constraint; anchors there move coverage by less
# than its detection threshold.
# RUNG ORDER WAS RE-DERIVED FOR FRONT-ONLY ANCHORS. The original order came
# from a twelve-ear sweep run with anchors spread over the whole template;
# with front-only anchors the ranking changed and p2p_128, tried second,
# clears only 3 of the 15 worst ears while p2p_4 clears 8. Measured per rung
# on those 15: p2p_4 8, p2p_2 8, p2p_64 7, p2p_32 5, p2p_128 3.
LADDER = [
    ("default",  {}, ()),
    ("p2p_4",    {"globalPoint2PointFittingWeight": 4.0}, ()),
    ("p2p_64",   {"globalPoint2PointFittingWeight": 64.0}, ()),
    ("p2p_2",    {"globalPoint2PointFittingWeight": 2.0}, ()),
    ("p2p_32",   {"globalPoint2PointFittingWeight": 32.0}, ()),
    ("p2p_128",  {"globalPoint2PointFittingWeight": 128.0}, ()),
    # anchor rescue: cheapest first (asc broke only 1 of 8 clean ears,
    # lobe broke 5), and each is also tried at the strongest p2p rung
    ("asc",      {}, (1, 3)),
    ("asc_128",  {"globalPoint2PointFittingWeight": 128.0}, (1, 3)),
    ("lobe",     {}, (15, 17, 19)),
    ("lobe_128", {"globalPoint2PointFittingWeight": 128.0}, (15, 17, 19)),
    ("both",     {}, (1, 3, 15, 17, 19)),
]


def ear_sources(ear, dirs):
    for fd, sd in dirs:
        if (pathlib.Path(fd) / f"{ear}.obj").exists():
            return pathlib.Path(fd), pathlib.Path(sd)
    raise FileNotFoundError(f"{ear} is in none of the fit directories")


def fit_once(base, template, left, scan_p, dst, params, work, project,
             project_dist):
    """Run one Wrap job and return the mesh in the scan's own units."""
    import trimesh
    proj = json.loads(json.dumps(base))
    byid = {v["nodeId"]: v for v in proj["nodes"].values()}
    rig = one(proj, "RigidAlignment")
    byid[rig["params"]["geomFloating"]["connectedNodeId"]]["params"]["fileName"]["value"] = str(template)
    byid[rig["params"]["geomFloating"]["connectedNodeId"]]["params"]["scale"]["value"] = 1
    byid[rig["params"]["geomFixed"]["connectedNodeId"]]["params"]["fileName"]["value"] = str(scan_p)
    byid[rig["params"]["geomFixed"]["connectedNodeId"]]["params"]["scale"]["value"] = 1
    for n in nodes_by_type(proj, "ProjectGeom"):
        n["params"]["limitDist"]["value"] = True
        n["params"]["dist"]["value"] = project_dist
    for n in nodes_by_type(proj, "SelectPointPairs"):
        n["params"]["pointsLeft"]["value"] = left[0]
        n["params"]["pointsRight"]["value"] = left[1]
    one(proj, "SaveGeom")["params"]["fileName"]["value"] = str(dst)
    rewire(proj, project=project, wrap_params=params)
    proj["timeline"] = {"current": 0, "min": 0, "max": 0}
    pj = pathlib.Path(dst).parent / "work" / (pathlib.Path(dst).stem + ".wrap")
    pj.parent.mkdir(parents=True, exist_ok=True)
    json.dump(proj, open(pj, "w"), indent=4)
    r = subprocess.run([str(WRAPCMD), "compute", str(pj), "-s", "0", "-e", "0"],
                       capture_output=True, text=True, timeout=3600)
    bad = [l for l in (r.stdout + r.stderr).splitlines() if l.startswith("ERROR")]
    if bad or not pathlib.Path(dst).exists():
        raise RuntimeError(bad[0] if bad else "no output written")
    wm = trimesh.load(dst, process=False, force="mesh")
    V = np.asarray(wm.vertices) / work
    F = np.asarray(wm.faces)
    # a torn mesh passes a distance check: a few edges tens of times the
    # median length are a stretched fit, not a close one
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    el = np.linalg.norm(V[e[:, 0]] - V[e[:, 1]], axis=1)
    torn = int((el > 10 * np.median(el)).sum())
    trimesh.Trimesh(vertices=V, faces=F, process=False).export(dst)
    return V, F, torn


def refit(ear, cfg):
    """Walk the ladder for one ear; stop at the first fold-free fit."""
    import trimesh
    fd, sd = ear_sources(ear, cfg["dirs"])
    Vfit, _ = load_mesh(fd / f"{ear}.obj")
    Vs, Fs = load_mesh(sd / f"{ear}.obj")
    if ear.endswith("_right"):
        Vs, Fs = Vs * np.array([1.0, 1.0, -1.0]), Fs[:, ::-1]
    mesh = trimesh.Trimesh(vertices=Vs, faces=Fs, process=False)

    def anchors(extra):
        """Template-side and scan-side anchors for a rung.

        The scan-side position of every anchor is read from this ear's
        existing automatic fit, so an anchor without a counterpart there
        would be silently dropped -- which is how an earlier run ended up
        fitting on three points. Extra anchors are template VERTICES, so
        they always have one.
        """
        idx = list(cfg["idx"])
        for j in extra:
            v = int(cfg["lm_vert"][j])
            if v not in idx:
                idx.append(v)
        cl, _, st = trimesh.proximity.closest_point(mesh, Vfit[idx])
        bb = trimesh.triangles.points_to_barycentric(Vs[Fs[st]], cl)
        return ([cfg["left_of"][v] for v in idx],
                [[int(st[j]), float(bb[j][0]), float(bb[j][1])]
                 for j in range(len(idx))])
    work = cfg["work_size"] / float(np.ptp(Vs, axis=0).max())
    out = pathlib.Path(cfg["out"])
    (out / "work").mkdir(parents=True, exist_ok=True)
    scan_p = out / "work" / f"{ear}_scan.obj"
    trimesh.Trimesh(vertices=Vs * work, faces=Fs, process=False).export(scan_p)

    trace, best = [], None
    for rung, params, extra in LADDER[:cfg["max_rungs"]]:
        left, right = anchors(extra)
        dst = out / f"{ear}.obj"
        t0 = time.perf_counter()
        try:
            V, F, torn = fit_once(cfg["base"], cfg["template"],
                                  (left, right), scan_p, dst, params,
                                  work, cfg["project"], cfg["project_dist"])
        except Exception as e:
            trace.append((rung, None, str(e)[:100]))
            continue
        # SCORE THE EXIT AT BASE RESOLUTION. Wrap's output is subdivided,
        # but base_topology, the canonical corpus, the PCA and the template
        # all use the first len(Vt) vertices with the template's own faces.
        # The two resolutions disagree and sometimes anti-correlate, so a
        # ladder that exits on the subdivided count can stop on a fit that is
        # still folded in the only mesh anything downstream reads.
        nb, Ft_ = cfg["n_base"], cfg["base_faces"]
        ang_full = fold_stats(V, F)
        ang = fold_stats(V[:nb], Ft_) if len(V) > nb else ang_full
        n90 = int((ang > 90).sum())
        # accuracy() is a closest-point query over the whole scan, ~1s, and
        # it is only ever read to break a tie on equal fold counts. Skipping
        # it on rungs that cannot win saves roughly a second per rung across
        # the corpus for no change in which fit is chosen.
        if best is None or n90 <= best["folds"]:
            acc = accuracy(V, F, Vs, Fs)
        else:
            acc = (float("inf"), float("nan"), float("nan"), float("nan"))
        rec = dict(rung=rung, folds=n90, folds_full=int((ang_full > 90).sum()),
                   anchors=len(right),
                   maxdih=float(ang.max()), maxdih_full=float(ang_full.max()),
                   torn=torn, scan2fit=acc[0], p95=acc[1], cov=acc[3],
                   sec=time.perf_counter() - t0)
        trace.append(rec)
        if best is None or (n90, rec["scan2fit"]) < (best["folds"],
                                                     best["scan2fit"]):
            best = dict(rec)
            # keep the winning mesh; a later rung would overwrite the file
            import shutil
            shutil.copy(dst, out / "work" / f"{ear}_best.obj")
        if n90 == 0 and torn == 0:
            break
    if best is None:
        return dict(ear=ear, ok=False, trace=trace)
    import shutil
    shutil.copy(out / "work" / f"{ear}_best.obj", out / f"{ear}.obj")
    return dict(ear=ear, ok=True, tries=len(trace), **best, trace=trace)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fits", default=str(WRAP / "corpus/wrapped3_hutubs"))
    p.add_argument("--scans", default=str(WRAP / "corpus/ears4_hutubs"))
    p.add_argument("--fits2", default=str(WRAP / "corpus/wrapped3_sonicom"))
    p.add_argument("--scans2", default=str(WRAP / "corpus/ears3_sonicom"))
    p.add_argument("--template", default=str(WRAP / "templates/ear_L_template.obj"))
    p.add_argument("--base", default=str(WRAP / "ear_wrap_pp12_left.wrap"))
    p.add_argument("--out", required=True)
    p.add_argument("--ears", nargs="+", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--points", type=int, default=14)
    p.add_argument("--front-only", action="store_true",
                   help="place the anchors only on front-facing template "
                        "vertices. Confirmed on 20 ears never used in "
                        "tuning: base folds -72%%, better on 17, worse on 0, "
                        "p < 0.0001, for +0.0016 pp of scan>fit. The back's "
                        "geometry still matters and must stay -- deleting it "
                        "is 3.5x worse -- it is only its control points that "
                        "drag the fit onto a hair bridge.")
    p.add_argument("--work-size", type=float, default=100.0)
    p.add_argument("--project", action="store_true",
                   help="keep ProjectGeom. Off by default: it buys a 5.4x "
                        "tighter fit only where the fit was already good, and "
                        "folds the mesh doing it")
    p.add_argument("--project-dist", type=float, default=0.2)
    p.add_argument("--max-rungs", type=int, default=len(LADDER))
    p.add_argument("--jobs", type=int, default=1)
    a = p.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    Vt, Ft = load_mesh(a.template)
    if a.front_only:
        from ear3d.wrap import vertex_normals
        N = vertex_normals(Vt, Ft)
        mean_p = WRAP / "templates/ear_mean_v4.obj"
        if mean_p.exists() and str(pathlib.Path(a.template).resolve()) != str(mean_p.resolve()):
            from ear3d.wrap import rigid_fit
            from gt_landmark_check import read_marks, DEFAULT_MARKS
            Vm, Fm = load_mesh(mean_p)
            tm, bm, _ = read_marks(DEFAULT_MARKS)
            Lm = np.einsum("ij,ijk->ik", bm[:55], Vm[Fm[tm[:55]]])
            R0, t0, s0 = rigid_fit(Vt, Vm, with_scale=True)
            Vi = (Vt @ R0.T) * s0 + t0
            nr = [int(np.argmin(np.linalg.norm(Vi - Lm[j], axis=1)))
                  for j in range(55)]
        else:
            nr = list(range(55))
        f = N[nr].mean(0)
        f /= np.linalg.norm(f)
        sub = np.flatnonzero((N @ f) > 0.3)
        idx = sub[spread_vertices(Vt[sub], a.points)]
        print(f"front-only anchors: {len(sub)}/{len(Vt)} front-facing vertices")
    else:
        idx = spread_vertices(Vt, a.points)
    cn = corner_barycentric(Ft)
    uv = [(1.0, 0.0), (0.0, 1.0), (0.0, 0.0)]
    left = [[int(cn[v][0]), *map(float, uv[cn[v][1]])] for v in idx]
    left_of = {int(v): [int(cn[v][0]), *map(float, uv[cn[v][1]])]
               for v in range(len(Vt))}
    if not a.front_only:                     # lm_vert is set above when front-only
        from ear3d.wrap import rigid_fit as _rf
        from gt_landmark_check import read_marks as _rm, DEFAULT_MARKS as _DM
        _Vm, _Fm = load_mesh(WRAP / "templates/ear_mean_v4.obj")
        _t, _b, _ = _rm(_DM)
        _L = np.einsum("ij,ijk->ik", _b[:55], _Vm[_Fm[_t[:55]]])
        _R, _tt, _s = _rf(Vt, _Vm, with_scale=True)
        _Vi = (Vt @ _R.T) * _s + _tt
        nr = [int(np.argmin(np.linalg.norm(_Vi - _L[j], axis=1)))
              for j in range(55)]
    dirs = [(a.fits, a.scans), (a.fits2, a.scans2)]
    ears = a.ears or []
    if a.all:
        for fd, _ in dirs:
            ears += [f.stem for f in sorted(pathlib.Path(fd).glob("*.obj"))]
    cfg = dict(dirs=dirs, idx=idx, left=left, base=json.load(open(a.base)),
               template=str(pathlib.Path(a.template).resolve()), out=str(out),
               work_size=a.work_size, project=a.project,
               project_dist=a.project_dist * a.work_size,
               max_rungs=a.max_rungs, base_faces=Ft, n_base=len(Vt),
               left_of=left_of, lm_vert=list(nr))
    print(f"{len(ears)} ears, ladder {[r for r, _, _ in LADDER[:a.max_rungs]]}",
          flush=True)

    res = []
    if a.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            futs = {ex.submit(refit, e, cfg): e for e in ears}
            for f in as_completed(futs):
                r = f.result()
                res.append(r)
                print(f"  {r['ear']:14s} {r.get('rung','-'):8s} "
                      f"{r.get('tries',0)} fit(s)  folds {r.get('folds','-')}"
                      f"  scan>fit {r.get('scan2fit',float('nan')):.4f}%",
                      flush=True)
    else:
        for e in ears:
            r = refit(e, cfg)
            res.append(r)
            print(f"  {r['ear']:14s} {r.get('rung','-'):8s} "
                  f"{r.get('tries',0)} fit(s)  folds {r.get('folds','-')}"
                  f"  scan>fit {r.get('scan2fit',float('nan')):.4f}%",
                  flush=True)

    ok = [r for r in res if r.get("ok")]
    clean = [r for r in ok if r["folds"] == 0]
    print(f"\n{len(clean)}/{len(res)} fold-free; "
          f"mean folds {np.mean([r['folds'] for r in ok]):.2f}; "
          f"mean fits/ear {np.mean([r['tries'] for r in ok]):.1f}")
    from collections import Counter
    print("rung used:", dict(Counter(r["rung"] for r in ok)))
    json.dump(res, open(out / "adaptive.json", "w"), indent=1)


if __name__ == "__main__":
    main()
