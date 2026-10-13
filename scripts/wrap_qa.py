#!/usr/bin/env python3
"""Score a wrapped ear for defects, with and without the scan it came from.

This exists because the folding went unnoticed for the whole life of the
corpus. Every one of the 588 fits was folded -- a median of 42 edges at base
resolution -- and nothing in the pipeline said so. The distance-to-surface
checks could not: a folded fit sits ON the scan. Its vertices are within a
tenth of a millimetre of the measured surface and its face normals still agree
with it; what has gone wrong is the SPACING between vertices, not their
position, so every check that asks "how far is the mesh from the scan" passes.

So the checks here are mostly intrinsic -- computable from the fit alone, with
no scan and no ground truth -- which is what makes them usable on new data as
a gate rather than as a post-mortem.

WHAT A FOLD LOOKS LIKE, measured on the corpus (worst ears, base resolution):

    triangle area at a fold   0.33-0.53x the mesh median
    edge length at a fold     0.64-0.70x the mesh median
    face normal vs the scan   still +0.80 (only 3 of 22 flipped)

Vertices drift tangentially along a smooth surface and bunch up; the surplus
surface pleats. The scan is not creased there -- median dihedral 1.0 degrees,
nothing past 90 -- so a dihedral spike in the FIT with no matching ridge in
the SCAN is the signature, and `--scan` checks exactly that.

    python scripts/wrap_qa.py --fits corpus/escalated_v1/hutubs \
        --scans corpus/ears4_hutubs
    python scripts/wrap_qa.py --fits corpus/escalated_v1/hutubs --baseline
"""
import argparse
import collections
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
WRAP = ROOT.parent / "wrap_work"
sys.path.insert(0, str(ROOT))

from ear3d.wrap import boundary_vertices, load_mesh          # noqa: E402

# Defaults; --baseline recomputes them from a corpus and writes them out.
THRESHOLDS = dict(
    folds_base=0,            # any folded edge at base resolution is a defect
    max_dihedral=120.0,      # degrees; the clean mean reaches only 58
    min_area_ratio=0.15,     # smallest triangle, as a fraction of the median
    min_edge_ratio=0.25,
    max_torn=0,              # edges over 10x the median length
    max_scan_dist_pct=1.0,   # % of ear extent, vertex to scan surface
)


def topology(F):
    ea = collections.defaultdict(list)
    for t, f in enumerate(F):
        for i in range(3):
            ea[tuple(sorted((int(f[i]), int(f[(i + 1) % 3]))))].append(t)
    E = np.array([e for e, ts in ea.items() if len(ts) == 2])
    A = np.array([ts for ts in ea.values() if len(ts) == 2])
    return E, A


def measure(V, F, E, A, Vs=None, Fs=None):
    """Every number this file knows how to compute about one fit."""
    p = V[F]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    area = 0.5 * np.linalg.norm(n, axis=1)
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-30)
    dih = np.degrees(np.arccos(
        np.clip(np.einsum("ij,ij->i", n[A[:, 0]], n[A[:, 1]]), -1, 1)))
    el = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    med_e, med_a = np.median(el), np.median(area)
    out = dict(
        folds_base=int((dih > 90).sum()),
        near_folds=int((dih > 60).sum()),
        max_dihedral=float(dih.max()),
        min_area_ratio=float(area.min() / max(med_a, 1e-30)),
        min_edge_ratio=float(el.min() / max(med_e, 1e-30)),
        torn=int((el > 10 * med_e).sum()),
        # the bunching signature, localised: how compressed are the triangles
        # around whichever edges are worst
        area_ratio_at_worst=float(
            np.median(area[A[np.argsort(-dih)[:10]].ravel()]) / max(med_a, 1e-30)),
    )
    if Vs is not None:
        import trimesh
        sm = trimesh.Trimesh(vertices=Vs, faces=Fs, process=False)
        ext = float(np.ptp(Vs, axis=0).max())
        d = trimesh.proximity.closest_point(sm, V)[1] / ext * 100
        out["scan_dist_med_pct"] = float(np.median(d))
        out["scan_dist_max_pct"] = float(d.max())
        out["unsupported_pct"] = float(100 * (d > 1.0).mean())
        # the discriminating one: a crease in the fit with no ridge in the scan
        cen = V[F].mean(1)
        ti = trimesh.proximity.closest_point(sm, cen)[2]
        sn = sm.face_normals[ti]
        out["normal_agreement"] = float(np.median(np.einsum("ij,ij->i", n, sn)))
        out["flipped_faces"] = int((np.einsum("ij,ij->i", n, sn) < 0).sum())
    return out


def verdict(m, th):
    bad = []
    if m["folds_base"] > th["folds_base"]:
        bad.append(f"{m['folds_base']} folded edges")
    if m["max_dihedral"] > th["max_dihedral"]:
        bad.append(f"max dihedral {m['max_dihedral']:.0f}deg")
    if m["min_area_ratio"] < th["min_area_ratio"]:
        bad.append(f"collapsed triangle ({m['min_area_ratio']:.3f} of median)")
    if m["min_edge_ratio"] < th["min_edge_ratio"]:
        bad.append(f"collapsed edge ({m['min_edge_ratio']:.3f} of median)")
    if m["torn"] > th["max_torn"]:
        bad.append(f"{m['torn']} torn edges")
    if m.get("scan_dist_med_pct", 0) > th["max_scan_dist_pct"]:
        bad.append(f"median {m['scan_dist_med_pct']:.2f}% from the scan")
    return bad


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fits", required=True)
    p.add_argument("--scans", default=None,
                   help="optional; enables the fit-creased-but-scan-smooth test")
    p.add_argument("--template", default=str(WRAP / "templates/ear_L_template.obj"))
    p.add_argument("--baseline", action="store_true",
                   help="report the distribution instead of pass/fail, and "
                        "write thresholds at the 99th percentile of it")
    p.add_argument("--thresholds", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()

    th = dict(THRESHOLDS)
    if a.thresholds:
        th.update(json.load(open(a.thresholds)))
    Vt, Ft = load_mesh(a.template)
    E, A = topology(Ft)
    files = sorted(pathlib.Path(a.fits).glob("*.obj"))
    if a.limit:
        files = files[:a.limit]
    if not files:
        sys.exit(f"no .obj under {a.fits}")

    rows, failed = [], []
    for f in files:
        V, _ = load_mesh(f)
        V = V[:len(Vt)]                       # the base control mesh
        Vs = Fs = None
        if a.scans:
            sp = pathlib.Path(a.scans) / f"{f.stem}.obj"
            if sp.exists():
                Vs, Fs = load_mesh(sp)
                if f.stem.endswith("_right"):
                    Vs, Fs = Vs * np.array([1., 1., -1.]), Fs[:, ::-1]
        m = measure(V, Ft, E, A, Vs, Fs)
        m["ear"] = f.stem
        rows.append(m)
        bad = verdict(m, th)
        if bad:
            failed.append((f.stem, bad))
            if not a.quiet:
                print(f"  FAIL {f.stem:14s} " + "; ".join(bad))

    keys = [k for k in rows[0] if k != "ear"]
    print(f"\n{len(files)} fits, {len(failed)} flagged "
          f"({100*len(failed)/len(files):.0f}%)")
    print(f"{'metric':22s} {'median':>10s} {'p90':>10s} {'p99':>10s} {'worst':>10s}")
    for k in keys:
        v = np.array([r[k] for r in rows], float)
        worst = v.max() if k not in ("min_area_ratio", "min_edge_ratio",
                                     "normal_agreement") else v.min()
        print(f"{k:22s} {np.median(v):10.4f} {np.percentile(v,90):10.4f} "
              f"{np.percentile(v,99):10.4f} {worst:10.4f}")
    if a.baseline:
        nt = dict(th)
        for k, q in (("max_dihedral", 99), ("max_scan_dist_pct", 99)):
            if k.replace("max_", "").replace("_pct", "") in keys or k in keys:
                pass
        v = lambda k: np.array([r[k] for r in rows], float)
        nt["max_dihedral"] = float(np.percentile(v("max_dihedral"), 99))
        nt["min_area_ratio"] = float(np.percentile(v("min_area_ratio"), 1))
        nt["min_edge_ratio"] = float(np.percentile(v("min_edge_ratio"), 1))
        if "scan_dist_med_pct" in keys:
            nt["max_scan_dist_pct"] = float(
                np.percentile(v("scan_dist_med_pct"), 99))
        out = a.out or "wrap_qa_thresholds.json"
        json.dump(nt, open(out, "w"), indent=1)
        print(f"\nthresholds at the 1st/99th percentile of this corpus -> {out}")
        for k, x in nt.items():
            print(f"  {k:22s} {x}")
    if a.out and not a.baseline:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
