#!/usr/bin/env python3
"""Wrap one template onto every cut ear in the corpus, unattended.

No per-scan anchors. scripts/cut_ears.py already puts every ear in the same
frame, so the template starts centred and scaled to the ear, a rigid ICP settles
the pose against the surface, and the non-rigid fit does the rest.

ONE TEMPLATE, NOT TWO. The right-ear template in the project is not a mirrored
copy of the left with matching vertex order -- different faces, different
orientation -- so fitting left ears with one and right ears with the other would
put the two halves of the corpus in different parameterisations, and a shape
model built on the union would be meaningless. Instead every RIGHT ear is
mirrored into left form (and its triangle winding flipped, or the normals point
inward and the fit rejects every correspondence) and the single left template is
fitted to everything.

Each fit writes an OBJ in template topology plus a row of metrics. The residual
distribution is the thing to read afterwards: if it is unimodal, one template
serves the corpus; if it splits, that is a real shape clustering and template
variants earn their place.

    python scripts/wrap_corpus.py --ears ears/ --template T.obj --out wrapped/
"""
import argparse
import csv
import pathlib
import sys
import time
import traceback

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ear3d.wrap import (DEFAULT_SCHEDULE, edges_of, load_mesh,  # noqa: E402
                        nicp, project_to_surface, rigid_icp)


def mirror_to_left(V, F):
    """Mirror a right ear into left form. Winding is flipped to match.

    THE AXIS IS Z, the anteroposterior one, not the protrusion axis. Left and
    right ears mirror across the body's sagittal plane, but each side is cut in
    its own frame and those frames already differ by a half turn about the
    vertical; composing the two, and then the mapping into the ear frame, lands
    the mirror on z. Measured, on a subject's own two ears aligned to each
    other: mirroring z leaves them 0.5-0.9% of ear size apart, which is what one
    person's two ears should be, while mirroring x leaves them ~2% apart -- no
    better than not mirroring at all, which is what it was really doing.

    Negating one axis reverses every triangle's orientation, so without the
    winding flip the normals point into the mesh, the normal-compatibility test
    rejects every correspondence, and the fit fails with no obvious cause.
    """
    return V * np.array([1.0, 1.0, -1.0]), F[:, ::-1]


def fit_one(Vs, Fs, Vt, Ft, stages):
    """Template onto one ear. Returns (vertices, metrics)."""
    # start centred and scaled to the ear: both are already in the same frame,
    # so this only has to be close enough for the rigid ICP to take over
    s0 = float(np.ptp(Vs, axis=0).max() / np.ptp(Vt, axis=0).max())
    V0 = (Vt - Vt.mean(0)) * s0 + Vs.mean(0)
    V0, (R, t, s) = rigid_icp(V0, Ft, Vs, Fs)
    V = nicp(V0, Ft, Vs, Fs, anchors=None, schedule=DEFAULT_SCHEDULE[:stages],
             verbose=False)
    _, d = project_to_surface(V, Vs, Fs)
    ear = float(np.ptp(V, axis=0).max())
    # STRETCH is the metric that can actually fail. Surface gap cannot: the data
    # term pulls every vertex onto the scan, so a fit that slid onto the wrong
    # fold still scores ~0. Stretch asks how much the template had to distort to
    # get there -- a fit that slid, folded or collapsed has to stretch some
    # edges far past the rest, and that shows up where the gap says nothing.
    E = edges_of(Ft)
    e0 = np.linalg.norm(Vt[E[:, 0]] - Vt[E[:, 1]], axis=1)
    e1 = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    # Degenerate template edges (the quad mesh triangulates to a few near-zero
    # ones) would divide through to a ratio in the tens of thousands and make
    # the maximum pure noise, so they are excluded rather than clamped.
    ok = e0 > 0.05 * np.median(e0)
    r = e1[ok] / e0[ok]
    r = r / np.median(r)                  # scale-free: only the SPREAD matters
    return V, {
        "scale": round(s0 * s, 4),
        "ear_extent": round(ear, 2),
        "gap_median_pct": round(100 * float(np.median(d)) / ear, 3),
        "gap_p90_pct": round(100 * float(np.percentile(d, 90)) / ear, 3),
        "stretch_p95": round(float(np.percentile(r, 95)), 3),
        "stretch_max": round(float(r.max()), 3),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ears", required=True, help="directory of cut ears")
    p.add_argument("--template", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--csv", default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--stages", type=int, default=len(DEFAULT_SCHEDULE))
    a = p.parse_args()

    Vt, Ft = load_mesh(a.template)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ears = sorted(pathlib.Path(a.ears).glob("*.obj"))
    if a.limit:
        ears = ears[:a.limit]
    print(f"{len(ears)} ears, template {pathlib.Path(a.template).name} "
          f"({len(Vt)} verts)", flush=True)

    rows, failed = [], []
    t0 = time.perf_counter()
    for i, f in enumerate(ears):
        name = f.stem
        try:
            Vs, Fs = load_mesh(f)
            if name.endswith("_right"):
                Vs, Fs = mirror_to_left(Vs, Fs)
            if len(Vs) < 1000:
                raise RuntimeError(f"only {len(Vs)} vertices in the cut")
            V, m = fit_one(Vs, Fs, Vt, Ft, a.stages)
            import trimesh
            trimesh.Trimesh(vertices=V, faces=Ft, process=False).export(
                out / f"{name}.obj")
            m["name"] = name
            rows.append(m)
            print(f"  [{i+1}/{len(ears)}] {name:<22s} gap p90 "
                  f"{m['gap_p90_pct']:5.2f}%  stretch p95 {m['stretch_p95']:5.2f} "
                  f"max {m['stretch_max']:6.2f}  scale {m['scale']:.3f}", flush=True)
        except Exception as e:
            failed.append((name, f"{type(e).__name__}: {e}"))
            print(f"  [{i+1}/{len(ears)}] {name:<22s} FAILED: {e}", flush=True)
            traceback.print_exc()

    dt = time.perf_counter() - t0
    print(f"\n{len(rows)}/{len(ears)} wrapped in {dt/60:.1f} min "
          f"({dt/max(len(ears),1):.1f}s each)")
    if rows:
        st = np.array([r["stretch_p95"] for r in rows])
        print(f"stretch p95 across the corpus: median {np.median(st):.3f}  "
              f"p90 {np.percentile(st,90):.3f}  worst {st.max():.3f}")
        bad = sorted(rows, key=lambda r: -r["stretch_p95"])[:12]
        print("most distorted fits -- look at these before trusting them:")
        for r in bad:
            print(f"  {r['name']:<22s} stretch p95 {r['stretch_p95']:5.2f} "
                  f"max {r['stretch_max']:6.2f}  gap p90 {r['gap_p90_pct']:5.2f}%")
    if failed:
        print(f"\n{len(failed)} FAILED:")
        for n, why in failed:
            print(f"  {n}: {why}")
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["name", "scale", "ear_extent",
                                               "gap_median_pct", "gap_p90_pct",
                                               "stretch_p95", "stretch_max"])
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {a.csv}")


if __name__ == "__main__":
    main()
