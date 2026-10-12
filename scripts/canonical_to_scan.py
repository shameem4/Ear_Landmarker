#!/usr/bin/env python3
"""One verified transform per ear: canonical millimetres -> that ear's scan.

WHY THIS FILE EXISTS. The canonical corpus is already one frame at one scale
(588 ears, mm, centroid identical to 0.00, long axis on +y). What was missing
was a single trustworthy way back to the scan, and composing it by hand cost
three separate bugs in one afternoon -- each of which left the landmarks ON
the scan surface, so nothing downstream complained:

  the barycentrics address the CANONICAL mesh, not the faceform2 fit;
  each wrap generation pairs with its own cut generation;
  corpus/tf_mm_*/matrix omits the scan's centroid (5.56 mm on hutubs pp10).

So the composition happens exactly once, here, and is VERIFIED against the
scan before it is written. Downstream code loads a 4x4 and does not think
about datasets, units, mirrors or normalisations at all.

THE CHAIN, left to right:

  canonical mm --to_original--> the fit's own frame (native units)
               --mirror z-->    the cut's frame, for a RIGHT ear only, because
                                every right ear was mirrored into left form
                                before fitting so that one template serves
  --tf matrix--> the scan AS cut_ears.py saw it
  --un-normalise--> the scan's own coordinates
  --x1000 if metres--> millimetres

THE UN-NORMALISE STEP IS PER DATASET and is the part nothing records. cut_ears
verifies its matrix against `load_head` for hutubs, which CENTRES the mesh and
divides by its largest centred coordinate, and against the raw vertices for
sonicom. A matrix that inverts the cut perfectly -- those files all carry
verified_gap ~1e-17 -- still lands 450 mm away if this is skipped.

    python scripts/canonical_to_scan.py --dataset hutubs --out corpus/canon2scan
"""
import argparse
import csv
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
WRAP = ROOT.parent / "wrap_work"

# How cut_ears.py had normalised the scan when it saved the cut's matrix. The
# names are the two branches of its own verification, kept verbatim so the
# correspondence is checkable by reading one line of that file.
NORMALISE = {"hutubs": "load_head", "sonicom": "raw"}

# The scan is reached to better than this, or the row is not written. The wrap
# fits the ear to 0.002 mm and the cut inverts exactly, so a correct chain
# lands a few hundredths of a millimetre out; a tenth is already generous.
TOL_MM = 0.5


def un_normalise(V_scan, how):
    """(scale, offset) taking cut_ears' working coordinates back to the scan."""
    if how == "raw":
        return 1.0, np.zeros(3)
    if how == "load_head":
        c = V_scan.mean(0)
        return float(np.abs(V_scan - c).max()), c
    raise ValueError(how)


def compose(name, dataset, side, scan_cache):
    """The 4x4 canonical mm -> scan mm, plus what it was checked against."""
    import open3d as o3d

    cn = np.load(WRAP / f"corpus/canonical_esc/{dataset}/{name}.npz",
                 allow_pickle=True)
    tf = np.load(WRAP / f"corpus/tf_{dataset}/{name}.npz", allow_pickle=True)
    scan_path = str(tf["source"])

    if scan_path not in scan_cache:
        m = o3d.io.read_triangle_mesh(scan_path)
        if len(m.vertices) == 0:
            return None, scan_path, None
        V = np.asarray(m.vertices)
        # metres or millimetres: no head is under 10 mm tall by any measure
        to_mm = 1000.0 if float(np.ptp(V, axis=0).max()) < 10.0 else 1.0
        sc = o3d.t.geometry.RaycastingScene()
        hm = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(V * to_mm), m.triangles)
        sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(hm))
        scan_cache.clear()                       # one scan at a time; they are big
        scan_cache[scan_path] = (V, to_mm, sc)
    V_scan, to_mm, scene = scan_cache[scan_path]

    T0 = cn["to_original"]                       # canonical mm -> fit native
    M = tf["matrix"]                             # cut native -> normalised scan
    s0, c0 = un_normalise(V_scan, NORMALISE[dataset])
    mir = np.diag([1.0, 1.0, -1.0 if side == "right" else 1.0])

    # x_scan_mm = ((mir @ T0 x) @ M.R' + M.t) * s0 * to_mm + c0 * to_mm
    A = (mir @ T0[:3, :3])
    b = mir @ T0[:3, 3]
    A = M[:3, :3] @ A
    b = M[:3, :3] @ b + M[:3, 3]
    A = A * (s0 * to_mm)
    b = b * (s0 * to_mm) + c0 * to_mm
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = A, b

    g = o3d.io.read_triangle_mesh(
        str(WRAP / f"corpus/canonical_esc/{dataset}/{name}.obj"))
    X = np.asarray(g.vertices) @ A.T + b
    d = scene.compute_distance(o3d.core.Tensor(
        np.ascontiguousarray(X, np.float32))).numpy()
    return T, scan_path, (float(np.median(d)), float(np.percentile(d, 90)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", nargs="+", default=["hutubs", "sonicom"],
                   help="audioear has no scan to reach: ear-only clouds, no "
                        "corpus/tf_audioear, and nothing downstream wants it")
    p.add_argument("--out", default=str(WRAP / "corpus/canon2scan"))
    p.add_argument("--tol", type=float, default=TOL_MM)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()

    rows = [r for r in csv.DictReader(
        open(WRAP / "corpus/canonical_esc/manifest.csv", encoding="utf-8"))
        if r["dataset"] in a.dataset]
    # sorted by subject so both ears reuse one loaded scan
    rows.sort(key=lambda r: (r["dataset"], r["name"]))
    if a.limit:
        rows = rows[:a.limit]
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cache, written, bad, miss = {}, [], [], []
    for i, r in enumerate(rows):
        name, ds, side = r["name"], r["dataset"], r["side"]
        try:
            T, scan, res = compose(name, ds, side, cache)
        except FileNotFoundError as e:
            miss.append((name, str(e).split("/")[-1]))
            continue
        if T is None or res is None:
            miss.append((name, "scan unreadable"))
            continue
        med, p90 = res
        np.savez(out / f"{name}.npz", matrix=T, inverse=np.linalg.inv(T),
                 scan=scan, dataset=ds, side=side, units="mm",
                 resid_median_mm=med, resid_p90_mm=p90,
                 frame="canonical mm (one frame, one scale) -> this scan, mm",
                 note="x_scan_mm = matrix[:3,:3] @ x_canonical_mm + matrix[:3,3]")
        (written if med <= a.tol else bad).append((name, med, p90))
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(rows)}] {name} {med:.3f} mm", flush=True)

    allr = written + bad
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "resid_median_mm", "resid_p90_mm", "ok"])
        for n, m, q in sorted(allr):
            w.writerow([n, round(m, 4), round(q, 4), int(m <= a.tol)])

    print(f"\n{len(allr)} composed, {len(written)} within {a.tol} mm of the scan")
    if allr:
        d = np.array([m for _, m, _ in allr])
        print(f"  residual to the scan: median {np.median(d):.4f} mm, "
              f"p90 {np.percentile(d, 90):.4f}, max {d.max():.4f}")
    if bad:
        print(f"  {len(bad)} OVER TOLERANCE -- these are frame failures, not "
              f"data:")
        for n, m, q in sorted(bad, key=lambda x: -x[1])[:10]:
            print(f"    {n:<22} {m:8.3f} mm (p90 {q:.3f})")
    if miss:
        print(f"  {len(miss)} missing a piece: "
              f"{', '.join(n for n, _ in miss[:6])}"
              f"{' ...' if len(miss) > 6 else ''}")


if __name__ == "__main__":
    main()
