#!/usr/bin/env python3
"""A Wrap SelectPoints project -> the (triangle, barycentric) file propagation wants.

Keeps the hand-placed marks in Wrap, where a person can see and adjust them,
and hands the rest of the pipeline a plain array. The conversion is the one in
ear3d.wrap.wrap_barycentric, so the convention lives in a single place.

    python scripts/wrap_marks_to_npz.py --project mark_mean_v4.wrap \\
        --mesh ear_mean_v3.obj --out mean_v4_marks.npz
"""
import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ear3d.scheme import STRIPS                       # noqa: E402
from ear3d.wrap import load_mesh, wrap_barycentric    # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--mesh", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    V, F = load_mesh(a.mesh)
    d = json.load(open(a.project))
    sel = [n for n in d["nodes"].values() if n["nodeType"] == "SelectPoints"]
    if not sel:
        sys.exit(f"{a.project}: no SelectPoints node")
    pts = sel[0]["params"]["points"]["value"]
    n = len(pts)
    tri = np.zeros(n, int)
    bary = np.zeros((n, 3))
    # Names are carried through because an extended scheme's new points are
    # not covered by STRIPS, which describes the original 55 only. The key is
    # the only record of what a new point MEANS, so losing it here would leave
    # every downstream file with anonymous coordinates.
    names = [""] * n
    for k, q in pts.items():
        j = int(k[:2])
        names[j] = k
        u, v = float(q["u"]), float(q["v"])
        tri[j] = int(q["triangleInd"])
        bary[j] = (u, v, 1.0 - u - v)      # the order wrap_barycentric applies
    xyz = np.einsum("ij,ijk->ik", bary, V[F[tri]])

    # the same arithmetic by the shared helper, as a guard against this file
    # drifting from the one definition
    chk = wrap_barycentric(V, F, tri, bary[:, 0], bary[:, 1])
    err = float(np.abs(chk - xyz).max())
    if err > 1e-9:
        sys.exit(f"disagrees with wrap_barycentric by {err:.2e} mm")

    if "" in names:
        sys.exit("point indices are not contiguous from 00")
    np.savez(a.out, xyz=xyz, triangle=tri, barycentric=bary,
             strips=np.array(STRIPS), names=np.array(names))
    print(f"{n} marks from {pathlib.Path(a.project).name} -> {a.out}")
    print(f"  spanning {np.ptp(xyz, axis=0).round(1)} mm, "
          f"weights in [{bary.min():.3f}, {bary.max():.3f}]")
    if bary.min() < -1e-6:
        print("  WARNING: a negative weight means a point outside its triangle")


if __name__ == "__main__":
    main()
