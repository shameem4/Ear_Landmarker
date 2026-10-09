#!/usr/bin/env python3
"""Restate every cut -> head transform in millimetres, once, with provenance.

The saved transforms are in whatever units their cut script ran in:
cut_sonicom on the raw STL, so millimetres; cut_hutubs on load_head's copy,
which divides by that subject's head radius, so per-subject normalised units.
Every consumer has had to know which, re-read the original .ply to recover the
factor, and apply it in the right place. That is how a HUTUBS ear came to
measure larger than its own head, and how 'flat in depth' survived for weeks.

For a rigid transform the conversion is exact and trivial. With
x_head = R x_cut + t in native units, scaling both frames by k gives
x_head_mm = R x_cut_mm + k t. Only the translation changes.

Each output carries the whole chain -- source file, dataset, the factor, the
original matrix -- and is CHECKED by pushing points through both routes and
comparing, rather than trusted.

    python scripts/make_mm_transforms.py --tf corpus/tf_hutubs --out corpus/tf_mm_hutubs
"""
import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ear3d.frames import load_head                     # noqa: E402


def mm_factor(src, dataset):
    """Multiply native coordinates by this to get millimetres."""
    if dataset != "hutubs":
        return 1.0
    import open3d as o3d
    raw = np.asarray(o3d.io.read_triangle_mesh(src).vertices)
    if not len(raw):
        raise RuntimeError(f"could not read {src}")
    # exactly load_head's divisor; the .ply is in metres
    return float(np.abs(raw - raw.mean(0)).max()) * 1000.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tf", nargs="+", required=True)
    p.add_argument("--out", nargs="+", required=True)
    p.add_argument("--tol", type=float, default=1e-9)
    a = p.parse_args()
    if len(a.tf) != len(a.out):
        sys.exit("--tf and --out must pair up")

    rng = np.random.default_rng(0)
    for td, od in zip(a.tf, a.out):
        od = pathlib.Path(od)
        od.mkdir(parents=True, exist_ok=True)
        files = sorted(pathlib.Path(td).glob("*.npz"))
        worst, cache, n = 0.0, {}, 0
        for f in files:
            z = np.load(f, allow_pickle=True)
            M, src, ds = z["matrix"], str(z["source"]), str(z["dataset"])
            if src not in cache:
                cache[src] = mm_factor(src, ds)
            k = cache[src]
            Mmm = M.copy()
            Mmm[:3, 3] = M[:3, 3] * k

            # check: a point taken to the head natively then scaled must equal
            # the same point scaled then taken through the mm transform
            x = rng.normal(size=(64, 3)) * 20.0
            viaA = (x @ M[:3, :3].T + M[:3, 3]) * k
            viaB = (x * k) @ Mmm[:3, :3].T + Mmm[:3, 3]
            err = float(np.abs(viaA - viaB).max())
            worst = max(worst, err)
            if err > a.tol:
                sys.exit(f"{f.name}: the two routes disagree by {err:.2e} mm")

            np.savez(od / f.name, matrix=Mmm, to_mm=k, matrix_native=M,
                     source=src, dataset=ds, units="mm",
                     frame="cut (left form; right ears mirrored about z)",
                     mirror_right_about="z",
                     note="x_head_mm = R @ x_cut_mm + t; apply to a fit only "
                          "after undoing left-form mirroring for a right ear")
            n += 1
        print(f"{td} -> {od}: {n} transforms, worst round-trip {worst:.2e} mm")


if __name__ == "__main__":
    main()
