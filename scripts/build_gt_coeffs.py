#!/usr/bin/env python3
"""Ground-truth shape coefficients per ear, for supervising a shape head.

This existed as an undocumented .npz with no script behind it, which is its own
kind of gremlin: nobody could say how it was made or regenerate it after the
corpus changed. This is that script.

PROJECT THE MESH, NOT THE LANDMARKS. The obvious implementation -- least
squares of each ear's 55 landmarks against the landmark basis -- is WRONG, and
wrong in a way that looks fine. The PCA components are orthonormal over all
1188 vertices, but restricted to 55 of them they are not orthogonal at all, so
lstsq against them inflates the coefficients: measured, per-mode sd came back
at 1.42 instead of 1.0, which would have scaled every supervised target.
Projecting the full mesh onto the orthonormal basis and dividing by each
mode's sd gives coefficients that are standardised by construction, and those
are exactly the coefficients the landmark basis expects, because
build_landmark_basis samples the SAME components at the marks with the same sd
scaling.

POSE IS REMOVED FIRST, by the same Procrustes the PCA was built with.
Otherwise rotation is charged to shape, which is the gauge freedom head 2
already demonstrated.

WHAT A COEFFICIENT MEANS: one standard deviation of that mode, so a unit
gaussian prior is the right one. The per-mode sd is printed; far from 1 means
the basis and the corpus do not match.

    python scripts/build_gt_coeffs.py --pca ear_pca_v5_all_mm.npz \
        --fits corpus/canonical_esc/sonicom ... --out gt_coeffs_v7_mm.npz
"""
import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ear3d.wrap import base_topology, load_mesh, rigid_fit      # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pca", required=True)
    p.add_argument("--fits", nargs="+", required=True)
    p.add_argument("--template", required=True)
    p.add_argument("--modes", type=int, default=0, help="0 keeps all stored")
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    d = np.load(a.pca, allow_pickle=True)
    mean, comp, var = d["mean"], d["components"], d["variance"]
    k = a.modes or len(comp)
    comp, var = comp[:k], var[:k]
    Vt, Ft = load_mesh(a.template)

    files = [f for d_ in a.fits for f in sorted(pathlib.Path(d_).glob("*.obj"))]
    V, names = [], []
    for f in files:
        Vi, Fi = load_mesh(str(f))
        Vi, _ = base_topology(Vi, Fi, Vt, Ft, f.name)
        V.append(Vi)
        names.append(f.stem)
    V = np.asarray(V)

    # the same generalised Procrustes build_ear_pca used, onto the PCA's mean
    A = V.copy()
    for _ in range(a.iters):
        for i in range(len(A)):
            R, t, s = rigid_fit(A[i], mean)
            A[i] = s * A[i] @ R.T + t

    X = (A - mean).reshape(len(A), -1)
    raw = X @ comp.T                       # orthonormal basis -> true projection
    sd = np.sqrt(var)
    coeffs = raw / sd                      # standardised: 1 = one sd of a mode
    recon = mean.reshape(-1) + raw @ comp
    resid = np.linalg.norm((recon - (mean.reshape(-1) + X)).reshape(len(A), -1, 3),
                           axis=2).mean(1)
    size = float(np.ptp(mean, axis=0).max())

    emp = coeffs.std(0)
    print(f"{len(A)} ears x {k} modes from {pathlib.Path(a.pca).name}")
    print(f"  per-mode sd: mean {emp.mean():.3f}  "
          f"[{emp.min():.3f}, {emp.max():.3f}]   (1.0 expected)")
    if not 0.9 < emp.mean() < 1.1:
        sys.exit("  ! far from 1 -- basis and corpus do not match")
    print(f"  mesh residual outside these {k} modes: "
          f"{100 * resid.mean() / size:.3f}% of ear size "
          f"(worst {100 * resid.max() / size:.3f}%)")
    np.savez_compressed(a.out, coeffs=coeffs, names=np.array(names), sd=sd,
                        pca=str(a.pca), residual=resid)
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
