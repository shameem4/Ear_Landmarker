#!/usr/bin/env python3
"""Project the ear shape model onto the 55 landmarks.

A model that predicts shape coefficients does not need the 1188-vertex mesh at
every step -- only where the 55 marks end up. Each PCA component is a
displacement field over the template, so interpolating it at the marks'
barycentric coordinates gives a 55x3 displacement per mode, and the landmarks
are then mean + sum(c_i * basis_i). That is 55x3 of arithmetic instead of
1188x3, and it is exact rather than an approximation.

    python scripts/build_landmark_basis.py --pca ear_pca_v4_all.npz \\
        --marks mean_v4_marks.npz --template ear_mean.obj --out lm_basis.npz
"""
import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ear3d.wrap import load_mesh                      # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pca", required=True)
    p.add_argument("--marks", required=True)
    p.add_argument("--template", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--modes", type=int, default=0, help="0 keeps all stored")
    a = p.parse_args()

    d = np.load(a.pca, allow_pickle=True)
    mean, comp, var = d["mean"], d["components"], d["variance"]
    F = np.asarray(d["faces"])
    m = np.load(a.marks)
    tri, bary = m["triangle"], m["barycentric"]
    Vt, Ft = load_mesh(a.template)
    if not np.array_equal(F, Ft):
        sys.exit("the PCA was built on different faces from the template the "
                 "marks were placed on; the barycentric indices would not match")
    k = a.modes or len(comp)
    var_all = np.asarray(var, float)          # keep ALL of it for the fraction
    comp = comp[:k].reshape(k, len(mean), 3)
    var = var_all[:k]

    at = lambda V: np.einsum("ij,ijk->ik", bary, V[F[tri]])
    L0 = at(mean)                                   # (55, 3)
    B = np.stack([at(c) for c in comp])             # (k, 55, 3)

    # EVERYTHING IS SCALE-FREE. The PCA inherits the units of whichever ear the
    # Procrustes started from -- here HUTUBS normalised head units, not
    # millimetres -- and the network predicts a scale anyway, so the basis is
    # divided by the mean ear's own size. The components are also multiplied by
    # their standard deviation, so a coefficient of 1 means "one sd of this
    # mode" and a unit gaussian prior on the coefficients is the right one.
    size = float(np.ptp(L0, axis=0).max())
    sd = np.sqrt(var)
    L0 = (L0 - L0.mean(0)) / size
    B = B * sd[:, None, None] / size

    move = np.abs(B).reshape(k, -1).max(axis=1)     # per mode, in ear sizes
    np.savez_compressed(a.out, mean=L0, components=B, variance=var,
                        n_modes=k, ear_size=size, source=str(a.pca))
    print(f"{k} modes -> {a.out}")
    print(f"  basis normalised by the mean ear's size ({size:.3f} in the "
          f"PCA's own units); coefficients are in standard deviations")
    print(f"  one sd of a mode moves a landmark by, as a fraction of ear size:")
    print(f"    mode 1 {move[0]:.4f}, mode 5 {move[4]:.4f}, "
          f"mode {k} {move[-1]:.4f}")
    # against ALL the stored variance, not just the slice kept -- dividing the
    # kept modes by their own sum says 100% whatever k is
    print(f"  these {k} of {len(var_all)} stored modes carry "
          f"{100 * var.sum() / var_all.sum():.1f}% of the retained variance")


if __name__ == "__main__":
    main()
