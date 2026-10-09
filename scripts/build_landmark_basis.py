#!/usr/bin/env python3
"""Project the ear shape model onto the 55 landmarks.

A model that predicts shape coefficients does not need the 1188-vertex mesh at
every step -- only where the 55 marks end up. Each PCA component is a
displacement field over the template, so interpolating it at the marks'
barycentric coordinates gives a 55x3 displacement per mode, and the landmarks
are then mean + sum(c_i * basis_i). That is 55x3 of arithmetic instead of
1188x3, and it is exact rather than an approximation.

THE OUTPUT IS UNIT-HEIGHT, NOT MILLIMETRES, and the filename has to say so.
A basis written here as `*_mm.npz` sat on disk for a while and was read back
as millimetres more than once; the `--out` name is now checked, because the
cheap fix for a unit trap is to make the wrong name impossible rather than to
document it. `ear_size` carries the mean ear's height in the PCA's own units
(mm, for the canonical corpus), so the scale is recoverable -- multiply by it.

    python scripts/build_landmark_basis.py --pca ear_pca_v6_all_mm.npz \\
        --marks mean_v5_marks.npz --template ear_mean.obj \\
        --out lm_basis_v6_95_unitheight.npz
"""
import argparse
import pathlib
import re
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
    p.add_argument("--allow-misleading-name", action="store_true",
                   help="write a name this script considers a unit trap")
    a = p.parse_args()

    # The one guard. What this writes is divided by ear size, so a name ending
    # _mm is a false claim about the contents, and a reader who trusts it is
    # off by a factor of 65. Refusing costs nothing and the alternative was
    # four such files on disk.
    stem = pathlib.Path(a.out).stem
    if re.search(r"_mm$|_mm[._]|millimet", stem, re.I) and not a.allow_misleading_name:
        sys.exit(f"--out {a.out!r} names itself millimetres, and this script "
                 f"writes a UNIT-HEIGHT basis (mean ear height exactly 1.0). "
                 f"Use a name like lm_basis_v6_95_unitheight.npz. The scale is "
                 f"kept in `ear_size` if something needs mm back.")

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

    # EVERYTHING IS SCALE-FREE, AND THAT IS A CHOICE, not an inherited defect.
    # The canonical v5/v6 PCA this now reads IS in millimetres -- its mean ear
    # spans [21.6, 66.5, 42.1] mm. The division happens anyway because the
    # network predicts a scale of its own, so asking the shape head for size
    # too would have it fight the pose solver for the same degree of freedom.
    # (An earlier version of this comment claimed the PCA was in HUTUBS head
    # units; true of v4, false since the canonical rebuild.)
    #
    # `ear_size` below is measured BEFORE the division, so it is the mean ear's
    # height in the PCA's units and multiplying by it puts the basis back into
    # millimetres. Nothing is lost, but nothing downstream may assume mm.
    #
    # The components are also multiplied by their standard deviation, so a
    # coefficient of 1 means "one sd of this mode" and a unit gaussian prior on
    # the coefficients is the right one.
    size = float(np.ptp(L0, axis=0).max())
    sd = np.sqrt(var)
    L0 = (L0 - L0.mean(0)) / size
    B = B * sd[:, None, None] / size

    move = np.abs(B).reshape(k, -1).max(axis=1)     # per mode, in ear sizes
    np.savez_compressed(a.out, mean=L0, components=B, variance=var,
                        n_modes=k, ear_size=size, source=str(a.pca),
                        units="unit-height: mean ear height is exactly 1.0. "
                              "Multiply mean and components by `ear_size` "
                              f"({size:.3f}, the source PCA's units) for mm.")
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
