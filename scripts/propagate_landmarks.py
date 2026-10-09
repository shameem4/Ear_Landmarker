#!/usr/bin/env python3
"""Carry the 55 landmarks from the mean ear to every fitted ear in the corpus.

This is what the whole template-fitting track was for. Every fit shares the
mean's topology, so a landmark stored as (triangle, u, v) names the same
anatomy on all of them: read the triangle's three corners from a given fit,
weight them, and the point is placed. No detector, no rendering, no lifting.

The result is 3D landmarks for the corpus that owe nothing to the 2D model --
PROVIDED the marks on the mean were checked by a person. Placed automatically
they are model-seeded, and a model trained on them and then judged by them
would only be measuring its own agreement with itself.

    python scripts/propagate_landmarks.py --marks mean_landmarks.npz \\
        --fits corpus/wrapped3_hutubs corpus/wrapped3_sonicom --out lm3d.npz
"""
import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ear3d.scheme import STRIPS, STRIP_NAMES          # noqa: E402
from ear3d.wrap import base_topology, load_mesh       # noqa: E402


def place(V, F, tri, bary):
    """(triangle, barycentric) -> xyz on this particular mesh."""
    corners = V[F[tri]]                                # (k, 3, 3)
    return np.einsum("ij,ijk->ik", bary, corners)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--marks", required=True, help="mean_landmarks.npz")
    p.add_argument("--fits", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--reference", default=None,
                   help="the mean the marks were placed on, for a sanity check")
    p.add_argument("--template", required=True,
                   help="the mesh the marks were placed on. The fits MUST be "
                        "checked against this, not merely against each other: "
                        "Wrap's graph ends in SubdivideGeom, so its fits carry "
                        "6968 vertices and 13776 faces where the template has "
                        "1188 and 2296. Every fit then agrees with every other "
                        "fit, the marks' triangle indices are all in range, and "
                        "the landmarks land on the wrong anatomy in silence")
    a = p.parse_args()

    d = np.load(a.marks)
    tri, bary = d["triangle"], d["barycentric"]
    files = []
    for f in a.fits:
        files += sorted(pathlib.Path(f).glob("*.obj"))
    if not files:
        sys.exit("no fits found")

    Vt, Ft = load_mesh(a.template)
    V0, F = load_mesh(files[0])
    subdivided = False
    if not np.array_equal(F[:len(Ft)], Ft) or len(V0) != len(Vt):
        # A subdivided fit KEEPS the original vertices, in order, and appends
        # the new ones; verified by checking that the first len(Vt) of them
        # still span the whole mesh. That makes the base-resolution fit
        # recoverable without re-running anything -- but it has to be checked
        # per corpus rather than assumed, because a renumbering would pass
        # every other test here while quietly scrambling the marks.
        if len(V0) < len(Vt):
            sys.exit(f"fits have {len(V0)} vertices, fewer than the template's "
                     f"{len(Vt)}; the marks cannot apply")
        span = float(np.ptp(V0[:len(Vt)], axis=0).max())
        full = float(np.ptp(V0, axis=0).max())
        if span < 0.9 * full:
            sys.exit(f"the fits' first {len(Vt)} vertices span only "
                     f"{100*span/full:.0f}% of the mesh, so they are not the "
                     f"template's vertices -- the topology really differs")
        subdivided = True
        F = Ft
        print(f"fits are subdivided ({len(V0)} verts); taking the first "
              f"{len(Vt)} with the template's faces, which span "
              f"{100*span/full:.1f}% of the mesh")
    if tri.max() >= len(F):
        sys.exit(f"the marks reference triangle {tri.max()} but these fits have "
                 f"{len(F)} -- different topology, the marks do not apply")

    # NOT hardcoded 55: the scheme is extensible and an extended marks file
    # silently truncated here would propagate the first 55 and drop the rest.
    k_marks = len(tri)
    names, P = [], np.empty((len(files), k_marks, 3))
    for i, f in enumerate(files):
        V, Fi = load_mesh(f)
        if len(V) != len(V0) or (not subdivided and not np.array_equal(Fi, F)):
            sys.exit(f"{f.name}: topology differs from {files[0].name}")
        P[i] = place(V[:len(Vt)] if subdivided else V, F, tri, bary)
        names.append(f.stem)

    # UNITS ARE PER EAR, and they are not all the same: HUTUBS cuts are in
    # normalised head units and SONICOM in millimetres. Correspondence is
    # unaffected -- landmark k is the same anatomy everywhere -- but absolute
    # size is not comparable across the two, so each ear's own size is recorded
    # and a scale-free copy is stored alongside. Anything measuring real
    # dimensions must use one or the other deliberately.
    own = np.array([float(np.ptp(load_mesh(f)[0], axis=0).max()) for f in files])
    Pn = (P - P.mean(axis=1, keepdims=True)) / own[:, None, None]
    np.savez_compressed(a.out, landmarks=P, landmarks_scalefree=Pn,
                        ear_size=own, names=np.array(names),
                        triangle=tri, barycentric=bary, strips=np.array(STRIPS),
                        point_names=d["names"] if "names" in d else np.array([]))
    print(f"{len(files)} ears x {k_marks} landmarks -> {a.out}")
    # Only warn when the sizes really ARE inconsistent. This message used to
    # fire unconditionally, so it kept crying mixed units on a corpus that had
    # been converted -- a warning that is always on teaches people to ignore it.
    if own.max() / max(own.min(), 1e-9) > 3.0:
        print(f"  ear size spans {own.min():.3f} to {own.max():.1f} -- MIXED "
              f"UNITS, see landmarks_scalefree for a comparable set")
    else:
        print(f"  ear size {own.min():.1f} to {own.max():.1f} "
              f"(mean {own.mean():.1f}) -- consistent units")

    # Sanity, not accuracy: the spread of each strip should look like an ear,
    # and a landmark should sit on the surface of the mesh it was placed on.
    groups = [(lo, hi, STRIP_NAMES[i]) for i, (lo, hi) in enumerate(STRIPS)]
    if k_marks > STRIPS[-1][1] and "names" in d:
        # extended points are grouped by the suffix of their Wrap key
        import itertools
        tail = [str(x) for x in d["names"][STRIPS[-1][1]:]]
        base = STRIPS[-1][1]
        for nm, run in itertools.groupby(enumerate(tail),
                                         key=lambda z: z[1][3:]):
            run = list(run)
            groups.append((base + run[0][0], base + run[-1][0] + 1, nm))
    for lo, hi, nm in groups:
        span = np.ptp(Pn[:, lo:hi], axis=1)          # scale-free, so comparable
        print(f"  {nm:<26s} [{lo:2d}:{hi:2d}] span as a fraction of ear size: "
              f"median {np.median(span, axis=0).round(3)}")
    if a.reference:
        Vm, Fm = load_mesh(a.reference)
        ref = place(Vm, Fm, tri, bary)
        err = np.abs(ref - d["xyz"]).max()
        print(f"  round trip on the mean itself: {err:.2e} mm "
              f"(should be ~0 -- it is the same arithmetic)")


if __name__ == "__main__":
    main()
