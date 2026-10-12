#!/usr/bin/env python3
"""Build the 3D morphable ear model from a set of fitted templates.

Every fit shares the template's topology, so vertex i is the same anatomy on
all of them and they can simply be averaged and decomposed. The only real
decisions are what goes in and what pose is removed first.

POSE IS REMOVED WITH SCALE. A generalised Procrustes aligns every ear to the
running mean including a uniform scale, so the model describes SHAPE and not
how big someone's ear is. Size is still recorded per ear, so it can be put
back; leaving it in would spend the first mode on it.

LICENCE IS AN INPUT, not an afterthought. HUTUBS and AudioEar3D are CC BY;
SONICOM is CC BY-SA, which a model built from it inherits. --note records what
went in so a released file says what it is.

    python scripts/build_ear_pca.py --fits corpus/faceform2_hutubs \\
        --template templates/ear_mean.obj --out ear_pca.npz --note "..."
"""
import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ear3d.wrap import base_topology, load_mesh, rigid_fit   # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fits", nargs="+", required=True)
    p.add_argument("--template", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--note", default="")
    p.add_argument("--modes", type=int, default=80)
    p.add_argument("--iters", type=int, default=4)
    a = p.parse_args()

    Vt, Ft = load_mesh(a.template)
    files = []
    for d in a.fits:
        files += sorted(pathlib.Path(d).glob("*.obj"))
    if not files:
        sys.exit("no fits found")

    V = np.empty((len(files), len(Vt), 3))
    names, size = [], np.empty(len(files))
    for i, f in enumerate(files):
        Vi, Fi = load_mesh(f)
        Vi, _ = base_topology(Vi, Fi, Vt, Ft, f.name)
        V[i] = Vi
        size[i] = float(np.ptp(Vi, axis=0).max())
        names.append(f.stem)
    print(f"{len(files)} fits, {len(Vt)} vertices each")

    def centroid_size(M):
        return float(np.sqrt(((M - M.mean(0)) ** 2).sum()))

    # THE MEAN'S SIZE IS HELD FIXED. Scaling every ear onto the running mean
    # makes the next mean smaller, so an unnormalised loop shrinks ~1.1% per
    # iteration and never converges -- it drifts, and the iteration count
    # silently sets the basis scale. Every ear_pca_v5/v6 file was built that
    # way and is a few percent small. Pinning centroid size converges to
    # machine precision in three iterations.
    A = V.copy()
    mean = A[0]
    s0 = centroid_size(mean)
    for it in range(a.iters):
        for i in range(len(A)):
            R, t, s = rigid_fit(A[i], mean)
            A[i] = s * A[i] @ R.T + t
        new_mean = A.mean(0)
        c = new_mean.mean(0)
        new_mean = (new_mean - c) * (s0 / centroid_size(new_mean)) + c
        shift = float(np.linalg.norm(new_mean - mean, axis=1).mean())
        mean = new_mean
        print(f"  Procrustes iter {it}: shift {shift:.6f}, "
              f"centroid size {centroid_size(mean):.2f}")
    resid = np.linalg.norm(A - mean, axis=2).mean(axis=1)
    k = float(np.ptp(mean, axis=0).max())
    print(f"  after Procrustes: residual from the mean median "
          f"{100 * np.median(resid) / k:.2f}% of ear size, worst "
          f"{100 * resid.max() / k:.2f}%")

    X = (A - mean).reshape(len(A), -1)
    U, S, Wt = np.linalg.svd(X, full_matrices=False)
    var = S ** 2 / (len(X) - 1)
    cum = np.cumsum(var) / var.sum()
    modes = min(a.modes, len(S))
    for q in (0.90, 0.95, 0.99):
        print(f"  {int(np.searchsorted(cum, q)) + 1} modes reach "
              f"{100 * q:.0f}% of the variance")
    np.savez_compressed(a.out, mean=mean, faces=Ft, components=Wt[:modes],
                        variance=var[:modes], n_ears=len(files),
                        ear_size=size, names=np.array(names), note=a.note)
    print(f"wrote {a.out}: {modes} modes, {len(files)} ears")
    if a.note:
        print(f"  note: {a.note}")


if __name__ == "__main__":
    main()
