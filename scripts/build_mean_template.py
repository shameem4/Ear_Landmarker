#!/usr/bin/env python
"""Build the mean ear template from a canonical corpus, and replay the
landmark promotion onto it.

MEAN, NOT MEDIAN -- measured. A componentwise median CREATES folds (5 edges
past 90 degrees, max 131.7) because neighbouring vertices take their values
from different ears, and it is not rotation-equivariant. The geometric median
is fold-free but ties the plain mean for far more cost. Restricting to the
fold-free fits changes nothing, which is what showed the old template's folds
were never about outlier ears.

NO ALIGNMENT HERE. The canonical corpus is already one frame in true
millimetres; re-aligning with scale would silently discard the real size
distribution, which is exactly what canonicalise_corpus.py refuses to do.

THE PROMOTION IS REPLAYED, NOT RECOMPUTED, so the landmark-vertex template
keeps byte-identical connectivity across rebuilds and any face-index mask --
such as a hand-marked polygon exclusion -- stays valid.
"""
import argparse, glob, pathlib, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from ear3d.wrap import load_mesh                                   # noqa: E402
from landmark_template import replay                               # noqa: E402

WRAP = pathlib.Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/wrap_work")


def _load(f):
    return load_mesh(f)[0]


def folds(V, F):
    import collections
    adj = collections.defaultdict(list)
    for t, f in enumerate(F):
        for i in range(3):
            adj[tuple(sorted((int(f[i]), int(f[(i + 1) % 3]))))].append(t)
    A = np.array([ts for ts in adj.values() if len(ts) == 2])
    p = V[F]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-30
    return np.degrees(np.arccos(
        np.clip(np.einsum("ij,ij->i", n[A[:, 0]], n[A[:, 1]]), -1, 1)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--canonical", required=True)
    p.add_argument("--template", default=str(WRAP / "templates/ear_L_template.obj"),
                   help="topology donor: supplies the faces")
    p.add_argument("--out", default=str(WRAP / "templates/ear_mean_v4.obj"))
    p.add_argument("--recipe", default=str(WRAP / "templates/ear_lmv_v1.npz"))
    p.add_argument("--lmv-out", default=str(WRAP / "templates/ear_lmv_v3.obj"))
    p.add_argument("--sources", nargs="*",
                   default=["sonicom", "hutubs", "audioear"])
    p.add_argument("--max-folds", type=int, default=0,
                   help="refuse to write a mean with more than this many "
                        "edges past 90 degrees")
    a = p.parse_args()

    _, F = load_mesh(a.template)
    files = [f for ds in a.sources
             for f in sorted(glob.glob(str(pathlib.Path(a.canonical) / ds / "*.obj")))]
    if not files:
        print(f"no fits under {a.canonical}")
        return 1
    with ProcessPoolExecutor(max_workers=12) as ex:
        A = np.stack(list(ex.map(_load, files)))
    ext = np.array([np.ptp(V, 0).max() for V in A])
    print(f"{len(A)} canonical fits; ear extent mean {ext.mean():.2f} mm "
          f"sd {ext.std():.2f} range {ext.min():.1f}-{ext.max():.1f}")
    if ext.std() > 0.25 * ext.mean():
        print("  REFUSING: extent spread looks like mixed units, not one "
              "corpus. Canonicalise before averaging.")
        return 1

    M = A.mean(0)
    ang = folds(M, F)
    n90 = int((ang > 90).sum())
    print(f"mean: extent {np.ptp(M, 0).max():.2f} mm  centroid "
          f"{np.round(M.mean(0), 4)}  folds>90 {n90}  >60 "
          f"{int((ang > 60).sum())}  max {ang.max():.1f}")
    if n90 > a.max_folds:
        print(f"  REFUSING to write: {n90} folds > --max-folds {a.max_folds}. "
              f"A folded template is inherited by every fit made from it.")
        return 1

    with open(a.out, "w") as fh:
        fh.write(f"# canonical frame, millimetres; mean of {len(A)} fits\n")
        for q in M:
            fh.write(f"v {q[0]:.6f} {q[1]:.6f} {q[2]:.6f}\n")
        for t in F:
            fh.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")
    print(f"wrote {a.out}")

    rec = dict(np.load(a.recipe, allow_pickle=True))
    Vn, Fn = replay(M, rec), rec["faces"]
    an = folds(Vn, Fn)
    print(f"ear_lmv: {len(Vn)}v {len(Fn)}f  folds>90 {int((an > 90).sum())}  "
          f"max {an.max():.1f}")
    old = pathlib.Path(a.lmv_out)
    if old.exists():
        _, Fo = load_mesh(old)
        same = np.array_equal(np.asarray(Fn), Fo)
        print(f"  connectivity identical to the previous {old.name}: {same}"
              + ("" if same else "   <-- face-index masks are INVALID"))
    with open(a.lmv_out, "w") as fh:
        for q in Vn:
            fh.write(f"v {q[0]:.6f} {q[1]:.6f} {q[2]:.6f}\n")
        for t in Fn:
            fh.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")
    rec["template"] = str(a.out)
    np.savez(str(a.lmv_out).replace(".obj", ".npz"), **rec)
    print(f"wrote {a.lmv_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
