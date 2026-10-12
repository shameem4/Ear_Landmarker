#!/usr/bin/env python
"""Verify a hand-trimmed template and recover its mapping to the original.

Wrap may reorder or renumber on export, so the kept vertices are matched by
exact coordinate rather than by index.  That is only sound if the edit was
face deletion and nothing else, which is checked here rather than assumed:
every trimmed vertex must coincide with an original vertex, no original may
be claimed twice, and no new vertex may appear.

Reports what the cut costs: which landmarks are gone, and how close the
survivors sit to the new free boundary.  A landmark ON a free boundary is
unconstrained during the fit and flaps -- the failure that ruled out trim 2.
"""
import argparse, collections, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from ear3d.wrap import load_mesh                                   # noqa: E402

WRAP = pathlib.Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/wrap_work")


def area(V, F):
    a = V[F[:, 1]] - V[F[:, 0]]
    b = V[F[:, 2]] - V[F[:, 0]]
    return 0.5 * np.linalg.norm(np.cross(a, b), axis=1).sum()


def boundary(F):
    ec = collections.Counter()
    for t in F:
        for i in range(3):
            ec[tuple(sorted((int(t[i]), int(t[(i + 1) % 3]))))] += 1
    return np.unique([v for e, c in ec.items() if c == 1 for v in e])


def components(n, F):
    adj = collections.defaultdict(set)
    for t in F:
        for i in range(3):
            adj[int(t[i])].add(int(t[(i + 1) % 3]))
            adj[int(t[(i + 1) % 3])].add(int(t[i]))
    seen, c = set(), 0
    for v in range(n):
        if v in seen:
            continue
        c += 1
        st = [v]
        seen.add(v)
        while st:
            for w in adj[st.pop()]:
                if w not in seen:
                    seen.add(w)
                    st.append(w)
    return c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--template", default=str(WRAP / "templates/ear_lmv_v3.obj"))
    p.add_argument("--trimmed", required=True)
    p.add_argument("--n-landmarks", type=int, default=55)
    p.add_argument("--tol", type=float, default=1e-5,
                   help="coordinate match tolerance, as a fraction of extent. "
                        "Wrap's OBJ export rounds, so an untouched vertex "
                        "comes back ~4e-8 of extent away; 1e-5 absorbs that "
                        "and is still orders of magnitude below any real edit")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    V, F = load_mesh(a.template)
    Vt, Ft = load_mesh(a.trimmed)
    k = a.n_landmarks
    scale = float(np.ptp(V, axis=0).max())
    tol = a.tol * scale
    print(f"original {len(V)}v {len(F)}f   trimmed {len(Vt)}v {len(Ft)}f")

    # --- match by coordinate ------------------------------------------------
    from scipy.spatial import cKDTree
    d, idx = cKDTree(V).query(Vt)
    bad = d > tol
    if bad.any():
        print(f"\nFAIL: {bad.sum()} trimmed vertices do not coincide with an "
              f"original vertex (worst {d.max() / scale:.2e} of extent).")
        print("The edit was not face-deletion only -- vertices were moved, "
              "welded, or created. The corpus mapping cannot be recovered; "
              "this template would need a full re-fit.")
        return 1
    if len(set(idx.tolist())) != len(idx):
        dup = [v for v, c in collections.Counter(idx.tolist()).items() if c > 1]
        print(f"\nFAIL: {len(dup)} original vertices claimed more than once "
              f"-- vertices were welded.")
        return 1
    keep = np.sort(idx)
    print(f"face-only edit confirmed: every trimmed vertex matches an "
          f"original (worst {d.max() / scale:.2e} of extent, "
          f"tolerance {a.tol:g}; export rounding alone is ~4e-8)")

    # --- faces, in original numbering ---------------------------------------
    remap = -np.ones(len(V), int)
    remap[idx] = np.arange(len(Vt))
    back = -np.ones(len(Vt), int)
    back[remap[idx]] = idx
    Fo = back[Ft]                               # trimmed faces, original ids
    orig_key = {tuple(sorted(map(int, t))): i for i, t in enumerate(F)}
    kf, unknown = [], 0
    for t in Fo:
        key = tuple(sorted(map(int, t)))
        if key in orig_key:
            kf.append(orig_key[key])
        else:
            unknown += 1
    kf = np.array(sorted(kf))
    if unknown:
        print(f"\nFAIL: {unknown} trimmed faces are not original faces -- the "
              f"mesh was re-triangulated or subdivided.")
        return 1
    print(f"faces kept {len(kf)}/{len(F)}  ({100 * area(V, F[kf]) / area(V, F):.1f}% "
          f"of area)   vertices kept {len(keep)}/{len(V)}")

    # --- what the cut costs -------------------------------------------------
    lost = [j for j in range(k) if j not in set(keep.tolist())]
    print(f"\nlandmarks: {k - len(lost)}/{k} kept" +
          (f"   LOST: {lost}" if lost else ""))
    nc = components(len(Vt), Ft)
    bnd = boundary(Ft)
    print(f"components {nc}" + ("" if nc == 1 else "   <-- the cut split the mesh")
          + f"   boundary vertices {len(bnd)}")
    surv = [j for j in range(k) if j not in set(lost)]
    if surv and len(bnd):
        lmt = remap[np.array(surv)]
        dm = np.linalg.norm(Vt[lmt][:, None, :] - Vt[bnd][None, :, :],
                            axis=2).min(1) / scale * 100
        order = np.argsort(dm)
        print("\nclosest landmarks to the new boundary (% of ear extent):")
        for i in order[:6]:
            flag = "  <-- ON the boundary" if dm[i] < 0.5 else ""
            print(f"  lm{surv[i]:<3d} {dm[i]:5.2f}%{flag}")
        if dm.min() < 0.5:
            print("\nWARNING: a landmark sits on a free boundary. It will be "
                  "unconstrained during the fit and will flap. Leave a "
                  "couple of rings of polygons around it.")

    dst = pathlib.Path(a.out) if a.out else pathlib.Path(
        str(a.trimmed).replace(".obj", "_map.npz"))
    np.savez(dst, keep_verts=keep, keep_faces=kf,
             remap=remap, n_landmarks=k,
             template=str(a.template), trimmed=str(a.trimmed))
    print(f"\nwrote {dst}")
    print("  keep_verts  original vertex ids, sorted -- use to remap the "
          "fitted corpus, the PCA basis and base_topology")
    print("  keep_faces  original face ids -- use to score both arms over the "
          "same faces")
    return 0


if __name__ == "__main__":
    sys.exit(main())
