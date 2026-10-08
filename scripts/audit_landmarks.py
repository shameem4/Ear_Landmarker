#!/usr/bin/env python3
"""Check landmarks placed on a 3D ear against how the scheme is actually annotated.

Reads the points straight out of a Wrap project, so there is nothing to export
in between, and asks four things:

  on the surface   barycentric points cannot float, so this only catches a mesh
                   mismatch -- the marks belonging to a different ear
  ordering         a chain that doubles back has a point out of sequence
  position         where each point sits ALONG its chain, against the same
                   point in thousands of 2D annotations. This is the measure
                   that localises a mistake to one point
  spacing          the gaps, reported but NOT used to flag: gaps are correlated,
                   so one displaced point spoils two of them, and gap variance
                   is far wider than position variance. Judging by gaps flagged
                   17 points where position flagged 3, and position was right

What it cannot tell you is whether a point is on the right FEATURE. It measures
sequence and spacing along a chain, so a chain running neatly down the wrong
ridge passes everything here. Only eyes catch that.

    python scripts/audit_landmarks.py --project mark_mean_landmarks.wrap \\
        --mesh ear_mean.obj --corpus data/preprocessed/landmarks.npy
"""
import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ear3d.scheme import STRIPS, STRIP_NAMES, group_of  # noqa: E402
from ear3d.wrap import load_mesh, wrap_barycentric  # noqa: E402


def points_from_project(path, V, F):
    d = json.load(open(path))
    sel = [n for n in d["nodes"].values() if n["nodeType"] == "SelectPoints"]
    if not sel:
        sys.exit(f"{path}: no SelectPoints node")
    pts = sel[0]["params"]["points"]["value"]
    P = np.zeros((len(pts), 3))
    for k, q in pts.items():
        P[int(k[:2])] = wrap_barycentric(V, F, int(q["triangleInd"]),
                                         float(q["u"]), float(q["v"]))
    return P


def arc_fraction(pts):
    """Where each point sits along its own chain, 0 at the start, 1 at the end."""
    out = np.zeros(len(pts))
    for a, b in STRIPS:
        d = np.linalg.norm(np.diff(pts[a:b], axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(d)])
        out[a:b] = cum / cum[-1] if cum[-1] > 0 else 0.0
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--mesh", required=True)
    p.add_argument("--corpus", default="data/preprocessed/landmarks.npy")
    p.add_argument("--manifest", default=None,
                   help="the corpus manifest, so rows can be restricted by "
                        "source. Pass the same --sources the marks were made "
                        "with: a ruler and the thing it measures should come "
                        "from one corpus, and the subsets disagree enough to "
                        "flag different points")
    p.add_argument("--sources", nargs="*", default=None)
    p.add_argument("--band", type=float, default=5.0,
                   help="percentile band from the corpus, each side")
    p.add_argument("--save", default=None, help="write the points as .npy")
    a = p.parse_args()

    V, F = load_mesh(a.mesh)
    P = points_from_project(a.project, V, F)
    size = float(np.ptp(P, axis=0).max())
    print(f"{len(P)} points on {pathlib.Path(a.mesh).name}, "
          f"spanning {np.ptp(P, axis=0).round(1)} mm")

    import trimesh
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    _, dist, _ = trimesh.proximity.closest_point(mesh, P)
    print(f"  off the surface: max {dist.max():.4f} mm")


    L = np.load(a.corpus).reshape(-1, len(P), 2)
    if a.sources:
        import collections
        import csv
        rows = list(csv.DictReader(open(a.manifest)))
        if len(rows) != len(L):
            sys.exit(f"manifest has {len(rows)} rows, corpus has {len(L)}")
        src = np.array([r["source"] for r in rows])
        keep = np.isin(src, a.sources)
        print(f"  corpus restricted to {dict(collections.Counter(src[keep]))}")
        L = L[keep]
    A2 = np.array([arc_fraction(e) for e in L if np.isfinite(e).all()])
    A3 = arc_fraction(P)

    # Turning: judged against the corpus, NOT against a fixed angle. A flat 90
    # degree rule called the concha border "reversed" at a 92.9 degree turn
    # where the corpus median is 116 -- the border really does corner at the
    # antitragus, and the rule was wrong rather than the placement.
    def turns(pts):
        out = np.full(len(pts), np.nan)
        for lo, hi in STRIPS:
            v = np.diff(pts[lo:hi], axis=0)
            n = np.linalg.norm(v, axis=1, keepdims=True)
            v = v / np.where(n > 0, n, 1)
            out[lo+1:hi-1] = np.degrees(np.arccos(
                np.clip(np.einsum("ij,ij->i", v[:-1], v[1:]), -1, 1)))
        return out
    T3, T2 = turns(P), np.array([turns(e) for e in L])
    bad_order = []
    for j in range(len(P)):
        if not np.isfinite(T3[j]):
            continue
        col = T2[:, j][np.isfinite(T2[:, j])]
        if T3[j] > np.percentile(col, 99):
            bad_order.append(j)
            print(f"  point {j} ({group_of(j)}): turns {T3[j]:.0f} deg, corpus 99th percentile "
                  f"{np.percentile(col, 99):.0f} -- out of sequence?")

    # Chain LENGTH, which catches an endpoint placed too far out. The position
    # test cannot: the first and last points of a chain are 0 and 1 by
    # definition, so an over-long chain shows up only as its interior points
    # reading "early", which points at the wrong places.
    s3 = float(np.ptp(P, axis=0).max())
    bad_len = []
    print("\nchain length as a fraction of ear size:")
    for i, (lo, hi) in enumerate(STRIPS):
        ours = np.linalg.norm(np.diff(P[lo:hi], axis=0), axis=1).sum() / s3
        corp = np.array([np.linalg.norm(np.diff(e[lo:hi], axis=0), axis=1).sum()
                         / np.ptp(e, axis=0).max() for e in L])
        q1, q3 = np.percentile(corp, [a.band, 100 - a.band])
        out_of = not (q1 <= ours <= q3)
        if out_of:
            bad_len.append(STRIP_NAMES[i])
        print(f"  {STRIP_NAMES[i]:<14s} {ours:.3f}   corpus {np.median(corp):.3f} "
              f"[{q1:.3f}, {q3:.3f}]"
              + ("   <-- check this chain's ENDPOINTS" if out_of else ""))
    print(f"\nposition along each chain, against {len(A2)} annotations "
          f"({a.band:.0f}-{100-a.band:.0f}% band):")
    flagged = []
    for i, (lo, hi) in enumerate(STRIPS):
        for j in range(lo + 1, hi - 1):
            q1, q3 = np.percentile(A2[:, j], [a.band, 100 - a.band])
            med = float(np.median(A2[:, j]))
            if not (q1 <= A3[j] <= q3):
                flagged.append((j, A3[j], med))
                print(f"  point {j:2d} ({group_of(j)}): at {A3[j]:.3f}, "
                      f"corpus {med:.3f} [{q1:.3f}, {q3:.3f}] -> move it "
                      f"{'back' if A3[j] > med else 'forward'}")
    n = sum(b - a_ - 2 for a_, b in STRIPS)
    print(f"  {len(flagged)} of {n} interior points outside the band "
          f"(about {0.02 * a.band * n:.0f} expected by chance)")

    print("\nspacing, for information only:")
    for i, (lo, hi) in enumerate(STRIPS):
        seg = np.linalg.norm(np.diff(P[lo:hi], axis=0), axis=1)
        print(f"  {STRIP_NAMES[i]:<14s} mean {seg.mean():5.2f} mm  "
              f"min {seg.min():5.2f}  max {seg.max():5.2f}  "
              f"cv {seg.std()/seg.mean():.2f}")

    ok = not bad_order and not flagged and not bad_len and dist.max() < 1e-3
    print(f"\n{'PASSES' if ok else 'NEEDS ATTENTION'}: "
          f"{len(bad_order)} ordering, {len(flagged)} position, "
          f"{len(bad_len)} chain length, {int(dist.max() > 1e-3)} off-surface")
    print("None of this can tell you a point is on the WRONG FEATURE -- a chain "
          "running neatly down the wrong ridge passes every check here.")
    if a.save:
        np.save(a.save, P)
        print(f"wrote {a.save}")


if __name__ == "__main__":
    main()
