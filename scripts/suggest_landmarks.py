#!/usr/bin/env python3
"""Suggest where a misplaced landmark on the 3D mean belongs, from the 2D corpus.

The audit says a point is wrong but not where it should go. This does: the
thousands of 2D annotations fix the whole scheme's shape, so the points nobody
disputes determine the ones that are in doubt. A ridge regression over the
corpus, trained in a common Procrustes frame, predicts the suspect points from
the trusted ones; the prediction is then lifted back onto the mesh.

The lift is a RAY CAST along the LATERAL VIEW AXIS, and that is not a
convenience: the 2D scheme was standardised on frontal photographs, so a
landmark is DEFINED by where it falls in that projection. The 3D point that
means the same thing is therefore the first surface the ray meets -- what the
annotator could actually see. A point hidden behind the helix rim has no
frontal definition at all, which is a property of the scheme rather than a
failure of this tool, and the control below is what exposes it.

Handedness is resolved by measurement, not assumption: each corpus shape and the
mean's own projection are aligned with reflection allowed and the better fit
kept, so a corpus of mixed left and right ears costs nothing.

    python scripts/suggest_landmarks.py --project mark_mean_v2.wrap \\
        --mesh ear_mean_v3.obj --fix 8 9 10 14 --out suggest_v3.wrap
"""
import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ear3d.scheme import STRIPS, STRIP_NAMES, group_of  # noqa: E402
from ear3d.wrap import load_mesh, wrap_barycentric    # noqa: E402


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


def similarity(X, Y, reflect=True):
    """Best similarity taking X onto Y. Returns the transformed X."""
    cx, cy = X.mean(0), Y.mean(0)
    A, B = X - cx, Y - cy
    sa = np.sqrt((A ** 2).sum())
    U, S, Vt = np.linalg.svd(A.T @ B)
    R = U @ Vt
    if not reflect and np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
        S = S.copy()
        S[-1] *= -1
    k = S.sum() / (sa ** 2) if sa > 0 else 1.0
    return k * (X - cx) @ R + cy, (R, k, cx, cy)


def gpa(L, iters=5):
    """Generalised Procrustes over the corpus, reflection allowed per shape."""
    A = L.copy()
    ref = A[0] - A[0].mean(0)
    ref /= np.sqrt((ref ** 2).sum())
    for _ in range(iters):
        for i in range(len(A)):
            A[i] = similarity(A[i], ref)[0]
        ref = A.mean(0)
        ref -= ref.mean(0)
        ref /= np.sqrt((ref ** 2).sum())
    return A, ref


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--mesh", required=True)
    p.add_argument("--corpus", default="data/preprocessed/landmarks.npy")
    p.add_argument("--manifest", default=None,
                   help="the corpus manifest, so rows can be restricted by "
                        "source. Not needed for ordinary use: the iBUG rows are "
                        "licensed for non-commercial research, which is what "
                        "correcting annotations is, and the 55-point SCHEME is "
                        "a convention rather than their data. Reach for it if "
                        "you ever want to show a result standing on the "
                        "commissioned annotations alone")
    p.add_argument("--sources", nargs="*", default=None,
                   help="manifest sources to keep, e.g. audioear2d audioear3d")
    p.add_argument("--fix", type=int, nargs="+", required=True)
    p.add_argument("--out", required=True, help="the .wrap to write")
    p.add_argument("--alpha", type=float, default=1e-3, help="ridge strength")
    p.add_argument("--view", type=float, nargs=2, default=(-90.0, 0.0),
                   metavar=("YAW", "PITCH"),
                   help="the lateral view the scheme is defined in. Projection "
                        "here is orthographic: the ear is small against the "
                        "camera distance, and shape statistics over a corpus "
                        "are Procrustes-aligned anyway, so the perspective the "
                        "render pipeline uses would wash out")
    p.add_argument("--control", type=int, default=12,
                   help="points NOT in doubt that are re-predicted anyway, to "
                        "measure how far this method moves a point that was "
                        "already right. A suggestion smaller than that floor "
                        "is noise and should be ignored")
    a = p.parse_args()

    V, F = load_mesh(a.mesh)
    P = points_from_project(a.project, V, F)
    n = len(P)
    size = float(np.ptp(P, axis=0).max())
    fix = sorted(set(a.fix))
    keep = [j for j in range(n) if j not in fix]
    print(f"{n} points, repairing {fix} from the other {len(keep)}")

    L = np.load(a.corpus).reshape(-1, n, 2).astype(np.float64)
    if a.sources:
        import csv
        rows = list(csv.DictReader(open(a.manifest)))
        if len(rows) != len(L):
            sys.exit(f"manifest has {len(rows)} rows, corpus has {len(L)}")
        src = np.array([r["source"] for r in rows])
        kept = np.isin(src, a.sources)
        import collections
        print(f"sources: kept {dict(collections.Counter(src[kept]))}, "
              f"dropped {dict(collections.Counter(src[~kept]))}")
        L = L[kept]
    elif a.manifest:
        print("WARNING: --manifest given but no --sources, using every row")
    L = L[np.isfinite(L).all(axis=(1, 2))]
    A, ref = gpa(L)
    print(f"corpus: {len(A)} annotations, Procrustes frame set")

    # The mean's marks as the ANNOTATOR would have seen them: projected along
    # the lateral view axis, which is the frame the 55-point scheme is defined
    # in. An arbitrary best-fit plane would be a different projection from the
    # one the corpus was annotated in, and the two chains that overlap in depth
    # are exactly where that difference bites.
    from ear3d.camera import orbit_extrinsic
    from ear3d.config import DEFAULTS
    E = orbit_extrinsic(a.view[0], a.view[1], dict(DEFAULTS))
    axes, normal = E[:2, :3], E[2, :3]
    c3 = P.mean(0)
    flat = (P - c3) @ axes.T
    depth = (P - c3) @ normal
    print(f"  view yaw {a.view[0]:+.0f} pitch {a.view[1]:+.0f}: the marks span "
          f"{np.ptp(depth):.1f} mm in depth along it")

    # into the corpus frame using ONLY the trusted points
    _, (R, k, cx, cy) = similarity(flat[keep], ref[keep])
    into = lambda Q: k * (Q - cx) @ R + cy
    back = lambda Q: ((Q - cy) @ R.T) / k + cx
    mflat = into(flat)
    res = np.linalg.norm(mflat[keep] - ref[keep], axis=1)
    print(f"  trusted points land {res.mean():.4f} from the corpus mean shape "
          f"(frame units; worst {res.max():.4f})")

    # ridge from the trusted points to every point, so the suspect ones are
    # predicted and the rest serve as a check on the predictor itself
    Xt = A[:, keep].reshape(len(A), -1)
    Xt = np.hstack([Xt, np.ones((len(A), 1))])
    pred2 = np.zeros((n, 2))
    G = Xt.T @ Xt + a.alpha * len(A) * np.eye(Xt.shape[1])
    W = np.linalg.solve(G, Xt.T @ A.reshape(len(A), -1))
    x = np.hstack([mflat[keep].reshape(1, -1), [[1.0]]])
    pred2 = (x @ W).reshape(n, 2)
    cv = np.linalg.norm(A.reshape(len(A), -1) - Xt @ W, axis=0)
    print(f"  predictor residual on the corpus itself: "
          f"{np.sqrt((cv**2).sum()/len(A)/n/2):.4f} frame units rms")

    # back to the plane, then onto the mesh by ray cast along the normal
    import trimesh
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    plane = back(pred2) @ axes + c3
    far = 4.0 * size
    New = P.copy()
    rows = []
    for j in fix:
        o = plane[j] - normal * far
        hits, _, tri = mesh.ray.intersects_location([o], [normal])
        if len(hits):
            w = int(np.argmin((hits - o) @ normal))     # first visible surface
            New[j], t, how = hits[w], int(tri[w]), "ray"
            ray = float(abs((New[j] - P[j]) @ normal))
        else:
            close, dd, t = trimesh.proximity.closest_point(mesh, [plane[j]])
            New[j], t, how = close[0], int(t[0]), "nearest"
            ray = float(dd[0])
        rows.append((j, t, how, ray, float(np.linalg.norm(New[j] - P[j]))))

    # THE ERROR FLOOR. Re-predict points nobody flagged: whatever this moves
    # them by is the method's own noise, and a suggestion below it means
    # nothing. It is not one number -- the ray cast is accurate where the ear
    # is flat and hopeless in the concha, so the floor is reported against the
    # off-plane distance, which is the thing that predicts it.
    ctl = []
    step = max(1, len(keep) // a.control)
    for j in keep[::step][:a.control]:
        kp = [q for q in range(n) if q != j]
        Xc = np.hstack([A[:, kp].reshape(len(A), -1), np.ones((len(A), 1))])
        Wc = np.linalg.solve(Xc.T @ Xc + a.alpha * len(A) * np.eye(Xc.shape[1]),
                             Xc.T @ A[:, j])
        _, (Rc, kc, cxc, cyc) = similarity(flat[kp], ref[kp])
        mf = kc * (flat - cxc) @ Rc + cyc
        q2 = np.hstack([mf[kp].reshape(1, -1), [[1.0]]]) @ Wc
        pl = (((q2 - cyc) @ Rc.T) / kc + cxc) @ axes + c3
        hits, _, _ = mesh.ray.intersects_location(
            [pl[0] - normal * far], [normal])
        if len(hits):
            # FIRST hit, not nearest: the annotator clicked what they could see
            w = int(np.argmin((hits - (pl[0] - normal * far)) @ normal))
            ctl.append((j, float(np.linalg.norm(hits[w] - P[j])),
                        float(abs((hits[w] - P[j]) @ normal))))
        else:
            cl, dd, _ = trimesh.proximity.closest_point(mesh, pl)
            ctl.append((j, float(np.linalg.norm(cl[0] - P[j])), float(dd[0])))
    flat_ctl = [m for _, m, o in ctl if o < 3.0]
    floor = float(np.median(flat_ctl)) if flat_ctl else float("nan")
    worst = max(ctl, key=lambda r: r[1])
    print(f"\nerror floor, from {len(ctl)} points that were NOT in doubt:")
    print(f"  where the lift stays on the same surface (under 3 mm of depth "
          f"change): median {floor:.2f} mm, over {len(flat_ctl)} points")
    print(f"  worst: point {worst[0]} moves {worst[1]:.2f} mm at "
          f"{worst[2]:.2f} mm of depth change -- a jump that large means the "
          f"ray found a different sheet of the ear, so judge each suggestion "
          f"by its own depth change")

    print("\nsuggested moves:")
    for j, t, how, ray, moved in rows:
        strip = group_of(j)
        if ray > 3.0:
            verdict = "the point jumps to another surface -- CHECK BY EYE"
        elif moved < 2 * floor:
            verdict = f"below the {floor:.2f} mm floor, ignore"
        else:
            verdict = "real"
        print(f"  point {j:2d} ({strip:<22s}) moves {moved:5.2f} mm "
              f"({100*moved/size:4.1f}% of ear size), lifted by {how}, "
              f"{ray:.2f} mm in depth -- {verdict}")

    # what the repair does to the thing that was wrong: separation between the
    # outer helix and the inner helix
    o_lo, o_hi = STRIPS[0]
    i_lo, i_hi = STRIPS[1]
    def sep(Q, j):
        return float(np.linalg.norm(Q[i_lo:i_hi] - Q[j], axis=1).min()) / size
    print("\nouter-to-inner helix separation, fraction of ear size:")
    cs = np.array([[np.linalg.norm(e[i_lo:i_hi] - e[j], axis=1).min()
                    / np.ptp(e, axis=0).max() for j in range(o_lo, o_hi)]
                   for e in L])
    for j in range(o_lo, o_hi):
        was, now = sep(P, j), sep(New, j)
        col = cs[:, j - o_lo]
        lo2 = np.percentile(col, 2)
        if was < lo2 or j in fix:
            mark = "  <-- was touching" if was < lo2 else ""
            print(f"  point {j:2d}: {was:.3f} -> {now:.3f}   "
                  f"corpus median {np.median(col):.3f}, 2nd pct {lo2:.3f}{mark}")

    # write the project, changing only the repaired points
    d = json.load(open(a.project))
    sel = [nd for nd in d["nodes"].values() if nd["nodeType"] == "SelectPoints"][0]
    pts = sel["params"]["points"]["value"]
    key = {int(kk[:2]): kk for kk in pts}
    for j, t, how, ray, moved in rows:
        bary = trimesh.triangles.points_to_barycentric(
            V[F[t]][None], New[j][None])[0]
        q = pts[key[j]]
        q["triangleInd"] = t
        q["u"], q["v"] = float(bary[0]), float(bary[1])   # see wrap_barycentric
    json.dump(d, open(a.out, "w"), indent=4)
    print(f"\nwrote {a.out} -- open it in Wrap, the other {len(keep)} points "
          f"are untouched")


if __name__ == "__main__":
    main()
