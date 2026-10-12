#!/usr/bin/env python3
"""Re-label the rendered depth corpus for new marks, without re-rendering it.

NOTHING IN AN IMAGE DEPENDS ON THE MARKS. The renders are the head mesh, the
skin and the camera; moving a landmark changes only where the label sits in
the picture that is already on disk. Re-running ingest_depth3d would redraw
9123 views, and every one would come out the same.

HOW THE VIEW IS RECOVERED, WITHOUT REPRODUCING INGEST'S FRAME. A stored label
is the mark projected by the camera and then mapped into a crop by
(left, top, scale). Projection is a 3x4 camera and the crop is affine in
pixels, so the whole composition -- canonical millimetres straight to crop
coordinates -- is itself a 3x4 projective camera. Fifty-five correspondences
determine it by DLT, so the view is read back OUT of the old labels rather
than rebuilt from the chain that stored them. That matters: the chain is
exactly where three bugs lived today, and this never touches it.

AND IT CHECKS ITSELF. The recovered camera must reproduce the OLD labels it
was fitted to, to numerical precision. A frame error, a wrong mesh or a
mismatched ear cannot survive that, because the model is exact: anything that
does not fit is not a view.

DEPTH comes back the same way. Corpus z is root-relative and divided by the
crop's width in world units, so z = (zc - zbar) * k for one unknown k per
view, recovered by least squares against the old labels. The new marks have
their own zbar, and k scales as 1/zbar, so it is corrected rather than reused.

VISIBILITY IS CARRIED OVER, NOT RECOMPUTED. The flag comes from the render's
depth buffer, which is the one thing here that would need the renderer back.
The marks move about 2 mm on a 60 mm ear, so a point that was visible almost
always still is -- but it is an approximation, and the only one in this file.

    python scripts/relabel_depth3d.py --data data/depth3d \
        --marks wrap_work/templates/mark_mean_v6_spacing.npz --out data/depth3d_v7
"""
import argparse
import csv
import pathlib
import shutil
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
WRAP = ROOT.parent / "wrap_work"
sys.path.insert(0, str(ROOT))


def dlt(X, uv):
    """The 3x4 projective camera taking X (n,3) to uv (n,2). Exact for n>=6."""
    n = len(X)
    Xh = np.hstack([X, np.ones((n, 1))])
    A = np.zeros((2 * n, 12))
    A[0::2, 0:4] = Xh
    A[0::2, 8:12] = -uv[:, 0:1] * Xh
    A[1::2, 4:8] = Xh
    A[1::2, 8:12] = -uv[:, 1:2] * Xh
    _, _, Vt = np.linalg.svd(A)
    return Vt[-1].reshape(3, 4)


def apply_cam(P, X):
    Xh = np.hstack([X, np.ones((len(X), 1))])
    q = Xh @ P.T
    return q[:, :2] / q[:, 2:3], q[:, 2]


def marks_on(mesh_path, tri, bary, cache):
    import open3d as o3d
    if mesh_path not in cache:
        m = o3d.io.read_triangle_mesh(str(mesh_path))
        cache.clear()
        cache[mesh_path] = (np.asarray(m.vertices), np.asarray(m.triangles))
    V, F = cache[mesh_path]
    return np.einsum("ij,ijk->ik", bary, V[F[tri]])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/depth3d")
    p.add_argument("--marks", required=True, help="the NEW marks npz")
    p.add_argument("--old-marks", default=str(WRAP / "templates/mean_v5_marks.npz"),
                   help="the marks the corpus was built with")
    p.add_argument("--out", required=True)
    p.add_argument("--canonical", default=str(WRAP / "corpus/canonical_esc"))
    p.add_argument("--tol", type=float, default=1e-3,
                   help="max allowed reprojection of the OLD labels, in crop "
                        "units; above this the view was not recovered")
    p.add_argument("--link-images", action="store_true",
                   help="symlink the image directory instead of copying it")
    a = p.parse_args()

    src, out = pathlib.Path(a.data), pathlib.Path(a.out)
    rows = list(csv.DictReader(open(src / "manifest.csv", encoding="utf-8")))
    old_lm = np.load(src / "landmarks.npy")
    vis = np.load(src / "visible.npy")
    K = old_lm.shape[1]
    mo, mn = np.load(a.old_marks), np.load(a.marks)
    print(f"{len(rows)} views, {K} landmarks")
    print(f"old marks {pathlib.Path(a.old_marks).name}, "
          f"new {pathlib.Path(a.marks).name}")

    # the ears, in the order the manifest meets them, so one mesh is held
    order = sorted({r["ear"] for r in rows})
    canon = {}
    for e in order:
        for ds in ("hutubs", "sonicom", "audioear"):
            q = pathlib.Path(a.canonical) / ds / f"{e}.obj"
            if q.exists():
                canon[e] = q
                break
    missing = [e for e in order if e not in canon]
    if missing:
        sys.exit(f"no canonical mesh for {len(missing)} ears: {missing[:5]}")

    by_ear = {}
    for i, r in enumerate(rows):
        by_ear.setdefault(r["ear"], []).append(i)

    new_lm = np.array(old_lm, copy=True)
    cache, resid, zres, failed = {}, [], [], []
    for n_done, (ear, idxs) in enumerate(sorted(by_ear.items())):
        Lo = marks_on(canon[ear], mo["triangle"][:K], mo["barycentric"][:K], cache)
        Ln = marks_on(canon[ear], mn["triangle"][:K], mn["barycentric"][:K], cache)
        for i in idxs:
            xy_old, z_old = old_lm[i, :, :2], old_lm[i, :, 2]
            P = dlt(Lo, xy_old)
            uv_chk, w_old = apply_cam(P, Lo)
            e = float(np.abs(uv_chk - xy_old).max())
            resid.append(e)
            if e > a.tol:
                failed.append((i, e))
                continue
            # z = (w - wbar) * k, one k per view, read off the old labels
            dw = w_old - w_old.mean()
            k = float(dw @ z_old / max(dw @ dw, 1e-30))
            zres.append(float(np.abs(dw * k - z_old).max()))
            uv_new, w_new = apply_cam(P, Ln)
            # the crop's width in world units goes with zbar, so k does too
            kn = k * (w_old.mean() / w_new.mean())
            new_lm[i, :, :2] = uv_new
            new_lm[i, :, 2] = (w_new - w_new.mean()) * kn
        if (n_done + 1) % 100 == 0:
            print(f"  [{n_done+1}/{len(by_ear)}] ears", flush=True)

    resid, zres = np.array(resid), np.array(zres)
    print(f"\nview recovery, the check that matters:")
    print(f"  old labels reprojected: median {np.median(resid):.2e}, "
          f"max {resid.max():.2e} crop units")
    print(f"  old depth reproduced:   max {zres.max():.2e}")
    if failed:
        print(f"  {len(failed)} views NOT recovered above {a.tol} -- left "
              f"with their old labels")
    d = np.linalg.norm(new_lm[:, :, :2] - old_lm[:, :, :2], axis=2)
    dz = np.abs(new_lm[:, :, 2] - old_lm[:, :, 2])
    print(f"\nlabels moved: xy median {np.median(d):.4f} crop widths, "
          f"z median {np.median(dz):.4f}")
    inb = ((new_lm[:, :, :2] >= 0) & (new_lm[:, :, :2] <= 1)).all(2)
    print(f"  in frame: {100*inb.mean():.1f}% of points "
          f"(was {100*((old_lm[:,:,:2]>=0)&(old_lm[:,:,:2]<=1)).all(2).mean():.1f}%)")

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "landmarks.npy", new_lm.astype(np.float32))
    np.save(out / "visible.npy", vis)
    for f in ("manifest.csv", "train.csv", "val.csv", "test.csv",
              "pred_xy.npy", "pred_conf.npy", "pred_feat.npy"):
        if (src / f).exists():
            shutil.copy2(src / f, out / f)
    img = out / "images"
    if not img.exists():
        if a.link_images:
            img.symlink_to((src / "images").resolve(), target_is_directory=True)
        else:
            shutil.copytree(src / "images", img)
    print(f"\nwrote {out} -- images {'symlinked' if a.link_images else 'copied'}, "
          f"splits and pred_* carried over unchanged")


if __name__ == "__main__":
    main()
