#!/usr/bin/env python3
"""How much of our depth does an off-the-shelf model already give for free?

Our trained heads are compared against a flat predictor, which is a very low
bar. The honest bar is a monocular depth model that cost us nothing, and if it
clears skill 0.808 then the three-head comparison was answering the wrong
question.

RUN THIS WITH THE da3 VENV, not the mocap env:
    ../.venv_da3/bin/python scripts/da3_depth_baseline.py --n 300

Only da3-base and da3-small are used. Large, Giant and Nested are CC BY-NC,
which this corpus is deliberately free of.

TWO ALIGNMENTS, because DA3's depth is affine-invariant -- defined up to a
scale and offset per image -- while ours is a fixed root-relative quantity:

  per-image   a scale and offset fitted to the LABELS of that very image. This
              flatters it enormously and is not comparable to our heads; it is
              the ceiling of what DA3's depth ORDERING could give if something
              else supplied the scale.
  global      one scale and offset fitted on the training split and applied
              unchanged. This is the like-for-like number.
"""
import argparse
import csv
import pathlib
import sys

import numpy as np


def affine_fit(d, z, w):
    """Least squares a*d + b ~ z, weighted."""
    A = np.stack([d, np.ones_like(d)], 1) * w[:, None]
    sol, *_ = np.linalg.lstsq(A, z * w, rcond=None)
    return sol


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/depth3d")
    # Only these two. LARGE and GIANT are cc-by-nc-4.0, verified on the hub,
    # and this corpus is deliberately free of non-commercial terms.
    p.add_argument("--model", default="depth-anything/DA3-BASE",
                   choices=["depth-anything/DA3-BASE", "depth-anything/DA3-SMALL"])
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--fit-n", type=int, default=150)
    p.add_argument("--device", default="cuda")
    p.add_argument("--res", type=int, default=504)
    a = p.parse_args()

    import torch
    from PIL import Image
    from depth_anything_3.api import DepthAnything3

    d = pathlib.Path(a.data)
    L = np.load(d / "landmarks.npy", mmap_mode="r")
    V = np.load(d / "visible.npy", mmap_mode="r")
    rng = np.random.default_rng(0)

    def pick(split, k):
        rows = list(csv.DictReader(open(d / split)))
        sel = rng.choice(len(rows), min(k, len(rows)), replace=False)
        return [rows[i] for i in sel]

    model = DepthAnything3.from_pretrained(a.model).to(a.device).eval()
    print(f"{a.model} loaded on {a.device}", flush=True)

    def run(rows, tag):
        samp, gt, vis = [], [], []
        for i, r in enumerate(rows):
            im = Image.open(d / r["image_file"]).convert("RGB")
            with torch.no_grad():
                pr = model.inference([im], process_res=a.res)
            dep = np.asarray(pr.depth)
            dep = dep[0] if dep.ndim == 3 else dep
            H, W = dep.shape
            lm = np.asarray(L[int(r["idx"])])
            xi = np.clip((lm[:, 0] * W).astype(int), 0, W - 1)
            yi = np.clip((lm[:, 1] * H).astype(int), 0, H - 1)
            samp.append(dep[yi, xi])
            gt.append(lm[:, 2])
            vis.append(np.asarray(V[int(r["idx"])]))
            if i % 50 == 0:
                print(f"  {tag} {i}/{len(rows)}", flush=True)
        return np.array(samp), np.array(gt), np.array(vis)

    tr_d, tr_z, tr_v = run(pick("train.csv", a.fit_n), "fit")
    te_d, te_z, te_v = run(pick("test.csv", a.n), "test")

    base = np.abs(te_z[te_v]).mean()
    # global: one affine for everything, fitted on train only
    ab = affine_fit(tr_d[tr_v], tr_z[tr_v], np.ones(int(tr_v.sum())))
    g = te_d * ab[0] + ab[1]
    g = g - (g * te_v).sum(1, keepdims=True) / te_v.sum(1, keepdims=True)
    g_mae = np.abs(g - te_z)[te_v].mean()
    # per image: fitted to that image's own labels. Optimistic by construction.
    pi = np.empty_like(te_d)
    for i in range(len(te_d)):
        m = te_v[i]
        c = affine_fit(te_d[i][m], te_z[i][m], np.ones(int(m.sum())))
        pi[i] = te_d[i] * c[0] + c[1]
    p_mae = np.abs(pi - te_z)[te_v].mean()

    print(f"\n{len(te_d)} test views, flat-predictor baseline {base:.5f}")
    print(f"  DA3 + global affine (like for like) : {g_mae:.5f}  "
          f"skill {1 - g_mae / base:.3f}")
    print(f"  DA3 + per-image affine (optimistic) : {p_mae:.5f}  "
          f"skill {1 - p_mae / base:.3f}")
    print(f"\n  our head 1, trained                 : 0.01351  skill 0.808")
    print(f"  the shape basis ceiling             : 0.01194  skill 0.827")
    r = np.corrcoef(te_d[te_v], te_z[te_v])[0, 1]
    print(f"\n  correlation of DA3 depth with the truth: {r:+.3f}")
    print("  (a strong correlation with a weak global score would mean the")
    print("   ORDERING is right and only the per-image scale is missing)")


if __name__ == "__main__":
    main()
