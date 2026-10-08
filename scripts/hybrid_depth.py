#!/usr/bin/env python3
"""Do the heads combine, or is one of them simply best?

Two hybrids that need no training at all, so they can be answered before
anyone builds one:

  PROJECT ONTO THE SHAPE MANIFOLD. Head 1 predicts 55 free points; the shape
  basis can express an ear to 0.01194, which is BETTER than head 1's 0.01315.
  So the basis knows something head 1 does not. Fitting shape and pose to head
  1's own output -- not to the labels -- snaps it onto the manifold of real
  ears. If that improves the error, a hybrid is worth building and the basis
  is acting as a denoiser. If it degrades it, head 1's errors already lie
  along directions real ears vary in, and the manifold has nothing to add.

  AVERAGE THE HEADS. They fail differently -- one reads the image, one reads
  only the point configuration -- so an average can beat both. Equal weights
  and the best single weighting are both reported; if the best weight sits at
  an endpoint, the ensemble is not real.

    python scripts/hybrid_depth.py --heads head1=z=... head3b=feat=...
"""
import argparse
import csv
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def predict(data, split, head, ckpt):
    import torch
    from torch.utils.data import DataLoader
    from data.depth_dataset import EarDepthDataset
    from model.depth_module import (EarDepth3DModule, EarFeatureLifter3DModule,
                                    EarLifter3DModule, EarShape3DModule)
    cls = {"z": EarDepth3DModule, "shape": EarShape3DModule,
           "lifter": EarLifter3DModule, "feat": EarFeatureLifter3DModule}[head]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = cls.load_from_checkpoint(ckpt, map_location=dev).eval()
    ds = EarDepthDataset(data / split, data, augmentation=None,
                         with_pred=head in ("lifter", "feat"))
    P, Z, V = [], [], []
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=64, num_workers=4):
            b = {k: v.to(dev) for k, v in b.items()}
            k = m.hparams.num_landmarks
            t = b["landmarks"].view(-1, k, 3)
            if head == "lifter":
                z = m.model(b["pred_xy"].view(-1, k, 2),
                            b["pred_conf"] if m.model.use_conf else None)
                xy = b["pred_xy"].view(-1, k, 2)
            elif head == "feat":
                xy = b["pred_xy"].view(-1, k, 2)
                z = m.model.from_sampled(b["pred_feat"], xy,
                                         b["pred_conf"] if m.model.use_conf else None)
            else:
                o = m(b["image"]).view(-1, k, 3)
                xy, z = o[:, :, :2], o[:, :, 2]
            P.append(torch.cat([xy, z[:, :, None]], -1).cpu().numpy())
            Z.append(t.cpu().numpy())
            V.append(b["visible"].cpu().numpy())
    return np.concatenate(P), np.concatenate(Z), np.concatenate(V)


def fit_basis_to(pred, basis, iters=1500, lr=0.03):
    """Snap predictions onto the manifold of real ears. Fits to PRED, not GT."""
    import torch
    from model.shape_head import rot6d_to_matrix
    d = np.load(basis)
    L0 = torch.tensor(d["mean"], dtype=torch.float32)
    B = torch.tensor(d["components"], dtype=torch.float32)
    T = torch.tensor(pred, dtype=torch.float32)
    n = len(T)
    c = torch.zeros(n, len(B), requires_grad=True)
    r = torch.tensor([[1., 0, 0, 0, 1, 0]]).repeat(n, 1).requires_grad_()
    ls = torch.zeros(n, 1, requires_grad=True)
    t = torch.full((n, 2), 0.5, requires_grad=True)
    opt = torch.optim.Adam([c, r, ls, t], lr=lr)
    for _ in range(iters):
        R = rot6d_to_matrix(r)
        L = torch.einsum("bij,bkj->bki", R, L0 + torch.einsum("bm,mkj->bkj", c, B))
        s = ls.exp()[:, :, None]
        z = s[:, :, 0] * L[:, :, 2]
        out = torch.cat([s * L[:, :, :2] + t[:, None, :],
                         (z - z.mean(1, keepdim=True))[:, :, None]], -1)
        loss = (out - T).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if _ == iters - 1:
            print(f"    (fit residual to its target: {loss.item()**0.5:.5f})")
    return out.detach().numpy()


def mae(a, b, v):
    return float((np.abs(a - b) * v).sum() / v.sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/depth3d")
    p.add_argument("--split", default="test.csv")
    p.add_argument("--basis", required=True)
    p.add_argument("--heads", nargs="+", required=True)
    a = p.parse_args()

    d = pathlib.Path(a.data)
    preds, gt, vis = {}, None, None
    for spec in a.heads:
        name, head, ck = spec.split("=", 2)
        preds[name], gt, vis = predict(d, a.split, head, ck)
        print(f"{name:<8s} z MAE {mae(preds[name][:,:,2], gt[:,:,2], vis):.5f}",
              flush=True)
    base = mae(np.zeros_like(gt[:, :, 2]), gt[:, :, 2], vis)

    first = list(preds)[0]
    print(f"\n--- projecting {first} onto the shape manifold (fitted to its own "
          f"output, not the labels) ---")
    snapped = fit_basis_to(preds[first], a.basis)
    m0 = mae(preds[first][:, :, 2], gt[:, :, 2], vis)
    m1 = mae(snapped[:, :, 2], gt[:, :, 2], vis)
    print(f"  before {m0:.5f}  (skill {1-m0/base:.3f})")
    print(f"  after  {m1:.5f}  (skill {1-m1/base:.3f})   "
          f"{'IMPROVES' if m1 < m0 else 'DEGRADES'} by {abs(m1-m0)/m0*100:.1f}%")
    print(f"  x,y    {mae(preds[first][:,:,:2], gt[:,:,:2], vis[:,:,None]):.5f}"
          f" -> {mae(snapped[:,:,:2], gt[:,:,:2], vis[:,:,None]):.5f}")

    names = list(preds)
    if len(names) >= 2:
        print(f"\n--- averaging depth across heads ---")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                A, B = preds[names[i]][:, :, 2], preds[names[j]][:, :, 2]
                ws = np.linspace(0, 1, 21)
                errs = [mae(w * A + (1 - w) * B, gt[:, :, 2], vis) for w in ws]
                k = int(np.argmin(errs))
                half = mae(0.5 * (A + B), gt[:, :, 2], vis)
                edge = k in (0, 20)
                print(f"  {names[i]:>7s}+{names[j]:<7s} equal {half:.5f}  "
                      f"best w={ws[k]:.2f} {errs[k]:.5f}  "
                      f"{'NO real ensemble (best is an endpoint)' if edge else 'mixture beats both'}")


if __name__ == "__main__":
    main()
