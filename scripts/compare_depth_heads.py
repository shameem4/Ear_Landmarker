#!/usr/bin/env python3
"""Are the depth heads actually different? Paired bootstrap over SUBJECTS.

A single number per head invites a ranking it cannot support. Two sources of
uncertainty matter here and they are not the same:

  WHICH SUBJECTS are in the test set. Resampled here, over subjects rather
  than views, because the 21 views of one ear are nearly the same measurement
  and treating them as independent would shrink the interval by about sqrt(21)
  for no reason.

  WHICH TRAINING RUN you got. NOT measured here -- that needs several seeds
  per head. So these intervals are a lower bound on the real uncertainty, and
  two heads whose intervals overlap are certainly not separated.

Differences are PAIRED: the same resampled subjects are scored for both heads,
which removes the subject-difficulty variance that both share and is the only
honest way to ask whether one beats another.
"""
import argparse
import csv
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def per_view_error(data, split, head, ckpt):
    import torch
    from torch.utils.data import DataLoader
    from data.depth_dataset import EarDepthDataset
    from model.depth_module import (EarDepth3DModule, EarFeatureLifter3DModule,
                                    EarLifter3DModule, EarShape3DModule)
    cls = {"z": EarDepth3DModule, "shape": EarShape3DModule,
           "lifter": EarLifter3DModule, "feat": EarFeatureLifter3DModule}[head]
    m = cls.load_from_checkpoint(ckpt, map_location="cuda"
                                 if torch.cuda.is_available() else "cpu").eval()
    dev = next(m.parameters()).device
    ds = EarDepthDataset(data / split, data, augmentation=None,
                         with_pred=head in ("lifter", "feat"))
    out = []
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=64, num_workers=4):
            b = {k: v.to(dev) for k, v in b.items()}
            k = m.hparams.num_landmarks
            t = b["landmarks"].view(-1, k, 3)
            r = m._shared_step(b)
            # recompute per view rather than trusting the batch mean
            if head == "lifter":
                xy = b["pred_xy"].view(-1, k, 2)
                zp = m.model(xy, b["pred_conf"] if m.model.use_conf else None)
            elif head == "feat":
                zp = m.model.from_sampled(b["pred_feat"], b["pred_xy"].view(-1, k, 2),
                                          b["pred_conf"] if m.model.use_conf else None)
            else:
                zp = m(b["image"]).view(-1, k, 3)[:, :, 2]
            vis = b["visible"]
            e = ((zp - t[:, :, 2]).abs() * vis).sum(1) / vis.sum(1).clamp_min(1)
            out.append(e.cpu().numpy())
    return np.concatenate(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/depth3d")
    p.add_argument("--split", default="test.csv")
    p.add_argument("--heads", nargs="+", required=True,
                   help="name=head=ckpt triples, e.g. h1=z=runs/d_z/ckpt/x.ckpt")
    p.add_argument("--boots", type=int, default=5000)
    a = p.parse_args()

    d = pathlib.Path(a.data)
    rows = list(csv.DictReader(open(d / a.split)))
    subj = np.array([r["ear"].rsplit("_", 1)[0] for r in rows])
    uniq = np.unique(subj)
    gt = np.load(d / "landmarks.npy", mmap_mode="r")
    vis = np.load(d / "visible.npy", mmap_mode="r")
    idx = np.array([int(r["idx"]) for r in rows])
    base = np.array([np.abs(np.asarray(gt[i])[:, 2]).mean() for i in idx])

    errs = {}
    for spec in a.heads:
        name, head, ck = spec.split("=", 2)
        errs[name] = per_view_error(d, a.split, head, ck)
        print(f"{name:<12s} z MAE {errs[name].mean():.5f}", flush=True)

    rng = np.random.default_rng(0)
    groups = [np.flatnonzero(subj == s) for s in uniq]
    boot = {k: np.empty(a.boots) for k in errs}
    bb = np.empty(a.boots)
    for b in range(a.boots):
        pick = rng.integers(0, len(groups), len(groups))
        sel = np.concatenate([groups[i] for i in pick])
        for k in errs:
            boot[k][b] = errs[k][sel].mean()
        bb[b] = base[sel].mean()

    print(f"\n{len(uniq)} test subjects, {len(rows)} views, "
          f"{a.boots} bootstrap resamples over SUBJECTS")
    print(f"\n{'head':<12s} {'z MAE':>9s}  {'95% CI':>18s}  {'skill':>6s}")
    for k in errs:
        lo, hi = np.percentile(boot[k], [2.5, 97.5])
        print(f"{k:<12s} {errs[k].mean():9.5f}  [{lo:.5f}, {hi:.5f}]  "
              f"{1 - errs[k].mean() / base.mean():6.3f}")

    print("\npaired differences (same resampled subjects for both):")
    names = list(errs)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            dd = boot[names[i]] - boot[names[j]]
            lo, hi = np.percentile(dd, [2.5, 97.5])
            sep = "separated" if lo > 0 or hi < 0 else "NOT separated"
            print(f"  {names[i]:>10s} - {names[j]:<10s} "
                  f"{dd.mean():+.5f}  [{lo:+.5f}, {hi:+.5f}]  {sep}")


if __name__ == "__main__":
    main()
