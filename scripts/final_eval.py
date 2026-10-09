#!/usr/bin/env python3
"""Score the trained heads on validation AND test, side by side.

Both splits, deliberately. Checkpoints were selected on validation, so a
validation number is optimistic by an unknown amount -- and the amount is not
the same for every metric. Head 4's mesh skill is the case in point: 0.10-0.12
on validation against 0.05-0.06 on test, while its depth barely moves. Reading
only one split hid that, and an interim validation number was reported as a
result before this script existed.

`mesh_skill` is 1 - mesh_err/mesh_base, and mesh_base -- the error of simply
predicting the MEAN ear -- is a property of the split, not of the model. It is
printed so a skill difference between splits can be traced to the denominator
rather than assumed to be the model.

    python scripts/final_eval.py --data data/depth3d --basis lm_basis_v4.npz
"""
import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HEADS = {"z": "EarDepth3DModule", "shape": "EarShape3DModule",
         "lifter": "EarLifter3DModule", "feat": "EarFeatureLifter3DModule",
         "shapefeat": "EarShapeFeat3DModule"}


def score(ckpt, head, data, split, device, coeffs=""):
    import torch
    from torch.utils.data import DataLoader
    from data.depth_dataset import EarDepthDataset
    import model.depth_module as dm
    cls = getattr(dm, HEADS[head])
    m = cls.load_from_checkpoint(ckpt, map_location=device).eval().to(device)
    ds = EarDepthDataset(data / split, data,
                         augmentation=None,
                         with_pred=head in ("lifter", "feat"),
                         coeffs=coeffs if head in ("shape", "shapefeat") else "")
    keys = ("nme", "z_mae", "z_base", "mesh_err", "mesh_base")
    tot = {k: 0.0 for k in keys}
    seen = {k: 0 for k in keys}
    n = 0
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=64, num_workers=4):
            b = {k: v.to(device) for k, v in b.items()}
            r = m._shared_step(b)
            w = len(b["image"])
            n += w
            for k in keys:
                if k in r:
                    tot[k] += float(r[k]) * w
                    seen[k] += w
    out = {k: tot[k] / seen[k] for k in keys if seen[k]}
    out["z_skill"] = 1 - out["z_mae"] / out["z_base"]
    if "mesh_err" in out:
        out["mesh_skill"] = 1 - out["mesh_err"] / out["mesh_base"]
    out["params"] = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/depth3d")
    p.add_argument("--coeffs", default="/mnt/14BE47C2BE479ADE/Code/ear_stuff/"
                   "wrap_work/templates/gt_coeffs_v4.npz",
                   help="ground-truth coefficients, needed for mesh_err")
    p.add_argument("--splits", nargs="+", default=["val.csv", "test.csv"])
    p.add_argument("--models", nargs="+", help="name=head=ckpt triples",
                   default=["head 1=z=runs/final_z/ckpt/epoch=139-val_z_mae=0.0117.ckpt",
                            "head 2=shape=runs/final_shape/ckpt/epoch=10-val_z_mae=0.0284.ckpt",
                            "head 4 r1=shapefeat=runs/h4_r1/ckpt/epoch=147-val_z_mae=0.0134.ckpt",
                            "head 4 r2=shapefeat=runs/h4_r2/ckpt/epoch=145-val_z_mae=0.0135.ckpt"])
    a = p.parse_args()

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = pathlib.Path(a.data)
    cols = ["z MAE", "z skill", "NME", "mesh err", "mesh base", "mesh skill"]
    for split in a.splits:
        print(f"\n=== {split} ===")
        print(f"{'model':<14}" + "".join(f"{c:>11}" for c in cols) + f"{'params':>10}")
        for spec in a.models:
            name, head, ckpt = spec.split("=", 2)
            r = score(ckpt, head, d, split, dev, a.coeffs)
            row = [r["z_mae"], r["z_skill"], r["nme"],
                   r.get("mesh_err"), r.get("mesh_base"), r.get("mesh_skill")]
            print(f"{name:<14}" + "".join(
                f"{'-':>11}" if v is None else f"{v:>11.5f}" for v in row)
                + f"{r['params']:>10,}")


if __name__ == "__main__":
    main()
