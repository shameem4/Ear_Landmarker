#!/usr/bin/env python3
"""Score a checkpoint on each SOURCE separately, beside the pooled number.

The pooled number is the one that hides things. The depth corpus is 76%
SONICOM, so a pooled metric is three-quarters a statement about SONICOM, and
a model that is good at SONICOM and poor at HUTUBS reports as good. That is
not hypothetical here: a logistic regression on 40 shape coefficients names
the source of a HELD-OUT subject 93.9% of the time, so the two sources are
separable in exactly the signal the model reads.

Scoring is done by filtering the split's rows to one source and running the
unmodified scorer from final_eval.py on each subset. Doing it that way rather
than accumulating per-source sums inside the model keeps one definition of
every metric, including the skill denominators.

AND THE DENOMINATORS ARE PRINTED. z_skill is 1 - z_mae/z_base, and z_base --
the error of predicting z = 0 -- is a property of the SUBSET, not of the
model. If one source has flatter ears its baseline is smaller and its skill
looks worse at equal MAE. Without the baseline beside it a skill gap cannot be
told apart from a geometry difference.

    python scripts/eval_by_source.py --data data/depth3d_v5 \
        --splits bal_val.csv bal_test.csv \
        --models "balanced=z=runs/bal_z/ckpt/....ckpt" --n-points 55
"""
import argparse
import collections
import csv
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def subset_csv(split_csv, source, out_dir):
    """Write the rows of one source to their own csv; return path and count."""
    rows = list(csv.DictReader(open(split_csv)))
    if "source" not in rows[0]:
        sys.exit(f"{split_csv} has no 'source' column -- regenerate the split "
                 f"with scripts/split_depth3d.py")
    keep = [r for r in rows if r["source"] == source]
    p = out_dir / f"{pathlib.Path(split_csv).stem}__{source}.csv"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(keep)
    return p, keep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/depth3d_v5")
    p.add_argument("--splits", nargs="+", default=["bal_val.csv", "bal_test.csv"])
    p.add_argument("--models", nargs="+", required=True,
                   help="name=head=ckpt triples")
    p.add_argument("--n-points", type=int, default=0)
    p.add_argument("--coeffs", default="")
    p.add_argument("--tmp", default="", help="where the per-source csvs go")
    a = p.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from data.depth_dataset import EarDepthDataset
    import model.depth_module as dm
    # HEADS lives in final_eval.py and is loaded from there by path rather
    # than copied, so the head->class mapping has exactly one definition.
    # scripts/ is deliberately not a package, hence importlib and not import.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "final_eval", ROOT / "scripts" / "final_eval.py")
    fe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fe)
    HEADS = fe.HEADS

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = pathlib.Path(a.data)
    tmp = pathlib.Path(a.tmp) if a.tmp else d / "_by_source"
    tmp.mkdir(parents=True, exist_ok=True)

    def score(ckpt, head, split_csv):
        cls = getattr(dm, HEADS[head])
        m = cls.load_from_checkpoint(ckpt, map_location=dev).eval().to(dev)
        ds = EarDepthDataset(split_csv, d, augmentation=None,
                             with_pred=head in ("lifter", "feat"),
                             coeffs=a.coeffs if head in ("shape", "shapefeat") else "",
                             n_points=a.n_points)
        keys = ("nme", "z_mae", "z_base", "mesh_err", "mesh_base")
        tot = collections.Counter()
        seen = collections.Counter()
        with torch.no_grad():
            for b in DataLoader(ds, batch_size=64, num_workers=4):
                b = {k: v.to(dev) for k, v in b.items()}
                r = m._shared_step(b)
                w = len(b["image"])
                for k in keys:
                    if k in r:
                        tot[k] += float(r[k]) * w
                        seen[k] += w
        out = {k: tot[k] / seen[k] for k in keys if seen[k]}
        out["z_skill"] = 1 - out["z_mae"] / out["z_base"]
        if "mesh_err" in out:
            out["mesh_skill"] = 1 - out["mesh_err"] / out["mesh_base"]
        out["n"] = seen["z_mae"]
        return out

    # (name, width, decimals); n is an integer count, the rest are floats
    cols = [("n", 7, None), ("nme", 9, 5), ("z_mae", 9, 5), ("z_base", 9, 5),
            ("z_skill", 9, 4), ("mesh_err", 9, 5), ("mesh_skill", 11, 4)]

    def cell(c, w, dp, r):
        if c not in r:
            return f"{'-':>{w}}"
        return f"{r[c]:>{w}d}" if dp is None else f"{r[c]:>{w}.{dp}f}"
    for spec in a.models:
        name, head, ckpt = spec.split("=", 2)
        print(f"\n########## {name}  (head {head}) ##########")
        print(f"{'split':<18}{'source':<10}"
              + "".join(f"{c:>{w}}" for c, w, _ in cols))
        for split in a.splits:
            sp = d / split
            rows = list(csv.DictReader(open(sp)))
            srcs = sorted({r.get("source", "?") for r in rows})
            runs = [("pooled", sp)]
            if len(srcs) > 1:
                runs += [(s, subset_csv(sp, s, tmp)[0]) for s in srcs]
            for label, csv_path in runs:
                r = score(ckpt, head, csv_path)
                print(f"{split:<18}{label:<10}"
                      + "".join(cell(c, w, dp, r) for c, w, dp in cols),
                      flush=True)


if __name__ == "__main__":
    main()
