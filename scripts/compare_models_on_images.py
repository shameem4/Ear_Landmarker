#!/usr/bin/env python3
"""Overlay two landmarker runs on the same images, beside the annotation.

Built to look at the images the SAM silhouette check ranked worst, and ask
whether the models agree with the annotator or with each other.

TWO CONFOUNDS, BOTH FATAL TO A NAIVE READING, so they are printed on every
panel rather than buried here:

  SPLIT. data/manual is manual_occ_s42's OWN training corpus. On a train-split
  image that model has seen the annotation it is being scored against, so it
  will look better than it is. The panel title says which split each image is
  in; only val and test rows mean anything as a comparison.

  CONVENTION. v6_persp65 was trained on the iBUG-derived corpus, 55% of whose
  test split is collectionB, and manual_occ_s42 was not trained on that
  convention at all. A systematic offset between the two is as likely to be two
  annotation conventions as two accuracies.

So this is a picture to look at, not a measurement. scripts/eval_test.py and
eval_gt_free.py are the measurements.

    python scripts/compare_models_on_images.py --rank runs/sam_qa.csv --n 24
"""
import argparse
import csv
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scheme import STRIPS                                 # noqa: E402


def best_ckpt(run):
    d = ROOT / "runs" / "checkpoints" / run
    cks = [c for c in d.glob("*.ckpt") if "nme=" in c.name]
    if not cks:
        return None
    return min(cks, key=lambda c: float(c.stem.split("nme=")[1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/manual")
    p.add_argument("--rank", default="runs/sam_qa.csv",
                   help="csv with an idx column, already ordered")
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--runs", nargs=2, default=["manual_occ_s42", "v6_persp65"])
    p.add_argument("--out", default="runs/model_compare.png")
    a = p.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image
    from model.lightning_module import EarLandmarkerModule

    d = pathlib.Path(a.data)
    L = np.load(d / "landmarks.npy", mmap_mode="r")
    by_idx = {r["idx"]: r for r in csv.DictReader(open(d / "manifest.csv"))}
    split = {}
    for sp in ("train", "val", "test"):
        for r in csv.DictReader(open(d / f"{sp}.csv")):
            split[r["idx"]] = sp

    take = [r["idx"] for r in list(csv.DictReader(open(a.rank)))[:a.n]]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    models, sizes = {}, {}
    for run in a.runs:
        ck = best_ckpt(run)
        if ck is None:
            sys.exit(f"no checkpoint for {run}")
        m = EarLandmarkerModule.load_from_checkpoint(str(ck), map_location=dev)
        models[run] = m.eval().to(dev)
        sizes[run] = int(m.hparams.get("image_size", 192))
        print(f"{run:18s} {ck.name}  input {sizes[run]}")

    cols = 6
    rowsn = (len(take) + cols - 1) // cols
    fig, ax = plt.subplots(rowsn, cols, figsize=(2.7 * cols, 3.0 * rowsn))
    cmap = {a.runs[0]: "#ff2d95", a.runs[1]: "#2dd4ff"}
    nmes = {r: [] for r in a.runs}
    for t, gi in enumerate(take):
        b = ax.flat[t]
        r = by_idx[gi]
        im = Image.open(d / r["image_file"]).convert("RGB")
        W, H = im.size
        gt = np.asarray(L[int(gi)], float)
        b.imshow(im)
        for lo, hi in STRIPS:
            b.plot(gt[lo:hi, 0] * W, gt[lo:hi, 1] * H, "-", c="#39ff88", lw=1.3)
        b.plot(gt[:, 0] * W, gt[:, 1] * H, ".", c="#39ff88", ms=2.5)
        bits = []
        for run in a.runs:
            s = sizes[run]
            x = TF.normalize(TF.to_tensor(im.resize((s, s), Image.BILINEAR)),
                             [0.5] * 3, [0.5] * 3)[None].to(dev)
            with torch.no_grad():
                pr = models[run](x).reshape(-1, 2).cpu().numpy()
            for lo, hi in STRIPS:
                b.plot(pr[lo:hi, 0] * W, pr[lo:hi, 1] * H, "-",
                       c=cmap[run], lw=1.2, alpha=.9)
            e = np.linalg.norm(pr - gt, axis=1).mean()
            nmes[run].append(e)
            bits.append(f"{e:.3f}")
        b.set_title(f"{split.get(gi, '?')}  {' / '.join(bits)}", fontsize=7)
        b.axis("off")
    for b in ax.flat[len(take):]:
        b.axis("off")

    lab = " / ".join(f"{r}" for r in a.runs)
    fig.suptitle(
        f"GREEN = annotation.  MAGENTA = {a.runs[0]} (shipped).  "
        f"CYAN = {a.runs[1]}.\n"
        f"Title: split, then mean error to the annotation for {lab}.  "
        f"TRAIN rows are seen data for the shipped model.", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=135)
    print(f"\nwrote {out}")
    for sp in ("train", "val", "test", "all"):
        sel = [i for i, gi in enumerate(take)
               if sp == "all" or split.get(gi) == sp]
        if not sel:
            continue
        bits = "  ".join(f"{r} {np.mean([nmes[r][i] for i in sel]):.4f}"
                         for r in a.runs)
        print(f"  {sp:6s} n={len(sel):2d}   {bits}")


if __name__ == "__main__":
    main()
