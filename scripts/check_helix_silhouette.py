#!/usr/bin/env python3
"""Check the outer-helix annotations against an ear silhouette SAM finds itself.

Every check this project has on its landmarks is internal: turning angles,
position along a chain, chain length, all measured against other annotations
from the same corpus. When they agree they can still be jointly wrong, which is
what happened when a barycentric bug manufactured kinks in three successive
hand-corrected sets and every statistic endorsed them.

This is an outside opinion. The outer helix traces the ear's outer contour, so
where the ear meets background or hair those points should lie ON the
silhouette. SAM 2.1 is Apache 2.0 for code and weights, knows nothing about
this project, and was never trained on ears specifically.

THE PROMPT IS A BOX, NOT THE POINTS. Prompting with the landmarks and then
scoring the landmarks against the result would be close to circular; a padded
box fixes only where to look, and the contour that comes back is SAM's own.

Expect the attachment ends -- roughly points 0 and 19, where the ear meets the
head -- to sit off the boundary, because there is no edge there to lie on.

    python scripts/check_helix_silhouette.py --data data/preprocessed --n 60
"""
import argparse
import csv
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ear3d.scheme import STRIPS, group_of              # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/preprocessed")
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--sources", nargs="*", default=["audioear2d", "audioear3d"],
                   help="default is the commercially clean rows only")
    p.add_argument("--model", default="facebook/sam2.1-hiera-base-plus")
    p.add_argument("--pad", type=float, default=0.12, help="box padding")
    p.add_argument("--device", default="cpu")
    p.add_argument("--save", default="")
    a = p.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    import torch
    from PIL import Image
    from scipy.ndimage import distance_transform_edt
    from transformers import Sam2Model, Sam2Processor

    d = pathlib.Path(a.data)
    rows = list(csv.DictReader(open(d / "manifest.csv")))
    L = np.load(d / "landmarks.npy", mmap_mode="r")
    keep = [r for r in rows if r["source"] in a.sources]
    rng = np.random.default_rng(0)
    take = [keep[i] for i in rng.choice(len(keep), min(a.n, len(keep)), False)]
    print(f"{len(take)} images from {a.sources}")

    proc = Sam2Processor.from_pretrained(a.model)
    model = Sam2Model.from_pretrained(a.model).to(a.device).eval()

    lo, hi = STRIPS[0]
    dists, cover = [], []
    for r in take:
        im = Image.open(d / r["image_file"]).convert("RGB")
        W, H = im.size
        pts = np.asarray(L[int(r["idx"])], float)
        if not np.isfinite(pts).all():
            continue
        # the stored labels are NORMALISED to the crop, not pixels
        pts = pts * np.array([W, H])
        x0, y0 = pts.min(0)
        x1, y1 = pts.max(0)
        pw, ph = (x1 - x0) * a.pad, (y1 - y0) * a.pad
        box = [[[max(0, x0 - pw), max(0, y0 - ph),
                 min(W, x1 + pw), min(H, y1 + ph)]]]
        with torch.no_grad():
            inp = proc(images=im, input_boxes=box, return_tensors="pt").to(a.device)
            out = model(**inp, multimask_output=False)
            m = proc.post_process_masks(out.pred_masks.cpu(),
                                        inp["original_sizes"])[0][0, 0].numpy() > 0
        if m.sum() < 50:
            continue
        # distance from every pixel to the mask EDGE, inside and out
        edge = distance_transform_edt(m) + distance_transform_edt(~m)
        ear = max(y1 - y0, x1 - x0)
        xi = np.clip(pts[:, 0].astype(int), 0, W - 1)
        yi = np.clip(pts[:, 1].astype(int), 0, H - 1)
        dists.append(edge[yi, xi] / ear)                    # fraction of ear size
        cover.append(m[yi, xi])

    if not dists:
        sys.exit("no image produced a usable mask -- check the prompt box")
    D = np.asarray(dists)
    C = np.asarray(cover)
    print(f"\n{len(D)} images scored. Distance from each landmark to SAM's ear")
    print("boundary, as a fraction of ear size (0 = exactly on the silhouette):\n")
    print(f"{'point':>6s} {'group':<16s}{'median':>8s}{'p90':>8s}{'inside mask':>13s}")
    bad = []
    for j in range(lo, hi):
        med = float(np.median(D[:, j]))
        print(f"{j:>6d} {group_of(j):<16s}{med:8.3f}{np.percentile(D[:, j], 90):8.3f}"
              f"{100 * C[:, j].mean():12.0f}%")
        if med > 0.08:
            bad.append((j, med))
    inner = np.median(D[:, STRIPS[1][0]:STRIPS[1][1]])
    print(f"\nouter helix overall: median {np.median(D[:, lo:hi]):.3f} of ear size")
    print(f"inner helix, which should NOT be on the silhouette: {inner:.3f}")
    print(f"  -> the gap between those two is the check working; if they were")
    print(f"     equal the mask would be telling us nothing")
    if bad:
        print(f"\n{len(bad)} outer-helix points sit far from the boundary:")
        for j, m in bad:
            print(f"  point {j} ({group_of(j)}): {m:.3f}")
        print("  expected for the attachment ends, where there is no edge")
    if a.save:
        np.savez(a.save, dist=D, inside=C)
        print(f"\nwrote {a.save}")


if __name__ == "__main__":
    main()
