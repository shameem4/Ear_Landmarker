#!/usr/bin/env python3
"""Rank the manual annotations by how far their outer helix sits off SAM's ear.

WHY THIS SET. data/manual is model-seeded -- annotators corrected model
predictions -- so it inherits whatever the seeding model believed, and every
other check we have is measured against annotations from the same corpus. When
those agree they can still be jointly wrong, which is exactly what happened
when a barycentric bug manufactured kinks in three successive hand-corrected
sets and every internal statistic endorsed them. SAM 2.1 knows nothing about
this project, so it is the only outside opinion available.

check_helix_silhouette.py already established the aggregate on a 150-image
sample: outer helix 0.006 of ear size from SAM's boundary, inner helix 0.050.
This asks the per-image question instead -- WHICH annotations disagree -- and
ranks them worst-first for review.

THE PROMPT IS A BOX, NOT THE POINTS, so the contour that comes back is SAM's
own and scoring the points against it is not circular. The box does come from
the landmarks, which only fixes where to look.

WHAT THIS SCORE CANNOT DO is tell a bad annotation from a bad mask. Both show
up as distance. The guards below reject masks that are implausible on their own
terms (too small or large for the prompt box, fragmented, or barely overlapping
the region prompted) and those images are reported as UNSCORED rather than bad
-- a flagger that calls a SAM failure an annotation error is worse than no
flagger. The guards are not airtight, so the output is a review queue, not a
verdict: the contact sheet exists because the eye is the arbiter here.

ATTACHMENT ENDS ARE EXCLUDED by default. Points 0 and 1 are where the ear meets
the head and there is no edge for them to lie on; the earlier run measured them
as the worst outer-helix points for that reason, not because they are wrong.
Per-point medians are printed so the exclusion stays auditable.

    python scripts/sam_annotation_qa.py --data data/manual --device cuda
    python scripts/sam_annotation_qa.py --data data/manual --n 200 --sheet qa.png
"""
import argparse
import csv
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scheme import STRIPS, group_of              # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/manual")
    p.add_argument("--n", type=int, default=0, help="0 = the whole corpus")
    p.add_argument("--model", default="facebook/sam2.1-hiera-base-plus")
    p.add_argument("--pad", type=float, default=0.12, help="box padding")
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-points", type=int, nargs="*", default=[0, 1],
                   help="attachment end, where there is no silhouette to be on")
    p.add_argument("--flag", type=float, default=0.05,
                   help="score above this (fraction of ear size) is queued")
    p.add_argument("--out", default="runs/sam_qa.csv")
    p.add_argument("--sheet", default="", help="png contact sheet of the worst")
    p.add_argument("--sheet-n", type=int, default=24)
    p.add_argument("--save-dist", default="",
                   help="npz of the full per-point distance matrix, so the\n"
                        "ranking can be redone without re-running SAM -- the\n"
                        "first run lacked this and cost a 20-minute repeat")
    a = p.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    import torch
    from PIL import Image
    from scipy.ndimage import distance_transform_edt, label
    from transformers import Sam2Model, Sam2Processor

    d = pathlib.Path(a.data)
    rows = list(csv.DictReader(open(d / "manifest.csv")))
    L = np.load(d / "landmarks.npy", mmap_mode="r")
    if a.n:
        rng = np.random.default_rng(0)
        rows = [rows[i] for i in sorted(rng.choice(len(rows), min(a.n, len(rows)), False))]
    print(f"{len(rows)} images from {d}")

    proc = Sam2Processor.from_pretrained(a.model)
    model = Sam2Model.from_pretrained(a.model).to(a.device).eval()

    lo, hi = STRIPS[0]
    use = [j for j in range(lo, hi) if j not in a.skip_points]
    print(f"scoring outer-helix points {lo}-{hi - 1} except {a.skip_points}")

    recs, per_point, unscored = [], [], {}
    for i, r in enumerate(rows):
        if i and i % 250 == 0:
            print(f"  [{i}/{len(rows)}] {len(recs)} scored, "
                  f"{sum(unscored.values())} unscored")
        pts = np.asarray(L[int(r["idx"])], float)
        if not np.isfinite(pts).all():
            unscored["bad labels"] = unscored.get("bad labels", 0) + 1
            continue
        im = Image.open(d / r["image_file"]).convert("RGB")
        W, H = im.size
        pts = pts * np.array([W, H])          # labels are normalised to the crop
        x0, y0 = pts.min(0)
        x1, y1 = pts.max(0)
        pw, ph = (x1 - x0) * a.pad, (y1 - y0) * a.pad
        bx = [max(0., x0 - pw), max(0., y0 - ph), min(W, x1 + pw), min(H, y1 + ph)]
        with torch.no_grad():
            inp = proc(images=im, input_boxes=[[bx]], return_tensors="pt").to(a.device)
            out = model(**inp, multimask_output=False)
            m = proc.post_process_masks(out.pred_masks.cpu(),
                                        inp["original_sizes"])[0][0, 0].numpy() > 0

        # --- guards on the MASK, judged without reference to the annotation ---
        # Thresholds are MEASURED, not guessed, over 150 random images:
        #   largest component / whole mask   p1 0.572  p5 0.739  median 0.992
        #   largest component / box area     p5 0.241  median 0.560  p99 1.067
        # So specks are the norm and genuine splits are not -- only 1 image in
        # 150 fell below 0.50 dominance. An earlier 0.90 cut rejected 16% of
        # images for stray pixels, which is why the largest component is simply
        # TAKEN and dominance only has to rule out a mask in two real pieces.
        # The area bound is the load-bearing one: a padded box is ~0.56 ear by
        # area, so a mask filling it has grabbed the head or the hair.
        box_area = max(1.0, (bx[2] - bx[0]) * (bx[3] - bx[1]))
        lab, k = label(m)
        if k == 0:
            unscored["empty mask"] = unscored.get("empty mask", 0) + 1
            continue
        sizes = np.bincount(lab.ravel())[1:]
        if sizes.max() / sizes.sum() < 0.50:
            unscored["mask in two pieces"] = unscored.get("mask in two pieces", 0) + 1
            continue
        m = lab == (1 + int(sizes.argmax()))
        frac = m.sum() / box_area
        if not 0.15 <= frac <= 0.95:
            unscored["mask area implausible"] = unscored.get("mask area implausible", 0) + 1
            continue
        ys, xs = np.nonzero(m)
        ix0, iy0 = max(bx[0], xs.min()), max(bx[1], ys.min())
        ix1, iy1 = min(bx[2], xs.max()), min(bx[3], ys.max())
        inter = max(0., ix1 - ix0) * max(0., iy1 - iy0)
        mbox = max(1.0, (xs.max() - xs.min()) * (ys.max() - ys.min()))
        if inter / (box_area + mbox - inter) < 0.30:
            unscored["mask elsewhere"] = unscored.get("mask elsewhere", 0) + 1
            continue

        edge = distance_transform_edt(m) + distance_transform_edt(~m)
        ear = max(y1 - y0, x1 - x0)
        xi = np.clip(pts[:, 0].astype(int), 0, W - 1)
        yi = np.clip(pts[:, 1].astype(int), 0, H - 1)
        dist = edge[yi, xi] / max(ear, 1e-6)
        per_point.append(dist)
        recs.append({"idx": r["idx"], "image_file": r["image_file"],
                     "score": float(np.median(dist[use])),
                     "p90": float(np.percentile(dist[use], 90)),
                     "worst_point": int(use[int(np.argmax(dist[use]))]),
                     "worst_dist": float(dist[use].max()),
                     "mask_frac": float(frac)})

    if not recs:
        sys.exit("nothing scored -- check the prompt box and the guards")
    D = np.asarray(per_point)
    order = np.argsort([-z["score"] for z in recs])
    D = D[order]                      # keep D aligned with recs
    recs = [recs[i] for i in order]

    print(f"\n{len(recs)} scored, {sum(unscored.values())} unscored")
    for k, v in sorted(unscored.items(), key=lambda z: -z[1]):
        print(f"  {v:6d}  {k}")

    print("\nper-point median distance to SAM's boundary, fraction of ear size:")
    print(f"{'point':>6s} {'group':<16s}{'median':>8s}{'p90':>8s}")
    for j in range(lo, hi):
        mark = "  (excluded)" if j in a.skip_points else ""
        print(f"{j:>6d} {group_of(j):<16s}{np.median(D[:, j]):8.3f}"
              f"{np.percentile(D[:, j], 90):8.3f}{mark}")
    inner = np.median(D[:, STRIPS[1][0]:STRIPS[1][1]])
    print(f"\nouter helix (scored points) median {np.median(D[:, use]):.3f}")
    print(f"inner helix, which should NOT be on a silhouette: {inner:.3f}")
    print("  -> that gap is the check working; equal would mean the mask")
    print("     is telling us nothing about either")

    s = np.array([z["score"] for z in recs])
    qs = np.percentile(s, [50, 90, 99])
    print(f"\nper-image score: median {qs[0]:.3f}  p90 {qs[1]:.3f}  p99 {qs[2]:.3f}")
    n_flag = int((s > a.flag).sum())
    print(f"{n_flag} images ({100 * n_flag / len(s):.1f}%) score above {a.flag} "
          f"-- the review queue")

    if a.save_dist:
        np.savez(a.save_dist, dist=D,
                 idx=np.array([z["idx"] for z in recs]),
                 image_file=np.array([z["image_file"] for z in recs]))
        print(f"wrote {a.save_dist}")

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)
    print(f"wrote {out} (worst first)")

    if a.sheet:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        k = min(a.sheet_n, len(recs))
        cols = 6
        rowsn = (k + cols - 1) // cols
        fig, ax = plt.subplots(rowsn, cols, figsize=(2.2 * cols, 2.4 * rowsn))
        for t, z in enumerate(recs[:k]):
            b = ax.flat[t]
            im = Image.open(d / z["image_file"]).convert("RGB")
            W, H = im.size
            pts = np.asarray(L[int(z["idx"])], float) * np.array([W, H])
            b.imshow(im)
            b.plot(pts[lo:hi, 0], pts[lo:hi, 1], ".-", c="#ff2d55", ms=3, lw=0.8)
            b.plot(pts[z["worst_point"], 0], pts[z["worst_point"], 1], "o",
                   mfc="none", mec="yellow", ms=9, mew=1.5)
            b.set_title(f"{z['score']:.3f}  pt {z['worst_point']}", fontsize=7)
            b.axis("off")
        for b in ax.flat[k:]:
            b.axis("off")
        fig.suptitle("Worst outer-helix agreement with SAM's ear silhouette "
                     "(red = annotated outer helix, yellow = worst point)",
                     fontsize=9)
        fig.tight_layout()
        fig.savefig(a.sheet, dpi=130)
        print(f"wrote {a.sheet}")


if __name__ == "__main__":
    main()
