"""Find the same photo delivered twice under different `ec_*` names.

`ingest_manual_v3.py` splits on the file stem, so a re-delivered photo with a
new name can land in train while its original sits in test. On 2026-10-20
this had happened to 91 of the 1,000 test ears before any new data arrived,
and the expansion re-delivered 53 more photos (see RESULTS.md and
`data/v3_manual_exclude.txt`). Run this on every onboarding batch, BEFORE ingest.

Three stages, each cheap enough to run on everything:
  1. CLIP ViT-B/32 (local HF cache, GPU) proposes the K nearest images to
     each image, and to its MIRROR, since a flipped re-upload is common.
  2. SIFT on histogram-equalised greys + RANSAC similarity proves the pair
     shares geometry. Text overlays and diagrams make false inliers here, so
     the inlier count alone does NOT separate duplicates (25-40 is mixed).
  3. The decisive test uses the labels: push A's 55 landmarks through the
     fitted transform and measure the median miss against B's, in units of
     B's ear size. Same photo: < 0.1 for a new batch against the corpus, with
     nothing between 0.1 and 0.22. Inside the old corpus the labels came from
     different sessions, so same-photo pairs run up to ~0.2 and the band
     0.1-0.3 holds both: LOOK at `review.png` before trusting the cutoff.

Exclusion policy (written to --out-exclude, never applied by this script):
  - a confirmed pair with exactly one member in --keep-split: exclude the
    other one (the evaluation split is fixed, the copy goes);
  - a confirmed pair with exactly one member matching --new: exclude the new
    one; two new members: keep the first by name;
  - pairs inside train/val are reported only: they reweight, they don't leak.

Usage:
    python scripts/dedupe_corpus.py --src v3_manual_data \\
        --keep-split data/v3_manual/test.csv --new 'ec_r_*' --new 'ec_s_*' \\
        --out /var/tmp/claude/ear_landmarker/dedupe
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

MODEL = "openai/clip-vit-base-patch32"


def embed(paths, mirror, model, proc, bs=256):
    out = []
    for i in range(0, len(paths), bs):
        ims = [Image.open(p).convert("RGB") for p in paths[i:i + bs]]
        if mirror:
            ims = [ImageOps.mirror(im) for im in ims]
        x = proc(images=ims, return_tensors="pt")["pixel_values"].cuda().half()
        with torch.no_grad():
            f = model.get_image_features(pixel_values=x)
        if not torch.is_tensor(f):
            f = f.pooler_output
        out.append(torch.nn.functional.normalize(f.float(), dim=1).cpu())
    return torch.cat(out)


def candidates(paths, k):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(MODEL, local_files_only=True).cuda().eval().half()
    proc = CLIPProcessor.from_pretrained(MODEL, local_files_only=True)
    e = embed(paths, False, model, proc)
    pairs = set()
    for q in (e, embed(paths, True, model, proc)):
        for i0 in range(0, len(paths), 1024):            # chunked: N x N won't fit
            sim = q[i0:i0 + 1024] @ e.T
            for r in range(sim.shape[0]):
                sim[r, i0 + r] = -1                       # self
            for r, row in enumerate(sim.topk(k, dim=1).indices.tolist()):
                for j in row:
                    pairs.add(tuple(sorted((i0 + r, j))))
    return sorted(pairs)


class Verifier:
    def __init__(self, src):
        self.src = src
        self.sift = cv2.SIFT_create(nfeatures=1500)

    def feat(self, stem, flip):
        g = cv2.imread(str(self.src / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        w = g.shape[1]
        if flip:
            g = g[:, ::-1].copy()
        s = min(1.0, 400 / max(g.shape))
        g = cv2.equalizeHist(cv2.resize(g, None, fx=s, fy=s))
        k, d = self.sift.detectAndCompute(g, None)
        return np.float32([p.pt for p in k]) / s, d, w

    def __call__(self, pair):
        a, b = pair
        ka, da, _ = self.feat(a, False)
        A = np.loadtxt(self.src / f"{a}_annotation.txt")
        B = np.loadtxt(self.src / f"{b}_annotation.txt")
        best = {"a": a, "b": b, "lm_err": 9.9, "inliers": 0, "flip": False}
        for flip in (False, True):
            kb, db, wb = self.feat(b, flip)
            if da is None or db is None or len(da) < 2 or len(db) < 2:
                continue
            m = cv2.BFMatcher().knnMatch(da, db, k=2)
            g = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.75 * x[1].distance]
            if len(g) < 8:
                continue
            H, inl = cv2.estimateAffinePartial2D(
                ka[[x.queryIdx for x in g]], kb[[x.trainIdx for x in g]],
                method=cv2.RANSAC, ransacReprojThreshold=6)
            if H is None or inl.sum() < 10:
                continue
            Bt = B.copy()
            if flip:
                Bt[:, 0] = wb - 1 - Bt[:, 0]
            err = float(np.median(np.linalg.norm(A @ H[:, :2].T + H[:, 2] - Bt, axis=1))
                        / np.ptp(Bt, 0).max())
            if err < best["lm_err"]:
                best.update(lm_err=round(err, 4), inliers=int(inl.sum()), flip=flip)
        return best


def sheet(rows, src, path, cols=5):
    n = max(1, (len(rows) + cols - 1) // cols)
    g = Image.new("RGB", (cols * 200, n * 112), "white")
    for k, r in enumerate(rows):
        t = Image.new("RGB", (200, 112), "white")
        for m, st in enumerate((r["a"], r["b"])):
            im = Image.open(src / f"{st}.png").convert("RGB")
            im.thumbnail((95, 95))
            t.paste(im, (3 + 98 * m, 16))
        ImageDraw.Draw(t).text((2, 2), f'{r["lm_err"]:.3f} {r["a"][3:]} {r["b"][3:]}', fill="red")
        g.paste(t, (200 * (k % cols), 112 * (k // cols)))
    g.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=Path("v3_manual_data"))
    ap.add_argument("--keep-split", type=Path, help="split CSV whose ears must stay")
    ap.add_argument("--new", action="append", default=[], help="glob for the new batch")
    ap.add_argument("--cutoff", type=float, default=0.2, help="landmark-transfer error")
    ap.add_argument("--k", type=int, default=8, help="CLIP neighbours per image")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.src.glob("ec_*.png"))
    stems = [p.stem for p in paths]
    cand = candidates(paths, args.k)
    print(f"{len(stems)} images, {len(cand)} CLIP candidate pairs")
    with ThreadPoolExecutor(16) as ex:
        rows = list(ex.map(Verifier(args.src), [(stems[i], stems[j]) for i, j in cand]))
    rows = sorted((r for r in rows if r["inliers"]), key=lambda r: r["lm_err"])
    json.dump(rows, open(args.out / "pairs.json", "w"))
    hist = np.histogram([r["lm_err"] for r in rows], [0, .05, .1, .15, .2, .25, .3, 10])[0]
    print("geometry-verified pairs by landmark-transfer error "
          "[0,.05,.1,.15,.2,.25,.3,inf):", hist.tolist())

    dup = [r for r in rows if r["lm_err"] < args.cutoff]
    sheet([r for r in rows if 0.1 <= r["lm_err"] < 0.3][:60], args.src, args.out / "review.png")
    sheet(dup[:60], args.src, args.out / "confirmed_sample.png")

    keep = set()
    if args.keep_split:
        keep = {Path(r["image_file"]).stem for r in csv.DictReader(open(args.keep_split))}
    is_new = lambda s: any(fnmatch.fnmatch(s, g) for g in args.new)
    excl, leaks, inner = set(), 0, 0
    for r in dup:
        a, b = r["a"], r["b"]
        if (a in keep) != (b in keep):
            excl.add(b if a in keep else a); leaks += 1
        elif is_new(a) != is_new(b):
            excl.add(a if is_new(a) else b)
        elif is_new(a):
            excl.add(max(a, b))
        else:
            inner += 1
    (args.out / "exclude.txt").write_text("".join(f"{s}\n" for s in sorted(excl)))
    print(f"confirmed pairs {len(dup)}: {leaks} cross the kept split, "
          f"{inner} inside the rest (reported, not excluded)")
    print(f"proposed exclusions {len(excl)} -> {args.out / 'exclude.txt'}")
    print(f"LOOK at {args.out / 'review.png'} (the 0.1-0.3 band) before using them")


if __name__ == "__main__":
    main()
