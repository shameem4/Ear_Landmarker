"""Re-mark every row with the shipped landmarker and score the disagreement.

What this is and is NOT. The landmarker was trained on these annotations,
so its residual on a training row is memorisation, not an independent
opinion, and it cannot ADJUDICATE them -- that circularity has already
invalidated one analysis in this project. It is used here as a RANKER,
which is the use the literature supports: a row the model failed to
memorise DESPITE training on it is inconsistent with everything else in
the corpus, which is what a label error looks like (forgetting events,
area-under-margin). The measured recall for that screen was 92% at the
worst 30%.

So: distance from the model's prediction to the annotation, per point,
as a percentage of ear height, reported per strip and per split. Pinned to
the SHIPPED checkpoint, never find_best_checkpoint(), which would pick up
a half-trained experimental run.

    PYTHONNOUSERSITE=1 python scripts/qa_remark_corpus.py runs/remark
"""
import csv
import pathlib
import sys

import numpy as np
import torch
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from inference import LandmarkPredictor
# ear3d/__init__ pulls in open3d, which the borrowed env lacks; the strips are
# a constant, so load the module directly rather than through the package.
import importlib.util as _ilu
_s = _ilu.spec_from_file_location("_scheme", ROOT / "ear3d/scheme.py")
_m = _ilu.module_from_spec(_s); _s.loader.exec_module(_m)
STRIPS, STRIP_NAMES = _m.STRIPS, _m.STRIP_NAMES

SHIPPED = ROOT / "runs/checkpoints/manual_occ_s42/EarLandmarker_292_nme=0.0149.ckpt"
OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0

dev = "cuda" if torch.cuda.is_available() else "cpu"
lmk = LandmarkPredictor(SHIPPED, device=dev)
print(f"landmarker: {SHIPPED.name} on {dev}")

L = np.load(ROOT / "data/manual/landmarks.npy")
rows = list(csv.DictReader(open(ROOT / "data/manual/manifest.csv")))
split = {}
for nm in ("train", "val", "test"):
    for r in csv.DictReader(open(ROOT / "data/manual" / f"{nm}.csv")):
        k = r.get("idx") or r.get("index")
        if k is not None:
            split[int(k)] = nm

idx, D, EH = [], [], []
todo = range(len(rows) if not LIMIT else LIMIT)
for i in todo:
    g = L[i]
    if not np.isfinite(g).all():
        continue
    try:
        im = np.asarray(Image.open(ROOT / "data/manual"
                                   / rows[i]["image_file"]).convert("RGB"))
    except Exception:
        continue
    h, w = im.shape[:2]
    gt = g * np.array([w, h])
    eh = float(gt[:20, 1].max() - gt[:20, 1].min())
    if eh < 8:
        continue
    pr = lmk.predict(im)
    idx.append(i)
    D.append(np.linalg.norm(pr - gt, axis=1) / eh * 100)
    EH.append(eh)
    if (len(idx)) % 500 == 0:
        print(f"  {len(idx)}/{len(rows)}", flush=True)

idx = np.array(idx); D = np.array(D)
sp = np.array([split.get(i, "?") for i in idx])
np.savez(OUT / "remark.npz", idx=idx, d=D, split=sp, eh=np.array(EH))
print(f"\n{len(idx)} rows re-marked\n")
print(f"{'strip':<16}{'median':>9}{'p90':>8}{'p99':>8}")
for (lo, hi), nm in zip(STRIPS, STRIP_NAMES):
    v = np.median(D[:, lo:hi], 1)
    print(f"{nm:<16}{np.median(v):>9.2f}{np.percentile(v,90):>8.2f}"
          f"{np.percentile(v,99):>8.2f}")
out = np.median(D[:, :20], 1)
print(f"\n{'split':<10}{'n':>7}{'median':>9}{'p90':>8}{'p99':>8}  (outer chain)")
for nm in ("train", "val", "test"):
    m = sp == nm
    if m.sum():
        print(f"{nm:<10}{m.sum():>7}{np.median(out[m]):>9.2f}"
              f"{np.percentile(out[m],90):>8.2f}{np.percentile(out[m],99):>8.2f}")
print(f"\n{'threshold':>10}{'rows over':>11}{'% of corpus':>13}")
for t in (2, 3, 4, 5, 6, 8, 10, 15):
    print(f"{t:>10.0f}{(out > t).sum():>11d}{100*(out > t).mean():>12.1f}%")
