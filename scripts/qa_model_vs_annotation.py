"""Does our own landmarker disagree with the annotations SAM 3 flagged?

A model trained on 10,535 rows of which a few percent are wrong should
learn the majority convention and regress away from any single bad row.
So where it disagrees with an annotation, the annotation is the more
likely culprit -- but only where the row was HELD OUT. On a training row
the prediction is partly memorised and agreement proves nothing, which is
the whole reason [[manual-data-model-seeded]] exists.

So this splits every comparison by train vs val/test and reports them
apart. The held-out slice is the evidence; the training slice is there to
show how much memorisation flatters it.

Scored on the outer chain (points 0-19), as % of ear height, the same
units as every other QA number here.

    python scripts/qa_model_vs_annotation.py runs/measure
"""
import csv
import json
import pathlib
import sys

import numpy as np
import torch
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from inference import LandmarkPredictor, find_best_checkpoint

OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = find_best_checkpoint()
print(f"landmarker: {ck.parent.name}/{ck.name}")
lmk = LandmarkPredictor(ck, device=dev)

L = np.load(ROOT / "data/manual/landmarks.npy")
rows = list(csv.DictReader(open(ROOT / "data/manual/manifest.csv")))

split = {}
for nm in ("train", "val", "test"):
    for r in csv.DictReader(open(ROOT / "data/manual" / f"{nm}.csv")):
        k = r.get("idx") or r.get("index")
        if k is not None:
            split[int(k)] = nm
print(f"split covers {len(split)} rows")

flag = [int(x) for x in (ROOT / "runs/measure/rows_flagged.txt").read_text().split()]
ctrl = [int(x) for x in (ROOT / "runs/measure/rows_control.txt").read_text().split()]

res = {}
for grp, idxs in (("flagged", flag), ("control", ctrl)):
    for i in idxs:
        try:
            im = np.asarray(Image.open(ROOT / "data/manual"
                                       / rows[i]["image_file"]).convert("RGB"))
        except Exception:
            continue
        g = L[i]
        if not np.isfinite(g).all():
            continue
        h, w = im.shape[:2]
        gt = g[:20] * np.array([w, h])
        span = float(gt[:, 1].max() - gt[:, 1].min())
        if span < 8:
            continue
        pr = lmk.predict(im)[:20]
        d = np.linalg.norm(pr - gt, axis=1) / span * 100
        res[str(i)] = dict(grp=grp, split=split.get(i, "?"),
                           med=float(np.median(d)), mx=float(d.max()),
                           p90=float(np.percentile(d, 90)))
    print(f"  {grp}: {sum(1 for v in res.values() if v['grp']==grp)} scored")

json.dump(res, open(OUT / "model_vs_annotation.json", "w"))

print(f"\n{'group':<10}{'split':<10}{'n':>6}{'median':>9}{'p90':>8}{'max':>8}")
for grp in ("control", "flagged"):
    for sp in ("train", "held-out"):
        v = [x for x in res.values() if x["grp"] == grp
             and ((x["split"] == "train") == (sp == "train"))
             and x["split"] != "?"]
        if not v:
            continue
        print(f"{grp:<10}{sp:<10}{len(v):>6}"
              f"{np.median([x['med'] for x in v]):>9.1f}"
              f"{np.median([x['p90'] for x in v]):>8.1f}"
              f"{np.median([x['mx'] for x in v]):>8.1f}")
print("\n(distance from the ANNOTATION, % of ear height. Large on flagged +"
      "\n held-out = the model disagrees with an annotation it never saw,"
      "\n which is evidence against the annotation. On train rows the model"
      "\n has seen the row, so small numbers there mean nothing.)")
