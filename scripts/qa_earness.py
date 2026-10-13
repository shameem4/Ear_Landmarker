"""Score every row for "is there actually an ear here" with BlazeEar.

Row 7397 is a blurred shoulder and a red-striped shirt. There is no ear in
it, and the 55 annotated points trace nothing. Nothing in the QA stack so
far can say that: the silhouette assumes an ear and segments the nearest
blob, the landmarker assumes an ear and places 55 points on whatever is
there, and both the gate and the clash review measure the SHAPE of an
annotation, not whether its subject exists.

BlazeEar can, and it is the right instrument for a reason that matters:
it was trained on Open Images and Roboflow ear boxes, NOT on this corpus's
annotations, so its opinion is genuinely independent -- unlike the
landmarker, whose residual is partly memorisation of the thing being
audited. "Is there an ear here at all" is also a GLOBAL question, which is
where a whole-image detector confidence is strong and where it was weak
when asked about occlusion, which is local.

The one real difficulty is domain. BlazeEar runs on FACE crops, where an
ear is a small object with context around it; these images are tight ear
crops where the ear fills the frame. So each crop is pasted onto a canvas
PAD times larger, edge-padded, which puts the ear back at a face-crop-like
scale. The detector runs single-stage: there is no face to find here.

    PYTHONNOUSERSITE=1 python scripts/qa_earness.py runs/earness
"""
import csv
import pathlib
import sys

import numpy as np
import torch
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from inference import EarDetector, BLAZEEAR_DIR, DETECTOR_WEIGHTS

PAD = 2.2          # canvas size as a multiple of the crop
OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0

dev = "cuda" if torch.cuda.is_available() else "cpu"
det = EarDetector(BLAZEEAR_DIR / DETECTOR_WEIGHTS, device=dev,
                  confidence=0.01,          # score everything; threshold later
                  two_stage=False)
print(f"detector: {DETECTOR_WEIGHTS} on {dev}, single-stage, pad {PAD}x")

rows = list(csv.DictReader(open(ROOT / "data/manual/manifest.csv")))
L = np.load(ROOT / "data/manual/landmarks.npy")


def padded(im):
    """Paste the crop into the middle of a larger edge-padded canvas."""
    a = np.asarray(im)
    h, w = a.shape[:2]
    ph, pw = int(h * (PAD - 1) / 2), int(w * (PAD - 1) / 2)
    return np.pad(a, ((ph, ph), (pw, pw), (0, 0)), mode="edge")


def iou(b, g):
    x0 = max(b[0], g[0]); y0 = max(b[1], g[1])
    x1 = min(b[2], g[2]); y1 = min(b[3], g[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    i = (x1 - x0) * (y1 - y0)
    return i / ((b[2]-b[0])*(b[3]-b[1]) + (g[2]-g[0])*(g[3]-g[1]) - i)


idx, conf, ov = [], [], []
todo = range(len(rows) if not LIMIT else LIMIT)
for i in todo:
    try:
        im = Image.open(ROOT / "data/manual" / rows[i]["image_file"]).convert("RGB")
    except Exception:
        continue
    w, h = im.size
    a = padded(im)
    ph, pw = (a.shape[0] - h) // 2, (a.shape[1] - w) // 2
    d = det.detect(a)
    g = L[i, :20]
    if np.isfinite(g).all():
        gb = np.array([g[:, 0].min()*w + pw, g[:, 1].min()*h + ph,
                       g[:, 0].max()*w + pw, g[:, 1].max()*h + ph])
    else:
        gb = None
    if len(d) == 0:
        idx.append(i); conf.append(0.0); ov.append(0.0)
    else:
        # detect() returns [ymin, xmin, ymax, xmax, confidence]
        c = d[:, 4]
        k = int(np.argmax(c))
        b = np.array([d[k, 1], d[k, 0], d[k, 3], d[k, 2]])
        idx.append(i); conf.append(float(c[k]))
        ov.append(float(iou(b, gb)) if gb is not None else np.nan)
    if len(idx) % 500 == 0:
        print(f"  {len(idx)}/{len(rows)}", flush=True)

idx, conf, ov = np.array(idx), np.array(conf), np.array(ov)
np.savez(OUT / "earness.npz", idx=idx, conf=conf, iou=ov)
print(f"\n{len(idx)} rows scored")
print(f"confidence: median {np.median(conf):.3f}  p10 {np.percentile(conf,10):.3f}"
      f"  p1 {np.percentile(conf,1):.3f}  zero on {(conf==0).sum()} rows")
print(f"IoU(best box, annotation bbox): median {np.nanmedian(ov):.3f}")
print(f"\n{'threshold':>10}{'rows under':>12}{'% of corpus':>13}")
for t in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7):
    print(f"{t:>10.2f}{(conf < t).sum():>12d}{100*(conf < t).mean():>12.1f}%")
o = np.argsort(conf)
(OUT / "lowest_conf.txt").write_text(" ".join(str(int(i)) for i in idx[o[:300]]))
print(f"wrote {OUT/'lowest_conf.txt'} (300 lowest)")
