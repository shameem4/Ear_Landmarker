"""Ear silhouette from the locally-trained U-Net -- the LICENCE-CLEAN
fallback, not the best segmenter. Prefer qa_silhouette_sam3.py.

It wins every gate statistic and loses on the renders, which is worth
understanding before trusting any of those numbers again. On the same 40
rows it was dropped by the gates on 5.0% of images against SAM 3's 12.5%
and SAM 2.1's 30.8%, with the best interior/outer separation of the three
at 3.62x. Drawn against the ears it is blobby: it swallows an eye and a
rectangular slab of background on one row, puts a rectangle under the ear
on another, and is lumpy and off the rim on two more. It emits ear-ISH,
roughly ear-SIZED regions, and that is exactly what a gate on area,
coverage and hull overlap rewards. Gate pass rate measures "produced a
plausible blob", NOT "found the rim".

Its value is provenance: torchvision (BSD) on CC BY 4.0 data, trained
locally, nothing sent anywhere, nothing gated. With more data, or masks
distilled from SAM 3, it could become the shipped option. Today it is the
fallback.

Original note follows.

Same output schema as qa_silhouette_sam.py and qa_silhouette_florence.py, so
qa_find_bad_contours.py reads any of the three unchanged. Weights come from
scripts/train_ear_segmenter.py: torchvision code (BSD) on the CC BY 4.0
Roboflow set, trained locally, so nothing is sent anywhere and no licence is
stretched. CC BY requires crediting SaifulULL wherever this is published.

Unlike SAM this model only knows "ear", so it needs no box and no point --
but it is run on the SAME padded landmark crop SAM gets, so the two see
identical pixels and their masks can be compared directly rather than
confounded with framing.

    python scripts/qa_silhouette_unet.py /tmp/unet_screen 1500 /tmp/idx.txt
"""
import csv
import json
import pathlib
import sys

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.spatial import ConvexHull
from matplotlib.path import Path as MPath

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from scheme import GROUPS, GROUP_NAMES          # noqa: E402
from train_ear_segmenter import UNetR18, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402

OUT = pathlib.Path(sys.argv[1])
N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
IDX_FILE = sys.argv[3] if len(sys.argv) > 3 else None
CKPT = sys.argv[4] if len(sys.argv) > 4 else "runs/earseg_v1/best.pt"
OUT.mkdir(parents=True, exist_ok=True)

G = {n: list(range(lo, hi)) for n, (lo, hi) in zip(GROUP_NAMES, GROUPS)}
OUTER = G["ascending_helix"] + G["descending_helix"] + G["helix"] + G["ear_lobe"]
INNER = (G["ascending_inner_helix"] + G["descending_inner_helix"]
         + G["inner_helix"] + G["tragus"] + G["canal"] + G["concha"])

dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(ROOT / CKPT, map_location=dev)
SZ = ck.get("size", 256)
model = UNetR18(pretrained=False).to(dev).eval()
model.load_state_dict(ck["model"])
print(f"loaded {CKPT}  val IoU {ck.get('val_iou', float('nan')):.4f}  "
      f"pretrained-encoder={ck.get('pretrained')}")

L = np.load(ROOT / "data/manual/landmarks.npy")
rows = list(csv.DictReader(open(ROOT / "data/manual/manifest.csv")))
if IDX_FILE:
    pick = [int(x) for x in pathlib.Path(IDX_FILE).read_text().split()][:N]
    print(f"auditing {len(pick)} listed rows from {IDX_FILE}")
else:
    pick = np.random.default_rng(0).choice(len(rows), min(N, len(rows)), replace=False)

res, dropped = [], []
for c, i in enumerate(pick):
    i = int(i)
    try:
        im = Image.open(ROOT / "data/manual" / rows[i]["image_file"]).convert("RGB")
    except Exception:
        continue
    w, h = im.size
    xy = L[i] * np.array([w, h])
    if not np.isfinite(xy).all():
        continue
    lo, hi = xy.min(0), xy.max(0)
    eh = float(hi[1] - lo[1])
    if eh < 8:
        continue
    pad = 0.15 * (hi - lo)
    x0, y0 = int(max(0, lo[0] - pad[0])), int(max(0, lo[1] - pad[1]))
    x1, y1 = int(min(w, hi[0] + pad[0])), int(min(h, hi[1] + pad[1]))
    if x1 - x0 < 8 or y1 - y0 < 8:
        continue
    crop = im.crop((x0, y0, x1, y1)).resize((SZ, SZ), Image.BILINEAR)
    a = (np.asarray(crop, np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    t = torch.from_numpy(a.transpose(2, 0, 1))[None].to(dev)
    with torch.inference_mode():
        pr = torch.sigmoid(model(t))[0, 0].float().cpu().numpy()
    small = (pr > 0.5)
    if not small.any():
        dropped.append(i); continue
    mk = np.zeros((h, w), bool)
    up = np.asarray(Image.fromarray(small.astype(np.uint8) * 255)
                    .resize((x1 - x0, y1 - y0), Image.NEAREST)) > 127
    mk[y0:y1, x0:x1] = up
    lab, n = ndimage.label(mk)
    if n == 0:
        dropped.append(i); continue
    sizes = ndimage.sum(mk, lab, range(1, n + 1))
    if sizes.max() / max(sizes.sum(), 1.0) < 0.80:
        dropped.append(i); continue
    mk = lab == (1 + np.argmax(sizes))
    frac = float(mk.mean())
    if frac < 0.02 or frac > 0.95:
        dropped.append(i); continue
    yy, xx = np.mgrid[0:h, 0:w]
    try:
        hull = xy[ConvexHull(xy).vertices]
    except Exception:
        continue
    inhull = MPath(hull).contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(h, w)
    covered = float((mk & inhull).sum() / max(inhull.sum(), 1))
    bloat = float(mk.sum()) / max(float(inhull.sum()), 1.0)
    if covered < 0.60 or bloat > 2.0:
        dropped.append(i); continue
    sd = ndimage.distance_transform_edt(~mk) - ndimage.distance_transform_edt(mk)
    px = np.clip(np.round(xy[:, 0]).astype(int), 0, w - 1)
    py = np.clip(np.round(xy[:, 1]).astype(int), 0, h - 1)
    dd = sd[py, px] / eh * 100.0
    res.append(dict(idx=i, iou=float("nan"), frac=frac, covered=covered,
                    bloat=bloat,
                    outer_abs=float(np.abs(dd[OUTER]).mean()),
                    inner_abs=float(np.abs(dd[INNER]).mean()),
                    outer_in=float((dd[OUTER] < 0).mean()),
                    per=dd.tolist()))
    if (c + 1) % 200 == 0:
        print(f"  {c+1}/{len(pick)}", flush=True)

json.dump(res, open(OUT / "sam2_silhouette.json", "w"))
print(f"\nU-Net failures dropped by the gates: {len(dropped)} "
      f"({len(dropped)/max(len(dropped)+len(res),1)*100:.1f}%)")
if res:
    D = np.array([r["per"] for r in res])
    o = np.abs(D[:, OUTER]).mean(); n_ = np.abs(D[:, INNER]).mean()
    print(f"n={len(res)} images   mask area {np.mean([r['frac'] for r in res])*100:.1f}% of crop")
    print(f"OUTER |dist| {o:.2f}%   INNER |dist| {n_:.2f}%   "
          f"interior/outer ratio {n_/max(o,1e-9):.2f}x")
print("wrote", OUT / "sam2_silhouette.json")
