"""Run the silhouette correction over every GATED row, and keep the result.

This is the production pass behind qa_sam_correct.py's measurement. That
script sampled 120 gated rows, corrupted them synthetically, and showed
the normal-component correction recovers 100% of them with none harmed;
this one applies the same correct() to the REAL annotation on all 6,107
rows the gate admits, and stores the corrected outer chain so a dataset
can be built from it without touching data/manual.

Nothing here is written back into the corpus. The output is an npz of
corrected points plus the per-point normal offsets, which is what the
clash review and the dataset writer consume.

Resumable: the npz is rewritten every CHUNK rows, and a re-run skips
rows already in it. SAM 3 is the whole cost.

    "$CODE_UTILITIES/guard_run.sh" 24G 36000 runs/corrected/pass.log \
        python scripts/qa_correct_outer_pass.py runs/corrected
"""
import csv
import importlib.util as ilu
import pathlib
import sys
import time

import numpy as np
import torch
from matplotlib.path import Path as MPath
from PIL import Image
from scipy import ndimage
from transformers import Sam3Model, Sam3Processor

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_sp = ilu.spec_from_file_location("_sc", ROOT / "scripts/qa_sam_correct.py")
_sc = ilu.module_from_spec(_sp); _sp.loader.exec_module(_sc)

CHUNK = 100
OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
NPZ = OUT / "outer.npz"
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0

L = np.load(ROOT / "data/manual/landmarks.npy")
rows = list(csv.DictReader(open(ROOT / "data/manual/manifest.csv")))
pick = [int(x) for x in (ROOT / "runs/gate/proper_rows.txt").read_text().split()]

done, acc_idx, acc_pts, acc_off, acc_bnd = set(), [], [], [], []
if NPZ.exists():
    z = np.load(NPZ)
    acc_idx, acc_pts, acc_off = list(z["idx"]), list(z["pts"]), list(z["off"])
    cuts = np.cumsum(z["blen"])[:-1]
    acc_bnd = list(np.split(z["bnd"], cuts))
    done = set(int(i) for i in acc_idx)
    print(f"resuming: {len(done)} rows already corrected")
todo = [i for i in pick if i not in done]
if LIMIT:
    todo = todo[:LIMIT]
print(f"{len(todo)} rows to do")

dev = "cuda" if torch.cuda.is_available() else "cpu"
proc = Sam3Processor.from_pretrained("facebook/sam3")
sam = Sam3Model.from_pretrained("facebook/sam3", dtype=torch.float16).to(dev).eval()


def flush():
    np.savez(NPZ, idx=np.array(acc_idx), pts=np.array(acc_pts),
             off=np.array(acc_off),
             bnd=np.concatenate(acc_bnd) if acc_bnd else np.zeros((0, 2), np.float32),
             blen=np.array([len(b) for b in acc_bnd]))


t0 = time.time()
nomask = 0
for c, i in enumerate(todo):
    try:
        im = Image.open(ROOT / "data/manual" / rows[i]["image_file"]).convert("RGB")
    except Exception:
        continue
    w, h = im.size
    g = L[i] * np.array([w, h])
    if not np.isfinite(g).all():
        continue
    eh = float(g[:20, 1].max() - g[:20, 1].min())
    yy, xx = np.mgrid[0:h, 0:w]
    inann = MPath(g[:20]).contains_points(
        np.c_[xx.ravel(), yy.ravel()]).reshape(h, w)
    inp = proc(images=im, text="ear", return_tensors="pt").to(dev)
    with torch.inference_mode():
        o = sam(**inp)
    mk, best = None, -1.0
    try:
        r = proc.post_process_instance_segmentation(
            o, threshold=0.3, mask_threshold=0.5, target_sizes=[(h, w)])[0]
        for m_ in np.asarray(r["masks"].cpu()):
            m_ = np.asarray(m_) > 0
            if m_.ndim != 2 or not m_.any():
                continue
            v = (m_ & inann).sum() / max((m_ | inann).sum(), 1)
            if v > best:
                best, mk = v, m_
    except Exception:
        pass
    if mk is None:
        nomask += 1
        continue
    lab, nn = ndimage.label(mk)
    if nn == 0:
        nomask += 1
        continue
    mk = lab == (1 + np.argmax(ndimage.sum(mk, lab, range(1, nn + 1))))
    B = _sc.boundary_polyline(mk)
    if B is None or len(B) < 20:
        nomask += 1
        continue
    # cache the boundary, normalised, so every later correction experiment is
    # free: SAM is 99% of the cost of this pass and the mask is all it gives.
    if len(B) > 800:
        B = B[np.round(np.linspace(0, len(B) - 1, 800)).astype(int)]
    fixed, off = _sc.correct(g[:20], mk, eh)
    if fixed is None:
        nomask += 1
        continue
    acc_idx.append(i)
    acc_pts.append(fixed / np.array([w, h]))     # normalised, as landmarks.npy
    acc_off.append(off)
    acc_bnd.append((B / np.array([w, h])).astype(np.float32))
    if (c + 1) % CHUNK == 0:
        flush()
        el = time.time() - t0
        print(f"  {c+1}/{len(todo)}  {el/(c+1):.2f}s/row  "
              f"eta {(len(todo)-c-1)*el/(c+1)/60:.0f} min  nomask {nomask}",
              flush=True)
flush()
off = np.array(acc_off)
mv = np.abs(off)
print(f"\n{len(acc_idx)} rows corrected, {nomask} without a usable mask")
print(f"points moved (|offset| > {_sc.TOL}%): "
      f"{100*(mv > _sc.TOL).mean():.1f}% of all points")
print(f"per-row median |offset|: median {np.median(np.median(mv,1)):.2f}%  "
      f"p90 {np.percentile(np.median(mv,1),90):.2f}%")
print(f"rows where nothing moved: {100*(mv.max(1) <= _sc.TOL).mean():.1f}%")
