"""Active Shape Model fit: the template as a prior, no VLM, no API cost.

The cutoff diagnostic showed SAM 3's mask boundary already sits on an
edge (1.18x the rest of its own boundary, 2.4x in-ear texture), so
edge-driven repair has nowhere to go. A shape prior does: it can hold a
contour where the image is locally ambiguous, which is the one thing
SAM's mask cannot do for itself.

Classic Cootes ASM over the 55-point PDM from qa_build_pdm.py:

  init   similarity transform from SAM 3's mask (centroid, extent, PCA axis)
  data   outer chain  -> nearest point on the SAM mask boundary
         interior     -> strongest sato ridge along the point's normal
  model  fit pose + shape params to those targets, clamp b to +-3 sd
  repeat

The interesting case is the lobe. The data term there comes from SAM and
is therefore short; if the prior is doing its job it should REFUSE to pull
points 14-19 in, because that shape is unlikely given where the rest of
the contour sits. Whether it does is the measurement.

    python scripts/qa_asm_fit.py runs/asm ROWS.txt
"""
import csv
import json
import pathlib
import sys

import numpy as np
import torch
from matplotlib.path import Path as MPath
from PIL import Image
from scipy import ndimage
from scipy.spatial import ConvexHull
from skimage.filters import sato
from transformers import Sam3Model, Sam3Processor

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ear3d.scheme import STRIPS, STRIP_NAMES

OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
pick = [int(x) for x in pathlib.Path(sys.argv[2]).read_text().split()]
import importlib.util as _ilu
_sp = _ilu.spec_from_file_location("_ap", ROOT / "scripts/qa_build_appearance.py")
_ap = _ilu.module_from_spec(_sp); _sp.loader.exec_module(_ap)
PROF = np.load(ROOT / "runs/pdm/profiles.npz")
PM, PIC, PK, PSTEP = PROF["mean"], PROF["icov"], int(PROF["k"]), float(PROF["step"])
P = np.load(ROOT / "runs/pdm/pdm.npz")
MEAN, MODES, VAR = P["mean"], P["modes"], P["var"]
K = 20
SD = np.sqrt(VAR[:K])

dev = "cuda" if torch.cuda.is_available() else "cpu"
proc = Sam3Processor.from_pretrained("facebook/sam3")
sam = Sam3Model.from_pretrained("facebook/sam3", dtype=torch.float16).to(dev).eval()
L = np.load(ROOT / "data/manual/landmarks.npy")
rows = list(csv.DictReader(open(ROOT / "data/manual/manifest.csv")))


def similarity(src, dst):
    """Least-squares similarity (scale, rotation, translation) src -> dst."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    M = d0.T @ s0
    U, S, Vt = np.linalg.svd(M)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1, d]) @ Vt
    sc = (S[0] + d * S[1]) / max((s0 ** 2).sum(), 1e-9)
    return sc, R, mu_d - sc * (R @ mu_s)


def normals(pts, lo, hi):
    """Outward normals along a chain segment."""
    q = pts[lo:hi]
    t = np.gradient(q, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    return np.c_[-t[:, 1], t[:, 0]]


def _axis(pts):
    """Principal axis angle of a point set."""
    q = pts - pts.mean(0)
    u, _, _ = np.linalg.svd(q.T @ q / len(q))
    return np.arctan2(u[1, 0], u[0, 0])


def _rot(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def _search(grey, cur, eh, m=8):
    """Move each point to the offset whose profile best matches its model.

    This is the appearance half the first version lacked. Instead of pulling
    the outer chain to the nearest mask-boundary pixel and the interior to
    the strongest ridge -- neither of which can say whether a point is on
    the RIGHT structure -- each candidate offset is scored by Mahalanobis
    distance to that landmark's learned profile. Returns the targets and the
    total residual, which is also what scores a pose.
    """
    nrm = _ap.chain_normals(cur)
    step = PSTEP * eh
    tgt = cur.copy()
    cost = np.zeros(55)
    for j in range(55):
        best, bd = cur[j], np.inf
        for o in range(-m, m + 1):
            q = cur[j] + nrm[j] * (o * step)
            g = _ap.profile(grey, q, nrm[j], step, PK)
            d = g - PM[j]
            md = float(d @ PIC[j] @ d)
            if md < bd:
                bd, best = md, q
        tgt[j] = best
        cost[j] = bd
    return tgt, float(np.median(cost))


def _iterate(grey, cur, eh, iters=10):
    b = np.zeros(K)
    last = np.inf
    for _ in range(iters):
        tgt, cost = _search(grey, cur, eh)
        sc, R, t0 = similarity(MEAN + np.einsum("k,kij->ij", b, MODES[:K]), tgt)
        inv = (R.T @ (tgt - t0).T).T / max(sc, 1e-9)
        b = np.clip(np.einsum("kij,ij->k", MODES[:K], inv - MEAN),
                    -3 * SD, 3 * SD)
        new = sc * (R @ (MEAN + np.einsum("k,kij->ij", b, MODES[:K])).T).T + t0
        done = np.linalg.norm(new - cur) / max(eh, 1) < 1e-3
        cur, last = new, cost
        if done:
            break
    return cur, last


def fit_one(grey, mk, eh):
    """Fit from four initialisations and keep the best by DATA TERM.

    The first version of this had two bugs that made the method look eight
    times worse than it is: the PDM is built in a single handedness and
    nothing mirrored it per row, so ~44% of ears were fitted with a flipped
    template; and the mean shape was rotated by the mask's axis angle
    without first measuring the mean's OWN axis. Both are fixed by trying
    {mirror, not} x {axis, axis+pi} and scoring each with the image, never
    with the annotation -- scoring by ground truth would make this an
    oracle and not a method.
    """
    h, w = mk.shape
    ys, xs = np.nonzero(mk)
    c = np.array([xs.mean(), ys.mean()])
    pts = np.c_[xs, ys]
    a_mask = _axis(pts)
    # Scale by matching the model's OWN outer-chain extent to the mask's,
    # measured along the mask's principal axis. Scaling by sqrt of the
    # covariance eigenvalue (the first version) put a 24px model on a 100px
    # ear -- 4x too small, from which the nearest-boundary assignment is
    # degenerate and no amount of iterating recovers.
    proj = (pts - c) @ np.array([np.cos(a_mask), np.sin(a_mask)])
    scale = float(np.ptp(proj)) / max(np.ptp(MEAN[:20, 1]), 1e-6)
    bnd = mk ^ ndimage.binary_erosion(mk)
    by, bx = np.nonzero(bnd)
    B = np.c_[bx, by]
    if len(B) < 20:
        return None

    best_out, best_score = None, np.inf
    for mirror in (False, True):
        M0 = MEAN.copy()
        if mirror:
            M0[:, 0] = -M0[:, 0]
        a0 = _axis(M0)
        for extra in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            R0 = _rot(a_mask - a0 + extra)
            cur = scale * (R0 @ M0.T).T + c
            out, cost = _iterate(grey, cur, eh)
            # the pose is chosen by the SAME learned profile cost that drives
            # the search, which is the thing the shape-only version could not
            # do: a contour collapsed onto the mask boundary scored well
            # whichever way round it was.
            if cost < best_score:
                best_score, best_out = cost, out
    return best_out


res = []
for c_, i in enumerate(pick):
    im = Image.open(ROOT / "data/manual" / rows[i]["image_file"]).convert("RGB")
    w, h = im.size
    g = L[i]
    xy = g[:20] * np.array([w, h])
    if not np.isfinite(g).all():
        continue
    eh = float(xy[:, 1].max() - xy[:, 1].min())
    if eh < 32:
        continue
    yy, xx = np.mgrid[0:h, 0:w]
    try:
        hull = xy[ConvexHull(xy).vertices]
    except Exception:
        continue
    inhull = MPath(hull).contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(h, w)
    inp = proc(images=im, text="ear", return_tensors="pt").to(dev)
    with torch.inference_mode():
        o = sam(**inp)
    try:
        r = proc.post_process_instance_segmentation(
            o, threshold=0.3, mask_threshold=0.5, target_sizes=[(h, w)])[0]
        masks = np.asarray(r["masks"].cpu())
    except Exception:
        continue
    best, mk = -1.0, None
    for m_ in masks:
        m_ = np.asarray(m_) > 0
        if m_.ndim != 2 or not m_.any():
            continue
        v = (m_ & inhull).sum() / max((m_ | inhull).sum(), 1)
        if v > best:
            best, mk = v, m_
    if mk is None:
        continue
    lab, n = ndimage.label(mk)
    sz = ndimage.sum(mk, lab, range(1, n + 1))
    mk = lab == (1 + np.argmax(sz))

    grey = np.asarray(im.convert("L"), float) / 255.0

    out = fit_one(grey, mk, eh)
    if out is None:
        continue
    gt = g * np.array([w, h])
    d = {nm: float(np.median(np.linalg.norm(out[lo:hi] - gt[lo:hi], axis=1))
                   / eh * 100)
         for (lo, hi), nm in zip(STRIPS, STRIP_NAMES)}
    d["idx"] = i
    d["pts"] = (out / np.array([w, h])).tolist()   # normalised, for rendering
    res.append(d)
    if (c_ + 1) % 25 == 0:
        print(f"  {c_+1}/{len(pick)}", flush=True)

json.dump(res, open(OUT / "asm.json", "w"))
print(f"\n{len(res)} rows fitted\n{'strip':<16}{'median':>9}{'p90':>9}")
for nm in STRIP_NAMES:
    v = np.array([r[nm] for r in res])
    print(f"{nm:<16}{np.median(v):>9.1f}{np.percentile(v,90):>9.1f}")
print("\n(distance from the annotation, % of ear height."
      "\n Gemini shape fit, same metric, clean rows: 6.1 / 5.1 / 4.9 / 10.9)")
