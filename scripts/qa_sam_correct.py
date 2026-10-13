"""Correct the outer chain from SAM 3's silhouette, on gated ears only.

Why this is allowed here when "flaggers, not correctors" forbade it
everywhere else: that rule came from instruments with a ~5-6% of ear
height noise floor, which is coarser than the annotation being corrected.
Inside the gate the silhouette resolves to 0.93% median, p99 3.07%, which
is in the landmarker's league. The rule was never about the word
"instrument"; it was about the floor.

WHAT A SILHOUETTE CAN AND CANNOT FIX. It is a CURVE, not a set of
correspondences. It knows where the rim runs; it does not know which of
the 20 outer points belongs where along it. So the error is split into

  normal      distance from the point to the rim -- the silhouette sees
              this, and it is what a misplaced contour gets wrong
  tangential  position ALONG the rim, i.e. spacing -- the silhouette is
              blind to it, and it is 72% of squared error (see README)

and only the normal component is applied. Snapping each point to its
nearest boundary pixel would silently rewrite the tangential component
with whatever the projection happened to pick, collapsing neighbours at
high curvature and reordering them around the lobe.

Three guards, each for a failure the naive version has:
  - correspondence is forced MONOTONE in boundary arc length, so the
    corrected chain cannot cross itself or double back
  - the normal offsets are smoothed along j: real contour error is
    spatially smooth, an isolated spike is a projection artefact
  - a point is moved only where the offset exceeds TOL, so the 0.93%
    agreement of an already-good row is left alone instead of being
    jittered toward the mask's pixel staircase

    python scripts/qa_sam_correct.py runs/correct --mode {noharm,recover,fix}
"""
import argparse
import csv
import json
import pathlib
import sys

import numpy as np
import torch
from matplotlib.path import Path as MPath
from PIL import Image
from scipy import ndimage
from transformers import Sam3Model, Sam3Processor

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TOL = 2.0          # % of ear height; below this the point is left alone
SMOOTH = 1.0       # gaussian sigma along the chain, in points


def boundary_polyline(mk):
    """Ordered closed polygon of the mask boundary, via marching squares."""
    from skimage import measure
    cs = measure.find_contours(mk.astype(float), 0.5)
    if not cs:
        return None
    c = max(cs, key=len)
    return np.c_[c[:, 1], c[:, 0]]          # (x, y)


def correct(gt20, mk, eh, tol=TOL):
    """Return the corrected outer chain, and the applied normal offsets."""
    B = boundary_polyline(mk)
    if B is None or len(B) < 20:
        return None, None
    # nearest boundary vertex for each GT point, and its arc-length index
    d2 = ((gt20[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    j = np.argmin(d2, axis=1)
    # force the correspondence monotone around the closed curve: unwrap the
    # index sequence, then isotonic-regress it so it never goes backwards
    n = len(B)
    k = np.unwrap(j / n * 2 * np.pi) / (2 * np.pi) * n
    from sklearn.isotonic import IsotonicRegression
    order = 1 if k[-1] >= k[0] else -1
    k = IsotonicRegression(increasing=(order == 1)).fit_transform(
        np.arange(len(k)), k)
    tgt = B[np.clip(np.round(k).astype(int) % n, 0, n - 1)]

    # split the correction into normal and tangential, keep only normal
    t = np.gradient(gt20, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    nrm = np.c_[-t[:, 1], t[:, 0]]
    delta = tgt - gt20
    off = (delta * nrm).sum(1)                       # signed, pixels
    off = ndimage.gaussian_filter1d(off, SMOOTH, mode="nearest")
    off_pct = off / eh * 100
    apply = np.abs(off_pct) > tol
    out = gt20 + nrm * np.where(apply, off, 0.0)[:, None]
    return out, off_pct


def tps_warp(src, dst, pts, lam=1e-3):
    """Thin-plate spline taking src->dst, evaluated at pts.

    Used to PROPAGATE an outer-chain correction inward instead of
    predicting the interior. Prediction was measured and rejected: a PDM
    conditioned on points 0-19 reconstructs 20-54 to 3.22%, no better than
    a similarity-fitted mean shape (3.14%) and three times worse than the
    annotation itself, so the outer chain carries no interior shape
    information. Propagation keeps the interior annotation and only moves
    it the way its neighbourhood moved, which discards nothing.

    lam regularises, so a noisy outer correction does not produce a wild
    interior displacement.
    """
    def U(r2):
        return np.where(r2 > 1e-12, r2 * np.log(np.maximum(r2, 1e-12)), 0.0)
    n = len(src)
    d2 = ((src[:, None, :] - src[None, :, :]) ** 2).sum(-1)
    K = U(d2) + lam * np.eye(n)
    P = np.c_[np.ones(n), src]
    A = np.zeros((n + 3, n + 3))
    A[:n, :n] = K; A[:n, n:] = P; A[n:, :n] = P.T
    out = np.zeros((len(pts), 2))
    for c in range(2):
        rhs = np.r_[dst[:, c], np.zeros(3)]
        try:
            w = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            return pts.copy()
        d2p = ((pts[:, None, :] - src[None, :, :]) ** 2).sum(-1)
        out[:, c] = U(d2p) @ w[:n] + np.c_[np.ones(len(pts)), pts] @ w[n:]
    return out


def fit_similarity_warp(src, dst):
    """The 4-DOF transform that best explains src->dst. Applied to anything.

    This is the conservative alternative to the TPS: a thin-plate spline
    has about as many degrees of freedom as there are control points, so it
    fits the NOISE in the per-point silhouette corrections and carries that
    noise inward. A similarity has four, estimated from all 20 points at
    once, so the noise averages down. When the defect is a contour that
    inflated or shifted as a whole -- which is the direction every real
    defect took -- this is the right model of it.
    """
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(d0.T @ s0)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1, d]) @ Vt
    sc = (S[0] + d * S[1]) / max((s0 ** 2).sum(), 1e-9)
    return lambda pts: sc * (R @ pts.T).T + (mu_d - sc * (R @ mu_s))


def fit_affine_warp(src, dst):
    """6-DOF least squares: allows shear and anisotropic scale."""
    A = np.c_[src, np.ones(len(src))]
    M, *_ = np.linalg.lstsq(A, dst, rcond=None)
    return lambda pts: np.c_[pts, np.ones(len(pts))] @ M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--mode", choices=["noharm", "recover", "inner"],
                    default="noharm")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--warp", choices=["tps", "similarity", "affine"],
                    default="tps")
    ap.add_argument("--corrupt", choices=["global", "local"],
                    default="global")
    ap.add_argument("--tol", type=float, default=TOL)
    a = ap.parse_args()
    OUT = pathlib.Path(a.out); OUT.mkdir(parents=True, exist_ok=True)

    L = np.load(ROOT / "data/manual/landmarks.npy")
    rows = list(csv.DictReader(open(ROOT / "data/manual/manifest.csv")))
    pick = [int(x) for x in (ROOT / "runs/gate/proper_rows.txt").read_text().split()]
    rng = np.random.default_rng(0)
    pick = rng.choice(pick, min(a.n, len(pick)), replace=False).tolist()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = Sam3Processor.from_pretrained("facebook/sam3")
    sam = Sam3Model.from_pretrained("facebook/sam3",
                                    dtype=torch.float16).to(dev).eval()
    before, after, moved, resid = [], [], [], []
    for c, i in enumerate(pick):
        im = Image.open(ROOT / "data/manual" / rows[i]["image_file"]).convert("RGB")
        w, h = im.size
        g = L[i] * np.array([w, h])
        if not np.isfinite(g).all():
            continue
        eh = float(g[:20, 1].max() - g[:20, 1].min())
        truth = g[:20].copy()
        truth55 = g.copy()
        start = truth.copy()
        start55 = truth55.copy()
        if a.mode in ("recover", "inner"):
            # the failure this is meant to undo: the contour inflated off
            # the ear, which is the direction every real defect took
            ctr = truth.mean(0)
            start = ctr + (truth - ctr) * 1.12
            # the whole annotation inflates together, which is how the real
            # defects look: the interior drifts with the contour
            start55 = ctr + (truth55 - ctr) * 1.12
            if a.corrupt == "local":
                # a defect that is NOT a similarity: one stretch of the
                # contour wanders off while the rest stays put. A global
                # transform cannot represent this, which is the point of
                # testing it.
                start = truth.copy(); start55 = truth55.copy()
                seg = slice(4, 12)
                nrm0 = truth[seg] - truth.mean(0)
                nrm0 /= np.maximum(np.linalg.norm(nrm0, axis=1, keepdims=True), 1e-9)
                start[seg] = truth[seg] + nrm0 * 0.13 * eh
                start55[seg] = start[seg]
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
            continue
        lab, nn = ndimage.label(mk)
        if nn == 0:
            continue
        mk = lab == (1 + np.argmax(ndimage.sum(mk, lab, range(1, nn + 1))))
        fixed, off = correct(start, mk, eh, a.tol)
        if fixed is None:
            continue
        if a.mode == "inner":
            # score the INTERIOR only: 20-54, carried by the warp the outer
            # correction defines. The interior is never replaced.
            # Is this correction actually a TRANSFORMATION? Fit a
            # similarity to it and look at what is left over. A contour
            # that inflated as a whole is explained by 4 parameters and
            # leaves little; a contour with one wandering stretch is not a
            # transformation at all, and propagating it damages a correct
            # interior (measured: 100% of rows made worse).
            fs = fit_similarity_warp(start, fixed)
            resid.append(float(np.sqrt(((fixed - fs(start)) ** 2)
                                       .sum(1).mean())) / eh * 100)
            if a.warp == "tps":
                warped = tps_warp(start, fixed, start55[20:])
            else:
                f = (fit_similarity_warp if a.warp == "similarity"
                     else fit_affine_warp)(start, fixed)
                warped = f(start55[20:])
            b = np.median(np.linalg.norm(start55[20:] - truth55[20:],
                                         axis=1)) / eh * 100
            af = np.median(np.linalg.norm(warped - truth55[20:],
                                          axis=1)) / eh * 100
            before.append(b); after.append(af)
            moved.append(np.median(np.linalg.norm(
                warped - start55[20:], axis=1)) / eh * 100)
            if (c + 1) % 40 == 0:
                print(f"  {c+1}/{len(pick)}", flush=True)
            continue
        b = np.median(np.linalg.norm(start - truth, axis=1)) / eh * 100
        af = np.median(np.linalg.norm(fixed - truth, axis=1)) / eh * 100
        before.append(b); after.append(af)
        moved.append(np.median(np.linalg.norm(fixed - start, axis=1)) / eh * 100)
        if (c + 1) % 40 == 0:
            print(f"  {c+1}/{len(pick)}", flush=True)

    b, af, mv = map(np.array, (before, after, moved))
    print(f"\nmode={a.mode}  n={len(b)}  tol={a.tol}%")
    print(f"  error BEFORE correction   median {np.median(b):6.2f}%")
    print(f"  error AFTER  correction   median {np.median(af):6.2f}%")
    print(f"  the correction moved      median {np.median(mv):6.2f}%")
    print(f"  rows made WORSE           {100*(af > b + 0.1).mean():5.0f}%")
    print(f"  rows made better          {100*(af < b - 0.1).mean():5.0f}%")
    if resid:
        r = np.array(resid)
        print(f"  similarity-fit residual   median {np.median(r):5.2f}%"
              f"  p90 {np.percentile(r,90):5.2f}%")
    json.dump(dict(before=b.tolist(), after=af.tolist(), moved=mv.tolist()),
              open(OUT / f"correct_{a.mode}.json", "w"))


if __name__ == "__main__":
    main()
