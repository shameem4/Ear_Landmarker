"""Learn Cootes profile models -- the appearance half the ASM was missing.

The shape-only ASM reached 26.9% on the outer chain against Gemini's 6.1%,
and the renders showed why: every catastrophic fit was a POSE failure, and
sweeping rotations did not fix it because the data term could not SCORE a
pose. A shape collapsed onto the mask boundary looks good whichever way
round it is. Telling poses apart needs to know what the image looks like
at each landmark, which is the "A" in AAM and the piece this adds.

Classic Cootes local appearance: at each landmark, sample a 1-D profile
along the shape normal, take its derivative and normalise by the sum of
absolute values (which removes brightness and contrast), then store the
mean profile and the inverse covariance per landmark. At fit time a
candidate offset is scored by Mahalanobis distance to that model.

Sampling step is a fraction of EAR HEIGHT, not pixels, so the model is
scale-invariant; normals come from the same function the fitter uses, so
training and search agree.

Mirrored to one handedness -- image as well as shape, since a profile is
an image measurement and flipping the points alone would learn the mirror
of the texture.

    python scripts/qa_build_appearance.py runs/pdm 2500
"""
import csv
import pathlib
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scheme import STRIPS

# argv is parsed in main(), not at import: the ASM fitter imports this module
# for chain_normals() and profile(), and must not inherit its command line.
K = 5                    # profile half-length -> 2K+1 samples
STEP = 0.015             # sample spacing, in units of ear height


def chain_normals(pts):
    """Unit normals for all 55 points, taken strip by strip."""
    n = np.zeros_like(pts)
    for lo, hi in STRIPS:
        q = pts[lo:hi]
        if hi - lo < 2:
            n[lo:hi] = np.array([1.0, 0.0])
            continue
        t = np.gradient(q, axis=0)
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
        n[lo:hi] = np.c_[-t[:, 1], t[:, 0]]
    return n


def profile(grey, pt, nrm, step, k):
    """Normalised derivative profile of length 2k+1 centred on pt."""
    t = np.arange(-k, k + 2) * step          # 2k+2 samples -> 2k+1 derivatives
    q = pt[None, :] + nrm[None, :] * t[:, None]
    v = ndimage.map_coordinates(grey, [q[:, 1], q[:, 0]], order=1, mode="nearest")
    d = np.diff(v)                                  # derivative, 2k+1 long
    s = np.abs(d).sum()
    return d / s if s > 1e-6 else d


def main():
    OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
    NMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 2500
    L = np.load(ROOT / "data/manual/landmarks.npy")
    rows = list(csv.DictReader(open(ROOT / "data/manual/manifest.csv")))
    bad = {int(x) for x in (ROOT / "runs/sam3_full/bad_rows.txt").read_text().split()}
    ok = np.isfinite(L).all((1, 2))
    ok &= ~np.isin(np.arange(len(L)), list(bad))
    idx = np.flatnonzero(ok)
    rng = np.random.default_rng(0)
    idx = rng.choice(idx, min(NMAX, len(idx)), replace=False)
    print(f"{len(idx)} clean rows for the profile model")

    acc = [[] for _ in range(55)]
    used = 0
    for c, i in enumerate(idx):
        try:
            im = Image.open(ROOT / "data/manual" / rows[i]["image_file"]).convert("L")
        except Exception:
            continue
        g = np.asarray(im, float) / 255.0
        h, w = g.shape
        pts = L[i] * np.array([w, h])
        eh = float(pts[:20, 1].max() - pts[:20, 1].min())
        if eh < 40:                     # too small for a meaningful profile
            continue
        x, y = pts[:20, 0], pts[:20, 1]
        area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
        if area > 0:                    # mirror image AND points together
            g = g[:, ::-1].copy()
            pts = pts.copy()
            pts[:, 0] = (w - 1) - pts[:, 0]
        nrm = chain_normals(pts)
        step = STEP * eh
        for j in range(55):
            acc[j].append(profile(g, pts[j], nrm[j], step, K))
        used += 1
        if (c + 1) % 400 == 0:
            print(f"  {c+1}/{len(idx)}  used {used}", flush=True)

    mean = np.zeros((55, 2 * K + 1))
    icov = np.zeros((55, 2 * K + 1, 2 * K + 1))
    for j in range(55):
        A = np.array(acc[j])
        mean[j] = A.mean(0)
        C = np.cov(A.T)
        C += np.eye(len(C)) * (1e-3 * np.trace(C) / len(C) + 1e-9)
        icov[j] = np.linalg.inv(C)
    np.savez(OUT / "profiles.npz", mean=mean, icov=icov, k=K, step=STEP, n=used)
    print(f"\nwrote {OUT/'profiles.npz'} from {used} images")
    # how distinctive is each landmark? mean Mahalanobis of a random other
    # landmark's profile against this one -- higher means more discriminative
    d = []
    for j in range(55):
        A = np.array(acc[j])[:200]
        o = np.array(acc[(j + 7) % 55])[:200]
        da = np.einsum("ni,ij,nj->n", A - mean[j], icov[j], A - mean[j]).mean()
        do = np.einsum("ni,ij,nj->n", o - mean[j], icov[j], o - mean[j]).mean()
        d.append(do / max(da, 1e-9))
    d = np.array(d)
    print(f"separability (other-landmark / own Mahalanobis): "
          f"median {np.median(d):.2f}, worst {d.min():.2f} at point {d.argmin()}")
    print("(>1 means the profile model can tell this landmark from another)")


if __name__ == "__main__":
    main()
