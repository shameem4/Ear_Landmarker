"""Build a point distribution model of the ear from the clean annotations.

The cutoff diagnostic settles which repairs are possible: SAM 3's mask
boundary near the lobe sits on a gradient 1.18x the rest of its own
boundary and 2.4x the texture inside the ear, so it is already ON an
edge. Every edge-driven method -- GrabCut, snakes, region growing,
matting -- converges to the edge it can see, which is the one SAM already
found. Only a SHAPE PRIOR can carry a contour through evidence that is
locally ambiguous.

That is an Active Shape Model (Cootes), and the corpus is big enough to
build a real one rather than the single median outline the VLM was seeded
with: 10,003 clean 55-point shapes, Procrustes-aligned, PCA.

The median template already in runs/shape_template is mode 0 of this, so
this is the same prior with its variability attached -- which is what
lets a fit deform instead of only translating and scaling.

    python scripts/qa_build_pdm.py runs/pdm
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)

L = np.load(ROOT / "data/manual/landmarks.npy")
ok = np.isfinite(L).all((1, 2))
bad = {int(x) for x in (ROOT / "runs/sam3_full/bad_rows.txt").read_text().split()}
ok &= ~np.isin(np.arange(len(L)), list(bad))
X = L[ok].copy()
print(f"{len(X)} clean shapes")

# one handedness, as in the template build
x, y = X[:, :20, 0], X[:, :20, 1]
A = 0.5 * np.sum(x * np.roll(y, -1, 1) - np.roll(x, -1, 1) * y, axis=1)
flip = A > 0 if (A > 0).sum() < (A < 0).sum() else A < 0
X[flip, :, 0] = -X[flip, :, 0]
print(f"mirrored {flip.sum()}")


def procrustes(S, iters=5):
    """Generalised Procrustes: centre, scale, then iterate rotation to mean."""
    S = S - S.mean(1, keepdims=True)
    S = S / np.linalg.norm(S, axis=(1, 2), keepdims=True)
    m = S[0]
    for _ in range(iters):
        # optimal rotation of each shape onto the current mean
        M = np.einsum("nij,ik->njk", S, m)          # (n,2,2)
        U, _, Vt = np.linalg.svd(M)
        d = np.sign(np.linalg.det(np.einsum("nij,njk->nik", U, Vt)))
        D = np.zeros_like(U)
        D[:, 0, 0] = 1; D[:, 1, 1] = d
        R = np.einsum("nij,njk,nkl->nil", U, D, Vt)
        S = np.einsum("nij,njk->nik", S, R)
        m = S.mean(0)
        m /= np.linalg.norm(m)
    return S, m


S, mean = procrustes(X)
print(f"aligned; mean shape norm {np.linalg.norm(mean):.3f}")

F = (S - mean).reshape(len(S), -1)
C = F.T @ F / (len(F) - 1)
w, V = np.linalg.eigh(C)
order = np.argsort(w)[::-1]
w, V = w[order], V[:, order]
w = np.maximum(w, 0)
cum = np.cumsum(w) / w.sum()
print(f"\n{'modes':>7}{'variance explained':>21}")
for k in (1, 2, 3, 5, 10, 15, 20, 30, 40):
    print(f"{k:>7}{cum[k-1]*100:>20.1f}%")
K = int(np.searchsorted(cum, 0.95) + 1)
print(f"\n{K} modes reach 95% of shape variance")

np.savez(OUT / "pdm.npz", mean=mean, modes=V[:, :40].T.reshape(40, 55, 2),
         var=w[:40], n=len(S))
print(f"wrote {OUT/'pdm.npz'} (mean + 40 modes)")
