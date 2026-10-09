#!/usr/bin/env python3
"""Where on the ear do the 55 points fail to pin the surface down, and which
new points would fix it most?

This is optimal sensor placement, not anatomy. The 55-point scheme was designed
for FRONTAL PHOTOGRAPHS, where only contours are visible, so every point sits
on a ridge or a border and nothing samples a surface. Now that the project has
a 476-ear shape model with dense correspondence, the question "which features
should the scheme gain" has a measurable answer: the places where knowing all
55 points still leaves the surface uncertain.

THE MODEL. With shape coefficients b and the linear basis V = mean + C b, the
55 landmarks are L = A b (A = C sampled at the 55 barycentric sites). Observing
L with isotropic noise sigma gives a Gaussian posterior on b

    Sigma = (A^T A / sigma^2 + diag(1/var))^-1

and vertex v's reconstruction uncertainty is trace(C_v Sigma C_v^T): the
expected squared error at that vertex given perfect knowledge of all 55 points
to within sigma. High values mark surface the scheme cannot see.

WHY A PRIOR, AND WHY IT MATTERS. 55 points carry 165 numbers against 80
coefficients, so noiseless least squares would be exactly determined and this
map would be identically zero -- the question only has content under noise, and
the prior is what keeps the badly-conditioned directions finite rather than
exploding. sigma is quoted as a fraction of ear size so it can be set from the
model's measured error (NME 0.0106 on renders) rather than invented.

GREEDY SELECTION is then straightforward: adding a landmark at vertex v appends
three rows to A, so the posterior update is a rank-3 downdate and the candidate
that removes the most total uncertainty can be scored exactly. Greedy is not
optimal for subset selection, but the objective is submodular-ish here and the
alternative is combinatorial.

WHAT THIS DOES NOT ANSWER: whether a human can SEE the chosen point on a
photograph. A point in the middle of the concha floor is informative and
unannotatable. Treat the output as a shortlist to judge by eye, not a scheme.

    python scripts/where_to_add_landmarks.py --add 15
"""
import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TPL = pathlib.Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/wrap_work/templates")


def sampling_matrix(tri, bary, faces, n_vert):
    """(55, n_vert): each landmark as a barycentric mix of three vertices."""
    S = np.zeros((len(tri), n_vert))
    for i, (t, w) in enumerate(zip(tri, bary)):
        S[i, faces[int(t)]] = w
    return S


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pca", default=str(TPL / "ear_pca_v4_all.npz"))
    p.add_argument("--marks", default=str(TPL / "mean_v4_marks.npz"))
    p.add_argument("--sigma", type=float, default=0.0106,
                   help="landmark noise as a fraction of ear size; the default "
                        "is head 1's measured NME on the renders")
    p.add_argument("--add", type=int, default=15)
    p.add_argument("--out", default="runs/landmark_gaps.npz")
    a = p.parse_args()

    z = np.load(a.pca, allow_pickle=True)
    mean, faces = z["mean"], z["faces"]
    nv = len(mean)
    C = z["components"].reshape(-1, nv, 3)            # (modes, nv, 3)
    var = z["variance"]
    # Ear size comes from the TEMPLATE, not from the stored `ear_size` array:
    # that array is in each dataset's native units (SONICOM mm, HUTUBS metres)
    # and its mean is meaningless. The PCA is a Procrustes shape space, so the
    # mean's own long-axis extent is the only scale that matches `components`.
    ear = float(np.ptp(mean, axis=0).max())
    sigma = a.sigma * ear
    m = np.load(a.marks, allow_pickle=True)
    S = sampling_matrix(m["triangle"], m["barycentric"], faces, nv)
    print(f"template {nv} verts, {len(C)} modes, shape-space ear size {ear:.4f}")
    print(f"landmark noise sigma = {a.sigma} x ear = {sigma:.5f}\n")

    Cf = C.transpose(1, 2, 0)                         # (nv, 3, modes)
    prior = np.diag(var)

    def info(rows):
        """rows (k, nv) -> Fisher information contribution A^T A / sigma^2."""
        A = np.einsum("kv,vam->kam", rows, Cf).reshape(-1, len(C))
        return A.T @ A / sigma ** 2

    def unc(Sig):
        """trace(C_v Sig C_v^T) per vertex."""
        return np.einsum("vam,mn,van->v", Cf, Sig, Cf)

    P0 = np.diag(1.0 / var)
    P = info(S) + P0
    Sig = np.linalg.inv(P)
    u = unc(Sig)
    up = unc(prior)
    print(f"knowing nothing:      rms {np.sqrt(up.mean()):.5f}  "
          f"({100 * np.sqrt(up.mean()) / ear:.2f}% of ear)")
    print(f"with the 55 points:   rms {np.sqrt(u.mean()):.5f}  "
          f"({100 * np.sqrt(u.mean()) / ear:.2f}% of ear)  "
          f"-- removes {100 * (1 - np.sqrt(u.mean() / up.mean())):.1f}%")
    print(f"  worst vertex {100 * np.sqrt(u.max()) / ear:.2f}% of ear\n")

    # Greedy. Adding a landmark AT vertex v appends the three rows C[:, v, :],
    # so the information matrix takes a rank-3 update and nothing is rebuilt.
    chosen, cur = [], u.sum()
    for step in range(a.add):
        best, best_val, best_S = None, cur, None
        for v in range(nv):
            Cv = Cf[v]                                # (3, modes)
            Sg = np.linalg.inv(P + Cv.T @ Cv / sigma ** 2)
            val = unc(Sg).sum()
            if val < best_val:
                best, best_val, best_S = v, val, Sg
        if best is None:
            break
        Cv = Cf[best]
        P = P + Cv.T @ Cv / sigma ** 2
        chosen.append(best)
        print(f"  +{step + 1:2d}  vertex {best:5d}  total uncertainty "
              f"{cur:.4f} -> {best_val:.4f}  ({100 * (1 - best_val / cur):.1f}%)")
        cur = best_val

    Sig2 = np.linalg.inv(P)
    u2 = unc(Sig2)
    print(f"\nafter {len(chosen)} added points: rms {np.sqrt(u2.mean()):.5f} "
          f"({100 * np.sqrt(u2.mean()) / ear:.2f}% of ear), "
          f"down {100 * (1 - np.sqrt(u2.mean() / u.mean())):.1f}%")
    np.savez(a.out, uncertainty=u, after=u2, chosen=np.array(chosen),
             mean=mean, faces=faces, marks_xyz=m["xyz"], ear=ear)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
