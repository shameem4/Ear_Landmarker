"""Review the inner helix against the corrected outer rim: too close OR too far.

The outer correction moves points 0-19 and leaves 20-54 alone, which is
the conservative choice but not a free one: the two chains are no longer
guaranteed to be consistent with each other. Two things can go wrong, and
both qualify for review --

  too close / crossing  the rim moved inward past the inner helix, so the
                        annotation now says the helix fold is outside the
                        ear outline, which is anatomically impossible
  too far               the rim moved outward and left an implausibly wide
                        band of plain helix, which is the signature of a
                        rim pushed into hair

So the measurement is the CLEARANCE between the chains: the signed
distance from each inner-helix point to the outer polyline, positive
inside, as a percentage of ear height. A reference distribution is built
per inner-helix index from the ORIGINAL annotations on gated rows -- the
clearance at index 20 is not the clearance at index 30, and a single
global threshold would flag the narrow end of every ear. Rows are then
flagged by robust z-score (median/MAD) against that reference, in either
direction, plus any clearance <= 0 as an outright crossing.

Reported before AND after, because the distinction matters: an anomaly
present before the correction is a pre-existing annotation defect, one
that appears only after is damage the correction did.

    python scripts/qa_outer_inner_clash.py runs/corrected --z 3.0
"""
import argparse
import csv
import pathlib
import sys

import numpy as np
from matplotlib.path import Path as MPath

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

INNER = slice(20, 35)


def nearest_on_open(P, V):
    """Nearest point on the OPEN polyline V, per point of P.

    Open, not closed, and this is the whole correctness of the metric. The
    outer chain runs from the tragus (0) round the rim to the lobe end (19)
    and stops; it is NOT a closed outline. Closing it -- as the first
    version of this did -- draws a straight chord across the front of the
    ear, and the inner helix starts (point 20, the crus) right on top of
    that chord. The result was 14.8% of the ORIGINAL annotations scored as
    "inner helix outside the ear", which was the chord, not the ear.

    Returns the foot point, the distance, and a flag for feet that landed
    on a free END of the chain, where the outer chain has simply run out
    and a clearance is not defined.
    """
    A, B = V[:-1], V[1:]
    AB = B - A
    AP = P[:, None, :] - A[None, :, :]
    den = (AB ** 2).sum(-1)[None, :]
    t = (AP * AB[None]).sum(-1) / np.maximum(den, 1e-12)
    tc = np.clip(t, 0, 1)
    Q = A[None] + tc[..., None] * AB[None]
    d = np.linalg.norm(P[:, None, :] - Q, axis=-1)
    k = np.argmin(d, 1)
    n = np.arange(len(P))
    atend = ((k == 0) & (tc[n, k] <= 0.0)) | ((k == len(A) - 1) & (tc[n, k] >= 1.0))
    return Q[n, k], d[n, k], k, atend


def clearance(pts55, eh):
    """Signed clearance of each inner-helix point, % of ear height.

    Positive = inside the rim. The sign comes from the local inward normal
    of the nearest outer segment, inward being the side the outer chain's
    own centroid is on -- not from a point-in-polygon test, which needs a
    closed curve the annotation does not have.

    NaN where the foot lands on a free end of the chain.
    """
    O, I = pts55[:20], pts55[INNER]
    Q, d, k, atend = nearest_on_open(I, O)
    seg = O[1:] - O[:-1]
    seg = seg / np.maximum(np.linalg.norm(seg, axis=1, keepdims=True), 1e-9)
    nrm = np.c_[-seg[:, 1], seg[:, 0]][k]
    inward = O.mean(0) - Q
    sign = np.sign((nrm * inward).sum(1))
    sign[sign == 0] = 1.0
    c = (I - Q)
    out = (c * nrm).sum(1) * sign / eh * 100
    out[atend] = np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--z", type=float, default=3.0)
    a = ap.parse_args()
    RUN = pathlib.Path(a.run)

    L = np.load(ROOT / "data/manual/landmarks.npy")
    rows = list(csv.DictReader(open(ROOT / "data/manual/manifest.csv")))
    z = np.load(RUN / "outer.npz")
    idx = z["idx"].astype(int)
    newpts = z["pts"]

    CB, CA, EH = [], [], []
    for k, i in enumerate(idx):
        w, h = int(rows[i]["width"]), int(rows[i]["height"])
        g = L[i] * np.array([w, h])
        eh = float(g[:20, 1].max() - g[:20, 1].min())
        c = g.copy(); c[:20] = newpts[k] * np.array([w, h])
        CB.append(clearance(g, eh))
        CA.append(clearance(c, eh))
        EH.append(eh)
    CB, CA = np.array(CB), np.array(CA)

    # reference: per-index median and MAD of the ORIGINAL clearances
    med = np.nanmedian(CB, 0)
    mad = np.nanmedian(np.abs(CB - med), 0) * 1.4826
    mad = np.maximum(mad, 1e-6)
    ZB, ZA = (CB - med) / mad, (CA - med) / mad

    print(f"n={len(idx)} gated+corrected rows, inner-helix clearance "
          f"(% of ear height)\n")
    print(f"{'pt':>4}{'median':>9}{'MAD':>8}{'p1':>8}{'p99':>8}")
    for j in range(CB.shape[1]):
        print(f"{20+j:>4}{med[j]:>9.2f}{mad[j]:>8.2f}"
              f"{np.nanpercentile(CB[:,j],1):>8.2f}{np.nanpercentile(CB[:,j],99):>8.2f}")

    def tally(C, Z, nm):
        cross = np.nanmin(np.where(np.isnan(C), np.inf, C), 1) <= 0
        Zf = np.where(np.isnan(Z), 0.0, Z)
        close = (Zf <= -a.z).any(1) & ~cross
        far = (Zf >= a.z).any(1) & ~cross & ~close
        print(f"\n{nm:>9}  crossing {cross.sum():5d} ({100*cross.mean():.1f}%)"
              f"   too close {close.sum():5d} ({100*close.mean():.1f}%)"
              f"   too far {far.sum():5d} ({100*far.mean():.1f}%)")
        return cross, close, far
    cb = tally(CB, ZB, "before")
    ca = tally(CA, ZA, "after")

    flag_b = cb[0] | cb[1] | cb[2]
    flag_a = ca[0] | ca[1] | ca[2]
    print(f"\n  flagged before only (correction FIXED it) {int((flag_b & ~flag_a).sum())}"
          f"\n  flagged after only  (correction CAUSED it) {int((~flag_b & flag_a).sum())}"
          f"\n  flagged both                              {int((flag_b & flag_a).sum())}")

    worst = np.maximum(np.nan_to_num(np.abs(ZA)).max(1),
                       np.where(np.nanmin(CA, 1) <= 0, 99, 0))
    with open(RUN / "clash.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["idx", "image_file", "min_before", "min_after",
                     "worstz_before", "worstz_after", "kind", "new"])
        for k, i in enumerate(idx):
            kind = ("cross" if ca[0][k] else "close" if ca[1][k]
                    else "far" if ca[2][k] else "")
            wr.writerow([i, rows[i]["image_file"], f"{np.nanmin(CB[k]):.2f}",
                         f"{np.nanmin(CA[k]):.2f}", f"{np.nanmax(np.abs(ZB[k])):.2f}",
                         f"{np.nanmax(np.abs(ZA[k])):.2f}", kind,
                         int(flag_a[k] and not flag_b[k])])
    for nm, m in zip(("cross", "close", "far"), ca):
        o = idx[m][np.argsort(-worst[m])]
        (RUN / f"clash_{nm}.txt").write_text(" ".join(map(str, o)))
        print(f"  wrote clash_{nm}.txt  ({m.sum()} rows, worst first)")
    np.savez(RUN / "clearance.npz", idx=idx, before=CB, after=CA,
             med=med, mad=mad)
    print(f"  wrote {RUN/'clash.csv'} and clearance.npz")


if __name__ == "__main__":
    main()
