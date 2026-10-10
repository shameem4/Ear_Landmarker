#!/usr/bin/env python3
"""Score a monocular depth map as the depth source, against a real scan.

THE ROUTE THIS TESTS. The shipped 2D landmarker gives x, y. Instead of
recovering z by back-projecting onto a mesh (unreliable: rays just outside the
pinna hit the scalp) or regressing it with a learned head (five of those have
now lost to z = 0 on photographs), take z from a general monocular depth
model at the landmark pixels. The mesh then only grades the answer.

WHY THE SCALE IS FITTED ON THE HEAD AND NOT ON THE EAR. A monocular model is
free in global scale and offset, so some normalisation is unavoidable --
but normalising on the ear would fit the very relief being measured and
every model would score well. So the affine is solved on the valid pixels
OUTSIDE the ear and applied to the ear unchanged. The head is a big easy
structure any such model gets roughly right, which makes this the strict
version of the test: the model must put the ear's relief at the correct
amplitude given a scale it was pinned to elsewhere.

THREE READINGS, because they fail independently and one alone misleads.

  corr      direction only, and the one number no normalisation can flatter.
            A model that has the ear's shape but the wrong amplitude still
            scores high here; a model returning a smooth blob cannot.
  relief    sd of depth across the ear, as a fraction of ear height and in
            mm. This is the statistic that disqualified HSRD-100's
            photogrammetry (1.8 mm against this scan's 3.69 mm), so it is
            reported the same way for continuity.
  z_skill   1 - MAE/mean|z| in the corpus's own z units, at the 55 landmark
            pixels, directly comparable with every depth head measured in
            this project. The headline row fits NOTHING: corpus z is
            invariant to a global scale on the depth map and MoGe emits
            metric points, so its raw output is usable as it stands. The
            oracle-scale row beside it is the ceiling a downstream solver
            could reach by choosing the amplitude per ear.

NEAREST-NEIGHBOUR SAMPLING, DELIBERATELY. Bilinear sampling of a depth map
blends across the silhouette and invents a depth that is on neither surface;
at this resolution the ear is ~200 px tall, so nearest costs almost nothing
and removes that failure entirely.

    python scripts/moge_lift_check.py --dump supersecret/moge/ear0
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# BJ002's ear heights, measured off the scan in the earlier pass of this work
# and quoted here only to put the relief in mm. Nothing scored below depends
# on it; the fraction-of-ear-height column is the unit-safe one.
EAR_MM = {0: 57.6, 1: 54.9}

# What scan_depth_check.py scores on these very dumps, so the comparison in
# the footer is to the SAME ear and not to whichever one was measured first.
# ((z_skill, corr) for head 1, then for the mean ear with a solved pose.)
REF = {0: ((+0.277, 0.641), (-0.101, 0.237)),
       1: ((-0.051, 0.245), (-0.297, 0.206))}


def corpus_z(zc, fx_px, crop_px):
    """View-space depth -> the corpus's z: root-relative, crop-width units.

    Depends on fx and the mean depth ONLY through their ratio, which is why
    two sources with different assumed cameras can be compared here at all.
    """
    zb = float(np.mean(zc))
    return (zc - zb) * fx_px / (crop_px * zb)


def sample_nn(img, uv):
    """Nearest-neighbour sample; NaN outside the frame."""
    h, w = img.shape[:2]
    x = np.rint(uv[:, 0]).astype(int)
    y = np.rint(uv[:, 1]).astype(int)
    ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    out = np.full(len(uv), np.nan)
    out[ok] = img[y[ok], x[ok]]
    return out


def scalp_mask(zc, tol):
    """The same rule the mesh harness uses: a sample well behind the median
    of the sampled set is the scalp beside the pinna, not the ear."""
    f = np.isfinite(zc)
    if f.sum() < 3:
        return np.zeros(len(zc), bool)
    med = float(np.median(zc[f]))
    span = float(np.ptp(zc[f]))
    return f & (zc > med + tol * max(span, 1e-9))


def affine_fit(x, y):
    """Least-squares a, b with a*x + b ~= y."""
    A = np.stack([x, np.ones_like(x)], 1)
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a), float(b)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump", required=True,
                   help="a directory from scan_depth_check.py --dump-render")
    p.add_argument("--moge", default="moge_full.npz",
                   help="which MoGe output in that directory to score. The "
                        "FULL render is the default because a monocular model "
                        "needs the head for context; the crop is there to "
                        "test whether that context actually matters")
    p.add_argument("--scalp-tol", type=float, default=0.25)
    p.add_argument("--pad", type=float, default=0.15,
                   help="the ear box is grown by this fraction before the "
                        "scale fit EXCLUDES it, so no ear pixel leaks into "
                        "the region that sets the scale")
    a = p.parse_args()

    d = pathlib.Path(a.dump)
    F = np.load(d / "frame.npz", allow_pickle=True)
    M = np.load(d / a.moge)

    depth = F["depth_full"].astype(float)          # exact, inf background
    lm2, side = F["lm2"], float(F["side"])
    box = F["crop_box"]
    fx_true = float(F["fx"])
    ear = int(F["ear"])
    H, W = depth.shape

    mg, mask = M["depth"].astype(float), M["mask"].astype(bool)
    if a.moge.endswith("crop.npz"):
        # the crop was cut from the render, so put its map back on the full
        # grid to keep one coordinate system for everything below
        full = np.full((H, W), np.nan)
        fm = np.zeros((H, W), bool)
        x0, y0, x1, y1 = [int(v) for v in box]
        import PIL.Image as I
        r = np.asarray(I.fromarray(mg).resize((x1 - x0, y1 - y0), I.NEAREST))
        rm = np.asarray(I.fromarray(mask.astype(np.uint8))
                        .resize((x1 - x0, y1 - y0), I.NEAREST)).astype(bool)
        full[y0:y1, x0:x1], fm[y0:y1, x0:x1] = r, rm
        mg, mask = full, fm
    if mg.shape != depth.shape:
        sys.exit(f"MoGe map is {mg.shape}, the render is {depth.shape}")
    fx_moge = float(M["intrinsics"][0, 0]) * mg.shape[1]

    # ---- regions ----------------------------------------------------------
    (bx0, by0), (bx1, by1) = lm2.min(0), lm2.max(0)
    ew, eh = bx1 - bx0, by1 - by0
    yy, xx = np.mgrid[0:H, 0:W]
    in_ear = ((xx >= bx0) & (xx <= bx1) & (yy >= by0) & (yy <= by1))
    grown = ((xx >= bx0 - a.pad * ew) & (xx <= bx1 + a.pad * ew) &
             (yy >= by0 - a.pad * eh) & (yy <= by1 + a.pad * eh))
    valid = np.isfinite(depth) & mask & np.isfinite(mg)
    fit_px = valid & ~grown
    ear_px = valid & in_ear
    if fit_px.sum() < 500 or ear_px.sum() < 500:
        sys.exit(f"too few pixels: {fit_px.sum()} for the fit, "
                 f"{ear_px.sum()} on the ear")

    # ---- the scale, from the head only ------------------------------------
    sc, sh = affine_fit(mg[fit_px], depth[fit_px])
    head_res = float(np.std(sc * mg[fit_px] + sh - depth[fit_px]))
    # mm per render unit, from the one length that is independently known:
    # this ear is EAR_MM mm tall and spans `eh` pixels at mean depth z, so it
    # is eh/fx*z render units tall. z shares x's units, so the same ratio
    # converts depth.
    mm_per_unit = EAR_MM[ear] / (eh / fx_true * float(np.mean(depth[ear_px])))

    # ---- ON-PINNA ONLY, and this is the whole point of the dense test -----
    # The landmark BOX is not the ear: it also contains scalp beside the
    # pinna and whatever lies behind the silhouette, which sit a full
    # ear-depth further back. Scored over the raw box, truth's relief comes
    # out 11.3 mm against this scan's measured 3.69 mm of pinna relief, and
    # the correlation is then dominated by one easy step edge -- any model
    # that knows an ear protrudes scores above 0.95 without resolving any
    # relief at all. So the dense region is narrowed to the depth band the
    # ON-PINNA landmarks occupy, using the same rule flag_scalp applies to
    # the sparse points, and the step edge is excluded rather than measured.
    zl = sample_nn(depth, lm2)
    fin = np.isfinite(zl)
    lo = float(np.min(zl[fin]))
    hi = float(np.median(zl[fin]) + a.scalp_tol * np.ptp(zl[fin]))
    pinna = ear_px & (depth >= lo) & (depth <= hi)
    if pinna.sum() < 500:
        sys.exit(f"only {pinna.sum()} on-pinna pixels")
    ear_px = pinna

    t_ear = depth[ear_px]
    m_ear = sc * mg[ear_px] + sh
    relief_t, relief_m = float(np.std(t_ear)), float(np.std(m_ear))
    ear_h_units = eh / fx_true * float(np.mean(t_ear))
    corr_dense = float(np.corrcoef(m_ear, t_ear)[0, 1])

    print(f"\n{d.name}  {'left' if ear == 0 else 'right'} ear, "
          f"MoGe map {a.moge}")
    print(f"  pixels: {fit_px.sum():,} set the scale (head, ear excluded), "
          f"{ear_px.sum():,} scored (ON-PINNA only)")
    print(f"  head fit: scale {sc:.4f}, residual sd {head_res * mm_per_unit:.2f} mm")
    print(f"\n  DENSE over the ear region")
    print(f"    relief sd   truth {relief_t * mm_per_unit:5.2f} mm   "
          f"MoGe {relief_m * mm_per_unit:5.2f} mm   "
          f"ratio {relief_m / max(relief_t, 1e-12):.3f}")
    print(f"    as a fraction of ear height ({ear_h_units * mm_per_unit:.1f} mm): "
          f"truth {relief_t / ear_h_units:.4f}  MoGe {relief_m / ear_h_units:.4f}")
    print(f"    corr        {corr_dense:+.3f}")

    # ---- at the 55 landmark pixels ----------------------------------------
    zt = sample_nn(depth, lm2)
    zm = sample_nn(mg, lm2)
    bad = scalp_mask(zt, a.scalp_tol) | ~np.isfinite(zt) | ~np.isfinite(zm)
    g = ~bad
    zt_c = corpus_z(zt[g], fx_true, side)
    zm_c = corpus_z(zm[g], fx_moge, side)
    base = float(np.mean(np.abs(zt_c)))
    mae = float(np.mean(np.abs(zm_c - zt_c)))
    corr_lm = float(np.corrcoef(zm_c, zt_c)[0, 1])
    # the ceiling a downstream solver could reach by choosing the amplitude
    # per ear; alpha* = corr * sd(true)/sd(pred) is that optimum
    al = corr_lm * np.std(zt_c) / max(np.std(zm_c), 1e-12)
    mae_o = float(np.mean(np.abs(al * zm_c - zt_c)))
    print(f"\n  AT THE 55 LANDMARK PIXELS   ({int(g.sum())}/{len(g)} usable, "
          f"{int(scalp_mask(zt, a.scalp_tol).sum())} on the scalp)")
    print(f"    signal mean|z| {base:.4f}   (render corpus 0.0702)")
    # NOTHING IS FITTED TO TRUTH IN THIS ROW. The affine above exists only
    # for the dense relief test; the landmark z is MoGe's raw metric depth.
    # It can be used raw because corpus z is exactly invariant to a global
    # scale on the depth map -- a cancels in (a*z - a*zbar)/(a*zbar) -- and
    # MoGe-2 emits a metric point map, so there is no unknown shift either.
    # This is therefore the DEPLOYABLE number, not an oracle-assisted one.
    print(f"    z_skill  raw, nothing fitted  {1 - mae / base:+.3f}")
    print(f"    z_skill  oracle scale a={al:5.2f}  {1 - mae_o / base:+.3f}")
    print(f"    corr     {corr_lm:+.3f}")
    h1, me = REF[ear]
    print(f"\n  for reference on these same pixels, this ear: head 1 "
          f"{h1[0]:+.3f} (corr {h1[1]:.3f}), mean ear + solved pose "
          f"{me[0]:+.3f} (corr {me[1]:.3f}),\n  z = 0 is 0.000 by construction")


if __name__ == "__main__":
    main()
