#!/usr/bin/env python3
"""Turn a photograph's 2D annotation into a DEPTH label, via its own 3D ear.

AudioEar3D ships, for 112 ears, a photograph AND a point-cloud scan of that
same ear. The project already has 95 solved 3D landmarks per ear
(lm3d_v5_audioear_mm.npz) and 55 human 2D annotations per photo. Those are in
correspondence by construction -- same scheme, same point order -- so the
camera can come from the landmarks directly and no feature matching is needed.
scripts/pnp_camera_recovery.py takes the SIFT-against-rendered-cloud route; it
was written before the 3D landmarks existed and is not needed for this.

WEAK PERSPECTIVE, AND THAT IS NOT A COMPROMISE. Full PnP with unknown focal is
badly conditioned on an ear: the thing is shallow, so a long lens far away and
a short lens close by fit almost equally well, and focal and distance are only
weakly separable. But look at what the label actually needs. The render corpus
defines depth as

    zn = (zc - zbar) * fx / (s_crop * zbar)

which depends on fx and zbar ONLY through the ratio fx/zbar -- the
pixels-per-millimetre scale at the ear. That ratio is exactly what a scaled
orthographic fit observes, and it is well determined even when fx and zbar
separately are not. The ambiguity cancels out of the quantity being measured.

So: centre both sets, least-squares the 2x3 map, project it to the nearest
scaled-orthonormal pair (SVD), complete the third row by cross product, and
read depth off it. What weak perspective throws away is the second-order
foreshortening across the ear's own depth, and the size of that error is not
argued here, it is MEASURED -- --validate renders the same solver against
renders where the true z is known.

    python scripts/fit_photo_depth.py --validate        # error floor, on renders
    python scripts/fit_photo_depth.py --build out.npz   # the photo benchmark
"""
import argparse
import csv
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CANAL = 39          # the scheme's single ear-canal point

TPL = pathlib.Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/wrap_work/templates")


# The canonical frame's outward direction, and it really is a CONSTANT. Over
# the 588 ears of lm3d_v6_all_mm the landmark surface normal agrees with this
# axis to within 7.6 degrees in the worst case and 0.6 degrees on average,
# left ears and right ears alike, because canonicalisation already put every
# ear in one frame. Only the overall sign had to be settled, which is one bit,
# fixed once against 1171 renders whose true depth is known.
CANON_OUT = np.array([-0.954982, -0.006270, -0.296599])


def outward_normal(X):
    """Which way is OUT of the head, i.e. where the camera must be.

    Scaled orthography cannot tell a shape from its mirror in depth -- both
    project identically -- so the solver has to be told which side it is
    seeing. Three rules were tried:

      "the concha is farther than the helix"  true of only 76% of real depths
      "the canal is the farthest point"       true of 89%
      the ear's own plane normal, oriented by the canal being innermost
                                              inverted 14% of EARS, which is
                                              what left 17.9% of views
                                              depth-reversed and the label
                                              error at 2.4x the signal

    The first two are corpus statistics and enforcing either would have
    silently corrupted one label in ten. The third failed for a subtler
    reason: the plane normal is fine, it was the CANAL test orienting it that
    flipped, and because it flips per ear it survived every per-view quality
    filter at a constant 18%.

    What actually works is not per-ear at all. Canonicalisation already
    aligned every ear to one frame, so the outward direction is a fixed axis
    and nothing needs estimating from the individual ear.
    """
    n = np.linalg.svd(X - X.mean(0))[2][-1]
    return n if n @ CANON_OUT > 0 else -n


def weak_perspective(X, x):
    """Scaled-orthographic pose from 3D X (N,3) to 2D x (N,2).

    Returns (z, scale, rot, residual): z is the depth of each point in the
    SAME units as x, already root-relative, which is the corpus convention.
    """
    Xc = X - X.mean(0)
    xc = x - x.mean(0)
    # least squares 2x3; lstsq not inv, the ear is shallow and X'X is
    # ill-conditioned along the viewing direction by construction
    M, *_ = np.linalg.lstsq(Xc, xc, rcond=None)      # (3,2)
    M = M.T                                          # (2,3)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    s = float(S.mean())                              # crop widths per mm
    R2 = U @ Vt                                      # nearest orthonormal rows
    r3 = np.cross(R2[0], R2[1])
    r3 /= np.linalg.norm(r3)
    # depth grows AWAY from the camera, so it must grow into the head
    if r3 @ outward_normal(X) > 0:
        r3 = -r3
    z = (Xc @ r3) * s
    z -= z.mean()                                    # root-relative
    resid = float(np.linalg.norm(Xc @ M.T - xc, axis=1).mean())
    return z, s, np.vstack([R2, r3]), resid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--validate", action="store_true",
                   help="measure the solver against renders, where z is known")
    p.add_argument("--identity", choices=["render", "photo"], default=None,
                   help="THE TEST THAT DECIDES whether a depth label is worth "
                        "anything: fit each 2D set to its OWN 3D ear and to a "
                        "stranger's, and compare. If the stranger fits as "
                        "well, the label is a generic ear under a recovered "
                        "pose and cannot tell a good 3D model from a bad one")
    p.add_argument("--build", default="",
                   help="write the AudioEar3D photo depth benchmark here")
    p.add_argument("--onnx", default="docs/EarLandmarker_web.onnx",
                   help="second, independent 2D source. Averaging it with "
                        "the hand annotation halves the part of the label "
                        "error that is independent between them")
    p.add_argument("--render-data", default="data/depth3d_v5")
    p.add_argument("--photo-data", default="data/preprocessed")
    p.add_argument("--lm3d-render", default=str(TPL / "lm3d_v6_all_mm.npz"))
    p.add_argument("--lm3d-photo", default=str(TPL / "lm3d_v5_audioear_mm.npz"))
    p.add_argument("--n", type=int, default=600, help="views, for --validate")
    p.add_argument("--noise", type=float, default=0.0,
                   help="px-equivalent gaussian noise on the 2D input, as a "
                        "fraction of crop width, to imitate annotator error")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if not (a.validate or a.build or a.identity):
        sys.exit("nothing to do: pass --validate, --identity or --build")

    K = 55
    rng = np.random.default_rng(a.seed)

    if a.validate:
        d = pathlib.Path(a.render_data)
        rows = list(csv.DictReader(open(d / "common_test.csv")))
        L = np.load(d / "landmarks.npy", mmap_mode="r")
        z3 = np.load(a.lm3d_render, allow_pickle=True)
        lut = {str(n): i for i, n in enumerate(z3["names"])}
        X3 = z3["landmarks"]
        take = [rows[i] for i in rng.choice(len(rows), min(a.n, len(rows)), False)]
        errs, rel, scal, res = [], [], [], []
        miss = 0
        for r in take:
            if r["ear"] not in lut:
                miss += 1
                continue
            gt = np.asarray(L[int(r["idx"])][:K], float)
            tz = gt[:, 2] - gt[:, 2].mean()
            xy = gt[:, :2]
            if a.noise:
                xy = xy + rng.normal(0, a.noise, xy.shape)
            X = np.asarray(X3[lut[r["ear"]]][:K], float)
            z, s, _, rr = weak_perspective(X, xy)
            errs.append(np.abs(z - tz).mean())
            rel.append(np.abs(z - tz).mean() / np.abs(tz).mean())
            scal.append(s)
            res.append(rr)
        if miss:
            print(f"{miss} views had no 3D ear and were skipped")
        e = np.array(errs)
        print(f"\n{len(e)} render views, true z known, {K} landmarks"
              + (f", 2D noise sd {a.noise}" if a.noise else ""))
        print(f"  depth label error   mean {e.mean():.5f}  median "
              f"{np.median(e):.5f}  p90 {np.percentile(e, 90):.5f}")
        print(f"  as a fraction of the signal it is labelling: "
              f"{np.mean(rel):.1%}")
        print(f"  2D reprojection residual {np.mean(res):.5f} of crop width")
        print(f"  recovered scale {np.mean(scal):.5f} crop widths per mm")
        print(f"\n  head 1's own depth error on these views is 0.0144, so a "
              f"label\n  error of {e.mean():.4f} is "
              f"{e.mean()/0.0144:.0%} of what is being measured.")

    if a.identity:
        rng2 = np.random.default_rng(a.seed)
        if a.identity == "render":
            d = pathlib.Path(a.render_data)
            rows = list(csv.DictReader(open(d / "common_test.csv")))
            L = np.load(d / "landmarks.npy", mmap_mode="r")
            z3 = np.load(a.lm3d_render, allow_pickle=True)
            lut = {str(n): i for i, n in enumerate(z3["names"])}
            rows = [r for r in rows if r["ear"] in lut]
            rows = [rows[i] for i in rng2.choice(len(rows),
                                                 min(a.n, len(rows)), False)]
            key = [lut[r["ear"]] for r in rows]
        else:
            d = pathlib.Path(a.photo_data)
            rows = [r for r in csv.DictReader(open(d / "manifest.csv"))
                    if r["source"] == "audioear3d"]
            L = np.load(d / "landmarks.npy", mmap_mode="r")
            z3 = np.load(a.lm3d_photo, allow_pickle=True)
            lut = {str(n): i for i, n in enumerate(z3["names"])}
            nm = [r["original_path"].split("::")[-1].replace(".json", "")
                  .replace("/", "_") for r in rows]
            keep = [i for i, n in enumerate(nm) if n in lut]
            rows = [rows[i] for i in keep]
            key = [lut[nm[i]] for i in keep]
        X3 = z3["landmarks"]
        other = rng2.permutation(key)
        rc, rw = [], []
        for r, mine, theirs in zip(rows, key, other):
            xy = np.asarray(L[int(r["idx"])][:K, :2], float)
            if a.noise:
                xy = xy + rng2.normal(0, a.noise, xy.shape)
            rc.append(weak_perspective(np.asarray(X3[mine][:K], float), xy)[3])
            rw.append(weak_perspective(np.asarray(X3[theirs][:K], float), xy)[3])
        c, w = float(np.median(rc)), float(np.median(rw))
        print(f"\n{a.identity}: {len(rows)} items"
              + (f", 2D noise {a.noise}" if a.noise else ""))
        print(f"  fitted to its OWN 3D ear   median residual {c:.5f}")
        print(f"  fitted to a STRANGER's     median residual {w:.5f}")
        print(f"  ratio {w / c:.2f}"
              + ("   <- the individual ear carries the fit"
                 if w / c > 1.5 else
                 "   <- A STRANGER FITS AS WELL. Any depth label from this "
                 "is\n           a generic ear under a recovered pose, not "
                 "this ear."))

    if a.build:
        d = pathlib.Path(a.photo_data)
        rows = [r for r in csv.DictReader(open(d / "manifest.csv"))
                if r["source"] == "audioear3d"]
        L = np.load(d / "landmarks.npy", mmap_mode="r")
        z3 = np.load(a.lm3d_photo, allow_pickle=True)
        lut = {str(n): i for i, n in enumerate(z3["names"])}
        X3 = z3["landmarks"]
        # TWO 2D SOURCES, AVERAGED. The hand annotation and the shipped 2D
        # model disagree enough that the depth labels they produce differ by
        # 0.0204 (42% of the signal), so neither alone is the truth. The
        # model is genuinely independent here: it was trained on data/manual
        # and has never seen an AudioEar3D photograph. Averaging cancels the
        # part of their error that is independent; what it cannot touch is
        # the part they share, and the floor below reports the total either
        # way rather than the flattering half.
        import onnxruntime as ort
        from PIL import Image
        sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])

        def predict_2d(path):
            im = Image.open(path).convert("RGB")
            t = np.asarray(im.resize((192, 192), Image.BILINEAR),
                           np.float32).transpose(2, 0, 1)[None] / 255.0
            return sess.run(None, {"image": ((t - 0.5) / 0.5).astype(np.float32)})[0][0]

        out_idx, out_z, out_name, out_res, out_s = [], [], [], [], []
        disagree = []
        for r in rows:
            # 'AudioEar3D.zip::001/left.json' -> '001_left'
            tail = r["original_path"].split("::")[-1].replace(".json", "")
            name = tail.replace("/", "_")
            if name not in lut:
                continue
            xy = np.asarray(L[int(r["idx"])][:K, :2], float)
            if not np.isfinite(xy).all():
                continue
            X = np.asarray(X3[lut[name]][:K], float)
            mod = predict_2d(d / r["image_file"])[:K]
            zh = weak_perspective(X, xy)[0]
            zm = weak_perspective(X, mod)[0]
            disagree.append(float(np.abs(zh - zm).mean()))
            z, s, _, rr = weak_perspective(X, 0.5 * (xy + mod))
            out_idx.append(int(r["idx"]))
            out_z.append(z)
            out_name.append(name)
            out_res.append(rr)
            out_s.append(s)
        Z = np.asarray(out_z, np.float32)
        np.savez(a.build, idx=np.asarray(out_idx), z=Z,
                 names=np.asarray(out_name), resid=np.asarray(out_res),
                 scale=np.asarray(out_s))
        print(f"\n{len(out_idx)} of {len(rows)} photographs got a depth label "
              f"-> {a.build}")
        print(f"  mean |z| (the baseline a flat predictor achieves) "
              f"{np.abs(Z).mean():.5f}")
        print(f"  2D reprojection residual: median "
              f"{np.median(out_res):.5f}, p90 "
              f"{np.percentile(out_res, 90):.5f} of crop width")
        dis = float(np.mean(disagree))
        indep = dis / np.sqrt(2)                 # per-source independent part
        # Total label error is NOT the independent part. Measured on renders
        # driven to this same fit residual, where true depth is known, it is
        # 0.0242 -- the shared systematic dominates and no amount of
        # averaging removes it. Quote the total.
        total = 0.0242
        base = float(np.abs(Z).mean())
        print(f"\n  LABEL NOISE FLOOR, which anyone using this must quote:")
        print(f"    two 2D sources disagree           {dis:.4f}")
        print(f"    -> independent part each          {indep:.4f}")
        print(f"    total label error (from renders)  {total:.4f}"
              f"  = {total / base:.0%} of the signal")
        print(f"    => z_skill on this set is CAPPED near "
              f"{1 - total / base:.2f}; a perfect model cannot score above it")
        print(f"    => useless for a model already good to ~0.014, usable for")
        print(f"       one in the 0.05-0.09 range, which is where head 1 sits")
        np.savez(a.build, idx=np.asarray(out_idx), z=Z,
                 names=np.asarray(out_name), resid=np.asarray(out_res),
                 scale=np.asarray(out_s), label_error=total,
                 two_source_disagreement=dis, zbase=base)


if __name__ == "__main__":
    main()
