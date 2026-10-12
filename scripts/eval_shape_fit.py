#!/usr/bin/env python3
"""Route B scoring: take a shape head's COEFFICIENTS, solve pose, read depth.

The head regresses shape and pose together, but pose is thrown away here on
purpose. A regressed pose has to survive the render-to-photo shift; a SOLVED
pose rides on the 2D model that already survives it (0.0143 NME on real
photographs). This is MediaPipe's split -- network gives shape and 2D, metric
3D comes from fitting -- and it is the reason fitting the plain MEAN ear
already scores +0.217 on the photo benchmark where head 1 scores -0.171.

Scale does not have to be handled. `weak_perspective` solves for the scale
factor, so feeding it the unit-height basis and feeding it the same shape in
millimetres give identical depth in crop units. Asserted below rather than
assumed.

DISJOINT 2D SOURCES, WHICH IS THE WHOLE CARE OF THIS SCRIPT. The shipped
benchmark's depth label was built by fitting each ear's true 3D to the AVERAGE
of the hand annotation and the shipped model's own 2D. Predicting with that
same shipped 2D would share pose error with the label and inflate every skill
number. So by default the label is recomputed from the HAND annotation alone
and predictions use the shipped model's 2D -- no shared term. --label-2d avg
reproduces the published benchmark instead, and is comparable to the numbers
in RESULTS at the cost of that shared error.

    python scripts/eval_shape_fit.py --models b2=runs/b2_sup1/ckpt/*.ckpt
    python scripts/eval_shape_fit.py --renders --models b2=...   # in domain
"""
import argparse
import csv
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.fit_photo_depth import weak_perspective   # noqa: E402

TPL = pathlib.Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/wrap_work/templates")


def paired_boot(a, b, den, n=4000, seed=7):
    """Skill difference a-b over the SAME items, with its own CI.

    Paired because per-ear variance dominates an unpaired interval on 112
    ears; the question is whether one model beats the other on these ears,
    not what either scores in the abstract.
    """
    a, b, den = map(np.asarray, (a, b, den))
    sa, sb = 1 - a.sum() / den.sum(), 1 - b.sum() / den.sum()
    rs, d = np.random.default_rng(seed), []
    for _ in range(n):
        i = rs.integers(0, len(a), len(a))
        d.append((1 - a[i].sum() / den[i].sum()) - (1 - b[i].sum() / den[i].sum()))
    d = np.asarray(d)
    return sa - sb, np.percentile(d, [2.5, 97.5]), float((d > 0).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=[], help="name=ckpt pairs")
    p.add_argument("--head1", default="", help="head 1 ckpt, for comparison")
    p.add_argument("--base2d", default="",
                   help="the shipped 2D Lightning checkpoint, needed by "
                        "--lifter for per-point confidence")
    p.add_argument("--lifter", default="",
                   help="head 3a ckpt: x,y -> z with NO image at all. Fed the "
                        "same shipped 2D the mean-ear fit gets, so it is the "
                        "learned counterpart of that geometric fit and the "
                        "only variant with nothing render-domain in its input")
    p.add_argument("--feat", default="",
                   help="head 3b ckpt: the SHIPPED 2D model frozen, with only "
                        "a lifter trained on top. Its x,y is therefore the "
                        "shipped model's, which already transfers to "
                        "photographs, so only the depth has to cross the gap")
    p.add_argument("--renders", action="store_true",
                   help="score in domain on render common_test instead")
    p.add_argument("--bench", default="data/photo3d/audioear3d_depth.npz")
    p.add_argument("--photo-data", default="data/preprocessed")
    p.add_argument("--render-data", default="data/depth3d_v5")
    p.add_argument("--basis", default=str(TPL / "lm_basis_v8_all_55_unitheight.npz"))
    p.add_argument("--lm3d-photo", default=str(TPL / "lm3d_v5_audioear_mm.npz"))
    p.add_argument("--onnx", default="docs/EarLandmarker_web.onnx")
    p.add_argument("--label-2d", choices=["hand", "avg"], default="hand")
    p.add_argument("--n", type=int, default=400, help="views, for --renders")
    p.add_argument("--n-points", type=int, default=55)
    p.add_argument("--alpha-sweep", action="store_true",
                   help="scale each model's predicted coefficients by a factor "
                        "and report skill against it. The diagnostic that "
                        "separates 'wrong shape' from 'no shape': by "
                        "construction alpha=0 is the mean ear, so if the best "
                        "alpha is 0 the prediction carries no usable signal, "
                        "and if it is >0 there is signal being over-trusted")
    a = p.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    import torch
    from PIL import Image
    from model.depth_module import EarDepth3DModule, EarShape3DModule

    K = a.n_points
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bd = np.load(a.basis, allow_pickle=True)
    mean_u, comp_u = bd["mean"][:K], bd["components"][:, :K]
    size = float(bd["ear_size"])

    # the assertion the docstring promises: the solver is scale-free
    _t = np.random.default_rng(0).normal(size=(K, 2)) * 0.1 + 0.5
    z_u = weak_perspective(mean_u - mean_u.mean(0), _t)[0]
    z_mm = weak_perspective((mean_u - mean_u.mean(0)) * size, _t)[0]
    assert np.abs(z_u - z_mm).max() < 1e-9, "solver is not scale-invariant"

    def shape_from(c):
        """(n_modes,) coefficients -> (K,3) centred shape, any consistent unit."""
        L = mean_u + np.einsum("m,mkj->kj", c, comp_u[:len(c)])
        return L - L.mean(0)

    # ---- the items, their 2D, and their depth label ----------------------
    import onnxruntime as ort
    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])

    def onnx_2d(path):
        im = Image.open(path).convert("RGB")
        t = np.asarray(im.resize((192, 192), Image.BILINEAR),
                       np.float32).transpose(2, 0, 1)[None] / 255.0
        return sess.run(None, {"image": ((t - 0.5) / 0.5).astype(np.float32)}
                        )[0][0][:K]

    items = []          # (image_path, xy_for_fitting, z_label)
    if a.renders:
        d = pathlib.Path(a.render_data)
        z3 = np.load(TPL / "lm3d_v6_all_mm.npz", allow_pickle=True)
        lut = {str(n): i for i, n in enumerate(z3["names"])}
        rows = [r for r in csv.DictReader(open(d / "common_test.csv"))
                if r["ear"] in lut]
        L = np.load(d / "landmarks.npy", mmap_mode="r")
        rng = np.random.default_rng(0)
        rows = [rows[i] for i in rng.choice(len(rows), min(a.n, len(rows)), False)]
        for r in rows:
            gt = np.asarray(L[int(r["idx"])][:K], float)
            items.append((d / r["image_file"], gt[:, :2],
                          gt[:, 2] - gt[:, 2].mean()))
        label_src, cap = "renderer (true z)", None
    else:
        d = pathlib.Path(a.photo_data)
        B = np.load(a.bench, allow_pickle=True)
        rows = {int(r["idx"]): r for r in csv.DictReader(open(d / "manifest.csv"))}
        L = np.load(d / "landmarks.npy", mmap_mode="r")
        z3 = np.load(a.lm3d_photo, allow_pickle=True)
        lut = {str(n): i for i, n in enumerate(z3["names"])}
        X3 = z3["landmarks"]
        cap = 1 - float(B["label_error"]) / float(B["zbase"])
        for gi, nm, zpub in zip(B["idx"], B["names"], B["z"]):
            r = rows[int(gi)]
            hand = np.asarray(L[int(gi)][:K, :2], float)
            if a.label_2d == "hand":
                # label from the HAND 2D and this ear's true 3D; prediction
                # will use the shipped model's 2D, so nothing is shared
                zl = weak_perspective(np.asarray(X3[lut[str(nm)]][:K], float),
                                      hand)[0]
            else:
                zl = np.asarray(zpub)[:K]
            items.append((d / r["image_file"], None, zl))
        label_src = ("hand 2D + true 3D (disjoint from the prediction)"
                     if a.label_2d == "hand" else
                     "published benchmark, avg 2D (SHARES error with predictions)")

    # the 2D each PREDICTION is fitted to: ground truth on renders (the pose
    # is not what is being tested there), the shipped model on photographs
    xy_fit = [it[1] if it[1] is not None else onnx_2d(it[0]) for it in items]
    den = np.array([np.abs(it[2]).sum() for it in items])
    print(f"\n{len(items)} items, {'renders' if a.renders else 'photographs'}"
          f"\n  depth label: {label_src}")
    if cap:
        print(f"  label noise caps z_skill near {cap:.2f}")
    print(f"  prediction 2D: {'ground truth' if a.renders else 'shipped ONNX'}")

    # ---- conditions ------------------------------------------------------
    res = {}

    def record(name, zs):
        res[name] = np.asarray(zs)
        print(f"  {name:<40s} {1 - res[name].sum()/den.sum():8.3f}")

    print(f"\n{'condition':<42s} {'z_skill':>8s}")
    print(f"  {'predict z = 0':<40s} {0.0:8.3f}")
    record("mean ear + solver  (MediaPipe-equivalent)",
           [np.abs(weak_perspective(shape_from(np.zeros(1)), x)[0] - it[2]).sum()
            for x, it in zip(xy_fit, items)])

    for spec in a.models:
        name, ckpt = spec.split("=", 1)
        m = EarShape3DModule.load_from_checkpoint(
            ckpt, map_location=dev).eval().to(dev)
        nmod = m.model.n_modes
        coef, e2e = [], []
        for (path, _, zl) in items:
            im = Image.open(path).convert("RGB")
            t = np.asarray(im.resize((192, 192), Image.BILINEAR),
                           np.float32).transpose(2, 0, 1)[None] / 255.0
            x = torch.from_numpy((t - 0.5) / 0.5).float().to(dev)
            with torch.no_grad():
                prm = m.model.parameters_from(x).cpu().numpy()[0]
                lm = m.model.decode(
                    torch.from_numpy(prm[None]).to(dev)).cpu().numpy()[0]
            coef.append(prm[:nmod])
            zp = lm[:K, 2] - lm[:K, 2].mean()
            e2e.append(np.abs(zp - zl).sum())
        C = np.asarray(coef)
        record(f"{name}: coeffs + solver",
               [np.abs(weak_perspective(shape_from(c), x)[0] - it[2]).sum()
                for c, x, it in zip(C, xy_fit, items)])
        record(f"{name}: end-to-end (REGRESSED pose)", e2e)
        if a.alpha_sweep:
            print(f"       alpha sweep on the predicted coefficients:")
            for al in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
                zs = np.array([
                    np.abs(weak_perspective(shape_from(c * al), x)[0]
                           - it[2]).sum()
                    for c, x, it in zip(C, xy_fit, items)])
                print(f"         alpha {al:4.2f}  z_skill "
                      f"{1 - zs.sum()/den.sum():7.3f}")
        print(f"       predicted coefficient sd {C.std():.3f} "
              f"(gt is 1.0; << 1 means it fell back toward the mean ear)")

    if a.lifter:
        from model.depth_module import EarLifter3DModule
        from model.lightning_module import EarLandmarkerModule
        lm_ = EarLifter3DModule.load_from_checkpoint(
            a.lifter, map_location=dev).eval().to(dev)
        # This lifter wants the 2D model's per-point confidence, which the
        # ONNX export does not return, so its x,y come from the shipped
        # LIGHTNING checkpoint instead. To keep the comparison honest the
        # mean-ear fit is then re-scored at those same x,y -- the two differ
        # only in how depth is produced from identical landmarks.
        if not a.base2d:
            sys.exit("--lifter needs --base2d, the shipped 2D checkpoint, "
                     "because this lifter consumes its confidence")
        b2 = EarLandmarkerModule.load_from_checkpoint(
            a.base2d, map_location=dev).eval().to(dev).model
        zs, zm, nvalid = [], [], 0
        for (path, _, zl) in items:
            im = Image.open(path).convert("RGB")
            tt = np.asarray(im.resize((192, 192), Image.BILINEAR),
                            np.float32).transpose(2, 0, 1)[None] / 255.0
            xin = torch.from_numpy((tt - 0.5) / 0.5).float().to(dev)
            with torch.no_grad():
                feat = b2._decode(xin)
                hm = b2.to_heatmap(feat)
                xyb = b2.soft_argmax(hm, b2.tau).view(-1, K, 2)
                cf = b2.heatmap_confidence(hm, b2.tau)
                zq = lm_.model(xyb, cf if lm_.model.use_conf else None)
            zq = zq.cpu().numpy()[0][:K]
            zq = zq - zq.mean()
            zs.append(np.abs(zq - zl).sum())
            xb = xyb.cpu().numpy()[0][:K]
            zmm = weak_perspective(shape_from(np.zeros(1)), xb)[0]
            zm.append(np.abs(zmm - zl).sum())
            nvalid += 1
        record("head 3a: x,y -> z, no image", zs)
        record("  (mean ear at the SAME x,y, for reference)", zm)

    if a.feat:
        from model.depth_module import EarFeatureLifter3DModule
        fm = EarFeatureLifter3DModule.load_from_checkpoint(
            a.feat, map_location=dev).eval().to(dev)
        zs, zf_pred = [], []
        for (path, _, zl) in items:
            im = Image.open(path).convert("RGB")
            tt = np.asarray(im.resize((192, 192), Image.BILINEAR),
                            np.float32).transpose(2, 0, 1)[None] / 255.0
            with torch.no_grad():
                xyf, _, zp = fm.from_image(
                    torch.from_numpy((tt - 0.5) / 0.5).float().to(dev))
            zq = zp.cpu().numpy()[0][:K]
            zq = zq - zq.mean()
            zf_pred.append(zq)
            zs.append(np.abs(zq - zl).sum())
        record("head 3b: frozen shipped 2D + lifter", zs)
        zf_pred = np.asarray(zf_pred)
        print(f"       predicted z sd {zf_pred.std():.4f} "
              f"(label sd {np.std([it[2] for it in items]):.4f})")

    if a.head1:
        m = EarDepth3DModule.load_from_checkpoint(a.head1, map_location=dev).eval().to(dev)
        zs = []
        for (path, _, zl) in items:
            im = Image.open(path).convert("RGB")
            t = np.asarray(im.resize((192, 192), Image.BILINEAR),
                           np.float32).transpose(2, 0, 1)[None] / 255.0
            with torch.no_grad():
                P = m(torch.from_numpy((t - 0.5) / 0.5).float().to(dev)
                      ).view(-1, 3).cpu().numpy()
            zp = P[:K, 2] - P[:K, 2].mean()
            zs.append(np.abs(zp - zl).sum())
        record("head 1: regressed depth", zs)

    # ---- the comparison that decides it ----------------------------------
    base = "mean ear + solver  (MediaPipe-equivalent)"
    others = [k for k in res if k != base]
    if others:
        print(f"\npaired bootstrap against the mean-ear fit, same items:")
        print(f"  {'condition':<40s} {'delta':>7s} {'95% CI':>18s} {'P(better)':>10s}")
        for k in others:
            dd, ci, pb = paired_boot(res[k], res[base], den)
            print(f"  {k:<40s} {dd:+7.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] {pb:10.2f}")
        print(f"\nRoute B only counts if 'coeffs + solver' beats the mean ear "
              f"with a CI clear of zero.")


if __name__ == "__main__":
    main()
