#!/usr/bin/env python3
"""Show what head 1 actually produces: a 3D landmark set, from three angles.

THREE PANELS PER EAR, because a front view cannot show the only thing that
distinguishes this model from the 2D one.

  front      the image with x, y drawn on it. This is all a 2D landmarker
             gives, and two ears with very different depth look the same here.
  above      the same points as (x, z), the ear seen from over the top. A
             depth error shows up as geometry -- a flattened concha, a helix
             that fails to stand off the head -- instead of as a number.
  rotated    x and z mixed by a 55 degree turn about the vertical. If the
             depths were noise this view would be a smear; it staying a
             recognisable ear is the thing worth looking at.

RUNS ON REAL PHOTOGRAPHS TOO, and that is the honest test. The model is
trained entirely on renders, so renders tell you it learned the corpus. There
is no depth ground truth for a photograph, so nothing can be scored there --
the prediction is shown alone and labelled as such. Reading those panels is a
judgement about whether the shape is plausible, not a measurement, and this
script does not pretend otherwise.

    python scripts/demo_3d.py --data data/depth3d_v5 --split common_test.csv \
        --ckpt runs/sb_a_prop/ckpt/epoch=132-val_z_mae=0.0140.ckpt --n-points 55
    python scripts/demo_3d.py --data data/preprocessed --split manifest.csv \
        --sources audioear2d --no-gt --ckpt ... --n-points 55
"""
import argparse
import csv
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ear3d.scheme import STRIPS                                 # noqa: E402

GT_C, PR_C = "#39ff88", "#ff2d95"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="common_test.csv")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n-points", type=int, default=0)
    p.add_argument("--n", type=int, default=6, help="ears to show")
    p.add_argument("--sources", nargs="*", default=[])
    p.add_argument("--no-gt", action="store_true",
                   help="for real photos: no depth label exists, show the "
                        "prediction alone and say so")
    p.add_argument("--angle", type=float, default=55.0)
    p.add_argument("--match-framing", nargs=2, type=float, default=None,
                   metavar=("FRAC_W", "FRAC_H"),
                   help="re-crop around the label box so the ear fills this "
                        "fraction of the crop. The render corpus frames the "
                        "ear at 0.354 x 0.677 of the crop and the real-photo "
                        "pipeline at 0.835 x 0.827, so a render-trained model "
                        "shown a photo straight from that pipeline is being "
                        "tested on a framing 2.4x tighter than anything it "
                        "saw. Pass 0.354 0.677 to take that difference away")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from PIL import Image
    from model.depth_module import EarDepth3DModule

    d = pathlib.Path(a.data)
    rows = list(csv.DictReader(open(d / a.split)))
    if a.sources:
        rows = [r for r in rows if r.get("source") in a.sources]
    if not rows:
        sys.exit(f"no rows left in {a.split} after --sources {a.sources}")
    L = np.load(d / "landmarks.npy", mmap_mode="r")
    has_z = L.shape[-1] == 3 and not a.no_gt

    m = EarDepth3DModule.load_from_checkpoint(
        a.ckpt, map_location="cuda" if torch.cuda.is_available() else "cpu").eval()
    dev = next(m.parameters()).device
    k = a.n_points or m.hparams.num_landmarks
    if k != m.hparams.num_landmarks:
        sys.exit(f"--n-points {k} but the checkpoint predicts "
                 f"{m.hparams.num_landmarks}")
    # Same re-centring as data/depth_dataset: z is measured from the mean of
    # the points IN USE. Skipping it would draw a depth bias that is an
    # artefact of the slice and nothing to do with the model.
    recentre = L.shape[-1] == 3 and k < L.shape[1]
    if a.match_framing and has_z:
        sys.exit("--match-framing with depth labels would need z rescaled "
                 "too: z is in CROP-WIDTH units, so changing the crop changes "
                 "what the label means. Not implemented, because the renders "
                 "already have the framing this flag exists to restore.")
    size = m.hparams.get("image_size", 192)

    rng = np.random.default_rng(a.seed)
    # One ear per subject, so the panels are different people rather than one
    # person at several yaws. Of that ear's views take the most FRONTAL one:
    # an ear already imaged obliquely is foreshortened in x, and turning it
    # another 55 degrees puts it edge-on, which hides the depth instead of
    # showing it. Nothing is being flattered -- a frontal view is the HARDEST
    # case for depth, because it is where x and y give least away.
    by_ear = {}
    for r in rows:
        by_ear.setdefault(r.get("ear", r["idx"]), []).append(r)
    def frontal(v):
        return sorted(v, key=lambda r: abs(float(r.get("yaw", 0) or 0)))[0]
    ears = sorted(by_ear)
    take = [frontal(by_ear[ears[i]])
            for i in rng.choice(len(ears), min(a.n, len(ears)), replace=False)]

    fig, ax = plt.subplots(3, len(take), figsize=(2.9 * len(take), 9.2))
    if len(take) == 1:
        ax = ax[:, None]

    for ci, r in enumerate(take):
        im = Image.open(d / r["image_file"]).convert("RGB")
        lab = (np.asarray(L[int(r["idx"])][:k, :2], float)
               if L.shape[-1] >= 2 else None)
        if a.match_framing and lab is not None:
            # Square crop on the label box, sized so the ear fills the asked
            # fraction. It uses the LABEL box, which is what a detector hands
            # the landmarker at inference, so what this measures is the
            # landmarker on its own framing and not the detector.
            fw, fh = a.match_framing
            W0, H0 = im.size
            P = lab * [W0, H0]
            (x0, y0), (x1, y1) = P.min(0), P.max(0)
            side = max((x1 - x0) / fw, (y1 - y0) / fh)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            box = (int(round(cx - side / 2)), int(round(cy - side / 2)),
                   int(round(cx + side / 2)), int(round(cy + side / 2)))
            im = im.crop(box)
            lab = (P - [box[0], box[1]]) / [box[2] - box[0], box[3] - box[1]]
        W, H = im.size
        # THE TURN HAS TO FOLLOW THE SIDE. A left and a right ear face
        # opposite ways, so one sign of rotation opens one and closes the
        # other. With a single sign every left ear in the figure came out
        # edge-on and every right ear came out clean, which reads as the
        # model being worse on left ears when it is only the camera.
        th = np.radians(a.angle if "_right" in str(r.get("ear", ""))
                        else -a.angle)
        x = torch.from_numpy(np.asarray(im.resize((size, size), Image.BILINEAR),
                                        np.float32).transpose(2, 0, 1) / 255.)
        x = ((x - 0.5) / 0.5)[None].to(dev)
        with torch.no_grad():
            pr = m(x).view(-1, 3).cpu().numpy()

        # A real-photo corpus has HUMAN x, y labels and no z at all. Show
        # them on the front panel -- that half is checkable and it is the
        # half a human annotated -- and leave the depth panels carrying the
        # prediction alone. Drawing nothing would throw away the one real
        # measurement available off the renders.
        gt2d = None if has_z else lab
        gt = None
        if has_z:
            gt = np.asarray(L[int(r["idx"])][:k], float)
            if recentre:
                gt[:, 2] -= gt[:, 2].mean()

        def strips(axis, P, c, lw=1.4):
            for lo, hi in STRIPS:
                axis.plot(P[lo:hi, 0], P[lo:hi, 1], "-", c=c, lw=lw, alpha=.92)
            axis.plot(P[:, 0], P[:, 1], ".", c=c, ms=3)

        # --- front ------------------------------------------------------
        b = ax[0, ci]
        b.imshow(im)
        ref2d = gt[:, :2] if gt is not None else gt2d
        if ref2d is not None:
            strips(b, ref2d * [W, H], GT_C)
        strips(b, pr[:, :2] * [W, H], PR_C)
        name = r.get("ear", r.get("source", ""))
        extra = ""
        if ref2d is not None:
            extra = "\nNME %.4f" % np.linalg.norm(
                pr[:, :2] - ref2d, axis=1).mean()
        if gt is not None:
            extra += "   z MAE %.4f" % np.abs(pr[:, 2] - gt[:, 2]).mean()
        elif ref2d is not None:
            extra += "   z: no label exists"
        b.set_title(f"{str(name)[:18]}{extra}", fontsize=7)
        b.axis("off")

        # --- from above, (x, z) -----------------------------------------
        t = ax[1, ci]
        if gt is not None:
            strips(t, gt[:, [0, 2]], GT_C, 1.3)
        strips(t, pr[:, [0, 2]], PR_C, 1.3)
        t.set_aspect("equal")
        t.invert_yaxis()
        t.set_title("from above  (x, z)", fontsize=7)
        t.tick_params(labelsize=5)
        t.grid(alpha=.25)

        # --- rotated about the vertical ---------------------------------
        q = ax[2, ci]
        def rot(P):
            return np.stack([P[:, 0] * np.cos(th) + P[:, 2] * np.sin(th),
                             P[:, 1]], 1)
        if gt is not None:
            strips(q, rot(gt), GT_C, 1.3)
        strips(q, rot(pr), PR_C, 1.3)
        q.set_aspect("equal")
        q.invert_yaxis()
        q.set_title(f"turned {np.degrees(th):+.0f}°", fontsize=7)
        q.tick_params(labelsize=5)
        q.grid(alpha=.25)

    if has_z:
        head = (f"head 1, {k} landmarks on HELD-OUT ears.  "
                f"label GREEN, prediction MAGENTA.")
    else:
        head = (f"head 1, {k} landmarks on REAL PHOTOGRAPHS (trained only on "
                f"renders).  human x,y label GREEN, prediction MAGENTA.\n"
                f"NME is a real measurement against the annotation; the DEPTH "
                f"is unlabelled and therefore unscored -- judge it by whether "
                f"the shape is plausible, not as a result.")
    fig.suptitle(head + "\nrows: image (x, y)  |  from above (x, z)  |  "
                 f"turned ±{a.angle:.0f}° about the vertical, sign "
                 f"following the ear side.  Lower rows share the image's scale.",
                 fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"wrote {out}  ({len(take)} ears)")


if __name__ == "__main__":
    main()
