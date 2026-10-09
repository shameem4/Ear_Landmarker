#!/usr/bin/env python3
"""What, measurably, makes a render not a photograph?

Head 1 scores 5.1x the achievable floor on real photographs, and the cause is
a domain gap rather than framing or annotation. "Make the renders more
realistic" is useless as an instruction, though -- it invites guessing. So
measure which image statistics separate the two populations, and let the size
of each gap say what to fix and in what order.

The statistics are deliberately simple and interpretable, because the point
is to decide what to change in the renderer, not to win a classification
contest: sharpness, high-frequency energy, noise, colour spread, how much of
the frame is background, and how uniform that background is.

A logistic probe on those features reports how separable the two sets are
overall. 1.00 means a render is always identifiable from these statistics
alone; 0.50 means they are indistinguishable. That number is the thing to
drive down, and it is the same kind of measure the dataset-fingerprint work
used (0.87 -> 0.49) rather than a new invented one.

    python scripts/render_vs_photo.py --n 400
"""
import argparse
import csv
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def feats(im):
    """Simple, nameable image statistics."""
    import cv2
    g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g = cv2.resize(g, (192, 192))
    rgb = cv2.resize(im, (192, 192)).astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    # noise: residual after a light blur, which a render has almost none of
    resid = g - cv2.GaussianBlur(g, (0, 0), 1.0)
    # high-frequency energy, normalised so it is not just contrast
    f = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean())))
    yy, xx = np.mgrid[:192, :192] - 96
    r = np.sqrt(yy ** 2 + xx ** 2)
    hi = f[r > 48].sum() / max(f.sum(), 1e-6)
    # border ring stands in for background: how varied is it, how coloured
    ring = np.r_[rgb[:12].reshape(-1, 3), rgb[-12:].reshape(-1, 3),
                 rgb[:, :12].reshape(-1, 3), rgb[:, -12:].reshape(-1, 3)]
    return {
        "sharpness (laplacian var)": float(lap.var()),
        "noise (resid sd)": float(resid.std()),
        "high-freq energy": float(hi),
        "saturation sd": float(rgb.std(axis=2).mean()),
        "background sd": float(ring.std(axis=0).mean()),
        "background colourfulness": float(np.abs(ring - ring.mean(1, keepdims=True)).mean()),
        "global contrast": float(g.std()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--render", default="data/depth3d_v5")
    p.add_argument("--render-split", default="common_test.csv")
    p.add_argument("--photo", default="data/manual")
    p.add_argument("--photo-split", default="test.csv")
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--augment", type=float, default=0.0,
                   help="apply data/photo_aug at this strength to the renders "
                        "before measuring, to see what it closes")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    import cv2                                           # noqa: F401
    from PIL import Image
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    rng = np.random.default_rng(a.seed)
    sets = {}
    for label, root, split in (("render", a.render, a.render_split),
                               ("photo", a.photo, a.photo_split)):
        d = pathlib.Path(root)
        rows = list(csv.DictReader(open(d / split)))
        take = [rows[i] for i in rng.choice(len(rows), min(a.n, len(rows)), False)]
        F = []
        arng = np.random.default_rng(a.seed + 1)
        for r in take:
            pil = Image.open(d / r["image_file"]).convert("RGB")
            if a.augment and label == "render":
                from data.photo_aug import photo_realism
                pil = photo_realism(pil, arng, a.augment)
            F.append(feats(np.asarray(pil)))
        sets[label] = F
        print(f"{label}: {len(F)} images from {root}/{split}")

    keys = list(sets["render"][0].keys())
    R = np.array([[f[k] for k in keys] for f in sets["render"]])
    P = np.array([[f[k] for k in keys] for f in sets["photo"]])
    print(f"\n{'statistic':<28}{'render':>12}{'photo':>12}{'ratio':>9}"
          f"{'separation d':>14}")
    order = []
    for j, k in enumerate(keys):
        mr, mp = R[:, j].mean(), P[:, j].mean()
        sd = np.sqrt((R[:, j].var() + P[:, j].var()) / 2) + 1e-9
        d = abs(mr - mp) / sd
        order.append((d, k, mr, mp))
    for d, k, mr, mp in sorted(order, reverse=True):
        print(f"{k:<28}{mr:>12.3f}{mp:>12.3f}{mr / (mp + 1e-9):>9.2f}{d:>14.2f}")

    X = np.vstack([R, P])
    y = np.r_[np.zeros(len(R)), np.ones(len(P))]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    acc = cross_val_score(clf, X, y, cv=5).mean()
    print(f"\nlogistic probe on these {len(keys)} statistics: "
          f"**{acc:.3f}** (0.50 = indistinguishable)")
    print("drive this toward 0.50; separation d says which knob pays most")


if __name__ == "__main__":
    main()
