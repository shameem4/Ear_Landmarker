#!/usr/bin/env python3
"""Rotate head 1's predicted 3D landmarks, next to the truth, as an animation.

Three still views can be cherry-picked; a full turn cannot. If the predicted
depths were noise the point set would come apart as it turns away from the
camera -- the helix would cross the concha, the lobe would swing through the
head -- and a still at the right angle would hide exactly that. Every frame
here is the same prediction seen from a different place, so the shape either
holds up all the way round or it does not.

GREEN is the label, MAGENTA the prediction. The view starts face-on, where
the model was given the image, and turns away from there; error in z shows up
as the two curves separating as the turn proceeds, by construction, because
at 0 degrees only x and y are visible.

    python scripts/demo_3d_turntable.py --ckpt ... --out runs/turntable.gif
"""
import argparse
import csv
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ear3d.scheme import STRIPS                                 # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/depth3d_v5")
    p.add_argument("--split", default="common_test.csv")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n-points", type=int, default=55)
    p.add_argument("--n", type=int, default=4, help="ears side by side")
    p.add_argument("--frames", type=int, default=48)
    p.add_argument("--sweep", type=float, default=70.0,
                   help="degrees either side of face-on")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/turntable.gif")
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
    L = np.load(d / "landmarks.npy", mmap_mode="r")
    k = a.n_points
    m = EarDepth3DModule.load_from_checkpoint(
        a.ckpt, map_location="cuda" if torch.cuda.is_available() else "cpu").eval()
    dev = next(m.parameters()).device
    size = m.hparams.get("image_size", 192)

    rng = np.random.default_rng(a.seed)
    by_ear = {}
    for r in rows:
        by_ear.setdefault(r["ear"], []).append(r)
    ears = sorted(by_ear)
    take = [sorted(by_ear[ears[i]], key=lambda r: abs(float(r.get("yaw", 0))))[0]
            for i in rng.choice(len(ears), min(a.n, len(ears)), replace=False)]

    data = []
    for r in take:
        im = Image.open(d / r["image_file"]).convert("RGB")
        x = torch.from_numpy(np.asarray(im.resize((size, size), Image.BILINEAR),
                                        np.float32).transpose(2, 0, 1) / 255.)
        x = ((x - 0.5) / 0.5)[None].to(dev)
        with torch.no_grad():
            pr = m(x).view(-1, 3).cpu().numpy()
        gt = np.asarray(L[int(r["idx"])][:k], float)
        if k < L.shape[1]:
            gt[:, 2] -= gt[:, 2].mean()
        # centre both on the label, so the turn is about the ear not the frame
        c = gt.mean(0)
        data.append((r["ear"], gt - c, pr - c, im,
                     float(np.abs(pr[:, 2] - gt[:, 2]).mean())))

    # right ears face the other way; turning both the same way puts one of
    # them edge-on, which reads as a worse prediction when it is only the view
    sgn = [1.0 if "_right" in n else -1.0 for n, *_ in data]
    frames = []
    for fi in range(a.frames):
        th = np.radians(a.sweep * np.sin(2 * np.pi * fi / a.frames))
        fig, ax = plt.subplots(1, len(data), figsize=(2.6 * len(data), 3.1))
        if len(data) == 1:
            ax = [ax]
        for i, (name, gt, pr, im, zm) in enumerate(data):
            t = th * sgn[i]
            R = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0],
                          [-np.sin(t), 0, np.cos(t)]])
            for P, col in ((gt, "#39ff88"), (pr, "#ff2d95")):
                Q = P @ R.T
                for lo, hi in STRIPS:
                    ax[i].plot(Q[lo:hi, 0], Q[lo:hi, 1], "-", c=col, lw=1.6)
                ax[i].plot(Q[:, 0], Q[:, 1], ".", c=col, ms=2.5)
            ax[i].set_xlim(-0.42, 0.42)
            ax[i].set_ylim(0.46, -0.46)
            ax[i].set_aspect("equal")
            ax[i].axis("off")
            ax[i].set_title(f"{name}\nz MAE {zm:.4f}", fontsize=7)
        fig.suptitle(f"head 1 on held-out ears — label GREEN, prediction "
                     f"MAGENTA\nturned {np.degrees(th):+.0f}° from the view "
                     f"the model was given", fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.9])
        fig.canvas.draw()
        frames.append(Image.fromarray(
            np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
        plt.close(fig)

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=1000 // 24, loop=0, optimize=True)
    print(f"wrote {out}  ({len(frames)} frames, {len(data)} ears)")


if __name__ == "__main__":
    main()
