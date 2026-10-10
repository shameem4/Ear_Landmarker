#!/usr/bin/env python3
"""Run MoGe-2 on one image and write its point map out as an npz.

A SEPARATE PROCESS ON PURPOSE. MoGe lives in .venv_cu130 (python 3.10, its
own torch) and this project runs in the mocap env; importing both into one
interpreter is not possible and pinning either to suit the other would be a
large change to serve one experiment. So this is run with the OTHER
interpreter and talks to the rest of the pipeline through a file.

MoGe-2 emits a POINT MAP, not a relative depth map: a 3D point per pixel in
camera space, metric up to a single global scale, plus the fov it assumed and
a mask of pixels it considers valid. `points[...,2]` is therefore directly
comparable with a renderer's view-space z, which is the whole reason this
model was chosen over a disparity model -- a disparity map would have to be
inverted and affine-fitted before it meant anything.

    .venv_cu130/bin/python scripts/moge_infer.py in.png out.npz
"""
import argparse
import sys

import numpy as np
import torch
from PIL import Image

MODEL = "Ruicheng/moge-2-vitl-normal"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--fov-x", type=float, default=None,
                    help="the image's TRUE horizontal field of view in "
                         "degrees. MoGe estimates this when it is not given, "
                         "and on an ear render it estimated 515.9 px of focal "
                         "length against the renderer's actual 643.4 -- a "
                         "wider camera than the real one. Its depth is "
                         "self-consistent with whatever it assumed, so the "
                         "error only shows when the point map is combined "
                         "with the TRUE camera, where it costs about 20% of "
                         "the relief amplitude. When the image was rendered, "
                         "the fov is known exactly and there is no reason to "
                         "let it be guessed")
    args = ap.parse_args()
    src, dst = args.src, args.dst
    from moge.model.v2 import MoGeModel

    im = Image.open(src).convert("RGB")
    a = np.asarray(im, np.float32) / 255.0
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = MoGeModel.from_pretrained(MODEL).to(dev).eval()
    x = torch.from_numpy(a).permute(2, 0, 1).to(dev)
    with torch.no_grad():
        # resolution_level 9 is the package default and the highest it
        # offers; an ear is a small part of the frame, so there is no case
        # for asking for fewer tokens here.
        # fp16 ONLY ON THE GPU. MoGe's infer defaults to use_fp16=True, and
        # on CPU that fails outright -- "Input type (c10::Half) and bias type
        # (float) should be the same" -- because CPU convolutions have no
        # half-precision path. It only shows up when the GPU is busy and this
        # falls back, which is exactly when nobody is watching.
        out = m.infer(x, resolution_level=9, apply_mask=True,
                      fov_x=args.fov_x, use_fp16=(dev == "cuda"))

    d = {k: v.float().cpu().numpy() for k, v in out.items()
         if torch.is_tensor(v) and v.dtype != torch.bool}
    d.update({k: v.cpu().numpy() for k, v in out.items()
              if torch.is_tensor(v) and v.dtype == torch.bool})
    np.savez_compressed(dst, size=np.asarray(im.size),
                        fov_x_given=(np.nan if args.fov_x is None else args.fov_x),
                        **d)
    print(f"wrote {dst}: " + ", ".join(
        f"{k}{tuple(v.shape)}" for k, v in d.items()))


if __name__ == "__main__":
    main()
