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
import sys

import numpy as np
import torch
from PIL import Image

MODEL = "Ruicheng/moge-2-vitl-normal"


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
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
        out = m.infer(x, resolution_level=9, apply_mask=True)

    d = {k: v.float().cpu().numpy() for k, v in out.items()
         if torch.is_tensor(v) and v.dtype != torch.bool}
    d.update({k: v.cpu().numpy() for k, v in out.items()
              if torch.is_tensor(v) and v.dtype == torch.bool})
    np.savez_compressed(dst, size=np.asarray(im.size), **d)
    print(f"wrote {dst}: " + ", ".join(
        f"{k}{tuple(v.shape)}" for k, v in d.items()))


if __name__ == "__main__":
    main()
