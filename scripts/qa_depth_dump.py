#!/usr/bin/env python3
"""Dump DA3 depth for a sample of data/manual, for the 2D QA checks.

Runs in ../.venv_da3, which is the only env with depth_anything_3, and
writes plain npz so the analysis can stay in the mocap env. DA3-SMALL and
DA3-BASE are Apache 2.0 for code and weights; the METRIC-LARGE tier is
CC BY-NC and must not be used for anything released.

.venv_da3 ships torch 2.11/cu13 WITHOUT libnvrtc-builtins.so.13.0, so the
second inference in a process dies with an nvrtc JIT error. Nothing needs
installing -- ../../exrcise_trainer/.venv_cu130 already carries that exact
library, so put it on the loader path and the whole batch runs in one
process:

    NVR=../../exrcise_trainer/.venv_cu130/lib/python3.10/site-packages/nvidia/cu13/lib
    LD_LIBRARY_PATH=$NVR ../.venv_da3/bin/python scripts/qa_depth_dump.py \
        --n 150 --out /tmp/depth
"""
import argparse
import csv
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="depth-anything/DA3-SMALL",
                   choices=["depth-anything/DA3-SMALL", "depth-anything/DA3-BASE"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--res", type=int, default=504)
    p.add_argument("--indices", nargs="*", type=int, default=None,
                   help="dump exactly these manifest rows instead of a "
                        "random sample.")
    p.add_argument("--skip-existing", action="store_true",
                   help="leave rows that already have an npz in --out alone, "
                        "so an interrupted dump can be resumed.")
    a = p.parse_args()
    import torch
    from PIL import Image
    from depth_anything_3.api import DepthAnything3
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(ROOT / "data/manual/manifest.csv")))
    pick = (np.array(a.indices) if a.indices else
            np.random.default_rng(a.seed).choice(len(rows),
                                                 size=min(a.n, len(rows)),
                                                 replace=False))
    if a.skip_existing:
        pick = [i for i in pick if not (out / f"{int(i)}.npz").exists()]
        print(f"{len(pick)} rows still to dump")
    m = DepthAnything3.from_pretrained(a.model).to("cuda").eval()
    n = 0
    for i in pick:
        rec = rows[int(i)]
        im = Image.open(ROOT / "data/manual" / rec["image_file"]).convert("RGB")
        with torch.no_grad():
            r = m.inference([im], process_res=a.res)
        d = np.asarray(r.depth)
        d = d[0] if d.ndim == 3 else d
        np.savez_compressed(out / f"{int(i)}.npz", depth=d.astype(np.float32),
                            idx=int(i), w=im.size[0], h=im.size[1])
        n += 1
        if n % 50 == 0:
            print(f"  {n}/{len(pick)}", flush=True)
    print(f"wrote {n} depth maps -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
