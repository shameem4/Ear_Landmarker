#!/usr/bin/env python3
"""Run the trained 2D landmarker over the depth corpus and store what it says.

The lifter is fed the 2D model's output at test time, so it should be trained
on the 2D model's output. Training it on ground truth and adding isotropic
gaussian noise -- which is what this replaces -- gets the error distribution
wrong in two ways that matter: real landmark error is CORRELATED between
neighbouring points on a contour, and it is biased ALONG the contour rather
than across it, because that is where the annotation ambiguity lies. A lifter
trained against round noise learns to average it away and then meets something
quite different.

Per-landmark confidence is stored alongside. It comes free from the heatmap's
concentration, and a point the model is unsure of is exactly a point whose
depth is ambiguous, so withholding it was pure loss.

    python scripts/precompute_xy.py --data data/depth3d --run manual_occ_s42
"""
import argparse
import csv
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/depth3d")
    p.add_argument("--run", default="manual_occ_s42")
    p.add_argument("--ckpt", default="")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--features", action="store_true",
                   help="also store the decoder features SAMPLED AT the predicted "
                        "landmarks. The base is frozen and the lifter heads run "
                        "without geometric augmentation, so those vectors never "
                        "change -- precomputing them turns head 3b from a "
                        "backbone forward per step into a table lookup, which is "
                        "the difference between trainable on a CPU and not. "
                        "9800 x 55 x 96 in float16 is about 100 MB.")
    a = p.parse_args()

    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    from eval_test import best_ckpt
    from model.lightning_module import EarLandmarkerModule

    ck = a.ckpt or best_ckpt(a.run)
    if not ck:
        sys.exit(f"no checkpoint for run {a.run!r}")
    print(f"using {ck}")
    mod = EarLandmarkerModule.load_from_checkpoint(ck, map_location=a.device).eval()
    net = mod.model.to(a.device)
    has_hm = hasattr(net, "heatmaps")

    d = pathlib.Path(a.data)
    rows = list(csv.DictReader(open(d / "manifest.csv")))

    class Imgs(Dataset):
        def __len__(self): return len(rows)
        def __getitem__(self, i):
            im = Image.open(d / rows[i]["image_file"]).convert("RGB")
            t = TF.to_tensor(im.resize((192, 192), Image.BILINEAR))
            return TF.normalize(t, [0.5] * 3, [0.5] * 3), int(rows[i]["idx"])

    dl = DataLoader(Imgs(), batch_size=a.batch, num_workers=a.workers)
    n = len(rows)
    xy = np.zeros((n, 55, 2), np.float32)
    conf = np.ones((n, 55), np.float32)
    feats = None
    with torch.no_grad():
        for k, (img, idx) in enumerate(dl):
            img = img.to(a.device)
            if has_hm:
                d_ = net._decode(img)
                hm = net.to_heatmap(d_)
                pred = net.soft_argmax(hm, getattr(net, "tau", 1.0))
                c = net.heatmap_confidence(hm, getattr(net, "tau", 1.0))
                conf[idx.numpy()] = c.cpu().numpy()
                if a.features:
                    from model.lifter import FeatureLifter
                    pts = pred.view(len(img), 55, 2).clamp(0.0, 1.0)
                    f = FeatureLifter.sample(d_, pts).cpu().numpy()
                    if feats is None:
                        feats = np.zeros((n, 55, f.shape[-1]), np.float16)
                    feats[idx.numpy()] = f.astype(np.float16)
            else:
                pred = net(img)
            xy[idx.numpy()] = pred.view(len(img), 55, 2).cpu().numpy()
            if k % 20 == 0:
                print(f"  {k * a.batch}/{n}", flush=True)

    np.save(d / "pred_xy.npy", xy)
    np.save(d / "pred_conf.npy", conf)
    if feats is not None:
        np.save(d / "pred_feat.npy", feats)
        print(f"  features {feats.shape} -> pred_feat.npy "
              f"({feats.nbytes / 1e6:.0f} MB)")
    gt = np.load(d / "landmarks.npy", mmap_mode="r")[:, :, :2]
    err = np.linalg.norm(xy - np.asarray(gt), axis=-1)
    print(f"\nwrote pred_xy.npy and pred_conf.npy for {n} views")
    print(f"  the 2D model's error on these renders: median {np.median(err):.4f} "
          f"of crop, p90 {np.percentile(err, 90):.4f}")
    print(f"  confidence: median {np.median(conf):.3f}, "
          f"10th pct {np.percentile(conf, 10):.3f}")
    print("  (this is the input distribution the lifter will actually meet)")


if __name__ == "__main__":
    main()
