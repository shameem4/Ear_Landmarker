"""Train a local ear-silhouette segmenter on the CC BY 4.0 Roboflow set.

SAM 2.1 is the bottleneck in the contour audit -- rejected on a third of
images, and only ~43% of the rest give a mask tight enough to judge a
landmark contour against. Florence-2 was tried as a second opinion and was
worse. This trains a small dedicated model instead, which is the one path
that needs no licence compromise:

  code    this file, plus torchvision -- BSD
  data    universe.roboflow.com/saifulull/human-ear-detection-qboz1 v3,
          590 images, CC BY 4.0, DOWNLOADED and trained locally. The hosted
          Roboflow model is not used: it would mean sending our images to a
          third party, and it rate-limits.

ATTRIBUTION IS REQUIRED. CC BY 4.0 obliges credit to SaifulULL for the
dataset wherever this model or its outputs are published.

ONE GREY AREA, deliberately left as a switch. The ResNet18 encoder is
ImageNet-pretrained: torchvision ships those weights under BSD, but
ImageNet's own terms are non-commercial research, and opinions differ on
whether that reaches derived weights. With 415 training images the
pretrained init is worth a lot, so it is the default; --scratch trains from
random init for a fully clean provenance. Measure both before deciding.

    python scripts/train_ear_segmenter.py \
        --data ../ear_seg_data/roboflow590/extracted --out runs/earseg_v1
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


class CocoEars(Dataset):
    """COCO-segmentation export -> (image, binary mask).

    Every image in this set carries exactly one ear polygon, so the mask is
    the union of that image's polygons and no instance logic is needed.
    """

    def __init__(self, root, split, size=256, train=False):
        self.dir = pathlib.Path(root) / split
        d = json.load(open(self.dir / "_annotations.coco.json"))
        self.imgs = {im["id"]: im for im in d["images"]}
        self.polys = {}
        for a in d["annotations"]:
            seg = a.get("segmentation") or []
            if seg:
                self.polys.setdefault(a["image_id"], []).extend(seg)
        self.ids = [i for i in self.imgs if i in self.polys]
        self.size, self.train = size, train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, k):
        rec = self.imgs[self.ids[k]]
        im = Image.open(self.dir / rec["file_name"]).convert("RGB")
        w, h = im.size
        canvas = Image.new("1", (w, h), 0)
        dr = ImageDraw.Draw(canvas)
        for pg in self.polys[self.ids[k]]:
            p = np.asarray(pg, np.float32).reshape(-1, 2)
            if len(p) >= 3:
                dr.polygon([tuple(z) for z in p], fill=1)
        mk = Image.fromarray(np.asarray(canvas, np.uint8) * 255)
        if self.train:
            if np.random.rand() < 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
                mk = mk.transpose(Image.FLIP_LEFT_RIGHT)
            ang = float(np.random.uniform(-20, 20))
            sc = float(np.random.uniform(0.85, 1.2))
            im = _affine(im, ang, sc, Image.BILINEAR)
            mk = _affine(mk, ang, sc, Image.NEAREST)
        im = im.resize((self.size, self.size), Image.BILINEAR)
        mk = mk.resize((self.size, self.size), Image.NEAREST)
        x = np.asarray(im, np.float32) / 255.0
        if self.train:
            x = np.clip(x * np.random.uniform(0.75, 1.3)
                        + np.random.uniform(-0.08, 0.08), 0, 1)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        y = (np.asarray(mk, np.float32) > 127).astype(np.float32)
        return torch.from_numpy(x.transpose(2, 0, 1)), torch.from_numpy(y)[None]


def _affine(img, deg, scale, resample):
    w, h = img.size
    out = img.rotate(deg, resample=resample, expand=False)
    nw, nh = int(w * scale), int(h * scale)
    out = out.resize((nw, nh), resample)
    canvas = Image.new(img.mode, (w, h), 0)
    canvas.paste(out, ((w - nw) // 2, (h - nh) // 2))
    return canvas


class Up(nn.Module):
    def __init__(self, cin, skip, cout):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv2d(cin + skip, cout, 3, padding=1), nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True))

    def forward(self, x, s=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if s is not None:
            x = torch.cat([x, s], 1)
        return self.c(x)


class UNetR18(nn.Module):
    """U-Net with a ResNet18 encoder, written against torchvision directly
    so the mocap env needs no new packages (segmentation_models_pytorch and
    albumentations would both have pulled dependency changes into the one
    env that has the project's full dep set)."""

    def __init__(self, pretrained=True):
        super().__init__()
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu)   # /2, 64
        self.pool = m.maxpool
        self.l1, self.l2, self.l3, self.l4 = m.layer1, m.layer2, m.layer3, m.layer4
        self.u4 = Up(512, 256, 256)
        self.u3 = Up(256, 128, 128)
        self.u2 = Up(128, 64, 64)
        self.u1 = Up(64, 64, 32)
        self.u0 = Up(32, 0, 16)
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.l1(self.pool(s0))
        s2 = self.l2(s1)
        s3 = self.l3(s2)
        s4 = self.l4(s3)
        x = self.u4(s4, s3)
        x = self.u3(x, s2)
        x = self.u2(x, s1)
        x = self.u1(x, s0)
        return self.head(self.u0(x))


def dice_bce(logit, y):
    p = torch.sigmoid(logit)
    num = 2 * (p * y).sum((1, 2, 3)) + 1.0
    den = p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1.0
    return F.binary_cross_entropy_with_logits(logit, y) + (1 - num / den).mean()


@torch.no_grad()
def evaluate(model, dl, dev):
    model.eval()
    inter = union = 0.0
    for x, y in dl:
        p = (torch.sigmoid(model(x.to(dev))) > 0.5).float().cpu()
        inter += (p * y).sum().item()
        union += ((p + y) >= 1).float().sum().item()
    return inter / max(union, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="extracted COCO export dir")
    ap.add_argument("--out", default="runs/earseg_v1")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--scratch", action="store_true",
                    help="random init instead of ImageNet-pretrained, for "
                         "fully clean weight provenance")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tr = CocoEars(a.data, "train", a.size, train=True)
    va = CocoEars(a.data, "valid", a.size)
    te = CocoEars(a.data, "test", a.size)
    print(f"train {len(tr)}  valid {len(va)}  test {len(te)}  device {dev}")
    dtr = DataLoader(tr, a.batch_size, shuffle=True, num_workers=a.workers,
                     drop_last=True, persistent_workers=a.workers > 0)
    dva = DataLoader(va, a.batch_size, num_workers=a.workers)
    dte = DataLoader(te, a.batch_size, num_workers=a.workers)
    model = UNetR18(pretrained=not a.scratch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    scaler = torch.amp.GradScaler(dev, enabled=dev == "cuda")
    best = -1.0
    for ep in range(a.epochs):
        model.train()
        tot = 0.0
        for x, y in dtr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(dev, enabled=dev == "cuda"):
                loss = dice_bce(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
        sched.step()
        iou = evaluate(model, dva, dev)
        flag = ""
        if iou > best:
            best = iou
            torch.save({"model": model.state_dict(), "val_iou": iou,
                        "pretrained": not a.scratch, "size": a.size},
                       out / "best.pt")
            flag = "  *"
        print(f"  ep {ep:3d}  loss {tot/max(len(dtr),1):.4f}  val IoU {iou:.4f}{flag}",
              flush=True)
    model.load_state_dict(torch.load(out / "best.pt")["model"])
    print(f"\nbest val IoU {best:.4f}   TEST IoU {evaluate(model, dte, dev):.4f}")
    print(f"wrote {out/'best.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
