"""What is actually achievable on these photographs, and where does head 1 sit?

Scores three things on the SAME held-out real photographs (preprocessed
test.csv, audioear2d rows, never in anyone's training set):

  shipped 2D ONNX, image as stored   -- the achievable floor, a model trained
                                        on exactly this kind of image
  head 1, image as stored            -- render-trained, wrong framing
  head 1, re-cropped to render framing -- render-trained, framing corrected,
                                        black padding incurred

Without the floor, head 1's number has nothing to be bad relative to.
"""
import csv, pathlib, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image
ROOT = pathlib.Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/Ear_Landmarker")
sys.path.insert(0, str(ROOT))
import torch, onnxruntime as ort
from model.depth_module import EarDepth3DModule

d = ROOT / "data/preprocessed"
rows = [r for r in csv.DictReader(open(d / "test.csv")) if r["source"] == "audioear2d"]
L = np.load(d / "landmarks.npy", mmap_mode="r")
print(f"{len(rows)} held-out audioear2d photographs")

sess = ort.InferenceSession(str(ROOT / "docs/EarLandmarker_web.onnx"),
                            providers=["CPUExecutionProvider"])
m = EarDepth3DModule.load_from_checkpoint(
    ROOT / "runs/sb_a_prop/ckpt/epoch=132-val_z_mae=0.0140.ckpt",
    map_location="cuda" if torch.cuda.is_available() else "cpu").eval()
dev = next(m.parameters()).device

def prep(im):
    return np.asarray(im.resize((192, 192), Image.BILINEAR),
                      np.float32).transpose(2, 0, 1)[None] / 255.

def head1(im):
    x = torch.from_numpy((prep(im) - 0.5) / 0.5).float().to(dev)
    with torch.no_grad():
        return m(x).view(-1, 3).cpu().numpy()

def shipped(im):
    return sess.run(None, {"image": ((prep(im) - 0.5) / 0.5).astype(np.float32)})[0][0]

def recrop(im, lab, fw, fh):
    W0, H0 = im.size
    P = lab * [W0, H0]
    (x0, y0), (x1, y1) = P.min(0), P.max(0)
    side = max((x1 - x0) / fw, (y1 - y0) / fh)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    box = (int(round(cx - side/2)), int(round(cy - side/2)),
           int(round(cx + side/2)), int(round(cy + side/2)))
    return im.crop(box), (P - [box[0], box[1]]) / [box[2]-box[0], box[3]-box[1]]

acc = {k: [] for k in ("ship", "h1_raw", "h1_crop")}
for r in rows:
    gt = np.asarray(L[int(r["idx"])][:55, :2], float)
    if not np.isfinite(gt).all():
        continue
    im = Image.open(d / r["image_file"]).convert("RGB")
    acc["ship"].append(np.linalg.norm(shipped(im) - gt, axis=1).mean())
    acc["h1_raw"].append(np.linalg.norm(head1(im)[:, :2] - gt, axis=1).mean())
    ci, lab = recrop(im, gt, 0.354, 0.677)
    acc["h1_crop"].append(np.linalg.norm(head1(ci)[:, :2] - lab, axis=1).mean())

print(f"\n{len(acc['ship'])} scored\n")
print(f"{'model / condition':<46}{'NME':>9}{'median':>9}{'vs floor':>10}")
floor = float(np.mean(acc["ship"]))
for lab, k in [("shipped 2D ONNX, image as stored  (the floor)", "ship"),
               ("head 1 (render-trained), image as stored", "h1_raw"),
               ("head 1, re-cropped to render framing", "h1_crop")]:
    v = np.array(acc[k])
    print(f"{lab:<46}{v.mean():9.5f}{np.median(v):9.5f}{v.mean()/floor:9.2f}x")
