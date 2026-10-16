"""Nail the SAM 2.1 box-prompt API on a couple of images before scaling."""
import sys, numpy as np, torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor
dev = "cuda" if torch.cuda.is_available() else "cpu"
mid = "facebook/sam2.1-hiera-base-plus"
proc = Sam2Processor.from_pretrained(mid)
model = Sam2Model.from_pretrained(mid, dtype=torch.float16).to(dev).eval()
print("loaded", mid, "on", dev)
im = Image.open(sys.argv[1]).convert("RGB")
w, h = im.size
L = np.load("data/manual/landmarks.npy")[int(sys.argv[2])]
xy = L * np.array([w, h])
lo, hi = xy.min(0), xy.max(0)
pad = 0.15 * (hi - lo)
box = [[float(lo[0]-pad[0]), float(lo[1]-pad[1]),
        float(hi[0]+pad[0]), float(hi[1]+pad[1])]]
inp = proc(images=im, input_boxes=[box], return_tensors="pt").to(dev)
with torch.inference_mode():
    out = model(**inp, multimask_output=True)
m = proc.post_process_masks(out.pred_masks.float().cpu(),
                            inp["original_sizes"])[0]
print("mask tensor", tuple(m.shape), "iou", out.iou_scores.float().cpu().numpy().ravel())
m = m.numpy()
best = int(np.argmax(out.iou_scores.float().cpu().numpy().ravel()))
mk = m.reshape(-1, *m.shape[-2:])[best] > 0
print(f"image {w}x{h}  mask covers {mk.mean()*100:.1f}% of the frame")
