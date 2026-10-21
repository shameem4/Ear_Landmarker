"""Score ground truth and any number of runs by how well the outer contour sits
on real image structure -- no ground truth in the judging.

REPLACES scripts/eval_gt_free.py (removed 2026-10-20) for this purpose. That script needs
BlazeEar/data/yolo11_ears, which no longer exists, so it reports "0 frames
usable" and nan for every model. This one works on the crops a split already
has, which also means it scores the SAME images a paired NME comparison
scored, rather than a different corpus.

The claim under test: on face-on ears v3 looks better than the GT itself, so
NME punishes it for disagreeing with a noisy label. NME cannot test that,
because NME IS agreement with the label. Edge response can: it asks whether a
contour lies on an actual intensity boundary, and it scores the GT on the same
footing as the predictions.

Outer helix only (points 0-19). That strip is the silhouette, where a real
boundary exists; interior points sit on soft ridges that gradient magnitude
does not reliably mark. Per-image normalisation so blur and exposure cannot
bias one model.

Limitation, same as scripts/eval_gt_free.py: edge response cannot tell the
helix rim from a hairline or jaw edge, so a contour that overshoots onto
another strong edge is rewarded, not penalised. Small differences are
inconclusive.
"""
import sys, csv, numpy as np, torch
from pathlib import Path
from PIL import Image
from scipy.ndimage import sobel, gaussian_filter, map_coordinates
from torch.utils.data import DataLoader
R = Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/Ear_Landmarker"); sys.path.insert(0,str(R))
from data.dataset import EarLandmarkDataset
from model.lightning_module import EarLandmarkerModule

D, S = R/"data/v3_manual", 192
CK = {"baseline": R/"runs/checkpoints/fixed_conv_s42/EarLandmarker_433_nme=0.0145.ckpt",
      "v3":       R/"runs/checkpoints/v3_cpu_fresh/EarLandmarker_097_nme=0.0123.ckpt"}
ds = EarLandmarkDataset(split_csv=D/"test.csv", data_dir=D, image_size=S)
rows = list(csv.DictReader(open(D/"test.csv")))
ld = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

P = {}
for n, ck in CK.items():
    m = EarLandmarkerModule.load_from_checkpoint(str(ck), map_location="cpu").eval()
    out = []
    with torch.no_grad():
        for b in ld:
            out.append(m.model(b["image"]).view(-1,55,2).numpy())
    P[n] = np.concatenate(out)
gt = np.stack([ds[i]["landmarks"].view(55,2).numpy() for i in range(len(ds))])
P["ground truth"] = gt

def edge_map(path):
    a = np.asarray(Image.open(path).convert("L").resize((S,S)), dtype=np.float64)
    a = gaussian_filter(a, 1.0)
    g = np.hypot(sobel(a, 0), sobel(a, 1))
    return g / (g.mean() + 1e-9)            # per-image normalised

OUTER = slice(0, 20)
scores = {k: [] for k in P}
for i, r in enumerate(rows):
    g = edge_map(D/r["image_file"])
    for k, pts in P.items():
        p = pts[i][OUTER] * S
        v = map_coordinates(g, [p[:,1], p[:,0]], order=1, mode="nearest")
        scores[k].append(v.mean())
for k in scores: scores[k] = np.array(scores[k])

print(f"edge response under the OUTER CONTOUR, n={len(rows)} ears")
print("higher = the contour sits on stronger real image structure\n")
for k in ("ground truth", "baseline", "v3"):
    print(f"  {k:14s} {scores[k].mean():.4f}   median {np.median(scores[k]):.4f}")
from scipy.stats import ttest_rel, wilcoxon
d = scores["v3"] - scores["baseline"]
t, p = ttest_rel(scores["v3"], scores["baseline"])
print(f"\n  v3 - baseline  {d.mean():+.4f}  t={t:+.2f}  p={p:.2e}  "
      f"(v3 higher on {100*(d>0).mean():.1f}% of ears)")
for k in ("baseline", "v3"):
    dd = scores[k] - scores["ground truth"]
    t2, p2 = ttest_rel(scores[k], scores["ground truth"])
    print(f"  {k:9s} - GT   {dd.mean():+.4f}  t={t2:+.2f}  p={p2:.2e}  "
          f"(above GT on {100*(dd>0).mean():.1f}% of ears)")
