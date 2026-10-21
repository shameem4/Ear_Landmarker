# Promoted from scratch (2026-10-19-v3/score_2x2.py) on 2026-10-20 when scratch was pruned;
# cited by the docs as the harness behind the stated numbers.
"""Score the data-size x flip-penalty 2x2 on the PRESERVED 1,000-ear test split.

All four cells are scored by the same code on the same rows, so the four
numbers are paired per image and a difference can be tested directly.

Checkpoint choice is by the STORED current_score, never by parsing nme= from
the filename: the documented tie bug makes the filename rank pick the earliest
(worst) of a tied group, and 3 of 5 runs differed when this was corrected.

Equivariance is measured on the RAW model (m.model), not through
LandmarkPredictor.predict(): FLIP_FUSION_DEFAULT=True averages the two arms, so
any measurement through the deployed path is 0 by construction.
"""
import sys, torch, numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from scipy import stats
R = Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/Ear_Landmarker"); sys.path.insert(0, str(R))
from data.dataset import EarLandmarkDataset
from model.lightning_module import EarLandmarkerModule
from model.losses import _contour_frames
from checkpoint_utils import best_checkpoint, checkpoint_score

D = R / "data/v3_manual"
RUNS = [("v3 small ctl",  "v3_gpu_s42", 4538, 0.0),
        ("v3 small fc05", "v3_fc05",    4538, 0.5),
        ("v3 GROWN ctl",  "v3g_ctl_s42", 6686, 0.0),
        ("v3 GROWN fc05", "v3g_fc05_s42", 6686, 0.5)]

def best_ckpt(run):
    """Delegate to the repo's checkpoint_utils.

    An earlier version of this script read ck["current_score"] at the TOP
    level. That key does not exist -- the monitored score lives under
    ck["callbacks"][<ModelCheckpoint>]["current_score"] -- so every candidate
    scored +inf and min() silently fell back to glob order. 3 of 4 cells took
    the wrong checkpoint. This is the documented tie bug, reproduced inside the
    script whose docstring claimed to avoid it; do not re-implement it here.
    """
    b = best_checkpoint(R / "runs/checkpoints" / run)
    return b, checkpoint_score(b)


ds = EarLandmarkDataset(split_csv=D/"test.csv", data_dir=D, image_size=192)
ld = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)

out = {}
for label, run, rows, fc in RUNS:
    ck, sc = best_ckpt(run)
    m = EarLandmarkerModule.load_from_checkpoint(str(ck), map_location="cuda").eval().cuda()
    P, T, V, Q = [], [], [], []
    with torch.no_grad():
        for b in ld:
            img = b["image"].cuda()
            p = m.model(img).view(-1, 55, 2)
            mir = m.model(torch.flip(img, dims=[3])).view(-1, 55, 2).clone()
            mir[:, :, 0] = 1.0 - mir[:, :, 0]          # continuous coords: 1-x
            P.append(p.cpu().numpy()); Q.append(mir.cpu().numpy())
            T.append(b["landmarks"].view(-1, 55, 2).numpy())
            V.append(b["visible"].numpy().astype(bool))
    P, Q = np.concatenate(P), np.concatenate(Q)
    T, V = np.concatenate(T), np.concatenate(V)
    tan, nrm = _contour_frames(torch.tensor(T))
    r = torch.tensor(P) - torch.tensor(T)
    nn = (r * nrm).sum(-1).abs().numpy()
    tt = (r * tan).sum(-1).abs().numpy()
    e = np.linalg.norm(P - T, axis=-1)
    per_img = lambda a: np.array([a[i][V[i]].mean() for i in range(len(a))]) * 192
    out[label] = dict(ckpt=ck.name, sc=sc, rows=rows, fc=fc,
                      err=per_img(e), nrm=per_img(nn), tan=per_img(tt),
                      eq=per_img(np.linalg.norm(P - Q, axis=-1)))

print(f"\npreserved test split: {len(ds)} ears, px at 192\n")
print(f"  {'run':16s} {'rows':>5s} {'fc':>4s}  {'equivar':>8s} {'err px':>7s} "
      f"{'normal':>7s} {'tangen':>7s}   ckpt")
for label, _, _, _ in RUNS:
    o = out[label]
    print(f"  {label:16s} {o['rows']:5d} {o['fc']:4.1f}  {o['eq'].mean():8.5f} "
          f"{o['err'].mean():7.3f} {o['nrm'].mean():7.3f} {o['tan'].mean():7.3f}"
          f"   {o['ckpt']}")

def cmp(a, b, what):
    A, B = out[a], out[b]
    print(f"\n{what}  ({a} -> {b})")
    for k in ("err", "nrm", "tan"):
        x, y = A[k], B[k]
        t = stats.wilcoxon(x, y)
        print(f"  {k:4s} {x.mean():6.3f} -> {y.mean():6.3f}  "
              f"{(y.mean()-x.mean())/x.mean()*100:+6.2f}%   p={t.pvalue:.1e}")
    print(f"  eq   {A['eq'].mean():.5f} -> {B['eq'].mean():.5f}  "
          f"{(B['eq'].mean()-A['eq'].mean())/A['eq'].mean()*100:+.1f}%")

cmp("v3 small ctl",  "v3 GROWN ctl",  "DATA EFFECT (penalty off both sides)")
cmp("v3 small ctl",  "v3 small fc05", "PENALTY EFFECT at 4,538 rows")
cmp("v3 GROWN ctl",  "v3 GROWN fc05", "PENALTY EFFECT at 6,686 rows")
cmp("v3 small fc05", "v3 GROWN fc05", "DATA EFFECT (penalty on both sides)")
cmp("v3 small ctl",  "v3 GROWN fc05", "BOTH vs NEITHER")

# Additivity: is the combined gain the sum of the two single gains?
b = out["v3 small ctl"]["err"].mean()
d = out["v3 GROWN ctl"]["err"].mean() - b
p = out["v3 small fc05"]["err"].mean() - b
both = out["v3 GROWN fc05"]["err"].mean() - b
print(f"\nADDITIVITY on error px:  data {d:+.3f}  penalty {p:+.3f}  "
      f"sum {d+p:+.3f}  OBSERVED {both:+.3f}  interaction {both-(d+p):+.3f}")
