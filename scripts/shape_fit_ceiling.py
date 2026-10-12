#!/usr/bin/env python3
"""What a 40-mode shape can reach, before training anything to predict one.

Route B asks a network for shape coefficients and solves pose geometrically.
Whether that is worth doing depends on a number no training run is needed to
get: feed the solver the TRUE coefficients and see what depth comes out. That
is the ceiling for any shape head, and it is cheap.

Run on renders, never on the photo benchmark. The benchmark's depth label was
itself produced by fitting each ear's true 3D to its 2D, so a "true shape"
condition there would be scoring the label against itself and would return
something near 1.0 for free. Render depth comes from the renderer and is
independent of any fit.

TWO COEFFICIENT SETS, AND THE DIFFERENCE MATTERS. Fitting the basis to the 55
landmarks by least squares reconstructs them better (0.710 mm against 1.710)
but needs coefficients averaging 3.2 standard deviations, because 40 modes are
near-degenerate at 55 points and the fit spends that freedom on landmark noise.
gt_coeffs, fitted to the full 1188-vertex mesh, is unit-sd by construction and
is what a network can actually be supervised on. The honest ceiling uses that
one -- and it is the HIGHER of the two, which is the opposite of what the
reconstruction residuals suggest.

    python scripts/shape_fit_ceiling.py
"""
import csv, pathlib, sys
import numpy as np
ROOT = pathlib.Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/Ear_Landmarker"); sys.path.insert(0,str(ROOT))
TPL = pathlib.Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/wrap_work/templates")
_R = ROOT
exec(open(ROOT/"scripts/fit_photo_depth.py").read().split("def main()")[0].split('"""',2)[2])
ROOT = _R; K = 55
B = np.load(TPL/"lm_basis_v8_all_55_unitheight.npz", allow_pickle=True)
size=float(B["ear_size"]); mean_mm=B["mean"]*size; comp_mm=B["components"]*size
A = comp_mm.reshape(len(comp_mm),-1).T; M=(mean_mm-mean_mm.mean(0)).ravel()
z3 = np.load(TPL/"lm3d_v6_all_mm.npz", allow_pickle=True)
names=[str(n) for n in z3["names"]]; X3=np.asarray(z3["landmarks"][:,:K],float)
lut={n:i for i,n in enumerate(names)}
g = np.load(TPL/"gt_coeffs_v7_mm.npz", allow_pickle=True)
gc = {str(n):c for n,c in zip(g["names"], g["coeffs"])}
D=(X3-X3.mean(1,keepdims=True)).reshape(len(X3),-1)
Cls,*_=np.linalg.lstsq(A,(D-M).T,rcond=None)
Cgt=np.stack([gc[n] for n in names]).T

d=pathlib.Path(ROOT/"data/depth3d_v5")
rows=[r for r in csv.DictReader(open(d/"common_test.csv")) if r["ear"] in lut]
L=np.load(d/"landmarks.npy",mmap_mode="r")
rng=np.random.default_rng(0); rows=[rows[i] for i in rng.choice(len(rows),min(400,len(rows)),False)]

def skill(getX, boot=False):
    num=[];den=[]
    for r in rows:
        gt=np.asarray(L[int(r["idx"])][:K],float); tz=gt[:,2]-gt[:,2].mean()
        zp=weak_perspective(getX(r),gt[:,:2])[0]
        num.append(np.abs(zp-tz).sum()); den.append(np.abs(tz).sum())
    num=np.array(num);den=np.array(den)
    s=1-num.sum()/den.sum()
    if not boot: return s,None
    bs=[]; rs=np.random.default_rng(7)
    for _ in range(2000):
        i=rs.integers(0,len(num),len(num)); bs.append(1-num[i].sum()/den[i].sum())
    return s,np.percentile(bs,[2.5,97.5])

print(f"render common_test, {len(rows)} views, fitted to GROUND-TRUTH 2D")
print(f"{'shape fed to the solver':<40s} {'z-skill':>8s}  {'95% CI':>16s}")
print("  -- supervisable target: mesh-PCA coefficients, unit sd --")
for sh in (0.0,0.25,0.5,0.75,1.0,1.25):
    lab = "mean ear (coeffs = 0)" if sh==0 else f"{sh:.2f} x gt_coeffs (mesh PCA)"
    s,ci = skill(lambda r,sh=sh:(M+A@(Cgt[:,lut[r['ear']]]*sh)).reshape(K,3), boot=True)
    print(f"  {lab:<38s} {s:8.3f}  [{ci[0]:.3f}, {ci[1]:.3f}]")
print("  -- NOT supervisable: landmark lstsq, 3.2 sd coefficients --")
for sh in (0.5,1.0):
    s,ci = skill(lambda r,sh=sh:(M+A@(Cls[:,lut[r['ear']]]*sh)).reshape(K,3), boot=True)
    print(f"  {f'{sh:.2f} x landmark-lstsq coeffs':<38s} {s:8.3f}  [{ci[0]:.3f}, {ci[1]:.3f}]")
s,ci = skill(lambda r: X3[lut[r["ear"]]], boot=True)
print(f"  {'full true 3D (oracle, unreachable)':<38s} {s:8.3f}  [{ci[0]:.3f}, {ci[1]:.3f}]")
