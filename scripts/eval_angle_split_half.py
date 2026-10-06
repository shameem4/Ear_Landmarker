"""Does a per-landmark best viewing angle generalise, or is it noise?

eval_angle_correlation.py shows each landmark has a yaw at which it is most
stable, and the map of those yaws looks structured -- the helix rim favouring one
direction, inner points the other. This asks whether that structure survives
being chosen on one set of heads and applied to another.

Marginally, and not enough to act on. Over 70 splits of 8 heads:

    face-on always                            3.15%   (sd 0.38)
    best angle per landmark, chosen on TRAIN  3.06%   (sd 0.27)
    oracle, chosen on the TEST heads          1.94%

Selection wins by 0.09 points, about 3% relative, on 47 of 70 splits. The oracle
is far better than either, so most of what a per-landmark angle could buy is not
reachable by choosing it from four subjects.

AN EARLIER VERSION OF THIS FILE CLAIMED THE OPPOSITE -- that selection was clearly
worse, on 67 of 70 splits. That was measured through the 180 degree camera roll
fixed in aab7e38, which rendered every orbited view upside down. On the corrected
camera every figure roughly halves and the sign flips. Treat any multi-view number
predating that fix as void.

The evaluation uses a LEAVE-ONE-OUT consensus: a view is scored against the
median of the OTHER views, never one it helped define. Without that, every view
is partly scored against itself.

Same caveat as the correlation it tests: this is precision, not accuracy.

Usage:
    python scripts/eval_angle_split_half.py
"""

import contextlib
import io
import itertools
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import ear3d, skin as sk
from ear3d.config import DEFAULTS, EAR3D_DIR
from inference import BLAZEEAR_DIR, DETECTOR_WEIGHTS, EarLandmarkerPipeline
from eval_test import best_ckpt
# derived from third-party 3D data, so it caches outside the repo
PKL = Path(tempfile.gettempdir()) / "angle_split_half.pkl"
YAWS=[-40,-30,-20,-10,0,10,20,30,40]
HEADS=["pp12","pp11","pp10","pp16","pp19","pp20","pp21","pp22"]
if not PKL.exists():
    pipe=EarLandmarkerPipeline(BLAZEEAR_DIR/DETECTOR_WEIGHTS,best_ckpt("manual_occ_s42"),
                               device="cuda",smooth=False)
    base=EAR3D_DIR/"3D head meshes"
    P=np.full((len(HEADS),len(YAWS),55,3),np.nan); OK=np.zeros((len(HEADS),len(YAWS),55),bool)
    EXT=np.full(len(HEADS),np.nan)
    for hi,name in enumerate(HEADS):
        mp=str(base/f"{name}_3DheadMesh.ply")
        if not Path(mp).exists(): print(f"{name}: missing"); continue
        cfg=dict(DEFAULTS, mesh=mp); buf=io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                m=ear3d.load_head(mp); sk.apply_skin(m,seed=0,tone=cfg["tone"])
                pose=ear3d.head_pose(m,cfg)
                if pose is None: raise SystemExit("no face")
                lat,vert,tr_r,tr_l=pose
                g,_,_,_,P0,hit0,_=ear3d.label_two_pass(m,lat,vert,cfg,pipe,tr_l)
        except SystemExit as e:
            print(f"{name}: skipped ({e})"); continue
        EXT[hi]=float(np.ptp(P0[hit0],axis=0).max()) if hit0.any() else np.nan
        for yi,yaw in enumerate(YAWS):
            E=ear3d.orbit_extrinsic(yaw,0,cfg)
            img,depth,cam=ear3d.render(g,cfg,E=E)
            res=pipe(img,timestamp=0.0)
            if not res: continue
            b=max(res,key=lambda z:z["confidence"])
            Q,hit,_=ear3d.backproject_snapped(depth,np.asarray(b["landmarks"],float),cam,cfg)
            P[hi,yi]=Q; OK[hi,yi]=hit & np.isfinite(Q).all(1)
        print(f"{name}: {int(OK[hi].any(1).sum())}/{len(YAWS)} views")
    pickle.dump(dict(P=P,OK=OK,EXT=EXT,YAWS=YAWS,HEADS=HEADS),open(PKL,"wb"))
D=pickle.load(open(PKL,"rb")); P,OK,EXT=D["P"],D["OK"],D["EXT"]
good=[i for i in range(len(HEADS)) if np.isfinite(EXT[i]) and OK[i].any()]
print(f"\n{len(good)} usable heads: {[HEADS[i] for i in good]}")

def loo_dev(hi):
    """Deviation of each view from the consensus of the OTHER views."""
    d=np.full((len(YAWS),55),np.nan)
    for k in range(55):
        for v in range(len(YAWS)):
            if not OK[hi,v,k]: continue
            others=[u for u in range(len(YAWS)) if u!=v and OK[hi,u,k]]
            if len(others)<3: continue
            c=np.median(P[hi,others,k],axis=0)
            d[v,k]=100*np.linalg.norm(P[hi,v,k]-c)/EXT[hi]
    return d
DEV={hi:loo_dev(hi) for hi in good}
Z=YAWS.index(0)
rows=[]
for train in itertools.combinations(good,len(good)//2):
    test=[h for h in good if h not in train]
    tr=np.nanmean(np.stack([DEV[h] for h in train]),axis=0)       # (yaw,55)
    te=np.nanmean(np.stack([DEV[h] for h in test]),axis=0)
    pick=np.array([int(np.nanargmin(tr[:,k])) if np.isfinite(tr[:,k]).any() else Z
                   for k in range(55)])
    sel=np.array([te[pick[k],k] for k in range(55)])              # chosen on TRAIN
    face=te[Z]                                                    # always face-on
    orac=np.nanmin(te,axis=0)                                     # chosen on TEST
    m=np.isfinite(sel)&np.isfinite(face)&np.isfinite(orac)
    rows.append((np.nanmean(face[m]),np.nanmean(sel[m]),np.nanmean(orac[m]),
                 int((pick!=Z).sum())))
a=np.array(rows)
print(f"\nsplit-half, {len(rows)} splits of {len(good)} heads "
      f"({len(good)//2} train / {len(good)-len(good)//2} test)")
print(f"  face-on always        {a[:,0].mean():.2f}%  (sd over splits {a[:,0].std():.2f})")
print(f"  best angle per landmark, CHOSEN ON TRAIN   {a[:,1].mean():.2f}%  "
      f"(sd {a[:,1].std():.2f})")
print(f"  oracle, chosen on TEST itself              {a[:,2].mean():.2f}%")
print(f"  landmarks given a non-zero angle: {a[:,3].mean():.0f} of 55")
diff=a[:,1]-a[:,0]
print(f"\n  selected minus face-on: {diff.mean():+.2f}% "
      f"(better on {(diff<0).sum()} of {len(diff)} splits)")
t,p=stats.ttest_rel(a[:,1],a[:,0])
print(f"  paired t over splits: t={t:+.2f}, p={p:.3f}  "
      f"(splits overlap, so this is indicative only)")
print(f"  oracle minus face-on:   {(a[:,2]-a[:,0]).mean():+.2f}%  <- the ceiling")
