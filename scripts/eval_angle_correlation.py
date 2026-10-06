"""Does a landmark's stability depend on the angle it is seen from?

Each ear is landmarked independently from a sweep of camera yaws and every result
is lifted to 3D through the depth buffer of its own render. A landmark's
consensus is the median of its positions across views; its deviation at a given
yaw is how far that view puts it from that consensus.

WHAT THIS MEASURES: precision, not accuracy. A landmark the model places in the
same wrong spot from every angle has zero deviation. It cannot say a position is
correct, only that the views concur -- which is still the only placement-adjacent
signal available here without human annotation.

A CONFOUND TO READ WITH: deviation is lowest at yaw 0, and 0 is the centre of the
sampled range, so it is also the view closest to the mean viewpoint. Part of that
dip is geometry of the sampling, not merit of the view. The asymmetry and the
per-landmark structure do not have that problem, which is why they are the more
interesting part of the output.

Writes a three-panel figure: every landmark against every view, the same by
strip, and the yaw each landmark is most stable at, drawn on the ear.

Usage:
    python scripts/eval_angle_correlation.py
"""

import io, contextlib, pickle, sys, tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import ear3d                                                   # noqa: E402
import skin as sk                                              # noqa: E402
from ear3d.config import DEFAULTS, EAR3D_DIR                   # noqa: E402
from eval_test import best_ckpt                                # noqa: E402
from inference import (BLAZEEAR_DIR, DETECTOR_WEIGHTS,         # noqa: E402
                       EarLandmarkerPipeline)

# derived from third-party 3D data, so it defaults outside the repo
PKL = Path(tempfile.gettempdir()) / "angle_correlation.pkl"
FIG = Path(tempfile.gettempdir()) / "angle_correlation.png"

pipe=EarLandmarkerPipeline(BLAZEEAR_DIR/DETECTOR_WEIGHTS,best_ckpt("manual_occ_s42"),
                           device="cuda",smooth=False)
YAWS=[-40,-30,-20,-10,0,10,20,30,40]
HEADS=["pp12","pp11","pp10","pp16","pp19","pp20"]
base=EAR3D_DIR/"3D head meshes"
dev=np.full((len(HEADS),len(YAWS),55),np.nan)
ref_lm=None
for hi,name in enumerate(HEADS):
    mp=str(base/f"{name}_3DheadMesh.ply")
    cfg=dict(DEFAULTS, mesh=mp)
    buf=io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            m=ear3d.load_head(mp); sk.apply_skin(m,seed=0,tone=cfg["tone"])
            pose=ear3d.head_pose(m,cfg)
            if pose is None: raise SystemExit("no face")
            lat,vert,tr_r,tr_l=pose
            g,_,_,lm0,P0,hit0,_=ear3d.label_two_pass(m,lat,vert,cfg,pipe,tr_l)
    except SystemExit as e:
        print(f"{name}: skipped ({e})"); continue
    if ref_lm is None: ref_lm=lm0
    P=np.full((len(YAWS),55,3),np.nan); ok=np.zeros((len(YAWS),55),bool)
    for yi,yaw in enumerate(YAWS):
        E=ear3d.orbit_extrinsic(yaw,0,cfg)
        img,depth,cam=ear3d.render(g,cfg,E=E)
        res=pipe(img,timestamp=0.0)
        if not res: continue
        b=max(res,key=lambda z:z["confidence"])
        Q,hit,_=ear3d.backproject_snapped(depth,np.asarray(b["landmarks"],float),cam,cfg)
        P[yi]=Q; ok[yi]=hit & np.isfinite(Q).all(1)
    ext=float(np.ptp(P0[hit0],axis=0).max()) if hit0.any() else 1.0
    for k in range(55):
        pts=P[ok[:,k],k]
        if len(pts)<3: continue
        c=np.median(pts,axis=0)
        for yi in range(len(YAWS)):
            if ok[yi,k]:
                dev[hi,yi,k]=100*np.linalg.norm(P[yi,k]-c)/ext
    n=int(ok.any(1).sum())
    print(f"{name}: {n}/{len(YAWS)} views landmarked")
pickle.dump(dict(dev=dev,YAWS=YAWS,HEADS=HEADS,ref_lm=ref_lm),open(PKL,"wb"))
md=np.nanmean(dev,axis=0)
print("\nmean deviation from consensus, % of ear extent, by yaw:")
print("  yaw " + " ".join(f"{y:>6d}" for y in YAWS))
for i,(a,b) in enumerate(ear3d.STRIPS):
    print(f"  {ear3d.STRIP_NAMES[i][:12]:<12s} " +
          " ".join(f"{np.nanmean(md[yi,a:b]):>5.1f}" for yi in range(len(YAWS))))


D=pickle.load(open(PKL,"rb"))
dev,YAWS,ref_lm=D["dev"],D["YAWS"],D["ref_lm"]
md=np.nanmean(dev,axis=0)                       # (yaw, 55)
best=np.array([YAWS[int(np.nanargmin(md[:,k]))] if np.isfinite(md[:,k]).any() else np.nan
               for k in range(55)])

# a face-on render to draw the landmarks on
cfg=dict(DEFAULTS, mesh=str(EAR3D_DIR/"3D head meshes"/"pp12_3DheadMesh.ply"))
m=ear3d.load_head(cfg["mesh"]); sk.apply_skin(m,seed=0,tone=cfg["tone"])
lat,vert,tr_r,tr_l=ear3d.head_pose(m,cfg)
g,img,cam,lm,P3,hit,_=ear3d.label_two_pass(m,lat,vert,cfg,pipe,tr_l)
x0,x1=lm[:,0].min()-40,lm[:,0].max()+40; y0,y1=lm[:,1].min()-40,lm[:,1].max()+40

fig=plt.figure(figsize=(17,8.5),facecolor="white")
gs=GridSpec(1,3,width_ratios=[1.35,1,1],wspace=0.22,left=0.05,right=0.97,top=0.9,bottom=0.1)

# --- 1. heatmap: every landmark against every view ---
ax=fig.add_subplot(gs[0])
im=ax.imshow(md.T,aspect="auto",cmap="magma_r",vmin=0,vmax=14,
             extent=[YAWS[0]-5,YAWS[-1]+5,54.5,-0.5])
ax.set_xlabel("camera yaw (deg)"); ax.set_ylabel("landmark")
ax.set_title("deviation from each landmark's cross-view consensus\n(% of ear extent, 6 heads)",
             fontsize=11)
for a,b in ear3d.STRIPS[:-1]:
    ax.axhline(b-0.5,color="white",lw=1.4)
for i,(a,b) in enumerate(ear3d.STRIPS):
    ax.text(YAWS[-1]+7,(a+b)/2-0.5,ear3d.STRIP_NAMES[i].replace("_","\n"),
            va="center",fontsize=8.5,color=[c for c in ear3d.STRIP_COLOURS[i]])
ax.set_xticks(YAWS)
fig.colorbar(im,ax=ax,fraction=0.035,pad=0.13,label="% of ear extent")

# --- 2. per-strip curves ---
ax=fig.add_subplot(gs[1])
for i,(a,b) in enumerate(ear3d.STRIPS):
    ax.plot(YAWS,np.nanmean(md[:,a:b],axis=1),"o-",color=ear3d.STRIP_COLOURS[i],
            label=ear3d.STRIP_NAMES[i],lw=2,ms=5)
ax.axvline(0,color="0.6",ls="--",lw=1)
ax.set_xlabel("camera yaw (deg)"); ax.set_ylabel("mean deviation (% of ear extent)")
ax.set_title("by strip",fontsize=11); ax.legend(fontsize=8.5); ax.grid(alpha=0.25)
ax.set_xticks(YAWS)

# --- 3. where on the ear each landmark is best seen from ---
ax=fig.add_subplot(gs[2])
ax.imshow(img); ax.set_xlim(x0,x1); ax.set_ylim(y1,y0); ax.axis("off")
sc=ax.scatter(lm[:,0],lm[:,1],c=best,cmap="coolwarm",vmin=-40,vmax=40,
              s=70,edgecolors="black",linewidths=0.6)
ax.set_title("yaw at which each landmark is most stable",fontsize=11)
fig.colorbar(sc,ax=ax,fraction=0.04,pad=0.02,label="best yaw (deg)")
fig.savefig(FIG,dpi=115,facecolor="white")
print(f"wrote {FIG}")
print(f"best-yaw distribution: {np.nanmin(best):.0f} to {np.nanmax(best):.0f}, "
      f"median {np.nanmedian(best):.0f}")
vals,counts=np.unique(best[np.isfinite(best)],return_counts=True)
print("  " + "  ".join(f"{int(v):+d}deg:{c}" for v,c in zip(vals,counts)))
