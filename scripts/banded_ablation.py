"""Re-report the placement ablation PER FACE-SIZE BAND.

BlazeEar's objection, from trainable_blazeface's confound: on head-filling
frames "lands on a head" and "lands on image content" are nearly the same event,
so an aggregate placement ablation is a mixture dominated by head-filling frames
and mostly measures composition. The discriminator is face size.

My first attempt at this was defective: it banded images by the TUNED model's
largest detection at threshold 0.10, and that arm emits ~35 boxes per image
including large spurious ones, so the max side is inflated and almost no frame
qualified as small-head (70 of 6000). Banding must come from a sober estimate of
head scale -- stock at its own 0.2 threshold -- not from the tail under test.
"""
from __future__ import annotations
import sys, copy, argparse, collections
from pathlib import Path
import numpy as np, cv2, torch
ROOT=Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/Ear_Landmarker")
TBF=Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/trainable_blazeface")
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"scripts"))
from inference import EarDetector, BLAZEEAR_DIR, DETECTOR_WEIGHTS
from score_face_arms import load_gt, iou_xyxy, frac_inside, run_arm
from shotgun_control import ShotgunLocal

FBANDS=[(0.00,0.20),(0.20,0.30),(0.30,0.45),(0.45,10.0)]
def fband(v):
    for lo,hi in FBANDS:
        if lo<=v<hi: return (lo,hi)
    return FBANDS[-1]

def main(limit, seed):
    items=load_gt(); rng=np.random.default_rng(seed)
    sel=rng.choice(len(items),min(limit,len(items)),replace=False)
    items=[items[i] for i in sorted(sel)]
    det=EarDetector(BLAZEEAR_DIR/DETECTOR_WEIGHTS,"cuda",None)
    stock=det.face; tuned=copy.deepcopy(stock)
    tuned.load_state_dict(torch.load(TBF/"runs/ckpt_peranchor/BlazeFace_final.pth",
                                     map_location="cpu")["model_state_dict"], strict=True)
    tuned.eval()
    shot=ShotgunLocal(copy.deepcopy(tuned), seed+2)
    ARMS={"stock":(stock,0.20),"tuned":(tuned,0.10),"shot_local":(shot,0.0)}
    st=collections.defaultdict(lambda: collections.defaultdict(lambda: dict(n=0,fm=0)))
    nimg=collections.Counter()
    for path,gts in items:
        bgr=cv2.imread(str(path))
        if bgr is None: continue
        frame=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB); H,W=frame.shape[:2]
        # head scale from STOCK at its own threshold, never from the tail
        stock.min_score_thresh=0.20
        f=stock.process(frame)
        if isinstance(f,torch.Tensor): f=f.cpu().numpy()
        f=np.atleast_2d(np.asarray(f)) if np.size(f) else np.zeros((0,5))
        scale=(max(max(d[3]-d[1],d[2]-d[0]) for d in f)/max(min(H,W),1)) if len(f) else 0.0
        fb=fband(scale); nimg[fb]+=1
        res={a:run_arm(det,fm,frame,[0.70],1.5,ft) for a,(fm,ft) in ARMS.items()}
        for a,(allb,crops) in res.items():
            for g in gts:
                st[a][fb]["n"]+=1
                if not any(frac_inside(g,[c[0],c[1],c[0]+c[2],c[1]+c[2]])>=0.5 for c in crops):
                    st[a][fb]["fm"]+=1
    print("head scale = stock's largest face side / frame short side, stock @0.20\n")
    print(f"{'band':>12} {'images':>7} {'ears':>6} "
          f"{'tuned cov':>10} {'stock cov':>10} {'null cov':>9} {'lift(t/null)':>13}")
    for fb in FBANDS:
        n=st["tuned"][fb]["n"]
        if not n: continue
        cov=lambda a: 100*(1-st[a][fb]["fm"]/st[a][fb]["n"])
        ct,cs,cn=cov("tuned"),cov("stock"),cov("shot_local")
        lift=(1-st["tuned"][fb]["fm"]/n)/max(1-st["shot_local"][fb]["fm"]/n,1e-9)
        nm=f"{fb[0]:.2f}-{fb[1]:.2f}" if fb[1]<10 else f"{fb[0]:.2f}+"
        print(f"{nm:>12} {nimg[fb]:>7} {n:>6} {ct:>9.1f}% {cs:>9.1f}% {cn:>8.1f}% {lift:>12.2f}x")
    tot=sum(st["tuned"][fb]["n"] for fb in FBANDS)
    f=lambda a: 100*(1-sum(st[a][fb]["fm"] for fb in FBANDS)/tot)
    lift=(f("tuned")/100)/max(f("shot_local")/100,1e-9)
    print(f"{'ALL':>12} {sum(nimg.values()):>7} {tot:>6} {f('tuned'):>9.1f}% "
          f"{f('stock'):>9.1f}% {f('shot_local'):>8.1f}% {lift:>12.2f}x")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--limit",type=int,default=4000)
    ap.add_argument("--seed",type=int,default=11); a=ap.parse_args(); main(a.limit,a.seed)
