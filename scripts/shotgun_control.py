"""The control the tail result demands: does a DUMB shotgun match the detector?

The 0.10-0.20 tail covers ears at 12.72% against a random-placement null of
12.81% (lift 1.0x, p=0.34) -- its placement carries no information. But it is
85.9% of all crops, ~38 per image, and uniquely delivers 148 ears. That says
the coverage gain may be VOLUME, not model quality.

If so, a shotgun with no detector at all -- crop windows at the same size
distribution and the same count per image, placed uniformly at random -- should
match tuned@0.10 on both coverage and delivered recall. If the shotgun matches,
the tuned detector's coverage advantage is not about faces, heads, or anything
it learned; it is about handing stage 2 more windows, and the same could be
bought without it.
"""
from __future__ import annotations
import sys, copy, argparse
from pathlib import Path
import numpy as np, cv2, torch
ROOT=Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/Ear_Landmarker")
TBF=Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/trainable_blazeface")
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"scripts"))
from inference import EarDetector, EarLandmarkerPipeline, BLAZEEAR_DIR, DETECTOR_WEIGHTS, _as_boxes
from score_face_arms import load_gt, iou_xyxy, frac_inside, run_arm, preds_at, BANDS, band

class ShotgunLocal:
    """Destroys PLACEMENT only, keeping each image's own box sizes.

    The per-crop placement null returned lift 1.0x, but it preserved each box's
    size in its own image -- and that size encodes head scale, so the null was
    leaking the very information under test. This arm keeps the real per-image
    size multiset and randomises only the centres, which is the clean
    decomposition: if it matches the detector, placement carries nothing; if it
    collapses, placement is informative and the earlier null was too generous.
    """
    def __init__(self, face_model, seed):
        self.fm=face_model; self.rng=np.random.default_rng(seed)
        self.min_score_thresh=0.0
    def process(self, frame):
        self.fm.min_score_thresh=0.10
        f=self.fm.process(frame)
        if isinstance(f,torch.Tensor): f=f.cpu().numpy()
        f=np.atleast_2d(np.asarray(f)) if np.size(f) else np.zeros((0,5))
        if not len(f): return f
        H,W=frame.shape[:2]
        out=f.copy()
        for i,d in enumerate(f):
            side=max(d[3]-d[1],d[2]-d[0])
            cx=self.rng.uniform(0,W); cy=self.rng.uniform(0,H)
            out[i,0]=cy-side/2; out[i,1]=cx-side/2
            out[i,2]=cy+side/2; out[i,3]=cx+side/2
            out[i,4]=1.0
        return out


class Shotgun:
    """Stands in for a face detector: same interface, no learning.

    Emits `count` boxes per image, sizes drawn from the empirical tuned size
    distribution, centres uniform over the frame. `min_score_thresh` exists
    only so run_arm can set it.
    """
    def __init__(self, sizes, count, seed):
        self.sizes=np.asarray(sizes,dtype=np.float64)
        self.count=count; self.rng=np.random.default_rng(seed)
        self.min_score_thresh=0.0
    def process(self, frame):
        H,W=frame.shape[:2]
        s=self.rng.choice(self.sizes,self.count)
        s=np.clip(s,8,min(H,W))
        cx=self.rng.uniform(0,W,self.count); cy=self.rng.uniform(0,H,self.count)
        out=np.zeros((self.count,5))
        out[:,0]=cy-s/2; out[:,1]=cx-s/2; out[:,2]=cy+s/2; out[:,3]=cx+s/2
        out[:,4]=1.0
        return out

def main(limit, seed, thrs):
    items=load_gt()
    rng=np.random.default_rng(seed)
    sel=rng.choice(len(items),min(limit,len(items)),replace=False)
    items=[items[i] for i in sorted(sel)]
    det=EarDetector(BLAZEEAR_DIR/DETECTOR_WEIGHTS,"cuda",None)
    stock=det.face; tuned=copy.deepcopy(stock)
    tuned.load_state_dict(torch.load(TBF/"runs/ckpt_peranchor/BlazeFace_final.pth",
                                     map_location="cpu")["model_state_dict"], strict=True)
    tuned.eval()
    # calibrate the shotgun to the tuned arm's own crop count and size spread
    tuned.min_score_thresh=0.10
    sizes=[]; counts=[]
    for path,_ in items[:400]:
        bgr=cv2.imread(str(path))
        if bgr is None: continue
        frame=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        f=tuned.process(frame)
        if isinstance(f,torch.Tensor): f=f.cpu().numpy()
        f=np.atleast_2d(np.asarray(f)) if np.size(f) else np.zeros((0,5))
        counts.append(len(f))
        for d in f: sizes.append(max(d[3]-d[1],d[2]-d[0]))
    cnt=int(round(np.mean(counts)))
    print(f"shotgun calibrated to tuned@0.10: {cnt} boxes/image, "
          f"size median {np.median(sizes):.1f} px (n={len(sizes)})")
    ARMS={"stock":(stock,0.20),"tuned":(tuned,0.10),
          "shotgun":(Shotgun(sizes,cnt,seed+1),0.0),
          "shot_local":(ShotgunLocal(copy.deepcopy(tuned),seed+2),0.0)}
    S={a:{t:dict(tp=0,fp=0,nd=0) for t in thrs} for a in ARMS}
    bandc={a:{b:dict(hit=0,n=0,facemiss=0) for b in BANDS} for a in ARMS}
    n_gt=n_img=0
    thr_ship=float(det.model.min_score_thresh)
    for path,gts in items:
        bgr=cv2.imread(str(path))
        if bgr is None: continue
        frame=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        n_img+=1; n_gt+=len(gts)
        res={a:run_arm(det,fm,frame,thrs,1.5,ft) for a,(fm,ft) in ARMS.items()}
        for a,(allb,crops) in res.items():
            for t in thrs:
                pred=preds_at(allb,t); S[a][t]["nd"]+=len(pred); used=set()
                for g in gts:
                    bi,bv=-1,0.0
                    for i,q in enumerate(pred):
                        if i in used: continue
                        v=iou_xyxy(g,q)
                        if v>=0.3 and v>bv: bi,bv=i,v
                    if bi>=0: used.add(bi); S[a][t]["tp"]+=1
                S[a][t]["fp"]+=len(pred)-len(used)
            sh=preds_at(allb,thr_ship)
            for g in gts:
                bd=band(max(g[2]-g[0],g[3]-g[1])); bc=bandc[a][bd]
                bc["n"]+=1
                bc["hit"]+=int(any(iou_xyxy(g,q)>=0.3 for q in sh))
                if not any(frac_inside(g,[c[0],c[1],c[0]+c[2],c[1]+c[2]])>=0.5 for c in crops):
                    bc["facemiss"]+=1
    print(f"\nimages {n_img}, ears {n_gt}\n")
    print(f"{'arm':>9} {'crops/img':>10} {'facemiss':>9} {'recall@0.70':>12}")
    for a in ARMS:
        fm=sum(bandc[a][b]["facemiss"] for b in BANDS)
        hit=sum(bandc[a][b]["hit"] for b in BANDS)
        print(f"{a:>9} {'-':>10} {fm:>9} {100*hit/n_gt:>11.2f}%")
    print(f"\nface-stage misses by band:")
    print(f"{'arm':>9} " + " ".join(f"{(str(l)+'-'+(str(h) if h<10**9 else '+')):>9}" for l,h in BANDS))
    for a in ARMS:
        print(f"{a:>9} " + " ".join(f"{bandc[a][b]['facemiss']:>9}" for b in BANDS))
    print(f"\n{'thresh':>7} " + " ".join(f"| {a+' rec':>12} {'prec':>7}" for a in ARMS))
    for t in thrs:
        row=""
        for a in ARMS:
            s=S[a][t]
            row+=f"| {100*s['tp']/n_gt:>11.2f}% {100*s['tp']/max(s['nd'],1):>6.2f}% "
        print(f"{t:>7.2f} {row}")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--limit",type=int,default=3000)
    ap.add_argument("--seed",type=int,default=11); a=ap.parse_args()
    main(a.limit,a.seed,[0.90,0.86,0.82,0.78,0.74,0.70])
