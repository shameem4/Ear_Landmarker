"""What ARE the false positives? The precision argument depends on the answer.

The gate recommendation assumed a false positive is "55 landmarks drawn on a
cheek". The first visual check suggested otherwise: the FPs looked like real
ears the ground truth had not labelled -- the second ear on a frontal face,
which BlazeEar bounded as "precision may be mildly pessimistic" but which is
load-bearing here, because if an FP is a correct detection then the precision
that `automated` buys is partly illusory.

Crop every FP, tile them by score band, and count by eye.
"""
from __future__ import annotations
import sys, argparse, collections
from pathlib import Path
import numpy as np, cv2, torch
ROOT=Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/Ear_Landmarker")
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"scripts"))
from inference import (EarLandmarkerPipeline, BLAZEEAR_DIR, DETECTOR_WEIGHTS,
                       find_best_checkpoint)
from score_face_arms import load_gt, iou_xyxy

def main(limit, seed, per_band, out):
    items=load_gt(); rng=np.random.default_rng(seed)
    sel=rng.choice(len(items),min(limit,len(items)),replace=False)
    items=[items[i] for i in sorted(sel)]
    pipe=EarLandmarkerPipeline(BLAZEEAR_DIR/DETECTOR_WEIGHTS, find_best_checkpoint(),
                               device="cuda", detector_confidence=0.70)
    bands={"hi (>=0.86, survives automated)":[], "lo (0.70-0.86, cut at 0.86)":[]}
    n_fp=collections.Counter(); seen=0
    for path,gts in items:
        if all(len(v)>=per_band for v in bands.values()): break
        bgr=cv2.imread(str(path))
        if bgr is None: continue
        seen+=1
        for r in pipe(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)):
            yn,xn,yx,xx=[float(v) for v in r["bbox"]]; sc=float(r["confidence"])
            if any(iou_xyxy(g,[xn,yn,xx,yx])>=0.3 for g in gts): continue
            key="hi (>=0.86, survives automated)" if sc>=0.86 else "lo (0.70-0.86, cut at 0.86)"
            n_fp[key]+=1
            if len(bands[key])>=per_band: continue
            pad=int(0.25*max(xx-xn,yx-yn))
            x0,y0=max(0,int(xn)-pad),max(0,int(yn)-pad)
            x1,y1=min(bgr.shape[1],int(xx)+pad),min(bgr.shape[0],int(yy if False else yx)+pad)
            crop=bgr[y0:y1,x0:x1]
            if crop.size==0: continue
            lm=np.asarray(r["landmarks"],dtype=np.float64)-np.array([x0,y0])
            c=crop.copy()
            for (x,y) in lm.astype(int): cv2.circle(c,(int(x),int(y)),2,(60,220,255),-1)
            c=cv2.resize(c,(176,176))
            cv2.putText(c,f"{sc:.2f}",(4,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),3)
            cv2.putText(c,f"{sc:.2f}",(4,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
            bands[key].append(c)
    o=Path(out); o.mkdir(parents=True,exist_ok=True)
    print(f"images scanned {seen}")
    for k,v in n_fp.items(): print(f"  {k}: {v} false positives")
    for k,ims in bands.items():
        if not ims: continue
        cols=8; r_=(len(ims)+cols-1)//cols
        sheet=np.full((r_*176, cols*176,3),30,np.uint8)
        for n,i in enumerate(ims):
            y,x=(n//cols)*176,(n%cols)*176; sheet[y:y+176,x:x+176]=i
        f=o/(("fp_hi" if k.startswith("hi") else "fp_lo")+".jpg")
        cv2.imwrite(str(f),sheet,[cv2.IMWRITE_JPEG_QUALITY,92])
        print(f"  -> {f}  ({len(ims)} crops)")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--limit",type=int,default=4000); ap.add_argument("--seed",type=int,default=21)
    ap.add_argument("--per-band",type=int,default=32); ap.add_argument("--out",default="vis_fp")
    a=ap.parse_args(); main(a.limit,a.seed,a.per_band,a.out)
