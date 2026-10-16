"""Look at what the gate choice actually costs and buys.

Every figure behind the profiles is detection-level: recall, precision, FP/image
at IoU 0.30. None of it says whether the boxes between 0.70 and 0.86 produce
USABLE LANDMARKS or merely usable IoU. So render each detection once, coloured
by which gate would keep it, with its real landmarks drawn:

  GREEN  true positive scoring >= 0.86  -- every profile keeps it
  YELLOW true positive scoring 0.70-0.86 -- `automated` DISCARDS this; the
         recall cost is only real if these landmarks are good
  RED    false positive scoring >= 0.86 -- survives even `automated`; the
         expensive error, a confident fit on a non-ear
  ORANGE false positive scoring 0.70-0.86 -- `automated` removes this; the
         benefit is only real if these are visibly wrong

Verdict to read off: are the YELLOW fits good (then a high gate throws away
working output) and are the ORANGE fits obviously wrong (then it buys real
quality)?
"""
from __future__ import annotations
import sys, argparse, collections
from pathlib import Path
import numpy as np, cv2, torch
ROOT=Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/Ear_Landmarker")
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"scripts"))
from inference import (EarLandmarkerPipeline, EarDetector, BLAZEEAR_DIR,
                       DETECTOR_WEIGHTS, find_best_checkpoint, PROJECT)
from score_face_arms import load_gt, iou_xyxy

LOW, HIGH = 0.70, 0.86
COLS={"keep":(80,220,80),"lost":(60,220,255),"fp_hi":(60,60,235),"fp_lo":(0,165,255)}
LBL={"keep":"kept by all","lost":"LOST at 0.86","fp_hi":"FP survives 0.86","fp_lo":"FP cut at 0.86"}

def draw(img, lm, box, kind, score):
    c=COLS[kind]
    x1,y1,x2,y2=[int(v) for v in box]
    cv2.rectangle(img,(x1,y1),(x2,y2),c,2)
    for (x,y) in lm.astype(int):
        cv2.circle(img,(int(x),int(y)),2,c,-1)
    t=f"{score:.2f} {LBL[kind]}"
    cv2.putText(img,t,(x1,max(12,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,0),3)
    cv2.putText(img,t,(x1,max(12,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,0.45,c,1)

def main(limit, seed, want, outdir, small_only):
    items=load_gt(); rng=np.random.default_rng(seed)
    sel=rng.choice(len(items),min(limit,len(items)),replace=False)
    items=[items[i] for i in sorted(sel)]
    pipe=EarLandmarkerPipeline(BLAZEEAR_DIR/DETECTOR_WEIGHTS,
                               find_best_checkpoint(), device="cuda",
                               detector_confidence=LOW)
    out=Path(outdir); out.mkdir(parents=True, exist_ok=True)
    tally=collections.Counter(); saved=0; rows=[]
    for path,gts in items:
        if saved>=want: break
        bgr=cv2.imread(str(path))
        if bgr is None: continue
        if small_only:
            # the band where the tilt lives: biggest GT ear under 48 px
            if max(max(g[2]-g[0],g[3]-g[1]) for g in gts) >= 48: continue
        res=pipe(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
        if not res: continue
        kinds=[]
        for r in res:
            # pipeline bbox is [ymin, xmin, ymax, xmax]; everything here is xyxy
            yn,xn,yx,xx=[float(v) for v in r["bbox"]]
            b=[xn,yn,xx,yx]; sc=float(r["confidence"])
            tp=any(iou_xyxy(g,b)>=0.3 for g in gts)
            k=("keep" if sc>=HIGH else "lost") if tp else ("fp_hi" if sc>=HIGH else "fp_lo")
            kinds.append((k,r,sc)); tally[k]+=1
        # keep only frames where the gate choice CHANGES something
        if not any(k in ("lost","fp_lo","fp_hi") for k,_,_ in kinds): continue
        vis=bgr.copy()
        for k,r,sc in kinds:
            yn,xn,yx,xx=[float(v) for v in r["bbox"]]
            draw(vis, np.asarray(r["landmarks"],dtype=np.float64),
                 [xn,yn,xx,yx], k, sc)
        h=max(1,vis.shape[0]); scale=min(1.0, 420/h)
        vis=cv2.resize(vis,None,fx=scale,fy=scale)
        f=out/f"{saved:02d}_{path.stem[:28]}.jpg"
        cv2.imwrite(str(f),vis,[cv2.IMWRITE_JPEG_QUALITY,92]); rows.append(f)
        saved+=1
    print("detections by what the gate does to them:")
    for k in ("keep","lost","fp_hi","fp_lo"):
        print(f"  {LBL[k]:>20}: {tally[k]}")
    # montage
    if rows:
        ims=[cv2.imread(str(f)) for f in rows]
        W=max(i.shape[1] for i in ims); H=max(i.shape[0] for i in ims)
        cols=3; r_=(len(ims)+cols-1)//cols
        sheet=np.full((r_*H, cols*W, 3), 30, np.uint8)
        for n,i in enumerate(ims):
            y,x=(n//cols)*H,(n%cols)*W
            sheet[y:y+i.shape[0], x:x+i.shape[1]]=i
        m=out/"montage.jpg"
        cv2.imwrite(str(m),sheet,[cv2.IMWRITE_JPEG_QUALITY,90])
        print(f"\nmontage -> {m}  ({len(ims)} frames)")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--limit",type=int,default=4000); ap.add_argument("--seed",type=int,default=9)
    ap.add_argument("--want",type=int,default=9); ap.add_argument("--out",default="vis")
    ap.add_argument("--small-only",action="store_true")
    a=ap.parse_args(); main(a.limit,a.seed,a.want,a.out,a.small_only)
