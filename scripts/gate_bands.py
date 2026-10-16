"""Which ears does raising DETECTOR_CONFIDENCE cost? Band recall per gate.

F1 over the whole corpus says 0.70 and 0.80 are within 0.6% of each other, so
the constant is not a defect. But a precision-weighted objective puts the
optimum near 0.86, and the question that decides whether that is acceptable is
WHICH ears are lost -- a uniform loss is a trade, a loss concentrated in small
ears is a systematic bias against subjects whose ears are small in frame.
"""
from __future__ import annotations
import sys, argparse, collections
from pathlib import Path
import numpy as np, cv2, torch
ROOT=Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/Ear_Landmarker")
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"scripts"))
from inference import EarDetector, BLAZEEAR_DIR, DETECTOR_WEIGHTS, _as_boxes
from score_face_arms import load_gt, iou_xyxy, preds_at, band, BANDS

GATES=[0.70,0.76,0.80,0.86]
def main(limit, seed):
    items=load_gt(); rng=np.random.default_rng(seed)
    sel=rng.choice(len(items),min(limit,len(items)),replace=False)
    items=[items[i] for i in sorted(sel)]
    det=EarDetector(BLAZEEAR_DIR/DETECTOR_WEIGHTS,"cuda",None)
    bc={g:collections.defaultdict(lambda: dict(n=0,hit=0)) for g in GATES}
    S={g:dict(tp=0,nd=0,fp=0) for g in GATES}
    n_img=n_gt=0
    for path,gts in items:
        bgr=cv2.imread(str(path))
        if bgr is None: continue
        frame=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB); n_img+=1; n_gt+=len(gts)
        faces=det.face.process(frame)
        if isinstance(faces,torch.Tensor): faces=faces.cpu().numpy()
        faces=np.atleast_2d(np.asarray(faces)) if np.size(faces) else np.zeros((0,5))
        out=[]; det.model.min_score_thresh=0.01
        for f in faces[:det.max_faces]:
            x0,y0,side=det._crop_window(f,det.expand,frame.shape)
            patch=frame[y0:y0+side,x0:x0+side]
            if patch.size==0: continue
            d=_as_boxes(det.model.process(patch))
            if len(d):
                d=d.copy(); d[:,:4]+=np.array([y0,x0,y0,x0],dtype=np.float64); out.append(d)
        det.model.min_score_thresh=0.70
        allb=np.concatenate(out,0) if out else np.zeros((0,5))
        for g in GATES:
            pred=preds_at(allb,g); S[g]["nd"]+=len(pred); used=set()
            for gt in gts:
                bd=band(max(gt[2]-gt[0],gt[3]-gt[1])); bc[g][bd]["n"]+=1
                bi,bv=-1,0.0
                for i,q in enumerate(pred):
                    if i in used: continue
                    v=iou_xyxy(gt,q)
                    if v>=0.3 and v>bv: bi,bv=i,v
                if bi>=0:
                    used.add(bi); S[g]["tp"]+=1; bc[g][bd]["hit"]+=1
            S[g]["fp"]+=len(pred)-len(used)
    print(f"images {n_img}, ears {n_gt}\n")
    hdr=" ".join(f"{g:>14.2f}" for g in GATES)
    print(f"{'band (px)':>11} {'ears':>6} {hdr}")
    for lo,hi in BANDS:
        bd=(lo,hi); n=bc[GATES[0]][bd]["n"]
        if not n: continue
        nm=f"{lo}-{hi}" if hi<10**9 else f"{lo}+"
        cells=[]
        base=100*bc[GATES[0]][bd]["hit"]/n
        for g in GATES:
            r=100*bc[g][bd]["hit"]/n
            cells.append(f"{r:>7.2f}% {r-base:>+6.2f}")
        print(f"{nm:>11} {n:>6} " + " ".join(cells))
    print(f"\n{'gate':>6} {'recall':>8} {'prec':>8} {'FP/img':>8}")
    for g in GATES:
        s=S[g]
        print(f"{g:>6.2f} {100*s['tp']/n_gt:>7.2f}% {100*s['tp']/max(s['nd'],1):>7.2f}% {s['fp']/n_img:>8.3f}")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--limit",type=int,default=6000)
    ap.add_argument("--seed",type=int,default=5); a=ap.parse_args(); main(a.limit,a.seed)
