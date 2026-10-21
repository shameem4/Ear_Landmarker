# Promoted from scratch (2026-10-15-face-nms/scale_curve.py) on 2026-10-20 when scratch was pruned;
# cited by the docs as the harness behind the stated numbers.
"""Is stage 2 bad at SMALL scale, or miscalibrated as a function of scale?

BlazeEar's render-side investigation found confidence degrading in BOTH
directions away from the training median (ear/frame 0.253). My photograph side
only ever compared two points, 68.0px hits against 45.4px misses, which cannot
tell a monotone small-is-hard effect from a peaked one. This bins recall by ear
size inside the 256px crop and looks at the whole curve, including the large end.
"""
import sys, csv, collections, argparse
from pathlib import Path
import numpy as np, cv2, torch
ROOT=Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/Ear_Landmarker"); B=Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/BlazeEar")
sys.path.insert(0,str(ROOT))
from inference import (EarDetector, EarLandmarkerPipeline, BLAZEEAR_DIR, DETECTOR_WEIGHTS, _as_boxes)
HUMAN=("GT","GT+EAR","GT+REVIEW")
def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.0
def frac_inside(e,c):
    ix1,iy1=max(e[0],c[0]),max(e[1],c[1]); ix2,iy2=min(e[2],c[2]),min(e[3],c[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1); a=(e[2]-e[0])*(e[3]-e[1])
    return inter/a if a>0 else 0.0
def main(limit,seed):
    by=collections.defaultdict(list)
    for r in csv.DictReader(open(B/"data/splits/master.csv")):
        if r["annotation_source"] not in HUMAN: continue
        x,y,w,h=float(r["x1"]),float(r["y1"]),float(r["w"]),float(r["h"])
        asp=w/max(h,1e-6)
        if w<6 or h<6 or w*h<100 or asp<0.15 or asp>2.0: continue
        by[r["image_path"]].append([x,y,x+w,y+h])
    items=[(B/"data/raw"/ip, bx) for ip,bx in by.items() if (B/"data/raw"/ip).exists()]
    rng=np.random.default_rng(seed); sel=rng.choice(len(items),min(limit,len(items)),replace=False)
    items=[items[i] for i in sorted(sel)]
    det=EarDetector(BLAZEEAR_DIR/DETECTOR_WEIGHTS,"cuda",None)
    recs=[]; n_total=0; n_nocrop=0; n_noface=0; n_outside=0
    for path,gts in items:
        bgr=cv2.imread(str(path))
        if bgr is None: continue
        frame=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        faces=det.face.process(frame)
        if isinstance(faces,torch.Tensor): faces=faces.cpu().numpy()
        faces=np.atleast_2d(np.asarray(faces)) if np.size(faces) else np.zeros((0,17))
        crops,out=[],[]
        det.model.min_score_thresh=0.01
        for face in faces[:det.max_faces]:
            x0,y0,side=det._crop_window(face,det.expand,frame.shape)
            crops.append((x0,y0,side))
            patch=frame[y0:y0+side,x0:x0+side]
            if patch.size==0: continue
            d=_as_boxes(det.model.process(patch))
            if len(d):
                d=d.copy(); d[:,:4]+=np.array([y0,x0,y0,x0],dtype=np.float64); out.append(d)
        det.model.min_score_thresh=0.70
        allb=np.concatenate(out,0) if out else np.zeros((0,5))
        keep=allb[allb[:,4]>=0.70] if len(allb) else allb
        pred=[[d[1],d[0],d[3],d[2]] for d in EarLandmarkerPipeline._nms(keep)]
        for g in gts:
            n_total+=1
            cov=[c for c in crops if frac_inside(g,[c[0],c[1],c[0]+c[2],c[1]+c[2]])>=0.5]
            if not cov:
                n_nocrop+=1
                # split the way BlazeEar's README does: no face detected at all,
                # versus a face found whose window did not reach the ear
                if len(crops)==0: n_noface+=1
                else: n_outside+=1
                continue
            x0,y0,side=cov[0]; size=max(g[2]-g[0],g[3]-g[1])*(256.0/side)
            hit=any(iou(g,q)>=0.3 for q in pred)
            best=0.0
            for b in allb:
                if iou(g,[b[1],b[0],b[3],b[2]])>=0.3: best=max(best,float(b[4]))
            faceside=side/det.expand          # the detector box before expansion
            recs.append((size,hit,best,max(g[2]-g[0],g[3]-g[1])/max(faceside,1e-6)))
    a=np.array([r[0] for r in recs]); h=np.array([r[1] for r in recs])
    sc=np.array([r[2] for r in recs]); ef=np.array([r[3] for r in recs])
    print(f"ground-truth ears {n_total}; delivered inside a crop {len(a)}; "
          f"face-stage misses {n_nocrop}")
    print(f"FACE-STAGE CEILING = {n_nocrop}/{n_total} = {100*n_nocrop/n_total:.2f}% of all ears"
          f"   (SAME sample, no cross-run arithmetic)")
    print(f"   of which  no face detected at all : {n_noface:4d}  {100*n_noface/n_total:.2f}%")
    print(f"             face found, ear outside : {n_outside:4d}  {100*n_outside/n_total:.2f}%\n")
    edges=[0,32,48,64,80,96,120,160,1e9]
    print(f"{'size band':>12} {'n':>5} {'recall':>8} {'misses':>7} "
          f"{'never-proposed':>15} {'under-scored':>13} {'ear/facebox':>12}")
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(a>=lo)&(a<hi)
        if m.sum()<15: continue
        miss=m&~h
        nm=miss.sum()
        npz=int((sc[miss]==0).sum()); und=nm-npz
        lab=f"{lo}-{hi if hi<1e9 else ''}"
        print(f"{lab:>12} {m.sum():>5} {100*h[m].mean():>7.1f}% {nm:>7} "
              f"{npz:>6} ({100*npz/max(nm,1):>4.0f}%) {und:>5} ({100*und/max(nm,1):>4.0f}%) "
              f"{np.median(ef[m]):>12.2f}")
    npm=(~h)&(sc==0)
    und=(~h)&(sc>0)
    print(f"\nPOOLED: delivered-in-crop {len(a)}, misses {int((~h).sum())}, "
          f"never-proposed {int(npm.sum())} ({100*npm.sum()/max((~h).sum(),1):.1f}% of misses), "
          f"under-scored {int(und.sum())}")
    if npm.sum():
        z=a[npm]
        print(f"NEVER-PROPOSED sizes (px in 256 crop): n={len(z)}  p10 {np.percentile(z,10):.1f}  "
              f"p25 {np.percentile(z,25):.1f}  median {np.median(z):.1f}  p75 {np.percentile(z,75):.1f}")
        print(f"  under 24px {100*(z<24).mean():.1f}%   under 32px {100*(z<32).mean():.1f}%   "
              f"under 48px {100*(z<48).mean():.1f}%")
    if und.sum():
        z=a[und]
        print(f"UNDER-SCORED sizes: n={len(z)}  median {np.median(z):.1f}  "
              f"under 32px {100*(z<32).mean():.1f}%   under 48px {100*(z<48).mean():.1f}%")
    print(f"\near/frame equivalent of the training median 0.253 -> {0.253*256:.0f}px in a 256 crop")
    print(f"observed ear sizes: p10 {np.percentile(a,10):.0f}  median {np.median(a):.0f}  p90 {np.percentile(a,90):.0f}")
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--limit",type=int,default=1500)
    ap.add_argument("--seed",type=int,default=4); a=ap.parse_args()
    main(a.limit,a.seed)
