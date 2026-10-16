"""Measure tuned/stock face-box scale ON OUR CORPUS, not theirs.

trainable_blazeface measured linear 1.117 on their val split. The crop side in
make_face_crops.py:66 is `expand * max(box_w, box_h)`, so this ratio multiplies
FACE_CROP_EXPAND directly -- but a box-convention ratio measured on one image
distribution is not automatically the ratio on another, and their corpus is not
mine. Pair faces both detectors find at IoU>=0.5 and take the ratio of the
quantity the crop actually uses: max(w,h).
"""
from __future__ import annotations
import sys, csv, collections, copy
from pathlib import Path
import numpy as np, cv2, torch
ROOT=Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/Ear_Landmarker")
B=Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/BlazeEar")
TBF=Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/trainable_blazeface")
sys.path.insert(0,str(ROOT))
from inference import EarDetector, BLAZEEAR_DIR, DETECTOR_WEIGHTS

def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.0

HUMAN=("GT","GT+EAR","GT+REVIEW")
def images():
    seen=set(); 
    for r in csv.DictReader(open(B/"data/splits/master.csv")):
        if r["annotation_source"] in HUMAN: seen.add(Path(r["image_path"]).name)
    idx={}
    for q in (B/"data/yolo11_ears_human/images").rglob("*"):
        if q.is_file() and q.name in seen: idx[q.name]=q
    return sorted(idx.values())

def faces(fm, frame):
    f=fm.process(frame)
    if isinstance(f,torch.Tensor): f=f.cpu().numpy()
    f=np.atleast_2d(np.asarray(f)) if np.size(f) else np.zeros((0,17))
    h,w=frame.shape[:2]
    return [[float(b[1])*1.0,float(b[0])*1.0,float(b[3])*1.0,float(b[2])*1.0] for b in f]

def main(limit, seed):
    items=images(); rng=np.random.default_rng(seed)
    sel=rng.choice(len(items),min(limit,len(items)),replace=False)
    items=[items[i] for i in sorted(sel)]
    det=EarDetector(BLAZEEAR_DIR/DETECTOR_WEIGHTS,"cuda",None)
    stock=det.face; tuned=copy.deepcopy(stock)
    tuned.load_state_dict(torch.load(TBF/"runs/ckpt_peranchor/BlazeFace_final.pth",
                                     map_location="cpu")["model_state_dict"], strict=True)
    tuned.eval()
    rs=[]; ra=[]; dy=[]; ns=nt=0
    for p in items:
        bgr=cv2.imread(str(p))
        if bgr is None: continue
        frame=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        S=faces(stock,frame); T=faces(tuned,frame)
        ns+=len(S); nt+=len(T)
        used=set()
        for s in S:
            bi,bv=-1,0.0
            for i,t in enumerate(T):
                if i in used: continue
                v=iou(s,t)
                if v>=0.5 and v>bv: bi,bv=i,v
            if bi<0: continue
            used.add(bi); t=T[bi]
            ss=max(s[2]-s[0],s[3]-s[1]); st=max(t[2]-t[0],t[3]-t[1])
            if ss<=0: continue
            rs.append(st/ss)
            ra.append(((t[2]-t[0])*(t[3]-t[1]))/max((s[2]-s[0])*(s[3]-s[1]),1e-9))
            dy.append(((t[1]+t[3])/2-(s[1]+s[3])/2)/max(s[3]-s[1],1e-9))
    r=np.array(rs); a=np.array(ra); d=np.array(dy)
    print(f"images {len(items)}, stock faces {ns}, tuned faces {nt}, paired {len(r)}")
    print(f"\nmax(w,h) ratio tuned/stock  -- the quantity the crop side uses")
    print(f"  median {np.median(r):.4f}   mean {r.mean():.4f}   p10 {np.percentile(r,10):.4f}"
          f"  p90 {np.percentile(r,90):.4f}")
    print(f"  bootstrap 95% CI on the median: ", end="")
    bs=[np.median(np.random.default_rng(k).choice(r,len(r))) for k in range(400)]
    print(f"[{np.percentile(bs,2.5):.4f}, {np.percentile(bs,97.5):.4f}]")
    print(f"area ratio   median {np.median(a):.4f}")
    print(f"centre dy/h  median {np.median(d):+.4f}")
    print(f"\ntheir linear 1.117 -> expand {1.5/1.117:.4f}")
    print(f"ours  {np.median(r):.4f} -> expand {1.5/np.median(r):.4f}")

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--limit",type=int,default=1500)
    ap.add_argument("--seed",type=int,default=7); a=ap.parse_args()
    main(a.limit,a.seed)
