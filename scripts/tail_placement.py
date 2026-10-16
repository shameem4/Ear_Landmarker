"""Is the low-score tail's value PLACEMENT rather than facefulness?

trainable_blazeface showed their 0.10-0.20 detections are not unlabelled faces:
of 39,822 that miss their RetinaFace ground truth, YuNet confirms 19 as faces
(0.0%). So "your metric mislabels real faces" -- my proposed mechanism -- is
dead. Their follow-up question is the one only an ear ground truth can answer:
do those boxes CLUSTER on heads, or scatter?

Test: for each tuned detection in a score band, build the crop window the
pipeline would build (expand 1.5) and ask whether it covers a ground-truth ear.
Compare against a NULL that keeps each window's size and the image, but moves
its centre uniformly at random -- so the null carries the same size
distribution, the same per-image ear count, and the same image dimensions, and
the only thing destroyed is WHERE the box landed.
"""
from __future__ import annotations
import sys, csv, collections, copy, argparse
from pathlib import Path
import numpy as np, cv2, torch
ROOT=Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/Ear_Landmarker")
B=Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/BlazeEar")
TBF=Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/trainable_blazeface")
sys.path.insert(0,str(ROOT))
from inference import EarDetector, BLAZEEAR_DIR, DETECTOR_WEIGHTS

def frac_inside(e,c):
    ix1,iy1=max(e[0],c[0]),max(e[1],c[1]); ix2,iy2=min(e[2],c[2]),min(e[3],c[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1); a=(e[2]-e[0])*(e[3]-e[1])
    return inter/a if a>0 else 0.0

HUMAN=("GT","GT+EAR","GT+REVIEW")
def load_gt():
    by=collections.defaultdict(list)
    for r in csv.DictReader(open(B/"data/splits/master.csv")):
        if r["annotation_source"] not in HUMAN: continue
        x,y,w,h=float(r["x1"]),float(r["y1"]),float(r["w"]),float(r["h"])
        asp=w/max(h,1e-6)
        if w<6 or h<6 or w*h<100 or asp<0.15 or asp>2.0: continue
        by[r["image_path"]].append([x,y,x+w,y+h])
    idx={}
    for q in (B/"data/yolo11_ears_human/images").rglob("*"):
        if q.is_file(): idx[q.name]=q
    out=[]
    for ip,boxes in by.items():
        c=idx.get(Path(ip).name)
        if c: out.append((c,boxes))
    return out

BANDS=[(0.10,0.20),(0.20,0.30),(0.30,0.50),(0.50,1.01)]
def main(limit, seed, expand):
    items=load_gt(); rng=np.random.default_rng(seed)
    sel=rng.choice(len(items),min(limit,len(items)),replace=False)
    items=[items[i] for i in sorted(sel)]
    det=EarDetector(BLAZEEAR_DIR/DETECTOR_WEIGHTS,"cuda",None)
    stock=det.face; tuned=copy.deepcopy(stock)
    tuned.load_state_dict(torch.load(TBF/"runs/ckpt_peranchor/BlazeFace_final.pth",
                                     map_location="cpu")["model_state_dict"], strict=True)
    tuned.eval(); tuned.min_score_thresh=0.10
    # per band: crops, crops covering >=1 ear, null covering, ears covered ONLY
    # by this band's crops
    st={b:dict(n=0,cov=0,null=0) for b in BANDS}
    uniq={b:0 for b in BANDS}
    n_img=n_gt=0
    for path,gts in items:
        bgr=cv2.imread(str(path))
        if bgr is None: continue
        frame=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB); H,W=frame.shape[:2]
        n_img+=1; n_gt+=len(gts)
        f=tuned.process(frame)
        if isinstance(f,torch.Tensor): f=f.cpu().numpy()
        f=np.atleast_2d(np.asarray(f)) if np.size(f) else np.zeros((0,5))
        wins={b:[] for b in BANDS}
        for d in f:
            s=float(d[4])
            bd=next((b for b in BANDS if b[0]<=s<b[1]), None)
            if bd is None: continue
            x0,y0,side=det._crop_window(d,expand,frame.shape)
            wins[bd].append((x0,y0,side))
        for bd,ws in wins.items():
            for (x0,y0,side) in ws:
                c=[x0,y0,x0+side,y0+side]
                st[bd]["n"]+=1
                if any(frac_inside(g,c)>=0.5 for g in gts): st[bd]["cov"]+=1
                # null: same size, same image, random centre
                rx=rng.integers(0,max(1,W-side+1)); ry=rng.integers(0,max(1,H-side+1))
                cn=[int(rx),int(ry),int(rx)+side,int(ry)+side]
                if any(frac_inside(g,cn)>=0.5 for g in gts): st[bd]["null"]+=1
        # ears covered ONLY by a given band
        for g in gts:
            covering=[bd for bd in BANDS
                      if any(frac_inside(g,[x0,y0,x0+s,y0+s])>=0.5 for (x0,y0,s) in wins[bd])]
            if len(covering)==1: uniq[covering[0]]+=1
    print(f"images {n_img}, ears {n_gt}, expand {expand}, tuned face thr 0.10\n")
    print(f"{'score band':>12} {'crops':>8} {'cover an ear':>13} {'null':>9} "
          f"{'lift':>7} {'ears only this band':>20}")
    from scipy import stats as sst
    for bd in BANDS:
        s=st[bd]
        if not s["n"]: continue
        p=s["cov"]/s["n"]; q=s["null"]/s["n"]
        lift=p/q if q>0 else float('inf')
        print(f"{bd[0]:>5.2f}-{bd[1]:<6.2f} {s['n']:>8} {100*p:>12.2f}% {100*q:>8.2f}% "
              f"{lift:>7.1f}x {uniq[bd]:>20}")
    tot=sum(st[b]["n"] for b in BANDS)
    tail=st[BANDS[0]]
    print(f"\nthe 0.10-0.20 tail is {100*tail['n']/max(tot,1):.1f}% of all crops")
    b=sst.binomtest(tail["cov"], tail["n"], tail["null"]/tail["n"]).pvalue if tail["n"] else 1
    print(f"tail coverage {100*tail['cov']/max(tail['n'],1):.2f}% vs null "
          f"{100*tail['null']/max(tail['n'],1):.2f}%  p={b:.3e}")
    print(f"ears reachable ONLY via a tail crop: {uniq[BANDS[0]]} "
          f"({100*uniq[BANDS[0]]/max(n_gt,1):.2f}% of all ears)")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--limit",type=int,default=3000)
    ap.add_argument("--seed",type=int,default=11); ap.add_argument("--expand",type=float,default=1.5)
    a=ap.parse_args(); main(a.limit,a.seed,a.expand)
