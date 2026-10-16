"""Find the tuned face threshold that matches stock's CROP BUDGET.

FACE_CROP_THRESHOLD=0.2 is denominated in stock's score units. The tuned model
scores faces differently, so at the same 0.2 it proposes 3.3x more faces on our
corpus -- each extra crop is another chance for stage 2 to fire on non-ear
scene, which is where the precision loss comes from. Same class of error as the
expand constant, on the score axis instead of the geometry axis.
"""
from __future__ import annotations
import sys, csv, copy
from pathlib import Path
import numpy as np, cv2, torch
ROOT=Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/Ear_Landmarker")
B=Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/BlazeEar")
TBF=Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/trainable_blazeface")
sys.path.insert(0,str(ROOT))
from inference import EarDetector, BLAZEEAR_DIR, DETECTOR_WEIGHTS

HUMAN=("GT","GT+EAR","GT+REVIEW")
def images():
    seen=set()
    for r in csv.DictReader(open(B/"data/splits/master.csv")):
        if r["annotation_source"] in HUMAN: seen.add(Path(r["image_path"]).name)
    return sorted(q for q in (B/"data/yolo11_ears_human/images").rglob("*")
                  if q.is_file() and q.name in seen)

def scores(fm, frame):
    f=fm.process(frame)
    if isinstance(f,torch.Tensor): f=f.cpu().numpy()
    f=np.atleast_2d(np.asarray(f)) if np.size(f) else np.zeros((0,17))
    # output is (N,5): ymin,xmin,ymax,xmax,score -- verified against a real frame
    assert f.shape[1]==5 or not len(f), f"unexpected face output shape {f.shape}"
    return [float(b[4]) for b in f]

def main(limit, seed):
    items=images(); rng=np.random.default_rng(seed)
    sel=rng.choice(len(items),min(limit,len(items)),replace=False)
    items=[items[i] for i in sorted(sel)]
    det=EarDetector(BLAZEEAR_DIR/DETECTOR_WEIGHTS,"cuda",None)
    stock=det.face; tuned=copy.deepcopy(stock)
    tuned.load_state_dict(torch.load(TBF/"runs/ckpt_peranchor/BlazeFace_final.pth",
                                     map_location="cpu")["model_state_dict"], strict=True)
    tuned.eval()
    # the detector already filters at 0.2; collect what survives and sweep upward
    allS=[]; allT=[]; n=0
    for p in items:
        bgr=cv2.imread(str(p))
        if bgr is None: continue
        frame=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB); n+=1
        allS+=scores(stock,frame); allT+=scores(tuned,frame)
    S=np.array(allS); T=np.array(allT)
    cap=8
    def per_img(arr,t): return min(1.0,1.0)*0  # placeholder
    print(f"images {n}")
    print(f"stock @0.20: {len(S)/n:.3f} faces/image   tuned @0.20: {len(T)/n:.3f}")
    target=len(S)/n
    print(f"\ntarget crop budget = stock's {target:.3f} faces/image")
    print(f"{'tuned thr':>10} {'faces/img':>10}")
    best=None
    for t in np.arange(0.20,0.999,0.01):
        v=(T>=t).sum()/n
        if best is None and v<=target: best=t
        if abs(t*100-round(t*100))<1e-6 and round(t*100)%5==0:
            print(f"{t:>10.2f} {v:>10.3f}")
    if best is None:
        print("\nno threshold in [0.2,1.0) reaches stock's budget -- the tuned\n"
              "model is confident on the extra faces, so they cannot be\n"
              "thresholded away without also discarding real ones.")
    else:
        print(f"\nmatched threshold (first at or below target): {best:.2f}"
              f"  -> {(T>=best).sum()/n:.3f} faces/image")
    print(f"\nscore quantiles  stock: p10 {np.percentile(S,10):.3f} med {np.median(S):.3f}")
    print(f"                 tuned: p10 {np.percentile(T,10):.3f} med {np.median(T):.3f}")

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--limit",type=int,default=2000)
    ap.add_argument("--seed",type=int,default=7); a=ap.parse_args()
    main(a.limit,a.seed)
