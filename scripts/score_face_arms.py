"""Score a tuned BlazeFace stage 1 against stock, on OUR ear corpus.

T4 depends on EAR recall through the two-stage pipeline, not face recall on a
face corpus. Both arms run in ONE pass over the same sampled images so the GT
denominator is identical and every ear is paired (McNemar on the discordant
pairs, not two independent proportions).

Bands are the GT ear box's max side in the ORIGINAL image, because an ear that
no crop covered has no size-in-crop -- and those are precisely the ears a
better stage 1 should rescue. 0-32 and 32-48 are never pooled.
"""
from __future__ import annotations
import sys, csv, collections, argparse, json
from pathlib import Path
import numpy as np, cv2, torch
ROOT=Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/Ear_Landmarker")
B=Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/BlazeEar")
TBF=Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/trainable_blazeface")
sys.path.insert(0,str(ROOT))
from inference import (EarDetector, EarLandmarkerPipeline, BLAZEEAR_DIR,
                       DETECTOR_WEIGHTS, _as_boxes)

BANDS=[(0,32),(32,48),(48,64),(64,80),(80,10**9)]
def band(px):
    for lo,hi in BANDS:
        if lo<=px<hi: return (lo,hi)
    return BANDS[-1]

def iou_xyxy(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.0

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
    # yolo11_ears/ was replaced by yolo11_ears_human/ (human-source basis),
    # nested one level deeper by source dataset. Roboflow basenames are unique
    # hashes -- verified 13,007 files, 13,007 unique names, 0 collisions -- so
    # resolve by basename and assert the uniqueness rather than assume it.
    import collections as _c
    idx=_c.defaultdict(list)
    for q in (B/"data/yolo11_ears_human/images").rglob("*"):
        if q.is_file(): idx[q.name].append(q)
    assert all(len(v)==1 for v in idx.values()), "ambiguous basename in corpus"
    out=[]; unresolved=0
    for ip,boxes in by.items():
        c=idx.get(Path(ip).name)
        if c: out.append((c[0],boxes))
        else: unresolved+=1
    print(f"GT basis: {len(out)} images resolved, {unresolved} image_paths not in "
          f"yolo11_ears_human ({100*unresolved/max(len(by),1):.1f}%)")
    return out

def run_arm(det, face_model, frame, thrs, expand, face_thr):
    """Return (all_boxes_yxyxs, crop_windows) for one face stage."""
    face_model.min_score_thresh=float(face_thr)
    faces=face_model.process(frame)
    if isinstance(faces,torch.Tensor): faces=faces.cpu().numpy()
    faces=np.atleast_2d(np.asarray(faces)) if np.size(faces) else np.zeros((0,17))
    thr_def=float(det.model.min_score_thresh)
    crops,out=[],[]
    det.model.min_score_thresh=0.01
    for face in faces[:det.max_faces]:
        x0,y0,side=det._crop_window(face,expand,frame.shape)
        crops.append((x0,y0,side))
        patch=frame[y0:y0+side,x0:x0+side]
        if patch.size==0: continue
        d=_as_boxes(det.model.process(patch))
        if len(d):
            d=d.copy(); d[:,:4]+=np.array([y0,x0,y0,x0],dtype=np.float64); out.append(d)
    det.model.min_score_thresh=thr_def
    allb=np.concatenate(out,0) if out else np.zeros((0,5))
    return allb, crops

def preds_at(allb, t):
    keep=allb[allb[:,4]>=t] if len(allb) else allb
    dets=EarLandmarkerPipeline._nms(keep)
    return [[d[1],d[0],d[3],d[2]] for d in dets]

def main(limit, seed, thrs, ckpt, exp_s, exp_t, fthr_s, fthr_t, cascade=False):
    items=load_gt()
    rng=np.random.default_rng(seed)
    sel=rng.choice(len(items),min(limit,len(items)),replace=False)
    items=[items[i] for i in sorted(sel)]

    det=EarDetector(BLAZEEAR_DIR/DETECTOR_WEIGHTS,"cuda",None)
    stock=det.face
    import copy
    tuned=copy.deepcopy(stock)
    sd=torch.load(ckpt, map_location="cpu")
    sd=sd.get("model_state_dict", sd)
    missing,unexpected=tuned.load_state_dict(sd, strict=True), None
    tuned.eval()
    # the load must actually change the weights -- a silent no-op would show as
    # two identical arms, which is also what "no effect" looks like.
    same=all(torch.equal(a,b) for a,b in zip(stock.state_dict().values(),
                                             tuned.state_dict().values()))
    assert not same, "tuned weights identical to stock -- load was a no-op"

    ARMS={"stock":stock,"tuned":tuned}
    if cascade: ARMS["cascade"]=tuned
    EXP={"stock":exp_s,"tuned":exp_t,"cascade":exp_t}
    FTHR={"stock":fthr_s,"tuned":fthr_t,"cascade":fthr_t}
    print(f"FACE_CROP_EXPAND: stock {exp_s}, tuned {exp_t}")
    print(f"FACE_CROP_THRESHOLD: stock {fthr_s}, tuned {fthr_t}")
    # per-threshold counters
    S={a:{t:dict(tp=0,fp=0,nd=0) for t in thrs} for a in ARMS}
    # per-band hit counts at the shipped threshold, plus pairing
    thr_ship=float(det.model.min_score_thresh)
    CMP="cascade" if cascade else "tuned"
    bandc={a:collections.defaultdict(lambda: dict(hit=0,n=0,facemiss=0)) for a in ARMS}
    pairs=collections.Counter()   # (stock_hit, CMP_hit) -> count
    band_pairs=collections.defaultdict(collections.Counter)
    n_gt=n_img=0
    gained=[]
    for path,gts in items:
        bgr=cv2.imread(str(path))
        if bgr is None: continue
        frame=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        n_img+=1; n_gt+=len(gts)
        res={}
        for a,fm in ARMS.items():
            if a=="cascade": continue
            res[a]=run_arm(det, fm, frame, thrs, EXP[a], FTHR[a])
        if "cascade" in ARMS:
            # The 0-32 px gain is a COVERAGE gain: the tuned model finds faces
            # stock misses. Everything else it costs is paid on images stock had
            # already covered. So spend it only where stock came up empty.
            # "stock found nothing" almost never fires -- the face-stage
            # misses are ears inside images where a face WAS found but did not
            # cover them, which no inference-time trigger can detect. So the
            # deployable form is the UNION of both detectors' crops: stage 2
            # runs on every crop either one proposes. Costs crops, buys
            # coverage. Boxes are already in frame coordinates, so the union is
            # a concatenation followed by the same NMS.
            sb,sc=res["stock"]; tb,tc=res["tuned"]
            ub=np.concatenate([sb,tb],0) if (len(sb) or len(tb)) else np.zeros((0,5))
            res["cascade"]=(ub, sc+tc)
        for a,(allb,crops) in res.items():
            for t in thrs:
                pred=preds_at(allb,t)
                S[a][t]["nd"]+=len(pred); used=set()
                for g in gts:
                    bi,bv=-1,0.0
                    for i,q in enumerate(pred):
                        if i in used: continue
                        v=iou_xyxy(g,q)
                        if v>=0.3 and v>bv: bi,bv=i,v
                    if bi>=0: used.add(bi); S[a][t]["tp"]+=1
                S[a][t]["fp"]+=len(pred)-len(used)
        shipped={a:preds_at(res[a][0],thr_ship) for a in ARMS}
        for g in gts:
            bd=band(max(g[2]-g[0],g[3]-g[1]))
            h={}
            for a in ARMS:
                allb,crops=res[a]
                hit=any(iou_xyxy(g,q)>=0.3 for q in shipped[a])
                h[a]=hit
                bc=bandc[a][bd]; bc["n"]+=1; bc["hit"]+=int(hit)
                if not any(frac_inside(g,[c[0],c[1],c[0]+c[2],c[1]+c[2]])>=0.5 for c in crops):
                    bc["facemiss"]+=1
            key=(h["stock"],h[CMP])
            pairs[key]+=1; band_pairs[bd][key]+=1
            if key==(False,True):
                gained.append({"image":str(path),"box_xyxy":[round(v,1) for v in g],
                               "band":f"{bd[0]}-{bd[1]}"})

    print(f"images {n_img}, ground-truth ears {n_gt}   (seed {seed})")
    print(f"shipped stage-2 threshold {thr_ship:.2f}, IoU>=0.30\n")
    print(f"{'band(px, source image)':>24} {'n':>6} {'stock':>8} {CMP:>8} {'delta':>8}"
          f" {'facemiss_s':>11} {'facemiss_t':>11}")
    for lo,hi in BANDS:
        bd=(lo,hi); s=bandc["stock"][bd]; u=bandc[CMP][bd]
        if not s["n"]: continue
        nm=f"{lo}-{hi}" if hi<10**9 else f"{lo}+"
        print(f"{nm:>24} {s['n']:>6} {100*s['hit']/s['n']:>7.2f}% {100*u['hit']/u['n']:>7.2f}%"
              f" {100*(u['hit']-s['hit'])/s['n']:>+7.2f} {s['facemiss']:>11} {u['facemiss']:>11}")
    ts=sum(v for k,v in pairs.items() if k[0]); tt=sum(v for k,v in pairs.items() if k[1])
    print(f"\npooled recall  stock {100*ts/n_gt:.2f}%   {CMP} {100*tt/n_gt:.2f}%"
          f"   delta {100*(tt-ts)/n_gt:+.2f} pp")
    b,c=pairs[(True,False)],pairs[(False,True)]
    print(f"discordant pairs: stock-only {b}, {CMP}-only {c}")
    if b+c:
        from scipy import stats as st
        p=st.binomtest(c, b+c, 0.5).pvalue
        print(f"McNemar (exact binomial on discordant pairs): p={p:.3e}")
    else:
        print("McNemar: no discordant pairs")
    print("\nper-band discordance (stock-only / tuned-only, exact p):")
    from scipy import stats as st
    for lo,hi in BANDS:
        bd=(lo,hi); pc=band_pairs[bd]
        if not sum(pc.values()): continue
        b2,c2=pc[(True,False)],pc[(False,True)]
        nm=f"{lo}-{hi}" if hi<10**9 else f"{lo}+"
        pv=st.binomtest(c2,b2+c2,0.5).pvalue if b2+c2 else float('nan')
        print(f"  {nm:>8}  {b2:>4} / {c2:>4}   p={pv:.3g}")
    print(f"\n{'thresh':>7} | {'stock recall':>13} {'prec':>8} {'FP/img':>8}"
          f" | {CMP+" recall":>13} {'prec':>8} {'FP/img':>8}")
    for t in thrs:
        row=[]
        for a in ("stock",CMP):
            s=S[a][t]
            row.append(f"{100*s['tp']/n_gt:>12.2f}% {100*s['tp']/max(s['nd'],1):>7.2f}% {s['fp']/n_img:>8.3f}")
        print(f"{t:>7.2f} | {row[0]} | {row[1]}")
    out=Path(__file__).with_name(f"gained_seed{seed}_{exp_s}_{exp_t}_{fthr_t}.json")
    out.write_text(json.dumps(gained, indent=1))
    print(f"\n{CMP}-only rescues manifest: {len(gained)} -> {out}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--limit",type=int,default=1500)
    ap.add_argument("--seed",type=int,default=1)
    ap.add_argument("--expand-stock",type=float,default=1.5)
    ap.add_argument("--expand-tuned",type=float,default=1.5)
    ap.add_argument("--cascade",action="store_true")
    ap.add_argument("--face-thresh-stock",type=float,default=0.2)
    ap.add_argument("--face-thresh-tuned",type=float,default=0.2)
    ap.add_argument("--thrs",default="0.80,0.70,0.60,0.50,0.40,0.30")
    ap.add_argument("--ckpt",default=str(TBF/"runs/ckpt_peranchor/BlazeFace_final.pth"))
    a=ap.parse_args()
    main(a.limit,a.seed,[float(x) for x in a.thrs.split(',')],a.ckpt,
         a.expand_stock,a.expand_tuned,a.face_thresh_stock,a.face_thresh_tuned,a.cascade)
