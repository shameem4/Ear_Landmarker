"""A/B the old and new tracker on REAL clips, and measure the temporal axis.

Synthetic sequences showed no regression and a motion-tolerance gain, but they
are sequences I designed, which is the weakest evidence available. These are
25 conversational clips -- real detector output, real dropouts, real motion.

Measures, per clip, for each tracker:
  * track switches per reported frame -- an id change means the ear's filter
    cold-starts, which is the visible failure a tracker exists to prevent;
  * tracks per ear-sequence -- fragmentation;
  * adjacent-frame IoU of the EMITTED box -- temporal steadiness, the axis
    this repo flagged as unmeasured and could not measure without clips;
  * no-detection frame rate -- shared by both arms, reported once.
"""
from __future__ import annotations
import sys, importlib.util, argparse, collections
from pathlib import Path
import numpy as np, cv2, torch
ROOT=Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/Ear_Landmarker")
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location(
    "old_smoothing","/var/tmp/claude/ear_landmarker/2026-10-15-tracker/old_smoothing.py")
old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
from model.smoothing import EarTracker as NewTracker
from inference import EarDetector, BLAZEEAR_DIR, DETECTOR_WEIGHTS

def iou(a,b):
    ix1,iy1=max(a[1],b[1]),max(a[0],b[0]); ix2,iy2=min(a[3],b[3]),min(a[2],b[2])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    ua=(a[3]-a[1])*(a[2]-a[0])+(b[3]-b[1])*(b[2]-b[0])-inter
    return inter/ua if ua>0 else 0.0

def replay_old(seq):
    tr=old.EarTracker(); out=[]
    for t,dets in seq:
        boxes=[np.asarray(d[:4],dtype=np.float64) for d in dets]
        ids=tr.assign(boxes)
        out.append([(i,tr.smooth_box(i,b,t)) for i,b in zip(ids,boxes)])
    return out

def replay_new(seq):
    tr=NewTracker(); out=[]
    for t,dets in seq:
        out.append([(i,d[:4]) for i,d in tr.update(list(dets),t)])
    return out

def stats(frames):
    """Switches, track count, and adjacent-frame IoU of the emitted box."""
    prev={}; switches=0; ids=set(); ious=[]; reported=0
    last_primary=None
    for f in frames:
        if not f: 
            continue
        reported+=1
        f=sorted(f,key=lambda r:-(r[1][3]-r[1][1]))      # largest box first
        tid,box=f[0]
        ids.add(tid)
        if last_primary is not None:
            if tid!=last_primary[0]: switches+=1
            else: ious.append(iou(box,last_primary[1]))
        last_primary=(tid,box)
        for i,b in f: prev[i]=b
    return switches, len(ids), (float(np.mean(ious)) if ious else float("nan")), reported

def main(clips_dir, limit, max_frames, stride):
    det=EarDetector(BLAZEEAR_DIR/DETECTOR_WEIGHTS,"cuda",None)
    paths=sorted(Path(clips_dir).glob("*.mp4"))[:limit]
    agg=collections.defaultdict(list); nodet_total=0; frames_total=0
    print(f"{'clip':>8} {'frames':>7} {'no-det':>7} "
          f"{'old sw':>7} {'old trk':>8} {'old IoU':>8} "
          f"{'new sw':>7} {'new trk':>8} {'new IoU':>8}")
    for p in paths:
        cap=cv2.VideoCapture(str(p)); fps=cap.get(cv2.CAP_PROP_FPS) or 30.0
        seq=[]; k=0; grabbed=0
        while grabbed<max_frames:
            ok,bgr=cap.read()
            if not ok: break
            if k % stride: k+=1; continue
            frame=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            seq.append((k/fps, list(det.detect(frame)))); grabbed+=1; k+=1
        cap.release()
        if not seq: continue
        nodet=sum(1 for _,d in seq if len(d)==0)
        o=stats(replay_old(seq)); n=stats(replay_new(seq))
        nodet_total+=nodet; frames_total+=len(seq)
        agg["o_sw"].append(o[0]); agg["n_sw"].append(n[0])
        agg["o_trk"].append(o[1]); agg["n_trk"].append(n[1])
        if not np.isnan(o[2]): agg["o_iou"].append(o[2])
        if not np.isnan(n[2]): agg["n_iou"].append(n[2])
        print(f"{p.stem:>8} {len(seq):>7} {nodet:>7} "
              f"{o[0]:>7} {o[1]:>8} {o[2]:>8.4f} {n[0]:>7} {n[1]:>8} {n[2]:>8.4f}")
    print(f"\n{'TOTAL':>8} frames {frames_total}, no-detection {nodet_total} "
          f"({100*nodet_total/max(frames_total,1):.1f}%)")
    print(f"  track switches   old {sum(agg['o_sw'])}  new {sum(agg['n_sw'])}")
    print(f"  distinct tracks  old {sum(agg['o_trk'])}  new {sum(agg['n_trk'])}")
    print(f"  adjacent-frame IoU of emitted box: old {np.mean(agg['o_iou']):.4f}  "
          f"new {np.mean(agg['n_iou']):.4f}")
    if len(agg["o_iou"])==len(agg["n_iou"]) and len(agg["o_iou"])>2:
        from scipy import stats as st
        d=np.array(agg["n_iou"])-np.array(agg["o_iou"])
        print(f"  paired per-clip IoU delta: {d.mean():+.4f}  "
              f"p={st.wilcoxon(d).pvalue:.3g}  better on {int((d>0).sum())}/{len(d)} clips")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--clips",default="/mnt/14BE47C2BE479ADE/Code/ear_stuff/BlazeEar/data/clips")
    ap.add_argument("--limit",type=int,default=25)
    ap.add_argument("--max-frames",type=int,default=200)
    ap.add_argument("--stride",type=int,default=1)
    a=ap.parse_args(); main(a.clips,a.limit,a.max_frames,a.stride)
