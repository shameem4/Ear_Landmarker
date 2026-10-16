"""Landmark jitter through the FULL pipeline, old tracker vs new, on clips.

BlazeEar asked for the jitter benchmark to be re-run against the IoU-0.3
association. It cannot answer: scripts/eval_jitter.py synthesises crop-window
wobble from single images and has zero references to the tracker, so it
measures the landmarker in isolation and never exercises association at all.

The question behind the ask is real, though, and has a mechanism. The landmark
smoother is per-track, so a track switch COLD-STARTS it -- the new filter
starts at the observation and the smoothing that was suppressing jitter is
gone for several frames. Fewer switches should therefore mean less landmark
jitter, and the clip A/B found 400 -> 342 switches. This measures the
consequence directly: per-point landmark displacement between consecutive
reported frames, end to end through detector -> tracker -> landmarker.
"""
from __future__ import annotations
import sys, importlib.util, argparse, collections
from pathlib import Path
import numpy as np, cv2, torch
ROOT=Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/Ear_Landmarker")
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location(
    "old_smoothing","/var/tmp/claude/ear_landmarker/2026-10-15-tracker/old_smoothing.py")
oldmod=importlib.util.module_from_spec(spec); spec.loader.exec_module(oldmod)
import model.smoothing as newmod
from inference import EarLandmarkerPipeline, BLAZEEAR_DIR, DETECTOR_WEIGHTS, find_best_checkpoint

class OldShim:
    """Adapts the previous assign/smooth_box tracker to the new update() call."""
    def __init__(self, **kw):
        self._t = oldmod.EarTracker(**kw)
    def reset(self): self._t.reset()
    def update(self, detections, t):
        boxes=[np.asarray(d[:4],dtype=np.float64) for d in detections]
        ids=self._t.assign(boxes)
        return [(i, np.concatenate([self._t.smooth_box(i,b,t), np.asarray(d,dtype=np.float64)[4:]]))
                for i,b,d in zip(ids,boxes,detections)]
    def smooth_landmarks(self,*a,**k): return self._t.smooth_landmarks(*a,**k)

def jitter_for(pipe, clip, max_frames):
    """Frame-to-frame displacement of the PRIMARY (largest) ear.

    Results carry no track id, and adding one to the public dict for a
    measurement would change a contract three repos consume. The largest ear
    per frame is what a viewer actually watches, and it is where a cold-started
    landmark filter shows up as a visible jump.
    """
    cap=cv2.VideoCapture(str(clip)); fps=cap.get(cv2.CAP_PROP_FPS) or 30.0
    pipe.reset()
    prev=None; disp=[]; k=0; reported=0
    while k<max_frames:
        ok,bgr=cap.read()
        if not ok: break
        res=pipe(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB), timestamp=k/fps)
        if res:
            reported+=1
            r=max(res,key=lambda q: float(q["bbox"][3]-q["bbox"][1]))
            lm=np.asarray(r["landmarks"],dtype=np.float64)
            c=lm.mean(0)
            if prev is not None:
                pl,pc=prev
                # only compare when it is plausibly the same ear: centroid
                # within half the ear's extent. A jump past that is a different
                # ear, not jitter, and averaging it in would measure the wrong
                # thing in BOTH arms.
                ext=float(max(np.ptp(lm[:,0]),np.ptp(lm[:,1])))
                if float(np.linalg.norm(c-pc)) < 0.5*max(ext,1.0):
                    disp.append(float(np.median(np.linalg.norm(lm-pl,axis=1))))
            prev=(lm,c)
        k+=1
    cap.release()
    return np.array(disp), 0, reported

def main(clips, limit, max_frames):
    pipe=EarLandmarkerPipeline(BLAZEEAR_DIR/DETECTOR_WEIGHTS, find_best_checkpoint(),
                               device="cuda")
    paths=sorted(Path(clips).glob("*.mp4"))[:limit]
    rows=[]
    print(f"{'clip':>6} {'frames':>7} {'old med px':>11} {'new med px':>11} {'delta %':>9}")
    for p in paths:
        pipe.tracker=OldShim()
        o,osw,n_o=jitter_for(pipe,p,max_frames)
        pipe.tracker=newmod.EarTracker()
        nw,nsw,n_n=jitter_for(pipe,p,max_frames)
        if not len(o) or not len(nw): continue
        mo,mn=float(np.median(o)),float(np.median(nw))
        rows.append((mo,mn))
        print(f"{p.stem:>6} {n_o:>7} {mo:>11.4f} {mn:>11.4f} {100*(mn-mo)/mo:>8.2f}%")
    a=np.array([r[0] for r in rows]); b=np.array([r[1] for r in rows])
    from scipy import stats as st
    d=b-a
    print(f"\nclips {len(rows)}   median per-point frame-to-frame displacement (px)")
    print(f"  old {a.mean():.4f}   new {b.mean():.4f}   delta {100*(b.mean()-a.mean())/a.mean():+.2f}%")
    print(f"  paired: better on {int((d<0).sum())}, worse on {int((d>0).sum())}, "
          f"Wilcoxon p={st.wilcoxon(d).pvalue:.4g}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--clips",default="/mnt/14BE47C2BE479ADE/Code/ear_stuff/BlazeEar/data/clips")
    ap.add_argument("--limit",type=int,default=25); ap.add_argument("--max-frames",type=int,default=200)
    a=ap.parse_args(); main(a.clips,a.limit,a.max_frames)
