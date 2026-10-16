"""One box tracker, owned here and mirrored in docs/ear_tracker.js.

Three consumers wanted the same thing and each had half of it. This repo's
demo had hysteresis (confirm over several frames, continue lower, coast a
short gap) with an eyeballed EMA for display. `Ear_Landmarker` had One Euro
box smoothing measured to cut jitter 66-80% while lowering error, with no
hysteresis at all. `headSize-gnm` vendored the second and wanted one
approach. The module lives here because the constants were measured here, on
7,500 frames of 25 conversational clips, and a constant is only meaningful
alongside the association rule it was measured with.

Two decisions are load-bearing and both came out of review rather than
design:

MATCHING SEES THE RAW BOX. `docs/index.html` used to compute IoU against the
smoothed box, under a comment saying it did not. Replaying the clips both
ways moved lateral confirmations 189 -> 212 and midline 9 -> 14, which is the
same magnitude as the effect the constants were chosen for. So `match_iou`,
`confirm_score`, `continue_score` and `min_hits` all mean what they were
measured to mean only if matching reads the last matched detection. The
smoother affects the emitted box and nothing else.

THE SMOOTHER IS FED ONLY ON MATCHED FRAMES. One Euro's catch-up depends on
it: when a detection reappears after a gap, `dt` spans the whole gap, the
alpha rises, and the filter jumps most of the way to the new observation.
Feeding it a repeated or coasted box at small `dt` instead reads as "the ear
stopped", drags the speed estimate toward zero and over-smooths the next real
motion -- sluggishness on exactly the frames after a flicker, which is what
the tracker exists to prevent. A coasting track emits its last smoothed
output unchanged. Pre-confirmation frames (`hits < min_hits`) ARE fed: those
are real detections of the same track, and withholding them would cold-start
the filter at the moment the box becomes visible.

    tracker = BoxTracker(smoother=OneEuroBoxSmoother)
    for t, detections in frames:
        for track in tracker.update(detections, t):
            draw(track)

Pass-through, for a consumer that wants association and smoothing but no
hysteresis, is configuration rather than a flag, so every consumer runs one
code path and differs only in constants:

    BoxTracker(confirm_score=0.70, continue_score=0.70, min_hits=1,
               max_misses=5, smoother=OneEuroBoxSmoother)

and filter the output on `track['matched']` to report only frames where a
detection actually arrived, while the track's filter state, id and any
per-track state a consumer hangs off the id survive a short dropout.
"""
import math

CONFIRM_SCORE = 0.80
CONTINUE_SCORE = 0.65
MIN_HITS = 3
MAX_MISSES = 3
MATCH_IOU = 0.3

BOX_KEYS = ('xmin', 'ymin', 'width', 'height')


def _alpha(cutoff, dt):
    tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
    return 1.0 / (1.0 + tau / max(dt, 1e-6))


class _OneEuro:
    """Scalar speed-adaptive low-pass. Ported from Ear_Landmarker's array form.

    Scalar rather than vector because the JS mirror has no array type and the
    two have to agree numerically, not just behaviourally.
    """

    def __init__(self, min_cutoff, beta, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def __call__(self, x, t):
        if self.x_prev is None or self.t_prev is None:
            self.x_prev, self.dx_prev, self.t_prev = x, 0.0, t
            return x
        dt = t - self.t_prev
        if dt <= 0:                      # duplicate or out-of-order frame
            return self.x_prev
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * ((x - self.x_prev) / dt) + (1.0 - a_d) * self.dx_prev
        a = _alpha(self.min_cutoff + self.beta * abs(dx_hat), dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


class OneEuroBoxSmoother:
    """One Euro in centre/size form, position and scale filtered separately.

    Corner coordinates would let the four edges drift independently and change
    the box's shape; centre/size cannot. Scale is smoothed harder than
    position because scale wobble is almost always noise while position often
    is not -- Ear_Landmarker's parameterisation and defaults, kept so that
    repo can adopt this file without re-tuning.
    """

    def __init__(self, min_cutoff=0.8, beta=0.03,
                 size_min_cutoff=0.4, size_beta=0.01):
        self._cx = _OneEuro(min_cutoff, beta)
        self._cy = _OneEuro(min_cutoff, beta)
        self._w = _OneEuro(size_min_cutoff, size_beta)
        self._h = _OneEuro(size_min_cutoff, size_beta)

    def smooth(self, box, t):
        cx = self._cx(box['xmin'] + box['width'] / 2.0, t)
        cy = self._cy(box['ymin'] + box['height'] / 2.0, t)
        w = max(self._w(box['width'], t), 1.0)
        h = max(self._h(box['height'], t), 1.0)
        return {'xmin': cx - w / 2.0, 'ymin': cy - h / 2.0,
                'width': w, 'height': h}


class EmaBoxSmoother:
    """Fixed-rate exponential blend on the corners, the demo's original.

    Kept so the two can be compared on the same frames rather than argued
    about, and because `k` has no timestep in it: at a different frame rate
    this smooths by a different amount for the same motion, which is the
    structural complaint against it.
    """

    def __init__(self, k=0.6):
        self.k = float(k)
        self._prev = None

    def smooth(self, box, t=None):
        if self._prev is None:
            self._prev = dict(box)
            return dict(box)
        k = self.k
        self._prev = {key: k * self._prev[key] + (1.0 - k) * box[key]
                      for key in BOX_KEYS}
        return dict(self._prev)


def iou(a, b):
    x0 = max(a['xmin'], b['xmin'])
    y0 = max(a['ymin'], b['ymin'])
    x1 = min(a['xmin'] + a['width'], b['xmin'] + b['width'])
    y1 = min(a['ymin'] + a['height'], b['ymin'] + b['height'])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = a['width'] * a['height'] + b['width'] * b['height'] - inter
    return inter / union if union > 0 else 0.0


class BoxTracker:
    """Detection-then-tracking hysteresis with pluggable box smoothing.

    Args:
        confirm_score: an unmatched candidate starts a track only at or above
            this. Measured at 0.80, deliberately below TTA_SCORE_THRESHOLD,
            because a track confirming over `min_hits` frames carries evidence
            a single frame does not.
        continue_score: candidates below this are not considered at all. 0.65
            as measured, i.e. a 0.15 margin under `confirm_score`.
        min_hits: frames a track must have matched before it is reported.
        max_misses: frames a track may coast without a match before it dies.
        match_iou: IoU floor for association. IoU rather than nearest-centre
            because the constants above were measured with it, and because it
            refuses a match that nearest-centre accepts when the box size
            changes sharply -- which is a turning head.
        smoother: a class or factory taking no required arguments and
            returning an object with `smooth(box, t)`. One instance per
            track. None means the emitted box is the raw detection.
        fps: used only to synthesise `t` when `update` is called without one.
            None means a caller that omits `t` gets `t=None` passed to the
            smoother, which `EmaBoxSmoother` ignores and `OneEuroBoxSmoother`
            cannot accept -- deliberately a loud failure, because a guessed
            frame rate is a wrong `dt`, a wrong alpha, and a filter that
            smooths differently in two languages for the same footage.
    """

    def __init__(self, confirm_score=CONFIRM_SCORE,
                 continue_score=CONTINUE_SCORE, min_hits=MIN_HITS,
                 max_misses=MAX_MISSES, match_iou=MATCH_IOU,
                 smoother=None, fps=None):
        self.confirm_score = confirm_score
        self.continue_score = continue_score
        self.min_hits = min_hits
        self.max_misses = max_misses
        self.match_iou = match_iou
        self.smoother = smoother
        self.fps = fps
        self.tracks = []
        self.next_id = 1
        self._frame = 0

    def reset(self):
        self.tracks = []
        self.next_id = 1
        self._frame = 0

    def _new_smoother(self):
        return self.smoother() if self.smoother is not None else None

    def update(self, detections, t=None):
        if t is None and self.fps:
            t = self._frame / float(self.fps)
        self._frame += 1

        candidates = sorted(
            [d for d in detections if d['confidence'] >= self.continue_score],
            key=lambda d: -d['confidence'])
        claimed = set()

        # Existing tracks claim first, highest-scoring candidate first, so two
        # tracks cannot absorb the same detection.
        for track in self.tracks:
            best, best_iou = -1, self.match_iou
            for i, cand in enumerate(candidates):
                if i in claimed:
                    continue
                v = iou(track['raw_box'], cand)
                if v >= best_iou:
                    best, best_iou = i, v
            if best < 0:
                track['misses'] += 1
                track['matched'] = False
                track['det_index'] = None
                continue
            claimed.add(best)
            det = candidates[best]
            track['raw_box'] = {k: det[k] for k in BOX_KEYS}
            track['box'] = (track['smoother'].smooth(track['raw_box'], t)
                            if track['smoother'] is not None
                            else dict(track['raw_box']))
            track['confidence'] = det['confidence']
            track['hits'] += 1
            track['misses'] = 0
            track['matched'] = True
            track['det_index'] = detections.index(det)

        for i, det in enumerate(candidates):
            if i in claimed or det['confidence'] < self.confirm_score:
                continue
            raw = {k: det[k] for k in BOX_KEYS}
            sm = self._new_smoother()
            self.tracks.append({
                'id': self.next_id,
                'raw_box': raw,
                'box': sm.smooth(raw, t) if sm is not None else dict(raw),
                'smoother': sm,
                'confidence': det['confidence'],
                'hits': 1,
                'misses': 0,
                'age': 0,
                'matched': True,
                'det_index': detections.index(det),
            })
            self.next_id += 1

        # Drop over-missed tracks FIRST, then report. Reporting before the
        # filter credits a track one frame it never appeared in.
        self.tracks = [tr for tr in self.tracks
                       if tr['misses'] <= self.max_misses]
        out = []
        for tr in self.tracks:
            tr['age'] += 1
            if tr['hits'] < self.min_hits:
                continue
            box = tr['box']
            out.append({
                'id': tr['id'], 'det_index': tr['det_index'],
                'matched': tr['matched'],
                'xmin': box['xmin'], 'ymin': box['ymin'],
                'width': box['width'], 'height': box['height'],
                'xmax': box['xmin'] + box['width'],
                'ymax': box['ymin'] + box['height'],
                'confidence': tr['confidence'],
                'hits': tr['hits'], 'misses': tr['misses'], 'age': tr['age'],
            })
        return out
