/**
 * Temporal smoothing for the live demo: detector boxes and landmarks.
 *
 * Port of model/smoothing.py. Kept deliberately close to it so the two can be
 * compared line by line; the Python side is the one covered by tests.
 *
 * Why this is here: the jitter benchmark found ~90% of frame-to-frame landmark
 * jitter comes from detector bbox wobble, and that no change inside the
 * landmarker moved it much. Smoothing is what actually attacks it, cutting
 * successive-frame jitter 66-80% while also REDUCING error against a
 * perfect-detector reference (i.e. it is not trading accuracy for smoothness).
 *
 * One Euro rather than a fixed-alpha EMA: an EMA trades jitter for lag at one
 * fixed ratio, so killing jitter means visible lag when the subject moves. One
 * Euro adapts its cutoff to observed speed - heavy smoothing when still, light
 * when moving.
 *
 * Reference: Casiez, Roussel & Vogel, "1 euro filter", CHI 2012.
 */

function lowPassAlpha(cutoff, dt) {
    const tau = 1.0 / (2.0 * Math.PI * Math.max(cutoff, 1e-6));
    return 1.0 / (1.0 + tau / Math.max(dt, 1e-6));
}

/** Speed-adaptive low-pass filter over a flat array of numbers. */
export class OneEuroFilter {
    constructor({ minCutoff = 1.0, beta = 0.05, dCutoff = 1.0 } = {}) {
        this.minCutoff = minCutoff;
        this.beta = beta;
        this.dCutoff = dCutoff;
        this.xPrev = null;
        this.dxPrev = null;
        this.tPrev = null;
    }

    reset() { this.xPrev = this.dxPrev = this.tPrev = null; }

    /**
     * @param {number[]} x observation
     * @param {number} t timestamp in SECONDS
     * @param {number[]|null} cutoffScale per-element multiplier on minCutoff
     */
    filter(x, t, cutoffScale = null) {
        if (this.xPrev === null) {
            this.xPrev = x.slice();
            this.dxPrev = new Array(x.length).fill(0);
            this.tPrev = t;
            return x.slice();
        }
        const dt = t - this.tPrev;
        if (dt <= 0) return this.xPrev.slice();   // duplicate / out-of-order frame

        const aD = lowPassAlpha(this.dCutoff, dt);
        const out = new Array(x.length);
        for (let i = 0; i < x.length; i++) {
            const dx = (x[i] - this.xPrev[i]) / dt;
            const dxHat = aD * dx + (1 - aD) * this.dxPrev[i];
            const scale = cutoffScale ? cutoffScale[i] : 1.0;
            const cutoff = this.minCutoff * scale + this.beta * Math.abs(dxHat);
            const a = lowPassAlpha(cutoff, dt);
            out[i] = a * x[i] + (1 - a) * this.xPrev[i];
            this.dxPrev[i] = dxHat;
        }
        this.xPrev = out.slice();
        this.tPrev = t;
        return out;
    }
}

/**
 * One Euro over 55 landmarks, weighted by the model's per-point confidence.
 *
 * Confidence comes from the heatmap head's spatial softmax: a peaked
 * distribution means the point is localised, a diffuse one means the model is
 * hedging. Uncertain points are smoothed harder.
 */
export class LandmarkSmoother {
    constructor({ minCutoff = 3.0, beta = 0.4, confStrength = 0.5,
                  confFloor = 0.25, confRef = 0.12 } = {}) {
        this.f = new OneEuroFilter({ minCutoff, beta });
        this.confStrength = confStrength;
        this.confFloor = confFloor;
        // Confidence from the model is concentrated well below 1; confRef maps
        // the useful part of its range onto [0,1] before it modulates smoothing.
        this.confRef = confRef;
    }

    reset() { this.f.reset(); }

    /**
     * @param {{x:number,y:number}[]} landmarks in frame pixels
     * @param {number} t seconds
     * @param {Float32Array|number[]|null} confidence per point
     */
    smooth(landmarks, t, confidence = null) {
        const flat = new Array(landmarks.length * 2);
        for (let i = 0; i < landmarks.length; i++) {
            flat[i * 2] = landmarks[i].x;
            flat[i * 2 + 1] = landmarks[i].y;
        }

        let scale = null;
        if (confidence && this.confStrength > 0) {
            scale = new Array(flat.length);
            for (let i = 0; i < landmarks.length; i++) {
                const c = Math.min(1, Math.max(0, confidence[i] / this.confRef));
                const s = Math.min(1, Math.max(
                    this.confFloor, 1 - this.confStrength * (1 - c)));
                scale[i * 2] = s;
                scale[i * 2 + 1] = s;
            }
        }

        const out = this.f.filter(flat, t, scale);
        return landmarks.map((_, i) => ({ x: out[i * 2], y: out[i * 2 + 1] }));
    }
}

/**
 * One Euro over a detector box, in centre/size form so smoothing cannot distort
 * its shape. Scale is smoothed harder than position, since box scale wobble is
 * almost always noise whereas position often is not.
 */
export class BoxSmoother {
    constructor({ minCutoff = 3.0, beta = 0.4,
                  sizeMinCutoff = 1.5, sizeBeta = 0.2 } = {}) {
        this.pos = new OneEuroFilter({ minCutoff, beta });
        this.size = new OneEuroFilter({ minCutoff: sizeMinCutoff, beta: sizeBeta });
    }

    reset() { this.pos.reset(); this.size.reset(); }

    smooth(box, t) {
        const cx = (box.xmin + box.xmax) / 2;
        const cy = (box.ymin + box.ymax) / 2;
        const w = Math.max(box.xmax - box.xmin, 1);
        const h = Math.max(box.ymax - box.ymin, 1);
        const [scx, scy] = this.pos.filter([cx, cy], t);
        const [sw, sh] = this.size.filter([w, h], t).map(v => Math.max(v, 1));
        return {
            xmin: scx - sw / 2, ymin: scy - sh / 2,
            xmax: scx + sw / 2, ymax: scy + sh / 2,
        };
    }
}

/**
 * Associates detections across frames so each ear keeps its own filter.
 *
 * Without association, two ears swap filter state between frames and smoothing
 * actively corrupts the output - each ear gets pulled toward the other's
 * history. Nearest-centre matching is ample for the 1-2 well-separated ears
 * this pipeline sees; it is not a general tracker.
 */
/**
 * Associates detections across frames and owns one filter pair per track.
 *
 * Association and box smoothing are BlazeEar's `BoxTracker`, loaded from
 * `ear_tracker.js` (vendored verbatim; see ../vendor/README.md). This file
 * keeps the per-track landmark smoothing, which that module does not do and
 * should not -- landmarks are this pipeline's output, boxes are the shared
 * part.
 *
 * Two deliberate deltas from the nearest-centre rule used before, both
 * measured in BlazeEar: matching is IoU rather than centre distance, and it
 * reads the RAW detection rather than the smoothed centre. Replayed against
 * the old rule, a single ear's track survives 44 px/frame of motion where the
 * old one broke at 34 -- matching its own lagging smoothed centre was costing
 * it tolerance.
 *
 * Defaults are near pass-through: no hysteresis (confirmScore ===
 * continueScore, minHits 1) and only matched tracks are consumed, so an ear is
 * reported on detected frames exactly as before. Real hysteresis
 * (0.80 / 0.65 / 3) is a separate, measured decision for this pipeline.
 *
 * `ear_tracker.js` is a classic script, so it publishes on globalThis; do not
 * add a bare `export` to it, which would silently disable BlazeEar's
 * require()-based parity test.
 */
import './ear_tracker.js';
const { BoxTracker, OneEuroBoxSmoother } = globalThis;

export class EarTracker {
    constructor({ maxMissed = 5, matchIou = 0.3, confirmScore = 0.70,
                  continueScore = 0.70, minHits = 1,
                  landmark = {}, box = {} } = {}) {
        // BoxTracker does `new this.smoother()` in JS (a constructor taking
        // no arguments) while the Python module accepts any zero-arg factory.
        // Bind this repo's box options by subclassing rather than passing a
        // factory, which would work in Python and throw here.
        const boxOpts = box;
        class BoundBoxSmoother extends OneEuroBoxSmoother {
            constructor() { super(boxOpts); }
        }
        this._bt = new BoxTracker({
            confirmScore, continueScore, minHits, maxMisses: maxMissed,
            matchIou, smoother: BoundBoxSmoother,
        });
        this._lm = new Map();
        this._landmarkOpts = landmark;
    }

    reset() { this._bt.reset(); this._lm.clear(); }

    /**
     * Associate and smooth. Returns [{ id, det }] for matched tracks only,
     * `det` being the caller's own detection object with the smoothed box
     * applied, so every other field survives untouched.
     */
    update(detections, t) {
        const dets = detections.map((d) => ({
            xmin: d.xmin, ymin: d.ymin,
            width: Math.max(d.xmax - d.xmin, 0),
            height: Math.max(d.ymax - d.ymin, 0),
            confidence: d.confidence ?? 1,
        }));
        const live = new Set();
        const out = [];
        for (const tr of this._bt.update(dets, t)) {
            live.add(tr.id);
            if (!tr.matched) continue;
            const src = detections[tr.detIndex];
            out.push({ id: tr.id, det: { ...src,
                xmin: tr.xmin, ymin: tr.ymin, xmax: tr.xmax, ymax: tr.ymax } });
        }
        // BoxTracker owns track lifetime; mirror its expiries so a landmark
        // filter cannot outlive its track and be handed to a reused id.
        for (const tid of [...this._lm.keys()]) if (!live.has(tid)) this._lm.delete(tid);
        return out;
    }

    smoothLandmarks(tid, landmarks, t, confidence) {
        let f = this._lm.get(tid);
        if (!f) { f = new LandmarkSmoother(this._landmarkOpts); this._lm.set(tid, f); }
        return f.smooth(landmarks, t, confidence);
    }
}
