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
export class EarTracker {
    constructor({ maxDistFrac = 0.6, maxMissed = 5,
                  landmarkOptions = {}, boxOptions = {} } = {}) {
        this.maxDistFrac = maxDistFrac;
        this.maxMissed = maxMissed;
        this.landmarkOptions = landmarkOptions;
        this.boxOptions = boxOptions;
        this.tracks = new Map();
        this.nextId = 0;
    }

    reset() { this.tracks.clear(); this.nextId = 0; }

    static centre(b) {
        return [(b.xmin + b.xmax) / 2, (b.ymin + b.ymax) / 2];
    }

    /** Greedy nearest-centre assignment; each track claimed at most once. */
    assign(boxes) {
        const taken = new Set();
        const ids = [];
        for (const box of boxes) {
            const [cx, cy] = EarTracker.centre(box);
            const size = Math.max(
                ((box.xmax - box.xmin) + (box.ymax - box.ymin)) / 2, 1);
            let bestId = null;
            let bestD = this.maxDistFrac * size;
            for (const [tid, tr] of this.tracks) {
                if (taken.has(tid)) continue;
                const d = Math.hypot(cx - tr.centre[0], cy - tr.centre[1]);
                if (d < bestD) { bestId = tid; bestD = d; }
            }
            if (bestId === null) {
                bestId = this.nextId++;
                this.tracks.set(bestId, {
                    box: new BoxSmoother(this.boxOptions),
                    lm: new LandmarkSmoother(this.landmarkOptions),
                    centre: [cx, cy],
                    missed: 0,
                });
            }
            taken.add(bestId);
            ids.push(bestId);
        }
        for (const [tid, tr] of [...this.tracks]) {
            if (taken.has(tid)) tr.missed = 0;
            else if (++tr.missed > this.maxMissed) this.tracks.delete(tid);
        }
        return ids;
    }

    smoothBox(tid, box, t) {
        const tr = this.tracks.get(tid);
        const out = tr.box.smooth(box, t);
        tr.centre = EarTracker.centre(out);
        return out;
    }

    smoothLandmarks(tid, landmarks, t, confidence) {
        return this.tracks.get(tid).lm.smooth(landmarks, t, confidence);
    }
}
