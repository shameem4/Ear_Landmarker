/**
 * One box tracker, mirroring utils/box_tracker.py. Keep them in step.
 *
 * Three consumers wanted the same thing and each had half of it: this repo's
 * demo had hysteresis with an eyeballed EMA, Ear_Landmarker had One Euro box
 * smoothing with no hysteresis, headSize-gnm vendored the second and wanted
 * one approach. The module lives in BlazeEar because the constants were
 * measured here, on 7,500 frames of 25 conversational clips, and a constant
 * only means something alongside the association rule it was measured with.
 *
 * Two decisions are load-bearing, both from review rather than design:
 *
 * MATCHING SEES THE RAW BOX. This file's predecessor, inline in index.html,
 * computed IoU against the smoothed box under a comment saying it did not.
 * Replaying the clips both ways moved lateral confirmations 189 -> 212 and
 * midline 9 -> 14, the same magnitude as the effect the constants were chosen
 * for. The smoother affects the emitted box and nothing else.
 *
 * THE SMOOTHER IS FED ONLY ON MATCHED FRAMES. One Euro's catch-up depends on
 * it: after a gap, dt spans the whole gap, alpha rises, and the filter jumps
 * most of the way to the new observation. Feeding it a repeated box at small
 * dt reads as "the ear stopped", drags the speed estimate toward zero and
 * over-smooths the next real motion -- sluggishness on exactly the frames
 * after a flicker. A coasting track emits its last smoothed box unchanged.
 * Pre-confirmation frames ARE fed: real detections of the same track, and
 * withholding them would cold-start the filter when the box becomes visible.
 *
 * CONSUMING THIS FILE. In the browser, load it with a plain `<script src>`
 * and read the globals; in node, `require` it; from an ES module, do a
 * side-effect `import './ear_tracker.js'` and then read `globalThis`. Please
 * do NOT add a bare `export` -- that makes the file an ES module, which is a
 * syntax error under `require()`, and the parity test that keeps this file
 * honest requires it. Field names differ by language on purpose, each
 * idiomatic to its own: `det_index` in Python, `detIndex` here. The parity
 * test maps between them explicitly rather than letting either drift.
 *
 * One Euro is the default because it was measured against the EMA on those
 * clips: jitter -24.6% against -12.4% from raw, better on 110 of 140 tracks,
 * 95% CI on the paired difference [-0.0013, -0.0008], at lag that is
 * indistinguishable (75 of 140, CI spans zero). Better on one axis, free on
 * the other.
 */
const TRACKER_DEFAULTS = {
    confirmScore: 0.80,
    continueScore: 0.65,
    minHits: 3,
    maxMisses: 3,
    matchIou: 0.3,
};

function oneEuroAlpha(cutoff, dt) {
    const tau = 1.0 / (2.0 * Math.PI * Math.max(cutoff, 1e-6));
    return 1.0 / (1.0 + tau / Math.max(dt, 1e-6));
}

/** Scalar speed-adaptive low-pass. Ported from Ear_Landmarker's array form. */
class OneEuroScalar {
    constructor(minCutoff, beta, dCutoff = 1.0) {
        this.minCutoff = minCutoff;
        this.beta = beta;
        this.dCutoff = dCutoff;
        this.xPrev = null;
        this.dxPrev = 0.0;
        this.tPrev = null;
    }

    filter(x, t) {
        if (this.xPrev === null || this.tPrev === null) {
            this.xPrev = x; this.dxPrev = 0.0; this.tPrev = t;
            return x;
        }
        const dt = t - this.tPrev;
        if (!(dt > 0)) return this.xPrev;   // duplicate or out-of-order frame
        const aD = oneEuroAlpha(this.dCutoff, dt);
        const dxHat = aD * ((x - this.xPrev) / dt) + (1.0 - aD) * this.dxPrev;
        const a = oneEuroAlpha(this.minCutoff + this.beta * Math.abs(dxHat), dt);
        const xHat = a * x + (1.0 - a) * this.xPrev;
        this.xPrev = xHat; this.dxPrev = dxHat; this.tPrev = t;
        return xHat;
    }
}

/**
 * One Euro in centre/size form, position and scale filtered separately.
 * Corner coordinates would let the four edges drift independently and change
 * the box's shape; centre/size cannot. Scale is smoothed harder because scale
 * wobble is almost always noise while position often is not.
 */
class OneEuroBoxSmoother {
    constructor(opts = {}) {
        const mc = opts.minCutoff ?? 0.8, b = opts.beta ?? 0.03;
        const smc = opts.sizeMinCutoff ?? 0.4, sb = opts.sizeBeta ?? 0.01;
        this._cx = new OneEuroScalar(mc, b);
        this._cy = new OneEuroScalar(mc, b);
        this._w = new OneEuroScalar(smc, sb);
        this._h = new OneEuroScalar(smc, sb);
    }

    smooth(box, t) {
        const cx = this._cx.filter(box.xmin + box.width / 2.0, t);
        const cy = this._cy.filter(box.ymin + box.height / 2.0, t);
        const w = Math.max(this._w.filter(box.width, t), 1.0);
        const h = Math.max(this._h.filter(box.height, t), 1.0);
        return {xmin: cx - w / 2.0, ymin: cy - h / 2.0, width: w, height: h};
    }
}

/**
 * Fixed-rate exponential blend on the corners, the demo's original. Kept so
 * the two can be compared rather than argued about, and because `k` has no
 * timestep in it: at a different frame rate this smooths by a different
 * amount for the same motion, which is the structural complaint against it.
 */
class EmaBoxSmoother {
    constructor(opts = {}) {
        this.k = opts.k ?? 0.6;
        this._prev = null;
    }

    smooth(box) {
        if (this._prev === null) {
            this._prev = {...box};
            return {...box};
        }
        const k = this.k;
        this._prev = {
            xmin: k * this._prev.xmin + (1 - k) * box.xmin,
            ymin: k * this._prev.ymin + (1 - k) * box.ymin,
            width: k * this._prev.width + (1 - k) * box.width,
            height: k * this._prev.height + (1 - k) * box.height,
        };
        return {...this._prev};
    }
}

function boxIou(a, b) {
    const x0 = Math.max(a.xmin, b.xmin);
    const y0 = Math.max(a.ymin, b.ymin);
    const x1 = Math.min(a.xmin + a.width, b.xmin + b.width);
    const y1 = Math.min(a.ymin + a.height, b.ymin + b.height);
    const inter = Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
    const union = a.width * a.height + b.width * b.height - inter;
    return union > 0 ? inter / union : 0;
}

class BoxTracker {
    /**
     * @param {Object} options
     *   confirmScore  an unmatched candidate starts a track only at or above
     *                 this; 0.80 as measured, deliberately below
     *                 TTA_SCORE_THRESHOLD because a track confirming over
     *                 minHits frames has evidence a single frame does not.
     *   continueScore candidates below this are not considered at all; 0.65,
     *                 a 0.15 margin under confirmScore.
     *   minHits       matched frames before a track is reported.
     *   maxMisses     frames a track may coast without a match.
     *   matchIou      IoU floor for association. IoU rather than
     *                 nearest-centre because the constants were measured with
     *                 it, and because it refuses a match that nearest-centre
     *                 accepts when box size changes sharply -- a turning head.
     *   smoother      factory returning {smooth(box, t)}, one per track. Null
     *                 emits the raw detection.
     *   fps           used only to synthesise t when update() is called
     *                 without one. Null means t is passed through as given --
     *                 a guessed frame rate is a wrong dt, a wrong alpha, and a
     *                 filter that smooths differently in the two languages.
     */
    constructor(options = {}) {
        // Unknown keys throw. Python's signature already rejects a
        // misspelling with a TypeError, and Object.assign accepted anything,
        // so the two languages disagreed about whether a typo was an error.
        // The concrete failure headsize v2 hit: passing this module's own
        // Python names ({confirm_score: 0.7, min_hits: 1}) to the JS, which
        // took none of them, silently ran 0.80 / minHits 3 and emitted
        // nothing on frame one. That is the same shape as the slider's
        // value="0.70" bug -- a constant nobody chose, running quietly -- and
        // it is likely precisely because Python is snake_case here and JS is
        // camelCase.
        const allowed = new Set([...Object.keys(TRACKER_DEFAULTS),
                                 'smoother', 'fps']);
        const unknown = Object.keys(options).filter(k => !allowed.has(k));
        if (unknown.length) {
            throw new TypeError(
                `BoxTracker: unknown option(s) ${unknown.join(', ')}. ` +
                `Expected ${[...allowed].join(', ')}. Note this file is ` +
                `camelCase and utils/box_tracker.py is snake_case.`);
        }
        Object.assign(this, TRACKER_DEFAULTS, options);
        this.smoother = options.smoother ?? null;
        this.fps = options.fps ?? null;
        this.tracks = [];
        this.nextId = 1;
        this._frame = 0;
    }

    reset() {
        this.tracks = [];
        this.nextId = 1;
        this._frame = 0;
    }

    update(detections, t = null) {
        if (t === null && this.fps) t = this._frame / this.fps;
        this._frame += 1;

        const candidates = detections
            .map((d, i) => ({d, i}))
            .filter(({d}) => d.confidence >= this.continueScore)
            .sort((a, b) => b.d.confidence - a.d.confidence);
        const claimed = new Set();

        // Existing tracks claim first, highest-scoring candidate first, so two
        // tracks cannot absorb the same detection.
        for (const track of this.tracks) {
            let best = -1, bestIou = this.matchIou;
            for (let j = 0; j < candidates.length; j++) {
                if (claimed.has(j)) continue;
                const v = boxIou(track.rawBox, candidates[j].d);
                if (v >= bestIou) { bestIou = v; best = j; }
            }
            if (best < 0) {
                track.misses++;
                track.matched = false;
                track.detIndex = null;
                continue;
            }
            claimed.add(best);
            const {d, i} = candidates[best];
            track.rawBox = {xmin: d.xmin, ymin: d.ymin,
                            width: d.width, height: d.height};
            track.box = track.smoother
                ? track.smoother.smooth(track.rawBox, t)
                : {...track.rawBox};
            track.confidence = d.confidence;
            track.hits++;
            track.misses = 0;
            track.matched = true;
            track.detIndex = i;
        }

        for (let j = 0; j < candidates.length; j++) {
            if (claimed.has(j)) continue;
            const {d, i} = candidates[j];
            if (d.confidence < this.confirmScore) continue;
            const rawBox = {xmin: d.xmin, ymin: d.ymin,
                            width: d.width, height: d.height};
            const sm = this.smoother ? new this.smoother() : null;
            this.tracks.push({
                id: this.nextId++,
                rawBox,
                box: sm ? sm.smooth(rawBox, t) : {...rawBox},
                smoother: sm,
                confidence: d.confidence,
                hits: 1, misses: 0, age: 0,
                matched: true, detIndex: i,
            });
        }

        // Drop over-missed tracks FIRST, then report. Reporting before the
        // filter credits a track one frame it never appeared in.
        this.tracks = this.tracks.filter(tr => tr.misses <= this.maxMisses);
        const out = [];
        for (const tr of this.tracks) {
            tr.age++;
            if (tr.hits < this.minHits) continue;
            out.push({
                id: tr.id, detIndex: tr.detIndex, matched: tr.matched,
                xmin: tr.box.xmin, ymin: tr.box.ymin,
                width: tr.box.width, height: tr.box.height,
                xmax: tr.box.xmin + tr.box.width,
                ymax: tr.box.ymin + tr.box.height,
                confidence: tr.confidence,
                hits: tr.hits, misses: tr.misses, age: tr.age,
            });
        }
        return out;
    }
}

// Node (the parity test requires this file) and the browser (index.html loads
// it as a classic script and reads the globals). Deliberately NOT an ES module
// with a bare `export`, because that is a syntax error under require() and the
// parity test is the thing that keeps this file honest.
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {BoxTracker, OneEuroBoxSmoother, EmaBoxSmoother,
                      boxIou, TRACKER_DEFAULTS};
}
if (typeof globalThis !== 'undefined') {
    globalThis.BoxTracker = BoxTracker;
    globalThis.OneEuroBoxSmoother = OneEuroBoxSmoother;
    globalThis.EmaBoxSmoother = EmaBoxSmoother;
    globalThis.TRACKER_DEFAULTS = TRACKER_DEFAULTS;
}
