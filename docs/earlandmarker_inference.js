/**
 * EarLandmarker Inference Module for JavaScript/Browser
 *
 * Full pipeline: BlazeEar detection -> ROI crop -> EarLandmarker -> 55 landmarks
 * Uses ONNX Runtime Web for both models.
 *
 * Usage:
 *   import { EarLandmarkerPipeline } from './earlandmarker_inference.js';
 *
 *   const pipeline = new EarLandmarkerPipeline();
 *   await pipeline.load('BlazeEar_web.onnx', 'EarLandmarker_web.onnx');
 *
 *   const results = await pipeline.detect(videoElement);
 *   // Returns: Array of { bbox, confidence, landmarks }
 */

import { EarTracker } from './smoothing.js';

const ort = (typeof window !== 'undefined' && window.ort) ||
            (typeof globalThis !== 'undefined' && globalThis.ort) ||
            (typeof require !== 'undefined' ? require('onnxruntime-web') : null);

// Landmark groups for visualization
const LINESTRIP_GROUPS = [
    // iBUG ear scheme (Zhou & Zaferiou, FG 2017). Previously labelled
    // helix/antihelix/concha/tragus -- three of those four were wrong:
    // "tragus" was the superior crus, and the real tragus (35-38) is inside
    // the strip that was called "concha".
    { name: 'outer helix',   start: 0,  end: 20, color: '#00FF00' },
    { name: 'inner helix',   start: 20, end: 35, color: '#FF8000' },
    { name: 'concha border', start: 35, end: 50, color: '#0080FF' },
    { name: 'superior crus', start: 50, end: 55, color: '#FF0080' },
];

const DETECTOR_INPUT_SIZE = 128;
const LANDMARKER_INPUT_SIZE = 192;
const ROI_EXPAND = 1.3;

// Adaptive ROI refinement. ROI_EXPAND alone cannot frame the ear, because the
// detector box is not a fixed fraction of it: on real captures the ratio of
// true ear extent to detector box extent runs 1.02-1.53. Where the box is
// tight, a 1.3x crop is SMALLER than the ear, so landmarks jam against the crop
// border and can never reach the rim. The ROI is therefore re-derived from the
// landmarks, which do know where the ear is -- the same ROI-from-landmarks
// refinement MediaPipe uses for face and hand tracking.
const TRAIN_OCCUPANCY = 0.777;  // ear extent / crop side, over all 5,870 training samples
const ROI_OCC_TOL = 0.06;       // skip refinement when occupancy is already this close
const ROI_SATURATED = 0.88;     // above this the ear is clipped, so extent under-reads
const ROI_SAT_BOOST = 1.25;     // ...so grow faster than the measurement implies
const ROI_MAX_REFINE = 1;       // refinement passes; 1 lands within 1% of the fixed point

/** Next ROI side so the ear lands at the training occupancy. */
function refineRoiSide(side, extent) {
    const occ = side > 0 ? extent / side : 0;
    if (occ > ROI_SATURATED) return side * (occ / TRAIN_OCCUPANCY) * ROI_SAT_BOOST;
    return extent / TRAIN_OCCUPANCY;
}


class EarLandmarkerPipeline {
    /**
     * @param {Object} options
     * @param {number} options.confidenceThreshold - Detection confidence (default: 0.70)
     * @param {number} options.iouThreshold - NMS IoU threshold (default: 0.3)
     */
    constructor(options = {}) {
        this.confidenceThreshold = options.confidenceThreshold ?? 0.70;
        this.iouThreshold = options.iouThreshold ?? 0.3;
        // Suppress a box whose overlap covers this much of the SMALLER box.
        // Calibrated on 104 duplicate pairs logged from a live webcam run: they
        // measured IoU 0.19-0.30 and IoMin 0.39-0.59, i.e. they sat just under
        // BOTH a 0.30 IoU and a 0.60 IoMin threshold, which is why an earlier
        // 0.60 setting never fired. 0.35 catches all 104 with margin below the
        // observed minimum of 0.388, while two genuinely different ears measure
        // ~0.00 (they are a head-width apart), so the two cases stay separable.
        this.ioMinThreshold = options.ioMinThreshold ?? 0.35;
        this.debug = options.debug ?? false;
        // Re-derive the ROI from the landmarks instead of trusting the detector
        // box. Pass { refineRoi: false } for the old single-pass behaviour.
        this.refineRoi = options.refineRoi ?? true;
        this._roiExpand = new Map();   // trackId -> expansion that worked last frame
        this.minAspectRatio = options.minAspectRatio ?? 0.35;
        this.maxAspectRatio = options.maxAspectRatio ?? 1.4;
        this.minSizeFrac = options.minSizeFrac ?? 0.03;
        this.maxSizeFrac = options.maxSizeFrac ?? 0.55;
        this.detectorSession = null;
        this.landmarkerSession = null;
        this.isLoaded = false;
        // Temporal smoothing. Box wobble is ~90% of frame-to-frame jitter, so
        // boxes are smoothed BEFORE cropping and landmarks after, in frame
        // coords. Pass { smooth: false } to disable.
        this.tracker = (options.smooth ?? true)
            ? new EarTracker(options.smoothing ?? {})
            : null;
    }

    /** Clear smoothing state, e.g. when switching between webcam and an image. */
    resetSmoothing() { if (this.tracker) this.tracker.reset(); }

    /**
     * Load both ONNX models
     * @param {string} detectorPath - Path to BlazeEar_web.onnx
     * @param {string} landmarkerPath - Path to EarLandmarker_web.onnx
     */
    async load(detectorPath, landmarkerPath, sessionOptions = {}) {
        if (!ort) {
            throw new Error('ONNX Runtime Web not found. Include ort.min.js.');
        }

        const defaultOptions = {
            executionProviders: ['wasm', 'webgl'],
            graphOptimizationLevel: 'all',
        };
        const opts = { ...defaultOptions, ...sessionOptions };

        [this.detectorSession, this.landmarkerSession] = await Promise.all([
            ort.InferenceSession.create(detectorPath, opts),
            ort.InferenceSession.create(landmarkerPath, opts),
        ]);

        this.isLoaded = true;
        console.log('EarLandmarker pipeline loaded');
        console.log('Detector outputs:', this.detectorSession.outputNames);
        console.log('Landmarker outputs:', this.landmarkerSession.outputNames);
    }

    /**
     * Run full pipeline: detect ears, crop, predict landmarks
     * @param {HTMLImageElement|HTMLVideoElement|HTMLCanvasElement} source
     * @returns {Promise<Array>} Array of { bbox, confidence, landmarks }
     */
    async detect(source, timestamp = null) {
        if (!this.isLoaded) throw new Error('Models not loaded. Call load() first.');

        const { canvas, width, height } = this._sourceToCanvas(source);
        const ctx = canvas.getContext('2d');

        // Stage 1: BlazeEar detection
        let detections = await this._runDetector(canvas, width, height);

        // Smooth the boxes before cropping: a steady crop means the landmarker
        // sees a consistent input frame to frame, which is where most of the
        // visible jitter comes from.
        const t = (timestamp ?? performance.now()) / 1000;
        let trackIds = [];
        if (this.tracker) {
            trackIds = this.tracker.assign(detections);
            detections = detections.map((d, i) => ({
                ...d, ...this.tracker.smoothBox(trackIds[i], d, t),
            }));
        }

        // Stage 2: For each detection, crop and run landmarker
        const results = [];
        for (let di = 0; di < detections.length; di++) {
            const det = detections[di];
            // Expand bbox for context
            const bw = det.xmax - det.xmin;
            const bh = det.ymax - det.ymin;
            const cx = (det.xmin + det.xmax) / 2;
            const cy = (det.ymin + det.ymax) / 2;
            // Seed from what this track needed last frame, so video settles to
            // one pass instead of paying for refinement on every frame.
            const tid = trackIds ? trackIds[di] : undefined;
            let side = Math.max(bw, bh) *
                (this._roiExpand.get(tid) ?? ROI_EXPAND);
            let roiX = cx, roiY = cy;
            let frameLandmarks = null, pointConfidence = null;

            for (let attempt = 0; attempt <= ROI_MAX_REFINE; attempt++) {
                // The ROI must stay SQUARE. Clamping it to the frame instead
                // would make the crop non-square, and resizing that to 192x192
                // stretches the ear along one axis -- a distortion the model
                // never saw in training. Near a frame edge that costs ~17%
                // accuracy, and 15% of real ears sit close enough to an edge to
                // trigger it. So the window is kept whole and the part outside
                // the frame is filled with grey 128, matching dataset.py.
                const n = Math.round(side);
                if (n < 16) break;
                const x1 = Math.round(roiX - side / 2);
                const y1 = Math.round(roiY - side / 2);

                const sx1 = Math.max(0, x1);
                const sy1 = Math.max(0, y1);
                const sx2 = Math.min(width, x1 + n);
                const sy2 = Math.min(height, y1 + n);
                if (sx2 - sx1 < 8 || sy2 - sy1 < 8) break;

                const cropCanvas = document.createElement('canvas');
                cropCanvas.width = n;
                cropCanvas.height = n;
                const cropCtx = cropCanvas.getContext('2d');
                cropCtx.fillStyle = 'rgb(128,128,128)';
                cropCtx.fillRect(0, 0, n, n);
                cropCtx.drawImage(canvas, sx1, sy1, sx2 - sx1, sy2 - sy1,
                                  sx1 - x1, sy1 - y1, sx2 - sx1, sy2 - sy1);

                const out = await this._runLandmarker(cropCanvas, n, n);
                pointConfidence = out.pointConfidence;
                frameLandmarks = out.landmarks.map(pt => ({
                    x: pt.x * n + x1,
                    y: pt.y * n + y1,
                }));

                if (!this.refineRoi || attempt === ROI_MAX_REFINE) break;

                const xs = frameLandmarks.map(q => q.x);
                const ys = frameLandmarks.map(q => q.y);
                const exX = Math.max(...xs) - Math.min(...xs);
                const exY = Math.max(...ys) - Math.min(...ys);
                const extent = Math.max(exX, exY);
                if (Math.abs(extent / n - TRAIN_OCCUPANCY) <= ROI_OCC_TOL) break;

                side = refineRoiSide(n, extent);
                roiX = (Math.min(...xs) + Math.max(...xs)) / 2;
                roiY = (Math.min(...ys) + Math.max(...ys)) / 2;
            }

            if (!frameLandmarks) continue;
            if (tid !== undefined && Math.max(bw, bh) > 0) {
                this._roiExpand.set(tid, side / Math.max(bw, bh));
            }

            // Smooth in frame coords, so the filter sees real motion rather
            // than motion induced by the crop moving underneath it.
            if (this.tracker) {
                frameLandmarks = this.tracker.smoothLandmarks(
                    trackIds[di], frameLandmarks, t, pointConfidence);
            }

            results.push({
                bbox: { xmin: det.xmin, ymin: det.ymin, xmax: det.xmax, ymax: det.ymax },
                confidence: det.confidence,
                landmarks: frameLandmarks,
                pointConfidence,
            });
        }

        return results;
    }

    /**
     * Run BlazeEar detector
     * @private
     */
    async _runDetector(canvas, width, height) {
        // Preprocess: resize to 256 with padding, then to 128
        const maxDim = Math.max(height, width);
        const scale = maxDim / 256.0;
        const newH = Math.round(height / scale);
        const newW = Math.round(width / scale);
        const padH1 = Math.floor((256 - newH) / 2);
        const padW1 = Math.floor((256 - newW) / 2);
        const padY = padH1 * scale;
        const padX = padW1 * scale;

        const canvas256 = document.createElement('canvas');
        canvas256.width = 256;
        canvas256.height = 256;
        const ctx256 = canvas256.getContext('2d');
        ctx256.fillStyle = 'black';
        ctx256.fillRect(0, 0, 256, 256);
        ctx256.drawImage(canvas, padW1, padH1, newW, newH);

        const canvas128 = document.createElement('canvas');
        canvas128.width = 128;
        canvas128.height = 128;
        canvas128.getContext('2d').drawImage(canvas256, 0, 0, 128, 128);

        const imgData = canvas128.getContext('2d').getImageData(0, 0, 128, 128);
        const pixels = imgData.data;
        const tensorData = new Float32Array(3 * 128 * 128);
        for (let i = 0; i < 128 * 128; i++) {
            tensorData[i] = pixels[i * 4];                         // R
            tensorData[128 * 128 + i] = pixels[i * 4 + 1];        // G
            tensorData[2 * 128 * 128 + i] = pixels[i * 4 + 2];    // B
        }

        const feeds = {
            'image': new ort.Tensor('float32', tensorData, [1, 3, 128, 128]),
            'scale': new ort.Tensor('float32', [scale], []),
            'pad_y': new ort.Tensor('float32', [padY], []),
            'pad_x': new ort.Tensor('float32', [padX], []),
        };

        const results = await this.detectorSession.run(feeds);
        const boxes = results.boxes.data;
        const scores = results.scores.data;

        // Filter by confidence and apply NMS
        const candidates = [];
        for (let i = 0; i < 896; i++) {
            if (scores[i] >= this.confidenceThreshold) {
                candidates.push({
                    ymin: boxes[i * 4], xmin: boxes[i * 4 + 1],
                    ymax: boxes[i * 4 + 2], xmax: boxes[i * 4 + 3],
                    confidence: scores[i],
                });
            }
        }
        candidates.sort((a, b) => b.confidence - a.confidence);

        const nmsed = this._nms(candidates);

        // Clamp and filter geometry
        for (const det of nmsed) {
            det.ymin = Math.max(0, Math.min(det.ymin, height));
            det.xmin = Math.max(0, Math.min(det.xmin, width));
            det.ymax = Math.max(0, Math.min(det.ymax, height));
            det.xmax = Math.max(0, Math.min(det.xmax, width));
        }

        return nmsed.filter(det => {
            const w = det.xmax - det.xmin;
            const h = det.ymax - det.ymin;
            if (h < 1) return false;
            const aspect = w / h;
            const sizeFrac = Math.max(w, h) / Math.max(width, height);
            return aspect >= this.minAspectRatio && aspect <= this.maxAspectRatio &&
                   sizeFrac >= this.minSizeFrac && sizeFrac <= this.maxSizeFrac;
        });
    }

    /**
     * Run EarLandmarker on a cropped ear ROI
     * @private
     */
    async _runLandmarker(cropCanvas, cropW, cropH) {
        // Resize to 192x192
        const canvas192 = document.createElement('canvas');
        canvas192.width = LANDMARKER_INPUT_SIZE;
        canvas192.height = LANDMARKER_INPUT_SIZE;
        canvas192.getContext('2d').drawImage(cropCanvas, 0, 0, LANDMARKER_INPUT_SIZE, LANDMARKER_INPUT_SIZE);

        const imgData = canvas192.getContext('2d').getImageData(0, 0, LANDMARKER_INPUT_SIZE, LANDMARKER_INPUT_SIZE);
        const pixels = imgData.data;
        const size = LANDMARKER_INPUT_SIZE;
        const tensorData = new Float32Array(3 * size * size);

        // Convert to CHW, normalize to [-1, 1]
        for (let i = 0; i < size * size; i++) {
            tensorData[i] = (pixels[i * 4] / 255.0 - 0.5) / 0.5;
            tensorData[size * size + i] = (pixels[i * 4 + 1] / 255.0 - 0.5) / 0.5;
            tensorData[2 * size * size + i] = (pixels[i * 4 + 2] / 255.0 - 0.5) / 0.5;
        }

        const feeds = {
            'image': new ort.Tensor('float32', tensorData, [1, 3, size, size]),
        };

        const results = await this.landmarkerSession.run(feeds);
        const lmData = results.landmarks.data;  // (1, 55, 2) flattened
        // The heatmap model also emits per-point confidence; a GAP-head model
        // exported earlier does not, so treat it as optional.
        const pointConfidence = results.confidence ? results.confidence.data : null;

        const landmarks = [];
        for (let i = 0; i < 55; i++) {
            landmarks.push({
                x: lmData[i * 2],      // normalized [0, 1]
                y: lmData[i * 2 + 1],
            });
        }
        return { landmarks, pointConfidence };
    }

    /** @private */
    _nms(candidates) {
        const selected = [];
        const suppressed = new Set();
        for (let i = 0; i < candidates.length; i++) {
            if (suppressed.has(i)) continue;
            selected.push(candidates[i]);
            for (let j = i + 1; j < candidates.length; j++) {
                if (suppressed.has(j)) continue;
                // Plain IoU misses the nested / strongly-offset duplicates the
                // detector produces on a single ear: a small box inside a large
                // one scores IoU = areaSmall/areaLarge, which falls under any
                // reasonable threshold once the larger box is ~3x the smaller.
                // Intersection-over-minimum catches exactly that case, while
                // staying near zero for two genuinely different ears, which are
                // far apart in frame.
                const iou = this._iou(candidates[i], candidates[j]);
                const iomin = this._ioMin(candidates[i], candidates[j]);
                if (iou > this.iouThreshold || iomin > this.ioMinThreshold) {
                    suppressed.add(j);
                }
            }
        }
        if (this.debug && selected.length > 1) {
            for (let i = 0; i < selected.length; i++) {
                for (let j = i + 1; j < selected.length; j++) {
                    console.log('[nms] surviving pair',
                        { a: this._boxStr(selected[i]), b: this._boxStr(selected[j]),
                          iou: +this._iou(selected[i], selected[j]).toFixed(3),
                          ioMin: +this._ioMin(selected[i], selected[j]).toFixed(3) });
                }
            }
        }
        return selected;
    }

    /** @private Intersection over the smaller box's area: catches containment. */
    _ioMin(a, b) {
        const x1 = Math.max(a.xmin, b.xmin), y1 = Math.max(a.ymin, b.ymin);
        const x2 = Math.min(a.xmax, b.xmax), y2 = Math.min(a.ymax, b.ymax);
        const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
        const areaA = (a.xmax - a.xmin) * (a.ymax - a.ymin);
        const areaB = (b.xmax - b.xmin) * (b.ymax - b.ymin);
        const minArea = Math.min(areaA, areaB);
        return minArea > 0 ? inter / minArea : 0;
    }

    /** @private */
    _boxStr(d) {
        return [Math.round(d.xmin), Math.round(d.ymin), Math.round(d.xmax),
                Math.round(d.ymax), +d.confidence.toFixed(3)].join(',');
    }

    /** @private */
    _iou(a, b) {
        const x1 = Math.max(a.xmin, b.xmin), y1 = Math.max(a.ymin, b.ymin);
        const x2 = Math.min(a.xmax, b.xmax), y2 = Math.min(a.ymax, b.ymax);
        const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
        const areaA = (a.xmax - a.xmin) * (a.ymax - a.ymin);
        const areaB = (b.xmax - b.xmin) * (b.ymax - b.ymin);
        const union = areaA + areaB - inter;
        return union > 0 ? inter / union : 0;
    }

    /** @private */
    _sourceToCanvas(source) {
        let width, height;
        if (source instanceof HTMLVideoElement) {
            width = source.videoWidth;
            height = source.videoHeight;
        } else if (source instanceof HTMLImageElement) {
            width = source.naturalWidth || source.width;
            height = source.naturalHeight || source.height;
        } else if (source instanceof HTMLCanvasElement) {
            return { canvas: source, width: source.width, height: source.height };
        } else {
            throw new Error('Unsupported source type');
        }
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        canvas.getContext('2d').drawImage(source, 0, 0);
        return { canvas, width, height };
    }

    /**
     * Draw results on a canvas
     * @param {CanvasRenderingContext2D} ctx
     * @param {Array} results - Output from detect()
     * @param {Object} options
     */
    drawResults(ctx, results, options = {}) {
        const lineWidth = options.lineWidth || 2;
        const pointRadius = options.pointRadius ?? 1.8;
        const pointLineWidth = options.pointLineWidth ?? 1;
        const showBbox = options.showBbox ?? true;
        const showConfidence = options.showConfidence ?? true;
        const fontSize = options.fontSize || 14;

        for (const r of results) {
            const { bbox, confidence, landmarks } = r;

            // Bounding box
            if (showBbox) {
                ctx.strokeStyle = '#00FF00';
                ctx.lineWidth = lineWidth;
                ctx.strokeRect(bbox.xmin, bbox.ymin,
                    bbox.xmax - bbox.xmin, bbox.ymax - bbox.ymin);

                if (showConfidence) {
                    ctx.font = `${fontSize}px Arial`;
                    const label = `${(confidence * 100).toFixed(1)}%`;
                    const tw = ctx.measureText(label).width;
                    ctx.fillStyle = '#00FF00';
                    ctx.fillRect(bbox.xmin, bbox.ymin - fontSize - 4, tw + 6, fontSize + 4);
                    ctx.fillStyle = '#000';
                    ctx.fillText(label, bbox.xmin + 3, bbox.ymin - 4);
                }
            }

            // Landmarks as connected linestrips
            for (const group of LINESTRIP_GROUPS) {
                const pts = landmarks.slice(group.start, group.end);
                ctx.strokeStyle = group.color;
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(pts[0].x, pts[0].y);
                for (let i = 1; i < pts.length; i++) {
                    ctx.lineTo(pts[i].x, pts[i].y);
                }
                ctx.stroke();

                // Unfilled rings: at 55 points on a small ear, filled discs merge
                // into a blob and hide where each landmark actually sits.
                ctx.lineWidth = pointLineWidth;
                for (const pt of pts) {
                    ctx.beginPath();
                    ctx.arc(pt.x, pt.y, pointRadius, 0, 2 * Math.PI);
                    ctx.stroke();
                }
            }
        }
    }

    async dispose() {
        this.detectorSession = null;
        this.landmarkerSession = null;
        this.isLoaded = false;
        if (this.tracker) this.tracker.reset();
    }
}


async function createPipeline(detectorPath, landmarkerPath, options = {}) {
    const pipeline = new EarLandmarkerPipeline(options);
    await pipeline.load(detectorPath, landmarkerPath);
    return pipeline;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { EarLandmarkerPipeline, createPipeline, LINESTRIP_GROUPS };
} else if (typeof window !== 'undefined') {
    window.EarLandmarkerPipeline = EarLandmarkerPipeline;
    window.createEarLandmarkerPipeline = createPipeline;
}

export { EarLandmarkerPipeline, createPipeline, LINESTRIP_GROUPS };
