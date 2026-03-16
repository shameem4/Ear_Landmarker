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

const ort = (typeof window !== 'undefined' && window.ort) ||
            (typeof globalThis !== 'undefined' && globalThis.ort) ||
            (typeof require !== 'undefined' ? require('onnxruntime-web') : null);

// Landmark groups for visualization
const LINESTRIP_GROUPS = [
    { name: 'helix',     start: 0,  end: 20, color: '#00FF00' },
    { name: 'antihelix', start: 20, end: 35, color: '#FF8000' },
    { name: 'concha',    start: 35, end: 50, color: '#0080FF' },
    { name: 'tragus',    start: 50, end: 55, color: '#FF0080' },
];

const DETECTOR_INPUT_SIZE = 128;
const LANDMARKER_INPUT_SIZE = 192;
const ROI_EXPAND = 1.3;


class EarLandmarkerPipeline {
    /**
     * @param {Object} options
     * @param {number} options.confidenceThreshold - Detection confidence (default: 0.70)
     * @param {number} options.iouThreshold - NMS IoU threshold (default: 0.3)
     */
    constructor(options = {}) {
        this.confidenceThreshold = options.confidenceThreshold ?? 0.70;
        this.iouThreshold = options.iouThreshold ?? 0.3;
        this.minAspectRatio = options.minAspectRatio ?? 0.35;
        this.maxAspectRatio = options.maxAspectRatio ?? 1.4;
        this.minSizeFrac = options.minSizeFrac ?? 0.03;
        this.maxSizeFrac = options.maxSizeFrac ?? 0.55;
        this.detectorSession = null;
        this.landmarkerSession = null;
        this.isLoaded = false;
    }

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
    async detect(source) {
        if (!this.isLoaded) throw new Error('Models not loaded. Call load() first.');

        const { canvas, width, height } = this._sourceToCanvas(source);
        const ctx = canvas.getContext('2d');

        // Stage 1: BlazeEar detection
        const detections = await this._runDetector(canvas, width, height);

        // Stage 2: For each detection, crop and run landmarker
        const results = [];
        for (const det of detections) {
            // Expand bbox for context
            const bw = det.xmax - det.xmin;
            const bh = det.ymax - det.ymin;
            const cx = (det.xmin + det.xmax) / 2;
            const cy = (det.ymin + det.ymax) / 2;
            const side = Math.max(bw, bh) * ROI_EXPAND;
            const x1 = Math.max(0, Math.round(cx - side / 2));
            const y1 = Math.max(0, Math.round(cy - side / 2));
            const x2 = Math.min(width, Math.round(cx + side / 2));
            const y2 = Math.min(height, Math.round(cy + side / 2));

            if (x2 - x1 < 16 || y2 - y1 < 16) continue;

            // Crop ROI
            const cropW = x2 - x1;
            const cropH = y2 - y1;
            const cropCanvas = document.createElement('canvas');
            cropCanvas.width = cropW;
            cropCanvas.height = cropH;
            cropCanvas.getContext('2d').drawImage(canvas, x1, y1, cropW, cropH, 0, 0, cropW, cropH);

            // Run landmarker on crop
            const landmarks = await this._runLandmarker(cropCanvas, cropW, cropH);

            // Map landmarks back to full frame coords
            const frameLandmarks = landmarks.map(pt => ({
                x: pt.x * cropW + x1,
                y: pt.y * cropH + y1,
            }));

            results.push({
                bbox: { xmin: det.xmin, ymin: det.ymin, xmax: det.xmax, ymax: det.ymax },
                confidence: det.confidence,
                landmarks: frameLandmarks,
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

        const landmarks = [];
        for (let i = 0; i < 55; i++) {
            landmarks.push({
                x: lmData[i * 2],      // normalized [0, 1]
                y: lmData[i * 2 + 1],
            });
        }
        return landmarks;
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
                if (this._iou(candidates[i], candidates[j]) > this.iouThreshold) {
                    suppressed.add(j);
                }
            }
        }
        return selected;
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
        const pointRadius = options.pointRadius || 3;
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

                ctx.fillStyle = group.color;
                for (const pt of pts) {
                    ctx.beginPath();
                    ctx.arc(pt.x, pt.y, pointRadius, 0, 2 * Math.PI);
                    ctx.fill();
                }
            }
        }
    }

    async dispose() {
        this.detectorSession = null;
        this.landmarkerSession = null;
        this.isLoaded = false;
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
