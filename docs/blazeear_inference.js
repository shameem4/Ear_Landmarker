/**
 * BlazeEar Inference Module for JavaScript/Browser
 * 
 * Pure JavaScript implementation of the BlazeEar ear detection pipeline.
 * Uses ONNX Runtime Web for model inference.
 * 
 * Usage:
 *   import { BlazeEarTwoStage } from './blazeear_inference.js';
 *
 *   const detector = new BlazeEarTwoStage();
 *   await detector.load('BlazeFace_web.onnx', 'BlazeEar_web.onnx');
 *
 *   // From video element, canvas, or ImageData
 *   const detections = await detector.detect(imageSource);
 *   // Returns: Array of { ymin, xmin, ymax, xmax, confidence },
 *   // carrying a faceCount property.
 *
 * BlazeEarTwoStage and createTwoStageDetector are the only exports. The
 * single-stage BlazeEarInference below is the building block they run, not an
 * entry point -- see docs/README.md for why it is not offered on its own.
 */

// Check for ONNX Runtime
const ort = (typeof window !== 'undefined' && window.ort) || 
            (typeof globalThis !== 'undefined' && globalThis.ort) ||
            (typeof require !== 'undefined' ? require('onnxruntime-web') : null);

/**
 * BlazeEar Ear Detection Pipeline
 */
class BlazeEarInference {
    /**
     * One detector pass over whatever it is handed. NOT exported: it is the
     * stage BlazeEarTwoStage runs twice, once on the frame for faces and once
     * per crop for ears, and it is not an entry point.
     *
     * Run over a whole frame on its own it scores mAP@0.5 0.3142 against the
     * two-stage pipeline's 0.5809, and on the images the face stage cannot
     * reach -- where it would be the only option -- it scores 0.0284, because
     * those ears are a median 4.4 px. There is no regime where it is the right
     * choice, so it is not offered as one.
     *
     * Create a BlazeEar detector instance
     * @param {Object} options - Configuration options
     * @param {number} options.confidenceThreshold - Minimum confidence for detections (default: 0.70)
     * @param {number} options.iouThreshold - IoU threshold for NMS (default: 0.3)
     */
    constructor(options = {}) {
        this.confidenceThreshold = options.confidenceThreshold ?? 0.70;
        this.iouThreshold = options.iouThreshold ?? 0.3;
        // Geometric filter, off by default to match the Python paths. With the
        // previous bounds it rejected 25.86% of real ears, and it ran only here
        // and in BlazeEar.process, so the deployed page and the measured
        // pipeline disagreed. Bounds are the 0.5/99.5 percentiles of the real
        // box distribution.
        this.geometryFilterEnabled = options.geometryFilterEnabled ?? false;
        this.minAspectRatio = options.minAspectRatio ?? 0.14;
        this.maxAspectRatio = options.maxAspectRatio ?? 2.57;
        this.minSizeFrac = options.minSizeFrac ?? 0.022;
        this.maxSizeFrac = options.maxSizeFrac ?? 0.733;
        this.inputSize = 128;
        this.session = null;
        this.isLoaded = false;
    }

    /**
     * Load the ONNX model
     * @param {string} modelPath - Path or URL to the ONNX model file
     * @param {Object} sessionOptions - ONNX Runtime session options
     */
    async load(modelPath, sessionOptions = {}) {
        if (!ort) {
            throw new Error('ONNX Runtime Web not found. Include ort.min.js or install onnxruntime-web');
        }

        const defaultOptions = {
            // Prefer WASM for better int64 support, fallback to WebGL
            executionProviders: ['wasm', 'webgl'],
            graphOptimizationLevel: 'all',
        };

        this.session = await ort.InferenceSession.create(
            modelPath,
            { ...defaultOptions, ...sessionOptions }
        );

        this.isLoaded = true;
        console.log('BlazeEar model loaded successfully');
        console.log('Input names:', this.session.inputNames);
        console.log('Output names:', this.session.outputNames);

        // Both shipped graphs output decoded boxes + scores and leave
        // thresholding and NMS to JavaScript. The e2e and "simple" variants
        // this used to accept were exported before v2 and have been removed;
        // fail loudly rather than take a graph whose outputs are not understood.
        const outputs = this.session.outputNames;
        if (!(outputs.includes('boxes') && outputs.includes('scores'))) {
            throw new Error(
                `Unsupported graph: expected outputs "boxes" and "scores", got ` +
                `[${outputs.join(', ')}]. Re-export with export_two_stage_web.py.`);
        }
    }

    /**
     * Preprocess an image for model input
     * @param {HTMLImageElement|HTMLVideoElement|HTMLCanvasElement|ImageData} source - Image source
     * @returns {Object} Preprocessed data with tensor and preprocessing params
     */
    preprocess(source) {
        // Get image data from various sources
        const { imageData, width, height } = this._getImageData(source);
        
        // Calculate preprocessing parameters
        const maxDim = Math.max(height, width);
        const scale = maxDim / 256.0;
        const newH = Math.round(height / scale);
        const newW = Math.round(width / scale);
        const padH = 256 - newH;
        const padW = 256 - newW;
        const padH1 = Math.floor(padH / 2);
        const padW1 = Math.floor(padW / 2);
        const padY = padH1 * scale;
        const padX = padW1 * scale;

        // Create 256x256 canvas for resize + pad
        const canvas256 = document.createElement('canvas');
        canvas256.width = 256;
        canvas256.height = 256;
        const ctx256 = canvas256.getContext('2d');
        ctx256.fillStyle = 'black';
        ctx256.fillRect(0, 0, 256, 256);

        // Draw resized image centered
        if (source instanceof ImageData) {
            // Create temporary canvas for ImageData
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = width;
            tempCanvas.height = height;
            tempCanvas.getContext('2d').putImageData(source, 0, 0);
            ctx256.drawImage(tempCanvas, padW1, padH1, newW, newH);
        } else {
            ctx256.drawImage(source, padW1, padH1, newW, newH);
        }

        // Create 128x128 canvas for final resize
        const canvas128 = document.createElement('canvas');
        canvas128.width = 128;
        canvas128.height = 128;
        const ctx128 = canvas128.getContext('2d');
        ctx128.drawImage(canvas256, 0, 0, 128, 128);

        // Get pixel data and convert to CHW float32 tensor
        const imgData = ctx128.getContext ? 
            ctx128.getImageData(0, 0, 128, 128) :
            canvas128.getContext('2d').getImageData(0, 0, 128, 128);
        
        const pixels = imgData.data;
        const tensorData = new Float32Array(3 * 128 * 128);

        // Convert RGBA to RGB CHW format (keep in [0, 255] range for the model)
        for (let y = 0; y < 128; y++) {
            for (let x = 0; x < 128; x++) {
                const srcIdx = (y * 128 + x) * 4;
                const r = pixels[srcIdx];
                const g = pixels[srcIdx + 1];
                const b = pixels[srcIdx + 2];

                tensorData[0 * 128 * 128 + y * 128 + x] = r;
                tensorData[1 * 128 * 128 + y * 128 + x] = g;
                tensorData[2 * 128 * 128 + y * 128 + x] = b;
            }
        }

        return {
            tensorData,
            scale,
            padY,
            padX,
            originalWidth: width,
            originalHeight: height
        };
    }

    /**
     * Get ImageData from various image sources
     * @private
     */
    _getImageData(source) {
        let width, height, imageData;

        if (source instanceof ImageData) {
            return { imageData: source, width: source.width, height: source.height };
        }

        if (source instanceof HTMLCanvasElement) {
            width = source.width;
            height = source.height;
            imageData = source.getContext('2d').getImageData(0, 0, width, height);
        } else if (source instanceof HTMLImageElement) {
            width = source.naturalWidth || source.width;
            height = source.naturalHeight || source.height;
            const canvas = document.createElement('canvas');
            canvas.width = width;
            canvas.height = height;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(source, 0, 0);
            imageData = ctx.getImageData(0, 0, width, height);
        } else if (source instanceof HTMLVideoElement) {
            width = source.videoWidth;
            height = source.videoHeight;
            const canvas = document.createElement('canvas');
            canvas.width = width;
            canvas.height = height;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(source, 0, 0);
            imageData = ctx.getImageData(0, 0, width, height);
        } else {
            throw new Error('Unsupported image source type');
        }

        return { imageData, width, height };
    }

    /**
     * Run detection on an image
     * @param {HTMLImageElement|HTMLVideoElement|HTMLCanvasElement|ImageData} source - Image source
     * @returns {Promise<Array>} Array of detection objects with {ymin, xmin, ymax, xmax, confidence}
     */
    async detect(source) {
        if (!this.isLoaded) {
            throw new Error('Model not loaded. Call load() first.');
        }

        // Preprocess
        const { tensorData, scale, padY, padX, originalWidth, originalHeight } = this.preprocess(source);

        // Create input tensor
        const imageTensor = new ort.Tensor('float32', tensorData, [1, 3, 128, 128]);

        const results = await this.session.run({
            'image': imageTensor,
            'scale': new ort.Tensor('float32', [scale], []),
            'pad_y': new ort.Tensor('float32', [padY], []),
            'pad_x': new ort.Tensor('float32', [padX], [])
        });

        // The graph returns all 896 decoded boxes; thresholding and NMS happen
        // here, because TopK and NMS emit int64 and ONNX Runtime Web rejects it.
        return this._processWebModelOutput(
            results.boxes.data,
            results.scores.data,
            originalWidth,
            originalHeight
        );
    }

    /**
     * Process web model output (boxes + scores) with JS NMS
     * @private
     */
    _processWebModelOutput(boxesData, scoresData, originalWidth, originalHeight) {
        const numAnchors = 896;
        const candidates = [];

        // Filter by confidence threshold
        for (let i = 0; i < numAnchors; i++) {
            const score = scoresData[i];
            if (score >= this.confidenceThreshold) {
                candidates.push({
                    ymin: boxesData[i * 4 + 0],
                    xmin: boxesData[i * 4 + 1],
                    ymax: boxesData[i * 4 + 2],
                    xmax: boxesData[i * 4 + 3],
                    confidence: score,
                    index: i
                });
            }
        }

        // Sort by confidence (descending)
        candidates.sort((a, b) => b.confidence - a.confidence);

        // Apply NMS
        const detections = this._nms(candidates, this.iouThreshold);

        // Clamp to image bounds, add convenience properties, then filter geometry
        for (const det of detections) {
            det.ymin = Math.max(0, Math.min(det.ymin, originalHeight));
            det.xmin = Math.max(0, Math.min(det.xmin, originalWidth));
            det.ymax = Math.max(0, Math.min(det.ymax, originalHeight));
            det.xmax = Math.max(0, Math.min(det.xmax, originalWidth));
            det.x = det.xmin;
            det.y = det.ymin;
            det.width = det.xmax - det.xmin;
            det.height = det.ymax - det.ymin;
            delete det.index;  // Remove internal property
        }

        return this._filterByGeometry(detections, originalWidth, originalHeight);
    }

    /**
     * Non-Maximum Suppression
     * @private
     */
    _nms(candidates, iouThreshold) {
        const selected = [];
        const suppressed = new Set();

        for (let i = 0; i < candidates.length; i++) {
            if (suppressed.has(i)) continue;

            const current = candidates[i];
            selected.push(current);

            for (let j = i + 1; j < candidates.length; j++) {
                if (suppressed.has(j)) continue;

                const other = candidates[j];
                const iou = this._computeIoU(current, other);

                if (iou > iouThreshold) {
                    suppressed.add(j);
                }
            }
        }

        return selected;
    }

    /**
     * Compute Intersection over Union
     * @private
     */
    _computeIoU(boxA, boxB) {
        const xA = Math.max(boxA.xmin, boxB.xmin);
        const yA = Math.max(boxA.ymin, boxB.ymin);
        const xB = Math.min(boxA.xmax, boxB.xmax);
        const yB = Math.min(boxA.ymax, boxB.ymax);

        const interWidth = Math.max(0, xB - xA);
        const interHeight = Math.max(0, yB - yA);
        const interArea = interWidth * interHeight;

        const areaA = (boxA.xmax - boxA.xmin) * (boxA.ymax - boxA.ymin);
        const areaB = (boxB.xmax - boxB.xmin) * (boxB.ymax - boxB.ymin);

        const unionArea = areaA + areaB - interArea;

        return unionArea > 0 ? interArea / unionArea : 0;
    }

    /**
     * Filter detections by aspect ratio and size relative to image dimensions.
     * @private
     */
    _filterByGeometry(detections, imageWidth, imageHeight) {
        if (!this.geometryFilterEnabled) return detections;
        const maxDim = Math.max(imageWidth, imageHeight);
        return detections.filter(det => {
            const w = det.xmax - det.xmin;
            const h = det.ymax - det.ymin;
            if (h < 1e-6) return false;
            const aspect = w / h;
            const sizeFrac = Math.max(w, h) / maxDim;
            return (
                aspect >= this.minAspectRatio &&
                aspect <= this.maxAspectRatio &&
                sizeFrac >= this.minSizeFrac &&
                sizeFrac <= this.maxSizeFrac
            );
        });
    }

    /**
     * Draw detections on a canvas
     * @param {CanvasRenderingContext2D} ctx - Canvas context to draw on
     * @param {Array} detections - Array of detection objects
     * @param {Object} options - Drawing options
     */
    drawDetections(ctx, detections, options = {}) {
        const color = options.color || '#00FF00';
        const lineWidth = options.lineWidth || 2;
        const showConfidence = options.showConfidence ?? true;
        const fontSize = options.fontSize || 14;

        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.font = `${fontSize}px Arial`;
        ctx.fillStyle = color;

        for (const det of detections) {
            // Draw bounding box
            ctx.strokeRect(det.xmin, det.ymin, det.width, det.height);

            // Draw confidence label
            if (showConfidence) {
                const label = `${(det.confidence * 100).toFixed(1)}%`;
                const labelWidth = ctx.measureText(label).width;
                
                // Background for label
                ctx.fillStyle = color;
                ctx.fillRect(det.xmin, det.ymin - fontSize - 4, labelWidth + 6, fontSize + 4);
                
                // Label text
                ctx.fillStyle = '#000000';
                ctx.fillText(label, det.xmin + 3, det.ymin - 4);
                ctx.fillStyle = color;
            }
        }
    }

    /**
     * Dispose of the model and free resources
     */
    async dispose() {
        if (this.session) {
            // ONNX Runtime Web doesn't have explicit dispose, but we can null the reference
            this.session = null;
            this.isLoaded = false;
        }
    }
}


/**
 * The two-stage pipeline: MediaPipe BlazeFace on the full frame, then the ear
 * model on a square crop around each detected face.
 *
 * This is the design the measurements picked. Asking one 128x128 detector to
 * find an ear in a whole frame gives it a median 14-pixel target; cropping to
 * a face first makes that 32 pixels, and on images no model trained on the
 * mAP@0.5 goes 0.3142 -> 0.5809.
 *
 * The cost is a recall ceiling: an ear whose face BlazeFace misses never
 * reaches the second stage, which is 1.3% of validation images at the default
 * threshold of 0.2 -- and 4.1% at 0.3, which is why the default is as low as
 * it is. Those ears are a median 4.4 px, so nothing else recovers them either.
 *
 * The two stages do NOT share anchors -- the face graph carries MediaPipe's
 * original squares, the ear graph the fitted ear priors -- but each has its
 * own baked in, so nothing here has to know about that.
 *
 * These must stay in step with FACE_CROP_* in utils/config.py; the Python
 * pipeline in evaluate_two_stage.py is the reference this mirrors.
 */
class BlazeEarTwoStage {
    constructor(options = {}) {
        this.expand = options.expand ?? 1.5;
        this.faceThreshold = options.faceThreshold ?? 0.2;
        this.maxFaces = options.maxFaces ?? 8;
        this.iouThreshold = options.iouThreshold ?? 0.3;
        this.face = new BlazeEarInference({
            confidenceThreshold: this.faceThreshold,
            iouThreshold: this.iouThreshold,
        });
        this.ear = new BlazeEarInference({
            confidenceThreshold: options.confidenceThreshold ?? 0.70,
            iouThreshold: this.iouThreshold,
        });
        this.isLoaded = false;
    }

    /**
     * @param {string} facePath - URL of the BlazeFace graph
     * @param {string} earPath - URL of the crop-trained ear graph
     */
    async load(facePath, earPath, sessionOptions = {}) {
        await Promise.all([
            this.face.load(facePath, sessionOptions),
            this.ear.load(earPath, sessionOptions),
        ]);
        this.isLoaded = true;
        return this;
    }

    /**
     * A square window at `expand` times the face box, clipped to the frame.
     * Square, because the ear model letterboxes its input anyway, and a
     * non-square crop would just spend resolution on padding.
     *
     * This duplicates crop_window() in make_face_crops.py, and the two must
     * agree or the browser crops differently from everything the model was
     * trained and measured on. The edge cases are pinned in
     * utils/tests/tests/test_face_crops.py::TestCropWindowIsPinned; both
     * implementations were checked against all of them.
     */
    _cropWindow(face, frameWidth, frameHeight) {
        const faceW = face.xmax - face.xmin;
        const faceH = face.ymax - face.ymin;
        // Clamp the side first so the window stays square wherever the frame
        // allows it, then shift the centre to keep it inside.
        const side = Math.min(
            this.expand * Math.max(faceW, faceH), frameWidth, frameHeight);
        const cx = (face.xmin + face.xmax) / 2;
        const cy = (face.ymin + face.ymax) / 2;
        const x0 = Math.round(Math.min(Math.max(cx - side / 2, 0), frameWidth - side));
        const y0 = Math.round(Math.min(Math.max(cy - side / 2, 0), frameHeight - side));
        return { x0, y0, side: Math.round(side) };
    }

    async detect(source) {
        if (!this.isLoaded) {
            throw new Error('Call load(facePath, earPath) before detect().');
        }

        const faces = await this.face.detect(source);
        const { width, height } = this.face._getImageData(source);
        if (faces.length === 0) {
            // No face means no crop, and therefore no detections. This is the
            // recall ceiling, reported rather than papered over: falling back
            // to a full-frame pass was measured and is a wash, contributing
            // about as many false positives as it recovers ears.
            return Object.assign([], { faceCount: 0 });
        }

        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const collected = [];

        for (const face of faces.slice(0, this.maxFaces)) {
            const { x0, y0, side } = this._cropWindow(face, width, height);
            if (side < 2) continue;
            canvas.width = side;
            canvas.height = side;
            ctx.clearRect(0, 0, side, side);
            ctx.drawImage(source, x0, y0, side, side, 0, 0, side, side);

            for (const ear of await this.ear.detect(canvas)) {
                // Crop coordinates back into the original frame.
                collected.push({
                    ...ear,
                    xmin: ear.xmin + x0, xmax: ear.xmax + x0,
                    ymin: ear.ymin + y0, ymax: ear.ymax + y0,
                });
            }
        }

        // Overlapping face crops can each report the same ear.
        const merged = this.ear._nms(
            collected.sort((a, b) => b.confidence - a.confidence),
            this.iouThreshold);
        return Object.assign(merged, { faceCount: faces.length });
    }

    drawDetections(ctx, detections, options = {}) {
        return this.ear.drawDetections(ctx, detections, options);
    }

    async dispose() {
        await Promise.all([this.face.dispose(), this.ear.dispose()]);
        this.isLoaded = false;
    }
}

/**
 * Load the two-stage pipeline. This is the entry point.
 * @returns {Promise<BlazeEarTwoStage>}
 */
async function createTwoStageDetector(facePath, earPath, options = {}) {
    const detector = new BlazeEarTwoStage(options);
    await detector.load(facePath, earPath);
    return detector;
}

// Export for different module systems
if (typeof module !== 'undefined' && module.exports) {
    // CommonJS
    module.exports = { BlazeEarTwoStage, createTwoStageDetector };
} else if (typeof window !== 'undefined') {
    // Browser global
    window.BlazeEarTwoStage = BlazeEarTwoStage;
    window.createBlazeEarTwoStageDetector = createTwoStageDetector;
}

export { BlazeEarTwoStage, createTwoStageDetector };
