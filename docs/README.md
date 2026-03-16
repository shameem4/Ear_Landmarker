# Ear Landmarker JavaScript Demo

Browser-based ear landmark detection using ONNX Runtime Web.
Two-stage pipeline: BlazeEar detects ears, EarLandmarker regresses 55 landmarks per ear.

## Quick Start

1. **Export the ONNX model** (if not already done):
   ```bash
   cd "Ear Landmarker"
   python export_onnx.py
   ```

2. **Copy the BlazeEar detector model**:
   ```bash
   cp ../BlazeEar/docs/BlazeEar_web.onnx docs/
   ```

3. **Start a local server**:
   ```bash
   python -m http.server 8000 -d docs
   ```

4. **Open in browser**: http://localhost:8000/

5. **Use the demo**:
   - Click "Start Webcam" for live detection + landmarks
   - Or upload an image for single-frame analysis

## Models

The demo loads two ONNX models:

| Model | Input | Output | Purpose |
|-------|-------|--------|---------|
| `BlazeEar_web.onnx` | (1, 3, 128, 128) | boxes (896, 4) + scores (896,) | Ear detection |
| `EarLandmarker_web.onnx` | (1, 3, 192, 192) | landmarks (1, 55, 2) | Landmark regression |

Pipeline: BlazeEar detection -> NMS -> ROI crop (1.3x expand) -> EarLandmarker -> 55 landmarks mapped to frame coordinates.

## Usage in Your Project

```html
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort.min.js"></script>

<script type="module">
import { EarLandmarkerPipeline } from './earlandmarker_inference.js';

const pipeline = new EarLandmarkerPipeline({
    confidenceThreshold: 0.70,
    iouThreshold: 0.3,
});

await pipeline.load('BlazeEar_web.onnx', 'EarLandmarker_web.onnx');

// Detect from video, canvas, or image element
const results = await pipeline.detect(videoElement);

// Each result: { bbox, confidence, landmarks }
// bbox: { xmin, ymin, xmax, ymax } in pixels
// landmarks: Array of 55 { x, y } in pixels
console.log(results);

// Draw on canvas
pipeline.drawResults(canvasCtx, results);
</script>
```

## API

### EarLandmarkerPipeline

```javascript
const pipeline = new EarLandmarkerPipeline(options);
```

**Options:**
- `confidenceThreshold` (default: 0.70) - Minimum detection confidence
- `iouThreshold` (default: 0.3) - NMS IoU threshold

**Methods:**
- `load(detectorPath, landmarkerPath)` - Load both ONNX models
- `detect(source)` - Run full pipeline on image/video/canvas
- `drawResults(ctx, results, options)` - Draw boxes and landmarks on canvas

**Draw options:**
- `lineWidth` (default: 2) - Bbox line width
- `pointRadius` (default: 3) - Landmark dot radius
- `showBbox` (default: true) - Draw bounding boxes
- `showConfidence` (default: true) - Show confidence labels
- `fontSize` (default: 14) - Label font size

### Landmark Groups

| Group | Indices | Color |
|-------|---------|-------|
| Helix | 0-19 | Green |
| Antihelix | 20-34 | Orange |
| Concha | 35-49 | Blue |
| Tragus | 50-54 | Pink |

## Regenerating the ONNX Model

```bash
python export_onnx.py
python export_onnx.py --checkpoint path/to/model.ckpt
```
