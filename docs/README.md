# Ear Landmarker JavaScript Demo

Browser-based ear landmark detection using ONNX Runtime Web.
Two-stage pipeline: BlazeEar detects ears, EarLandmarker regresses 55 landmarks per ear.

## Quick Start

1. **Export the ONNX model** (if not already done):
   ```bash
   cd "Ear Landmarker"
   python export_onnx.py
   ```

2. **Copy the detector models** (both stages, plus the JS that drives them):
   ```bash
   cp ../BlazeEar/docs/BlazeEar_web.onnx ../BlazeEar/docs/BlazeFace_web.onnx docs/
   cp ../BlazeEar/docs/blazeear_inference.js docs/
   ```
   Regenerate them from BlazeEar with `python export_two_stage_web.py`.

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
| `BlazeFace_web.onnx` | (1, 3, 128, 128) + scale/pad | boxes (N, 4) + scores (N,) | Face detection (stage 1) |
| `BlazeEar_web.onnx` | (1, 3, 128, 128) + scale/pad | boxes (N, 4) + scores (N,) | Ear detection on the face crop (stage 2) |
| `EarLandmarker_web.onnx` | (1, 3, 192, 192) | landmarks (1, 55, 2) + confidence (1, 55) | Landmark prediction |

Pipeline: BlazeEar detection -> NMS -> box smoothing -> ROI crop -> EarLandmarker
-> refine ROI from the landmarks and re-run if the framing is off -> 55 landmarks
mapped to frame coordinates -> landmark smoothing.

The ROI is re-derived from the landmarks rather than trusted from the detector
box, because the box is not a fixed fraction of the ear (1.02-1.53 on real
captures) and a too-tight crop clips the landmarks against its own border. Set
`refineRoi: false` for the old single-pass behaviour.

The landmarker exports two outputs. `landmarks` are normalised [0,1] crop
coordinates; `confidence` is a per-landmark score derived from the heatmap's
spatial spread, used to weight temporal smoothing so uncertain points are
smoothed harder. Opset 14.

## Usage in Your Project

```html
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort.min.js"></script>

<script type="module">
import { EarLandmarkerPipeline } from './earlandmarker_inference.js';

const pipeline = new EarLandmarkerPipeline({
    confidenceThreshold: 0.70,
    iouThreshold: 0.3,
    ioMinThreshold: 0.35,   // catches nested duplicate boxes
    smooth: true,           // temporal smoothing; set false for stills
});

await pipeline.load('BlazeFace_web.onnx', 'BlazeEar_web.onnx',
                   'EarLandmarker_web.onnx');

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
- `ioMinThreshold` (default: 0.35) - NMS intersection-over-minimum threshold.
  Suppresses a box nested inside another on the same ear, which plain IoU misses:
  a small box inside a 3x larger one scores IoU below any sane threshold. 0.35 is
  calibrated against 104 duplicate pairs logged from real webcam runs.
- `smooth` (default: true) - One Euro temporal smoothing of boxes and landmarks.
  Set `false` for single images; call `reset()` when switching sources.
- `smoothing` (default: {}) - Overrides passed to the underlying filters
  (see `smoothing.js`)
- `refineRoi` (default: true) - Re-derive the crop from the landmarks so the ear
  lands at the 0.777 occupancy the model was trained on. Costs a second
  landmarker pass only when the first framing is off; on video each track seeds
  from the previous frame, so the steady state is one pass.
- `debug` (default: false) - Log suppressed duplicate pairs to the console

**Methods:**
- `load(facePath, detectorPath, landmarkerPath)` - Load all three ONNX models.
  Detection is two-stage as of BlazeEar v2, so a face graph is required; calling
  it with two arguments throws rather than silently mis-detecting.
- `detect(source)` - Run full pipeline on image/video/canvas
- `reset()` - Clear smoothing/tracking state, e.g. webcam -> image
- `drawResults(ctx, results, options)` - Draw boxes and landmarks on canvas

**Draw options:**
- `lineWidth` (default: 2) - Bbox line width
- `pointRadius` (default: 1.8) - Landmark ring radius
- `pointLineWidth` (default: 1) - Landmark ring stroke width
- `showBbox` (default: true) - Draw bounding boxes
- `showConfidence` (default: true) - Show confidence labels
- `fontSize` (default: 14) - Label font size

### Landmark Groups

| Strip | Indices | Color |
|-------|---------|-------|
| Outer helix | 0-19 | Green |
| Inner helix | 20-34 | Orange |
| Concha border | 35-49 | Blue |
| Superior crus | 50-54 | Pink |

Names follow the iBUG ear scheme (Zhou & Zaferiou, FG 2017). Each strip spans
several iBUG regions -- the real tragus is 35-38, inside the concha-border
strip. See [../README.md](../README.md) for the full mapping.

## Regenerating the ONNX Model

```bash
python export_onnx.py
python export_onnx.py --checkpoint path/to/model.ckpt
```

The shipped model is `v6_persp65` (test NME 0.0292). See [../RESULTS.md](../RESULTS.md)
for why that checkpoint rather than the nominally better one.
