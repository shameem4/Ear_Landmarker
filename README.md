# Ear Landmarker

Real-time 55-point ear landmark regression using a BlazeBlock backbone (FaceMesh architecture pattern). Runs as a two-stage pipeline: BlazeEar detector finds ears, then EarLandmarker regresses landmarks on each crop.

## Pipeline

```
Webcam/Image -> BlazeEar detector (128x128) -> ROI crop (1.3x expand) -> EarLandmarker (192x192) -> 55 landmarks
```

The detector (BlazeEar, separate project) produces bounding boxes with NMS. Each box is expanded 30% for context, cropped, and fed to the landmarker.

## Architecture

EarLandmarker follows the MediaPipe FaceMesh pattern -- depthwise separable BlazeBlocks with skip connections, progressive channel expansion, and direct coordinate regression via sigmoid output.

| Stage   | Channels | Spatial | Blocks |
|---------|----------|---------|--------|
| conv0   | 3 -> 24  | 192->96 | 1 conv |
| stage0  | 24       | 96      | 2      |
| stage1  | 24 -> 48 | 96->48  | 4      |
| stage2  | 48 -> 96 | 48->24  | 4      |
| stage3  | 96 -> 128| 24->12  | 4      |
| stage4  | 128->192 | 12->6   | 3      |
| head    | 192->110 | GAP+FC  | -      |

- **312K parameters** -- designed for real-time inference
- Output: 55 x 2 coordinates in [0, 1], mapped back to frame pixels
- BlazeBlock: DepthwiseConv -> BN -> PointwiseConv -> BN -> Skip -> ReLU

## Landmark Layout

55 points organized as 4 linestrips:

| Group     | Indices | Points | Color (viz) |
|-----------|---------|--------|-------------|
| Helix     | 0-19    | 20     | Green       |
| Antihelix | 20-34   | 15     | Orange      |
| Concha    | 35-49   | 15     | Blue        |
| Tragus    | 50-54   | 5      | Pink        |

## Data

5,870 samples unified from 4 sources, deduplicated by image hash:

| Source       | Samples | Format     | Notes                          |
|--------------|---------|------------|--------------------------------|
| collectionB  | 3,153   | .pts       | Pre-cropped 256x256 ears       |
| AudioEar2D   | 2,000   | LabelMe    | Synthetic from FFHQ, 299x299  |
| collectionA  | 605     | .pts       | Full images, auto-cropped      |
| AudioEar3D   | 112     | LabelMe    | 3D-scanned ears, 187x186      |

Dropped: `coco_keypoint` (incompatible point ordering -- landmark indices don't match the linestrip convention used by other sources).

**Preprocessing pipeline:** `preprocess.py` ingests all formats, deduplicates, normalizes landmarks to [0,1], writes `landmarks.npy` (memory-mapped) + `manifest.csv`. `validate.py` runs 7 automated QA checks (image integrity, bounds, geometry, linestrip smoothness, statistical outliers, cross-source consistency, resolution audit). `split.py` creates stratified 85/15 train/val splits.

## Training

| Setting | Value |
|---------|-------|
| Loss | Wing loss (w=0.04, eps=0.01) |
| Optimizer | AdamW (lr=1e-3, wd=1e-4) |
| Schedule | Cosine annealing |
| Precision | 16-mixed AMP |
| Augmentation | Horizontal flip, translation (5%), rotation (+/-15 deg), color jitter, bbox jitter (10%) |
| Early stopping | Patience 50 on val/nme |

**Results:** val NME 0.0307 (~5.9px at 192px) after 345 epochs, no overfitting (train-val gap 0.0014).

## Usage

```bash
# Webcam demo
python inference.py webcam
python inference.py webcam --camera 1 --confidence 0.5

# Single image
python inference.py image path/to/ear.jpg
python inference.py image path/to/ear.jpg --output result.jpg

# Train
python train.py
python train.py --epochs 500 --batch-size 128 --lr 1e-3
python train.py --resume best

# Data pipeline
python data/preprocess.py
python data/validate.py
python data/split.py
```

## Performance

| Metric | Value |
|--------|-------|
| val NME | 0.0307 |
| GPU inference | 579 FPS |
| CPU inference | 226 FPS |
| Parameters | 312K |
| Input size | 192x192 |

## Dependencies

PyTorch, PyTorch Lightning, OpenCV, NumPy, Pillow, torchvision, tqdm

BlazeEar detector weights expected at `../BlazeEar/runs/checkpoints/BlazeEar_best.pth`.
