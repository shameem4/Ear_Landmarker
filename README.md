# Ear Landmarker

**[Live Demo](https://shameem4.github.io/Ear_Landmarker/)** | **[GitHub Repo](https://github.com/shameem4/Ear_Landmarker)**

Real-time 55-point ear landmark regression using a BlazeBlock backbone (FaceMesh architecture pattern). Runs as a two-stage pipeline: BlazeEar detector finds ears, then EarLandmarker predicts landmarks on each crop via a soft-argmax heatmap head.

**Test NME 0.029** (861 held-out samples) at 340K parameters. See [RESULTS.md](RESULTS.md) for the full v2 experiment log, including the seed-noise floor that several candidate improvements failed to clear.

## Pipeline

```
Webcam/Image -> BlazeEar detector (128x128) -> ROI crop (1.3x expand) -> EarLandmarker (192x192) -> 55 landmarks
```

The detector (BlazeEar, separate project) produces bounding boxes with NMS. Each box is expanded 30% for context, cropped, and fed to the landmarker. The 1.3x
expansion is not a guess -- measured against ground-truth boxes over 380 matched
ears it is the value that lands the ear at the landmarker's training occupancy of
0.777.

NMS suppresses on IoU **or intersection-over-minimum** (threshold 0.35). Plain IoU
cannot catch a small box nested inside a larger one on the same ear, which is what
produced doubled boxes and overlapping landmark sets in the live demo. The
threshold is calibrated against 104 duplicate pairs logged from real webcam runs,
9 of which are regression fixtures in `tests/test_nms.py`.

On video, boxes are smoothed **before** cropping and landmarks after, in frame
coordinates, with One Euro filters (Casiez et al., CHI 2012) -- box wobble is
roughly 90% of frame-to-frame jitter. This cuts jitter 66-80% and reduces error
at the same time. See `model/smoothing.py` and its JS port `docs/smoothing.js`.

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
| head    | -> 55    | see below | -    |

Two heads are implemented, selected with `--arch`:

| Head | Params | Test NME | How it predicts |
|------|--------|----------|-----------------|
| `heatmap` (default, shipped) | 340K | **0.0291** | Decoder fuses stage4->stage3->stage2 to 55 x 24x24 heatmaps, then soft-argmax (DSNT) |
| `gap` (v1 architecture) | 312K | 0.0301 | Global average pool -> FC -> 110 sigmoid coordinates |

The soft-argmax head is worth -4.0% test NME and -11.5% on the off-contour
(`nme_normal`) component, comfortably above the 1% seed-noise floor. It also
yields a free per-landmark confidence from heatmap spatial spread, exported as a
second ONNX output.

- Output: 55 x 2 coordinates in [0, 1], mapped back to frame pixels, plus 55 confidences
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

**Preprocessing pipeline:** `preprocess.py` ingests all formats, deduplicates, normalizes landmarks to [0,1], writes `landmarks.npy` (memory-mapped) + `manifest.csv`. `validate.py` runs 7 automated QA checks (image integrity, bounds, geometry, linestrip smoothness, statistical outliers, cross-source consistency, resolution audit). `split.py` creates stratified 70/15/15 train/val/test splits.

**Splits (v2):** 4,142 train / 861 val / 861 test. `val` drives early stopping and
checkpoint selection; `test` is touched once, at the end, and is the only number
suitable for reporting. AudioEar3D is kept entirely in train -- its 112 samples are
56 subjects x (left, right), and with horizontal flip on, splitting near-mirror pairs
across train/eval would leak.

## Training

| Setting | Value |
|---------|-------|
| Loss | Wing loss (w=0.04, eps=0.01) |
| Optimizer | AdamW (lr=1e-3, wd=1e-4) |
| Schedule | Cosine annealing |
| Precision | 16-mixed AMP |
| Augmentation | Horizontal flip, translation (5%), rotation (+/-15 deg), **perspective (65 deg)**, color jitter, bbox jitter (10%) |
| Early stopping | Patience 50 on val_nme |

Perspective augmentation cuts contour error at 50 deg yaw by 36% while leaving
the near-frontal test number unchanged -- the test split cannot see what it is
for. `--perspective-deg 65` is what the shipped model was trained with; 50 and 65
are not statistically separable, so neither is claimed better.

Optional, off by default: `--tangential-weight` discounts residual along the
ground-truth contour (where most label noise lives) and improves `nme_normal` but
not total NME; `--synthetic-ratio` mixes in the ingested synthetic renders, which
measurably regressed accuracy and should stay at 0. Both are documented in
[RESULTS.md](RESULTS.md).

Augmentation labels are covered by `tests/test_augmentation.py`, which asserts that
image content and landmark labels move together for every geometric augmentation.

**v1 results (superseded):** val NME 0.0307 (~5.9px at 192px) at epoch 345. This
number should not be used. Two reasons: the rotation augmentation transformed labels
by the inverse rotation, mislabelling every rotated sample by ~12.5px mean (p90
26.8px) at 192px; and the figure was the early-stopping/checkpoint-selection metric
on a two-way split, with no held-out test set. v2 re-baselines on the corrected
pipeline and reports test NME.

Fixing the rotation bug did **not** improve accuracy -- it cut jitter 19.5%. The
v2 accuracy gain comes from the soft-argmax head. [RESULTS.md](RESULTS.md) records
which changes cleared the seed-noise floor and which did not.

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
python train.py --run-name my_experiment    # isolates checkpoints/logs
python train.py --arch heatmap --perspective-deg 65    # the shipped configuration

# Score checkpoints on the held-out test split
python scripts/eval_test.py
python scripts/eval_test.py v6_persp65

# Tests (augmentation label correctness)
python -m pytest tests/ -v

# Data pipeline
python data/preprocess.py
python data/validate.py
python data/split.py
```

## Performance

| Metric | Value |
|--------|-------|
| **test NME, shipped web model** (`v6_persp65`) | **0.0293** (~5.6px at 192px) |
| test NME, best checkpoint (`v2_heatmap`) | 0.0291 |
| test NME, off-contour component | 0.0129 |
| Seed-to-seed spread (3 seeds) | +/- 0.0003 (~1% relative) |
| Parameters | 340K |
| Input size | 192x192 |

The web demo ships `v6_persp65` rather than the nominally better `v2_heatmap`
because the 0.0002 difference is under the seed spread, while perspective
augmentation's off-axis robustness (-36% contour error at 50 deg yaw) is real and
matters for webcam use.

Reported on the 861-sample held-out test split, scored once per run. Reproduce
with `python scripts/eval_test.py`. The v1 figure of 0.0307 was a validation
number on a two-way split and is not comparable; see [RESULTS.md](RESULTS.md).

FPS figures from v1 (579 GPU / 226 CPU) are not carried over -- the heatmap head
adds a decoder and they have not been re-measured.

## Dependencies

PyTorch, PyTorch Lightning, OpenCV, NumPy, Pillow, torchvision, tqdm

BlazeEar detector weights expected at `../BlazeEar/runs/checkpoints/BlazeEar_best.pth`.
