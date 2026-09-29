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

| Strip          | Indices | Points | Color (viz) | iBUG regions it spans |
|----------------|---------|--------|-------------|------------------------|
| Outer helix    | 0-19    | 20     | Green       | ascending helix 0-3, descending helix 4-7, helix 8-13, lobe 14-19 |
| Inner helix    | 20-34   | 15     | Orange      | ascending inner helix 20-24, descending inner helix 25-28, inner helix 29-34 |
| Concha border  | 35-49   | 15     | Blue        | tragus 35-38, canal 39, antitragus 40-42, concha 43-46, inferior crus 47-49 |
| Superior crus  | 50-54   | 5      | Pink        | superior crus 50-54 |

These four are the connected polylines used for drawing, contour losses and
smoothness checks. The per-point semantics come from the **iBUG ear annotation
scheme** (Zhou & Zaferiou, *Deformable Models of Ears in-the-wild*, FG 2017),
which is what collectionA/B and the AudioEar sets are labelled with;
`model.measure.IBUG_REGIONS` holds the authoritative mapping.

v1 named these strips helix / antihelix / concha / tragus, and three of the four
were wrong -- "tragus" was applied to the superior crus, while the real tragus
(35-38) sits inside the strip that was called "concha". Anything measuring by
those names measured the wrong structure. `tests/test_landmark_naming.py` locks
the mapping.

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

## Stability

Accuracy on still frames was never the whole problem -- the v1 model was visibly
jittery on video and occasionally drew two landmark sets on one ear. Three pieces
address that, all covered by tests.

**Temporal smoothing** (`model/smoothing.py`, JS port `docs/smoothing.js`).
One Euro filters (Casiez et al., CHI 2012), the same speed-adaptive low-pass
MediaPipe uses: heavy smoothing when a point is still, light when it moves fast,
so lag does not trade against stability. Box wobble is roughly 90% of
frame-to-frame jitter, so `BoxSmoother` runs *before* the crop (on centre and
size, not corners) and `LandmarkSmoother` after, in frame coordinates, weighted
by the model's per-landmark confidence. `EarTracker` associates ears across
frames by nearest centre so each keeps its own filter state.

Measured with `scripts/eval_smoothing.py`: **jitter down 66-80%**, and error
drops too rather than trading against it.

**Duplicate suppression.** NMS suppresses on IoU *or* intersection-over-minimum.
Plain IoU cannot catch a small box nested inside a larger one on the same ear --
that scores `areaSmall / areaLarge`, which falls below any sane threshold once
the outer box is ~3x the inner. The 0.35 IoMin threshold is calibrated against
104 duplicate pairs logged from real webcam runs; nine of them are regression
fixtures in `tests/test_nms.py`, and two genuinely different ears measure ~0.0.

**Index-free measurement** (`model/measure.py`). Anthropometry that does not
assume landmark *i* means the same thing in every annotation source: arc-length
resampling with PCHIP (monotone cubic) interpolation, caliper length, and width
perpendicular to the long axis. Needed because the sources disagree about where
along a contour a given index sits -- see the spacing-CV table in
[RESULTS.md](RESULTS.md).

## Loss functions

`model/losses.py`. Wing loss is the default. `TangentialWeightedWingLoss`
decomposes each residual into components normal to and tangential along the
ground-truth contour, and can discount the tangential part:

```bash
python train.py --tangential-weight 0.3 --monitor val_nme_normal
```

72% of squared error is tangential, and per-source tangential error tracks that
source's landmark spacing variability almost exactly -- it is annotation noise,
not model error. Down-weighting it improves `nme_normal` but not total NME, so it
ships as a flag that is off by default. `--spacing-weight` adds an anti-collapse
penalty for use with it. Both are analysed in [RESULTS.md](RESULTS.md).

## Experiment tooling

`scripts/` holds the harnesses behind the numbers in [RESULTS.md](RESULTS.md),
kept so the claims can be re-checked rather than taken on trust:

| Script | What it measures |
|--------|------------------|
| `eval_test.py` | Test NME for any saved checkpoint, via the module's own `test_step` |
| `eval_jitter.py` | Frame-to-frame jitter under photometric-only perturbation |
| `eval_smoothing.py` | Jitter and error with smoothing on vs off, at matched motion |
| `eval_contour.py` | Point-to-contour error vs yaw -- index-free, so it survives annotation disagreement |
| `eval_crop_framing.py` | Sensitivity to crop framing, on paired samples |
| `build_local_testset.py` | Detector recall and the ROI_EXPAND measurement, against ground-truth boxes |
| `ingest_synthetic.py` | Ingests the synthetic renders (off by default; it regressed accuracy) |
| `train_baseline_v1.py` | Re-trains v1 with the rotation bug restored, for a like-for-like baseline |

`data/canonicalize.py` is kept as a documented negative result -- contour
canonicalization deformed the shape 2.09px even with PCHIP, more than it removed.

## Tests

```bash
python -m pytest tests/ -q     # 181 tests
```

Not just smoke tests -- most encode a bug that actually shipped:

| File | Guards |
|------|--------|
| `test_augmentation.py` | Image content and labels move together for every geometric augmentation. Fails on 24 cases if the v1 rotation matrix is restored |
| `test_nms.py` | Nested duplicates collapse, two real ears do not merge; 9 fixtures are boxes logged from live runs |
| `test_smoothing.py` | Filter behaviour, tracker association, no lag blowup |
| `test_tangential_loss.py` | Normal/tangential decomposition, anti-collapse term |
| `test_measure.py` | Resampling and caliper measurement |
| `test_model.py` | Both heads, shapes, confidence output |
| `test_jitter_harness.py` | The jitter harness itself -- two earlier versions of it were wrong in ways that flattered the result |

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

## Where to revisit

This configuration is considered done, not finished. What stopped it was not
running out of ideas but running out of *resolution*: three seeds of an identical
setup spread +/- 0.0003 test NME (~1% relative), and the remaining candidate
changes were all smaller than that. Anything below the noise floor cannot be
validated, so continuing to tune here would produce commits that feel like
progress and are indistinguishable from reseeding.

Listed roughly by expected payoff.

**1. The detector, not the landmarker.** BlazeEar recalls **47.5%** of annotated
ears. A landmarker at 0.029 sitting behind a detector that misses half its
inputs is not the binding constraint on anything a user experiences -- half the
end-to-end failures are frames where no landmark is ever placed. Every remaining
percent of NME is worth less than a single point of detector recall. This is the
one item where the ceiling is high and the measurement is easy; it is separate
work only because it lives in another repo.

**2. The annotation-noise floor.** 72% of squared error is tangential, and
per-source tangential error tracks that source's landmark spacing CV almost
exactly (0.137 -> 2.74px, 0.250 -> 4.96px, 0.286 -> 7.44px). That is annotators
placing points at inconsistent arc-length positions along the same contour, not
the model being wrong. No architecture change lifts it. The routes out are
data-side, and both are real projects rather than tweaks:
  - re-annotate or arc-length-normalise a subset to establish how much of the
    0.029 is actually irreducible
  - train and report against a contour metric end to end, so the target stops
    rewarding the fitting of index noise

Until one of those happens, `nme_normal` (currently 0.0129) is the more honest
headline than total NME, and the gap between them is mostly label noise.

**3. Synthetic data, at a lower dose.** 50/50 regressed accuracy 1.7% with
complete seed separation, so that door is shut. But the epoch-20 crossover --
synthetic *ahead* by 0.0004 early, behind from epoch 40 on -- says the renders do
teach useful early structure and then stop paying for their share of the batch.
Two untried variants follow directly from that shape: a ratio around 0.25, or a
synthetic warm-start followed by real-only fine-tuning. Cheap to test; the
ingestion is already built and reverts by omitting a flag. Worth an afternoon,
not a week.

**4. Re-measure throughput.** The v1 FPS figures (579 GPU / 226 CPU) are not
carried over because the heatmap head adds a decoder and they have never been
re-measured. Nobody should quote a performance number this repo has not produced
since the architecture changed.

**What is deliberately not on this list:** further perspective-angle tuning
(persp50 and persp65 are 0.26 sigma apart -- not separable), contour
canonicalization (tried, deformed the shape more than it fixed), and ROI_EXPAND
(measured against ground truth over 380 ears; 1.30 was already correct). Each was
tested and closed out; reopening one needs a new argument, not another run.
[RESULTS.md](RESULTS.md) has the evidence for all three.

## Dependencies

PyTorch, PyTorch Lightning, OpenCV, NumPy, Pillow, torchvision, tqdm

BlazeEar detector weights expected at `../BlazeEar/runs/checkpoints/BlazeEar_best.pth`.
