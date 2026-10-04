# Ear Landmarker

**[Live Demo](https://shameem4.github.io/Ear_Landmarker/)** | **[GitHub Repo](https://github.com/shameem4/Ear_Landmarker)**

Real-time 55-point ear landmark regression using a BlazeBlock backbone (FaceMesh architecture pattern). Runs as a two-stage pipeline: BlazeEar detector finds ears, then EarLandmarker predicts landmarks on each crop via a soft-argmax heatmap head.

**Test NME 0.029** (861 held-out samples) at 340K parameters. See [RESULTS.md](RESULTS.md) for the full v2 experiment log, including the seed-noise floor that several candidate improvements failed to clear.

## Pipeline

```
Webcam/Image -> BlazeFace (128x128) -> face crop (1.5x) -> BlazeEar (128x128)
             -> ROI crop -> EarLandmarker (192x192) -> 55 landmarks
```

Detection is **two-stage**, as of BlazeEar v2. BlazeFace locates the head on the
full frame, and the ear model runs on a square crop at 1.5x the face box. A whole
frame squeezed into 128x128 leaves the median ear about 14 pixels across;
cropping to a face first makes it 32. Measured here on 500 annotated full scenes:

| detector | ear recall (IoU>=0.3) |
|----------|------------------------|
| pre-v2 single-stage | 47.6% |
| **v2 two-stage** | **78.6%** |

The cost is a recall ceiling -- an ear whose face BlazeFace misses never reaches
stage two, 1.4% of images here. BlazeEar measured a full-frame fallback for those
and found it a wash.

Detection is delegated to BlazeEar's own `blazeear_inference.js` and
`evaluate_two_stage.load_ear_model` rather than reimplemented. v2 replaced the
folded backbone with a trainable-BatchNorm one *and* fitted new ear anchors, so a
hand-rolled `BlazeEar()` either fails to load or silently decodes boxes at the
wrong scale. The browser ONNX contract changed too, from 896 raw anchors to
already-decoded, already-suppressed boxes.

Each detector box is then expanded for context, cropped, and fed to the
landmarker.

**The crop is derived from the landmarks, not from the detector box.** A fixed
expansion cannot work, because the detector box is not a fixed fraction of the
ear. Against ground truth on 500 annotated scenes the ratio of true ear extent to
box extent spans p10 0.83 to p90 1.15 (median 0.98), and on close-up captures it
reaches 1.53. Where the box is tight, a 1.3x crop is *smaller than the ear*, so
the landmarks jam against the crop border and can never reach the rim.

v2's boxes are more consistent than the pre-v2 detector's (p10-p90 0.83-1.15
against 0.72-1.28), so refinement fires less often -- but the spread still
straddles any single constant, and the tight tail is where the visible failures
were.

So the pipeline crops, predicts, measures the resulting occupancy, and re-crops
if the ear is not sitting at the 0.777 the model was trained on. This is the
ROI-from-landmarks refinement MediaPipe uses for face and hand tracking. One
refinement pass is enough.

Scored against real ground truth on the test split, with the detector box
artificially tightened to span the observed range:

| ear / det box | fixed 1.3x NME | adaptive NME | change |
|---------------|----------------|--------------|--------|
| 1.00 (box already covers the ear) | 0.0400 | 0.0399 | -0.3% |
| 1.20 | 0.0405 | 0.0407 | +0.6% |
| 1.40 | 0.0535 | 0.0417 | -22.1% |
| 1.53 (tightest observed) | 0.0657 | 0.0394 | **-39.9%** |

The claim is not that refinement is more accurate in general -- it is that the
pipeline stops caring how tight the detector box is. Adaptive holds ~0.040 across
the whole range while the fixed policy degrades 64%. Where the box already frames
the ear correctly it is a wash, within +/-0.6%.

Cost is bounded: framing that is already correct exits after one pass, and on
video each track seeds from the expansion that worked last frame, so the steady
state is one pass. Pass `refine_roi=False` (`refineRoi: false` in JS) for the old
single-pass behaviour.

The expansion is **clamped to 1.0-2.5x the detector box**, which is not cosmetic.
Seeding each frame from the previous one makes the refinement a feedback loop:
when the landmarker reports a saturated extent -- what it does on motion blur, an
occluded ear or a false-positive box -- the crop grows, the grown value is
cached, and the next frame starts larger still. Measured unclamped at ~1.53x per
frame, reaching 38x the detector box within eight frames with no recovery, since
each enlargement makes the ear smaller in the crop. The upper bound also keeps
the crop inside the band where over-wide framing is cheap.

Crops stay **square and grey-128 padded** at frame edges rather than being
clamped, since a clamped window is non-square and resizing it to 192x192
stretches the ear along one axis -- worth 17% NME on edge cases.

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

**The shipped model trains on `data/manual` alone** -- 10,535 samples whose
images were collected from the wild and vetted to exclude anything under a
non-commercial or otherwise non-permissive licence, and whose 55-point
annotations were commissioned as work-for-hire through a crowd-annotation
platform. That is what lets the weights ship under Apache-2.0; see
[NOTICE](NOTICE), including its disclosure that the annotation task was seeded
with predictions from an earlier model and corrected by hand.

Build it with `python scripts/ingest_manual.py`. The script re-crops each image
so the ear occupies a spread of the frame matching the older corpus (mean 0.777,
sd 0.093 -- a constant occupancy trains a model that only works at one scale),
and assigns splits by perceptual-hash GROUP rather than per sample, because
wild-collected images repeat the same ear and a near-duplicate across the test
boundary would inflate the score.

### The older corpus (`data/preprocessed`)

Still used for comparison and for the regression benchmark, **but it is
non-commercial** -- anything trained from it inherits that. 5,870 samples
unified from 4 sources, deduplicated by image hash:

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
# the shipped configuration
python train.py --data-dir data/manual --arch heatmap --perspective-deg 65

# Score checkpoints on the held-out test split
python scripts/eval_test.py
python scripts/eval_test.py v6_persp65
python scripts/eval_test.py --data-dir data/manual manual_occ_s42

# Compare models on real frames with NO ground truth, which is the only
# comparison the collectionB convention cannot distort
python scripts/eval_gt_free.py --limit 500 v6_persp65 manual_occ_s42

# Tests
python -m pytest tests/ -q

# Data pipeline
python data/preprocess.py
python data/validate.py
python data/split.py
```

### Refreshing the detector from BlazeEar

BlazeEar is a sibling repo; this one consumes its weights and its browser
pipeline. After updating it:

```bash
cp ../BlazeEar/docs/BlazeEar_web.onnx ../BlazeEar/docs/BlazeFace_web.onnx docs/
cp ../BlazeEar/docs/blazeear_inference.js docs/
python -m pytest tests/ -q
```

`docs/blazeear_inference.js` is a verbatim copy, not a fork -- do not edit it
here. The Python side reads BlazeEar's checkpoints directly through its own
loaders, so it needs no copy step; set `BLAZEEAR_DIR` if the repo is not a
sibling directory.

### Web demo

The browser pipeline is the same one the [live demo](https://shameem4.github.io/Ear_Landmarker/)
runs. To serve it locally:

```bash
python -m http.server 8000 -d docs
```

Then open `http://localhost:8000/`. Webcam capture needs a secure context, which
`localhost` counts as -- but a demo served from a LAN IP over plain HTTP will not
get camera permission, so use `localhost` or HTTPS.

To regenerate the ONNX model the page loads:

```bash
python export_onnx.py
python export_onnx.py --checkpoint runs/checkpoints/manual_occ_s42/EarLandmarker_292_nme=0.0149.ckpt
```

### Comparing the ROI refinement

Both pipelines take a flag that restores the old single-pass framing, which is
the easiest way to see what the refinement is doing:

```python
EarLandmarkerPipeline(det, lm, refine_roi=False)    # Python
```

```javascript
new EarLandmarkerPipeline({ refineRoi: false })     // JS, in docs/index.html
```

Point a webcam at an ear and watch the **top of the helix and the lobe** with the
flag off: where the detector box is tight, the contour stops short of the rim
because the crop is smaller than the ear. Vary your distance from the camera --
the ear-to-box ratio changes with scale, so correct framing should hold at every
distance, not just one. `ROI_MAX_REFINE` raises the pass count if one is ever not
enough.

## Performance

| Metric | Value |
|--------|-------|
| **shipped web model** | `manual_occ_s42` (clean provenance) |
| contour placement on 455 real frames, no ground truth | **4.067** vs 3.889 for `v6_persp65` (higher is better) |
| test NME on the old benchmark, shipped model | 0.0347 |
| test NME on the old benchmark, excluding collectionB | 0.0244 vs 0.0237 (`v6_persp65`) |
| test NME, previous shipped model (`v6_persp65`) | 0.0292 (~5.6px at 192px) |
| detector ear recall, IoU>=0.3 (BlazeEar v2 two-stage) | 78.6% |
| Seed-to-seed spread (3 seeds, old benchmark) | +/- 0.0003 (~1% relative) |
| Parameters | 340K |
| Input size | 192x192 |

**Read the first two rows together, because they disagree.** The shipped model
scores 19% WORSE than `v6_persp65` on the old test split and places contours
BETTER on real photographs. Both are true. 55% of that test split is
collectionB, whose annotation convention `v6_persp65` was trained on and this
model was not, so the benchmark rewards agreement with that convention rather
than accuracy. Excluding collectionB the gap falls to +3.1%; on audioear2d,
+0.9%. Judged with no ground truth at all -- 455 real full frames, contour
placement scored by image-gradient response -- three seeds of the shipped model
beat three seeds of `v6_persp65` with complete separation, p=0.05 exact (the
floor available to a 3-vs-3 design). RESULTS.md has the full argument and the
case against it.

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

**0. Data, before anything else.** The model overfits by 14.1% with augmentation
on, and 72% of its error is annotator spacing noise rather than anything it
controls. Capacity and architecture have both now been measured and neither is
the binding constraint. The open questions are whether a re-annotated subset with
an enforced arc-length convention would move the floor, whether the three source
collections' annotation guidelines are recoverable (their spacing CVs of 0.137 /
0.250 / 0.286 suggest three different conventions were merged), and whether real
captures can be added -- the one attempt to expand with synthetic data regressed
1.7%.

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

### Settled: could MediaPipe's FaceMesh have done this job?

**No.** Run, scored against a pre-registered rule; see [RESULTS.md](RESULTS.md).

| arm | params | init | test NME | vs control |
|-----|--------|------|----------|------------|
| control (EarLandmarker) | 340,167 | scratch | **0.02919** | -- |
| `fm_pre` | 206,942 | 84.6% MediaPipe | 0.03147 | +7.8% |
| `fm_scratch` | 206,942 | random | 0.03273 | +12.1% |

Swapping MediaPipe's regression head for this project's heatmap decoder
(`fm_heat_pre`, 141,991 params, still 81.6% pretrained) reaches **0.03087**, so
the 12.1% deficit decomposes as:

| component | worth |
|-----------|-------|
| MediaPipe pretrained weights | +3.83% |
| soft-argmax head over direct regression | +1.92% |
| **backbone / capacity** | **+5.75%** |

**What is NOT established: whether that last term is design or size.** The
control is 2.4x larger. A capacity-matched run of this project's own design at
142,935 params (`--width-mult 0.62`) was stopped at epoch 227 of 500, and where
it stood it was **dead level** with the MediaPipe copy on validation
(0.03133 vs 0.03133, no rank separation) -- from random initialisation against
81.6% pretrained weights. So on present evidence the two designs are
indistinguishable at equal size, and the shipped model's edge is substantially
its extra parameters. `scripts/run_capacity_matched.sh` would settle it.

### Why scaling was dropped as a direction

A width sweep above 1.0x was started and abandoned at 9% through. The reason was
the diagnosis, not the numbers:

- train/val gap **+14.1%** with augmentation ON, so wider in truth -- a
  data-limited model, where capacity costs generalisation.
- **72% of squared error is tangential**, tracking annotator spacing. If that is
  a floor, total remaining headroom is ~20%, shared across every possible change.

The binding constraints are **label consistency and real-data volume**, not model
size. See [RESULTS.md](RESULTS.md).

Original setup notes follow.

There is **no recorded rationale** for why v1 sized its backbone independently
rather than adopting MediaPipe's face landmark topology and its weights.
PROJECT.log states the architecture as a fact, the original commit message is a
feature list, and `blazeface_landmark.pth` has sat unreferenced in a sibling
repo throughout. The likeliest reading is that the deviation was never a
decision: v1 took the BlazeBlock idiom and sized the rest by hand.

`model/facemesh_ear.py` copies that topology exactly -- MediaPipe's weights load
into it with `strict=True`, and its 192x192 input happens to match
EarLandmarker's. The only change is unavoidable: the output conv emits
468 x 3 = 1404 channels, and ears need 55 x 2, so that one layer is resized and
left random. It is also the only layer carrying face-landmark *identity*, so
nothing reshapes face vertices into ear points -- what transfers is 175,152
params of generic feature extraction, 84.6% of the model.

| arm | params | init |
|-----|--------|------|
| `fm_scratch` | 206,942 | random -- is the architecture enough? |
| `fm_pre` | 206,942 | 84.6% MediaPipe -- do the free weights pay? |
| control | 340,167 | already run: 0.02919 +/- 0.00031 |

Expect the copy to start about 4% behind on head design alone: it regresses
coordinates directly from a 3x3 conv, the same family as the GAP+FC head this
project measured at 0.0301 against the heatmap's 0.0291. The pretrained features
have to win that back. A loss is a real result -- it says the independent design
earns its keep.

### Also queued: does backbone pretraining help?

Set up but not run -- `scripts/run_pretrain_experiment.sh`, pre-registered, six
runs.

The question was whether loading MediaPipe FaceMesh or BlazeEar weights to
bootstrap training would help. Measuring first showed it cannot be asked
directly: against the shipped backbone (5x5 depthwise, 24-48-96-128-192) only
**0.9% of parameters** are shape-compatible with any donor on disk, and FaceMesh
shares 15% by shape with *zero* name correspondence. The ladders are
incompatible end to end -- donor 3x3 depthwise and 24-28-32-36-42-48-56-64-72-80-88,
ours 5x5 and doubling.

So an A/B on the shipped architecture would read as "pretraining does not help"
when it only showed the weights never arrived. `--backbone blazeear` mirrors
BlazeEar v2's backbone1 block for block, which lets 59% of the backbone actually
transfer, and the experiment runs that architecture twice -- random init against
transferred -- changing nothing else.

Caveat worth keeping in view: the mirrored backbone is 82K params against the
shipped 340K, because the donor tops out at 88 channels. It answers "does
pretraining help", not "should this architecture ship".

**What is deliberately not on this list:** further perspective-angle tuning
(persp50 and persp65 are 0.26 sigma apart -- not separable), contour
canonicalization (tried, deformed the shape more than it fixed), and ROI_EXPAND
(measured against ground truth over 380 ears; 1.30 was already correct). Each was
tested and closed out; reopening one needs a new argument, not another run.
[RESULTS.md](RESULTS.md) has the evidence for all three.

## Licence

This repository's own source code is licensed under the **Apache License,
Version 2.0** -- see [LICENSE](LICENSE).

**The shipped model weights are Apache-2.0 too**, as of this version.
`docs/EarLandmarker_web.onnx` is trained on `data/manual` alone: 10,535 images
collected from the wild and vetted to exclude non-permissive licences, annotated
to the 55-point convention as commissioned work-for-hire. No iBUG or FFHQ data
is in it. Earlier versions of this file were trained on the corpus below and
were research-use only; they are superseded.

The 55-point scheme itself is a point ORDERING -- 0-19 outer helix, 20-34 inner
helix, 35-49 concha border, 50-54 superior crus -- not a dataset. Using the same
numbering does not make the annotations derived from iBUG's.

**Disclosed judgment call:** the annotation task was seeded with predictions from
an earlier model trained on the iBUG corpus, which annotators then corrected, and
iBUG's terms reach "any portion of derived data". The position taken here is that
hand-correction produces an independent work -- the final labels move more than a
pixel on 51% of points. [NOTICE](NOTICE) section 3 states this in full so anyone
relying on the weights can judge it themselves.

**`data/preprocessed` remains non-commercial**, so anything you train from it
inherits that:

| source | samples | licence | commercial use |
|--------|---------|---------|----------------|
| collectionB | 3,153 | [iBUG](https://ibug.doc.ic.ac.uk/resources/ibug-ears/): non-commercial research only | **no** |
| collectionA | 605 | iBUG: non-commercial research only | **no** |
| AudioEar2D | 2,000 | annotations [CC BY 4.0](https://zenodo.org/records/7592895), but images are [FFHQ](https://github.com/NVlabs/ffhq-dataset) (CC BY-NC-SA 4.0) | **no** |
| AudioEar3D | 112 | CC BY 4.0, provenance unverified | unclear |

98% of that corpus is non-commercial by two independent routes: iBUG forbids
exploiting "any portion of the annotations and **any portion of derived data**"
commercially, and a permissive licence on AudioEar2D's annotations does not make
the FFHQ pixels they annotate permissive. Train from `data/manual` instead.

[NOTICE](NOTICE) has the quoted terms and the full third-party inventory.
BlazeEar and trainable_blazeface have both been relicensed to Apache-2.0, so the
code side is now consistent; MediaPipe's contribution was always Apache-2.0.

## Dependencies

PyTorch, PyTorch Lightning, OpenCV, NumPy, Pillow, torchvision, tqdm

BlazeEar detector weights expected at `../BlazeEar/runs/checkpoints/BlazeEar_best.pth`.
