# Ear Landmarker

**[Live Demo](https://shameem4.github.io/Ear_Landmarker/)** | **[GitHub Repo](https://github.com/shameem4/Ear_Landmarker)**

Real-time 55-point ear landmark regression using a BlazeBlock backbone (FaceMesh architecture pattern). Runs as a two-stage pipeline: BlazeEar detector finds ears, then EarLandmarker predicts landmarks on each crop via a soft-argmax heatmap head.

## Status

**Shipped:** `v4_fc05_s42`, in the web demo and as `SHIPPED_RUN` in
`inference.py`, with flip fusion on by default. Trained on the deduplicated v3
corpus (`v3_manual_data/`, ingested by `scripts/ingest_manual_v3.py` after
`scripts/dedupe_corpus.py`) with `--flip-consistency 0.5`; about 14% lower
error than the previous default, `fixed_conv_s42`, replicated over three seeds.
[RESULTS.md](RESULTS.md) opens with the current state and holds every finding,
including the ones that were retracted.

**3D path model (branch `synthetic`):** `r2j_s42`, a separate, RENDER-ONLY
landmarker used by [Ear_3D_Lifter](https://github.com/shameem4/Ear_3D_Lifter)
on its own pipeline renders; the weights are
[`models/render_only/r2j_s42.ckpt`](models/render_only/r2j_s42.ckpt) (licence
basis in [NOTICE](NOTICE) section 5). It must never be used on photographs (photo test
error is about 5x v4's) or in the web demo; v4 stays the photo model. Details in
[Render-only 3D model](#render-only-3d-model).

## Scope

This repository detects the 55-point ear scheme in **2D**: the landmarker,
the BlazeEar detector in front of it, the training corpus and its QA, and
the browser demo.

Putting those landmarks into 3D -- depth, pose, the Wrap template pipeline
and the shape model -- moved to
[Ear_3D_Lifter](https://github.com/shameem4/Ear_3D_Lifter), and so did its
results -- `RESULTS_inherited.md` there holds that chapter whole, with
`RESEARCH_BRIEF.md` as its digest. `RESULTS.md` here keeps only the 2D work.
Two findings from the 3D effort stayed, because their subject is the 2D model:
the perspective-augmentation fidelity measurement, and the SAM 2.1 annotation
check.

## Pipeline

```
Webcam/Image -> BlazeFace (128x128) -> face crop (1.5x) -> BlazeEar (256x256)
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

### Render-only 3D model

Ear_3D_Lifter feeds the landmarker whole renders of 3D head scans, framed at
occupancy 0.75 from 25 fixed orbit views, which photo-trained models handle
worst on the lobe and upper helix. Since that path only ever sees its own
renders, it gets its own model, trained on those renders alone (decided
2026-10-22; the photo model is unaffected).

- **Data** (local only; the renders are of scan geometry, so `data/synthetic*`
  is gitignored and never published): `synthetic_v3r`, 444 ears from 222
  SONICOM/HUTUBS subjects x 4 skins x 25 views = 44,400 whole 600 px pipeline
  renders, 36% with decimated geometry; 212 ears hand-labelled in 3D by the
  user, 232 labelled by lm3d_v13. Plus `synthetic_v3j`, 140 of the training
  ears with skin-coloured jewellery (lobe hoops 50%, studs 30%, helix 20%),
  labels unchanged. Ingested by `scripts/ingest_synthetic_v1.py` (`--px 600`,
  `--carry-split`, `--addon`), split by subject 178/22/22; the 16 held-out
  heads and BJ002 are asserted absent.
- **Recipe:** init from v4 (`--init-weights`), renders only, lr 1e-3, early
  stopping on render val; `--bbox-jitter 0`, `--rotation-deg 5`, no
  perspective or occupancy reframing, `--degrade-prob 0.3`, flip consistency
  0.5, jewellery add-on at `--synthetic-ratio 0.32` (24% of the mix). Best
  checkpoint epoch 4: render val saturates fast.
- **Result** (Ear_3D_Lifter, end to end on the user's hand-corrected 32 ears
  of 16 held-out heads, view_table_v13, median mm):

  | model | clean | with synthetic jewellery | lobe clean / jewellery |
  |---|---|---|---|
  | v4 | 1.61 | -- | 2.02 / -- |
  | r1 (v3 renders) | 1.045 | 1.112 | 1.59 / 1.77 |
  | **r2j (v3r + jewellery)** | 1.089 | 1.061 | 1.43 / 1.52 |

  r2j vs r1: clean overall level (p=0.40), lobe -0.21 mm (p=0.034); with
  jewellery, lobe -0.42 mm (p=0.001). The real-hoop ear pp21_left goes from a
  worst point of 7.6 to 3.3 mm, judged improved by eye. Not immune: 3 of 15
  synthetic-hoop ears still lose 2.4-3.2 mm on their worst lobe point.
- **Weights:** [`models/render_only/r2j_s42.ckpt`](models/render_only/r2j_s42.ckpt),
  a Lightning checkpoint (load with `EarLandmarkerModule.load_from_checkpoint`,
  or Ear_3D_Lifter's `LandmarkPredictor`). Input: the whole square pipeline
  render, resized to 192.
- **Caveats:** one seed; the reference is the user's own placement, and 212
  of the training ears carry it (no held-out head was rendered), so this
  measures agreement with the user's convention.

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
frame-to-frame jitter, so box smoothing runs *before* the crop (on centre and
size, not corners) and `LandmarkSmoother` after, in frame coordinates, weighted
by the model's per-landmark confidence.

**Association and box smoothing are BlazeEar's shared `BoxTracker`**, vendored
at `vendor/box_tracker.py` and `docs/ear_tracker.js` and pinned to BlazeEar by
`tests/test_vendored_tracker.py`, so all three consumers run one
implementation. It associates on **IoU >= 0.3 against the raw detection** --
this repo previously used nearest centre within 0.6x box size against its own
*smoothed* centre, which lagged during motion and cost it tolerance: replayed,
a track survives 44 px/frame where the old rule broke at 34. `EarTracker` keeps
the per-track `LandmarkSmoother`, because landmarks are this pipeline's output
while boxes are the shared part. Settings are near pass-through -- no
hysteresis, matched tracks only -- so an ear is reported on detected frames
exactly as before; real hysteresis is a separate, unmade decision.

Measured with `scripts/eval_smoothing.py`: **jitter down 66-80%**, and error
drops too rather than trading against it. The tracker swap itself moved neither
landmark jitter (p=0.58) nor adjacent-frame box IoU (p=0.39), and cut track
switches 14.5%; see the top note.

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
python scripts/eval_test.py --data-dir data/manual fixed_conv_s42

# Compare models on real frames with NO ground truth, which is the only
# comparison the collectionB convention cannot distort
# (runs are set in CK at the top of the script)
python scripts/eval_edge_response.py

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
```

**Copy the graph and its JS together, always.** The pre-2026-10 JS hardcoded
`numAnchors = 896`, the anchor count at a 128 px input. The ear graph is now
256 px and emits 3584, so an ONNX refreshed on its own leaves the loop reading
the top fourteen rows of the anchor grid and quietly returning low-confidence
boxes from anchors that do not cover the subject. Nothing throws, and every
Python test still passes.

```bash
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
python export_onnx.py --checkpoint runs/checkpoints/fixed_conv_s42/EarLandmarker_433_nme=0.0145.ckpt
```

The browser demo applies the same flip fusion the Python pipeline does, so the
page is a faithful preview rather than an approximation. Verified by feeding
both the identical PNG: browser and Python agree to **all six decimals** on
every point, in both fused and single-arm modes. The mirror is taken on the
normalised tensor after the resize, exactly where `inference.py` takes it, which
is what makes that parity exact -- mirroring the crop canvas and redrawing is
not equivalent, because the resize is not flip-commutative at the edges.

The exported graph is fixed batch 1 (`dynamic_axes=None` in `export_onnx.py`),
so the browser runs the session **twice** rather than one batch of 2. That costs
more here than it does in Python: measured end-to-end on a 266x346 frame,
**97.0 ms against 77.9 ms, +24.5%**, where the same change on CUDA is +0.3%.
The browser runs WASM and the landmarker is a much larger share of the total
there. Pass `{ flipFusion: false }` to `EarLandmarkerPipeline` for the single
arm if you would rather have the frame rate.

**A dynamic-batch export does not help, and an earlier version of this section
wrongly said it would.** It was built and measured. The graph exports cleanly
with a symbolic batch axis -- the wrapper's `.view(1, 55, 2)` has to become
`.view(-1, 55, 2)` -- and it is numerically exact: batch-2 rows are bit-identical
to separate batch-1 runs, and the graph matches the shipped one to 1.2e-07. It
is simply not faster. Interleaved trials in the browser:

| | mean | sd |
|---|---|---|
| two sequential batch-1 runs (shipped) | 23.51 ms | 0.38 |
| one batch-2 run (dynamic graph) | 24.01 ms | 0.83 |

Batching **loses** 2.1%. The reason is structural rather than incidental:
`onnxruntime-web` needs WASM threads to exploit a batch, threads need
`SharedArrayBuffer`, and that needs cross-origin isolation via COOP and COEP
headers -- which GitHub Pages cannot send. The page reports
`crossOriginIsolated: false` and no `SharedArrayBuffer`, so the kernels run
sequentially and a batch of 2 is exactly twice the work plus a larger
allocation. `ort.env.wasm.numThreads` reads 4, but that is the requested value,
not the effective one.

## Performance

| Metric | Value |
|--------|-------|
| **shipped web model** | `fixed_conv_s42` (clean provenance, handedness bias fixed) |
| contour placement on 455 real frames, no ground truth | **4.067** vs 3.889 for `v6_persp65` (higher is better) |
| test NME on the old benchmark, shipped model | 0.0342 single-arm (0.0337 fused); 0.0346 for the previous `manual_occ_s42` |
| handedness bias (R-L antisymmetry, paired, audioear2d) | **-1.01% (p=0.46)** vs -17.05% (p=2.7e-19) before the fix |
| flip fusion, on by default | -1.83% nme_normal (p=0.009), +0.3% pipeline wall clock on CUDA |
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
