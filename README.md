# Ear Landmarker

**[Live Demo](https://shameem4.github.io/Ear_Landmarker/)** | **[GitHub Repo](https://github.com/shameem4/Ear_Landmarker)**

Real-time 55-point ear landmark regression using a BlazeBlock backbone (FaceMesh architecture pattern). Runs as a two-stage pipeline: BlazeEar detector finds ears, then EarLandmarker predicts landmarks on each crop via a soft-argmax heatmap head.

**Test NME 0.029** (861 held-out samples) at 340K parameters. See [RESULTS.md](RESULTS.md) for the full v2 experiment log, including the seed-noise floor that several candidate improvements failed to clear.

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
python scripts/eval_test.py --data-dir data/manual fixed_conv_s42

# Compare models on real frames with NO ground truth, which is the only
# comparison the collectionB convention cannot distort
python scripts/eval_gt_free.py --limit 500 manual_occ_s42 fixed_conv_s42

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

### Where fusion is applied matters, and the current placement leaves gain behind

Flip fusion here mirrors the normalised crop tensor *inside* `LandmarkPredictor`,
so the detector and the ROI loop both run once, on the un-mirrored frame. That
is deliberate -- it is what makes the fused result immune to the integer
crop-origin rounding in `square_roi_crop` -- but it means presentation effects
upstream of the crop are unreachable.

They are not negligible. Running the two-stage detector on 1200 full frames and
on their mirrors, un-mirroring and scoring at IoU>=0.3:

| | detector recall |
|---|---|
| as-is frame | 80.00% |
| mirrored frame | 79.58% |
| **union of the two** | **83.75%** |

Neither presentation is better; they are *differently wrong*. Per-box agreement
across presentations is median IoU 0.813, with **6.4% of boxes disagreeing below
IoU 0.1** -- found in one presentation, essentially absent in the other. So the
face stage is materially presentation-unstable, and up to **3.75 pp** of recall
sits in a frame-level union that in-landmarker fusion cannot reach by
construction: no averaging inside the landmarker recovers a crop that was never
made.

Note what this does *not* say about the -1.83% fusion gain recorded above. That
was measured on `data/preprocessed`, which is pre-cropped -- no detector, no ROI
loop -- and in that regime a frame-level mirror and a crop-level mirror are the
same operation. The two results are orthogonal, not competing. Ear_3D_Lifter
measures a frame-mirrored single pass beating this two-arm fusion on their
gross-error gate, which is consistent with both.

### Every Ear_3D_Lifter gate figure below is PROVISIONAL

> All of them -- the mixed-handedness table, the arm-selector oracle, and the
> `detector_frame_tta` recovery -- were measured on their renderer at a sun
> direction of x=+0.3, which they have since changed to x=+1.2 after sweeping
> it. That move is worth **-2.29 pp on 39 of 40 ears, p=2.0e-07**, taking their
> baseline from 3.49% to about 1.6%. For scale, the entire
> fusion-versus-replacement-versus-nothing spread on that gate is about 0.8 pp,
> so **the lighting change is larger than every TTA effect put together.**
>
> The plausible way the orderings break: mirroring may buy less once the
> occluding contour already has a strong edge, since weak-edge failure is what
> mirroring was compensating for. If so the 43% recovery and the p=0.058
> residual both need restating. The 112-ear audit is re-running at the new sun.
>
> The structural claims should survive regardless, because they are not about
> magnitude: a coordinate average is symmetric whatever the lighting, so
> fusion's zero variance across presentations is a property rather than a
> measurement. The *sizes* of every gap below are the provisional part.
>
> **Nothing in this touches the photograph-domain results** -- the NME figures,
> the detector recall, the bias measurements and the size-band analysis never
> involved their renderer.

### Symmetric rules survive a mixed corpus; asymmetric ones do not

The gap in all of the above was that every measurement came from a corpus that
is single-presentation (Ear_3D_Lifter's) or scored on crops (this one), so
nothing could say whether a mirror rule survives when handedness varies.
Ear_3D_Lifter built the missing test by relabelling an existing `--tta pair`
run: it already holds a full pipeline pass on each frame *and* on its mirror, so
presenting a seeded half of the views mirrored is a relabelling rather than a
re-render, and the mirror is an exact pixel permutation. 112 ears x 25 views,
200 draws, escape past 10 mm:

| rule | mean | sd | 2.5% | 97.5% |
|---|---|---|---|---|
| as-presented | 3.221% | 0.048 | 3.134% | 3.317% |
| always-mirror | 3.216% | 0.048 | 3.119% | 3.303% |
| **fusion** | **2.719%** | **0.000** | 2.719% | 2.719% |
| oracle (unreachable) | 1.871% | 0.000 | | |

**The single-arm mirror is worth -0.005 pp and beats doing nothing on exactly
50% of draws.** Its 0.53 pp gain on their single-presentation corpus is entirely
a handedness artefact -- mirroring half the ears *into* the unfavourable
presentation cancels the gain from the other half, exactly. **Fusion keeps its
full gain and has zero variance across draws**, because a coordinate average is
symmetric and is therefore literally the same arm whichever way a view was
presented.

That licenses fusion as a default here and retires *replacement* as a candidate
anywhere. It does not transfer the magnitude -- that is their render domain,
with their detector failures and a refinement loop firing on 81% of views where
this one never fires, so -1.83% remains the number for this domain. **What
transfers is the shape: a symmetric rule survives, an asymmetric one does not.**

It also retroactively settles a design choice in `detector_frame_tta`, which
merges the two presentations' detections through NMS rather than taking the
mirrored arm. A union is symmetric -- the same set whichever presentation you
start from -- so it falls in the surviving class. A single-arm mirrored detector
would not have.

**That prediction was then tested directly, and it holds exactly rather than
approximately.** Running the whole pipeline on a frame and on its mirror,
un-mirroring the second and comparing the box the detector settled on, over
6 ears x 5 yaws:

| | median box diff | p90 | max |
|---|---|---|---|
| `detector_frame_tta=True` | **0.00 px** | 0.00 | **0.00** |
| `detector_frame_tta=False` | 53.66 px | 121.69 | 134.88 |

With the union on, the chosen box is **bit-identical** whichever presentation
the pipeline is handed, on every view. With it off, the box moves by a median of
**54 px and up to 135 px on the same frame and the same ear**, decided by
nothing but which way the image was presented. On a 600 px frame that is not
jitter, it is a different crop -- and it is a far more concrete statement of
"differently wrong" than the recall figures above, because the ear is held
fixed and only the presentation varies.

Two caveats worth carrying. "Mixed" there is presentation randomised per
ear-view, while a real corpus has handedness correlated with the subject and
with everything else about the photograph, so the independence is idealised.
And the **oracle gap is large, real, and has no known route to it**:

| | escape past 10 mm | |
|---|---|---|
| fusion | 2.719% | |
| oracle per **view** | 2.118% | one decision per view |
| oracle per **point** | 1.871% | 55 decisions per view |
| **by detection confidence** | **3.192%** | the obvious selector |

Two things come out of that. Most of the gap survives being decided **once per
view** -- 0.601 pp of 0.848 pp -- so a practical rule would not need 55
decisions, which is what the per-point oracle had implied. And **the obvious
selector is worse than doing nothing**: picking the arm whose detection scored
higher lands at 3.192%, worse than fusion, worse than always-mirror, and barely
better than no TTA. It takes the mirrored arm on 75% of points, so it is
measuring the detector's response to presentation rather than which arm placed
the landmarks better. An older note in that project independently found
confidence selecting the good arm 32.3% of the time, on different data and
through different bugs, four months earlier.

So the prize is recorded as real with no route to it, rather than as an
opportunity -- because the attractive answer has now failed twice and the next
person to see a 0.85 pp ceiling will reach for the same signal.

`detector_frame_tta` stays off by default on cost grounds -- a second detector
pass at ~68% of pipeline wall clock -- not on presentation grounds.

### Full frame-level TTA: declined, deliberately

Running the *whole* pipeline twice -- detect, ROI refinement and landmarker, on
the frame and on its mirror -- is what Ear_3D_Lifter does outside this pipeline.
The only thing it adds over `detector_frame_tta` is letting the **ROI loop** see
the mirror, which matters because `refine_roi` re-derives the crop from the
model's own landmarks, so a better first pass compounds into a better crop.

**Declined, and the measurement is why.** Refinement fires on **81% of their
render views and 0% of photographs here** -- 1.000 landmarker calls per ear on
every fixture and every full-pipeline run, because the crop lands inside the
occupancy tolerance first time. The route the extra pass buys does not operate
in this domain.

It is also not a flag. `__call__` runs detection, then tracker box-smoothing,
then the crop/ROI loop, then tracker landmark-smoothing. A second pass would
double-update the tracker, since box smoothing sits *between* detection and
cropping, so doing it correctly means restructuring both passes to run
tracker-free and applying tracking once to the merged result -- surgery on the
main loop to buy a mechanism measured at zero here.

Nothing about presentation argues against it: a merged second pass is symmetric
and so falls in the class the mixed-handedness test says survives. The
objections are cost and the absent mechanism, which is why this is recorded as
a decision rather than an omission.

**The decline was then tested, and the residual is the size the reasoning
predicts.** Ear_3D_Lifter scored `detector_frame_tta` against their own
two-full-passes wrapper on their gross-error gate, 112 ears x 25 views:

| configuration | escape past 10 mm |
|---|---|
| their frame fusion (two full passes) | 2.719% |
| our fusion alone | 3.303% |
| **our fusion + `detector_frame_tta`** | **3.049%** |

The flag recovers **0.254 pp of the 0.584 pp gap, 43%**, p=1.3e-03, better on 66
of 112 ears. What remains is **+0.331 pp at p=0.058** -- real in direction, too
small to establish on 112 ears. That is exactly what the decline's reasoning
predicts: the flag takes the detector route and leaves the ROI route on the
table, and it would have been awkward for the account had the flag recovered
everything or nothing.

So the line this paragraph used to end on -- that consumers whose ROI loop fires
should keep their own frame-level wrapper -- turns out not to apply to the
consumer it was written for. The ROI route is real on their renders; it is just
not worth two full pipeline passes for 0.331 pp. They are retiring their wrapper
in favour of the flag.

**WebGPU was the obvious next lever and it is also rejected, on correctness
before speed.** Tested by loading `ort.webgpu.min.js` in an isolated iframe and
scoring all three graphs against the WASM backend on the same inputs:

| model | WASM | WebGPU | speedup | WASM-vs-WebGPU output |
|---|---|---|---|---|
| BlazeFace | 2.69 ms | 27.72 ms | **0.10x** | **max diff 13400** |
| BlazeEar | 10.0 ms | 29.7 ms | **0.34x** | **max diff 287** |
| landmarker | 11.9 ms | 10.0 ms | 1.19x | 2.98e-07 |
| modelled frame | 46.5 ms | 107.2 ms | **0.43x** | |

Only the landmarker both speeds up and stays correct. The two detectors are 3-10x
*slower* on WebGPU and return numerically different results, which disqualifies
the backend here whatever the timings say. A frame modelled from these parts is
2.3x slower overall. (Pinning also matters: the demo's `onnxruntime-web@1.18.0`
cannot use WebGPU at all -- its backend calls `adapter.requestAdapterInfo()`,
which current Chrome has removed -- so this needed 1.20.1 or 1.22.0 to test.)

One result from that run is worth keeping, because it is the counterfactual for
the section above: **on WebGPU, batching does pay** -- one batch-2 landmarker run
is 8.87 ms against 17.16 ms for two batch-1 runs, a 48.3% saving. So the
dynamic-batch export is not a bad idea in general, it is a bad idea *on WASM*.
If the backend ever becomes viable the two changes would compose, and fusion
would be close to free as it is on CUDA.

### What is actually left

Profiled per stage in the browser on a 266x346 frame, JS and canvas work is
0.33 ms of 91 ms -- there is nothing to win outside the models. The split is
BlazeFace 10.7 ms once, BlazeEar **16.7 ms per detected face**, landmarker
28.6 ms for both fusion arms.

That middle term looked like the one real opportunity, and it was measured
properly rather than assumed. **It costs recall and is rejected.**

BlazeFace is stock MediaPipe -- its front-facing 128 px weights at the original
anchor geometry -- and its NMS, like MediaPipe's own, suppresses on plain IoU at
0.30. That **structurally cannot** remove a nested box: one box inside another
scores `IoU = areaSmall / areaLarge`, which tends to zero as the ratio grows.
Measured on a profile frame, the three face boxes score IoU 0.159 and 0.225
against the box containing them -- under the threshold, so all three survive and
each costs a full BlazeEar pass. No setting of the existing knob fixes this:
lowering the threshold far enough would start merging genuinely distinct faces.

The repo already solved this exact failure for *ear* boxes with an
intersection-over-minimum test (`NMS_IOMIN_THRESH = 0.35`), and the nested face
boxes score IoMin 1.000, so the same rule removes them cleanly. Applying it to
the face stage, scored against ground-truth ear boxes on **2000 real labelled
frames** at the IoU>=0.3 the recall figure uses:

| variant | recall@0.3 | stage-2 calls | ears lost / gained | p |
|---|---|---|---|---|
| shipped | **75.65%** | 3687 | | |
| IoMin, keep highest score | 75.15% (-0.50 pp) | -10.6% | 10 / 0 | **0.002** |
| IoMin, keep larger box | 74.30% (-1.35 pp) | -10.8% | 27 / 0 | **<0.001** |

Both lose ears and gain none. Keeping the *larger* box -- which preserves
coverage and looks like the safe choice -- is **worse**, because a bigger crop
shrinks the ear inside it and that is precisely what the two-stage design exists
to avoid. The redundancy is not waste: a face box that another box contains
still frames the ear differently, and sometimes it is the only one that finds
it. `inference.py` already says as much, that overlapping face crops each report
the same ear and downstream NMS handles it; the cost of that correctness is the
duplicate pass.

So ~10.6% of stage-2 compute is available for ~0.5 pp of recall, significantly.
That trade is not taken here, since recall is the scarcer resource in a
two-stage detector whose documented ceiling is already an ear whose face was
missed. It is recorded so the next person does not re-derive it.

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

## The handedness bias

The model used to be measurably worse at left-presented ears than right. Finding
that took a while, the first explanation was wrong, and the fix retired a
conclusion this README had previously stated. The whole arc is worth recording,
because most of the difficulty was in the measurement rather than the bug.

### Why it stayed hidden

Two independent traps, and an earlier analysis fell into both and concluded
there was no bias at all.

**A balanced corpus cancels it.** The obvious test -- score the corpus as-is,
score it mirrored, compare -- is worthless on a handedness-balanced set. A bias
of `+d` on right ears appears as `-d` on left ones, and the pooled difference
goes to zero no matter how large `d` is. The test split is 146 right / 154 left,
so the pooled number read `+0.57%`, not significant, while its two halves were
`+9.3%` and `-7.7%` at `p < 1e-8`. **Always stratify by side.**

The per-side figures above are the *paired* quantity, flipped-arm minus as-is
arm: positive means mirroring the input makes that side worse, so the model
already favoured it. Right ears read `+9.3%` and left ears `-7.7%`, which is the
bias stated the other way round.

**Plain NME dilutes it ~6x.** About 72% of squared error here is tangential,
i.e. sliding along the contour, which is mostly annotation noise. A systematic
displacement shows up in the normal direction, so `nme_normal` sees it and plain
`nme` buries it. The effect was confined to the normal component; the tangential
component had `p = 0.79`.

**Pair within the row.** An unpaired test -- subgroup mean against the global
mean -- returns `|t| < 0.7` on the very same data, because row-to-row difficulty
swamps a few-percent shift. The statistic that works is per-row: the flipped
arm's `nme_normal` minus the as-is arm's. One of these measurements was built
unpaired first and had to be thrown away.

Side labels are free, incidentally, but get the polarity from anatomy rather
than from geometry. The canal (point 39) is anterior, so viewing a head from its
right side the nose -- and the canal -- falls on the image right: `canal_x >
rim_centroid_x` means a right ear. The signed area (shoelace) of the ordered
outer-rim polyline agrees with that on 860 of 861 rows, but its sign convention
is not self-evident and was adopted backwards here once, which silently swapped
every R/L attribution while leaving all magnitudes and p-values untouched.
Antisymmetry statistics will not catch that for you. The `_right` token in the
filenames is **not** a label either -- it is constant across all 10,535 rows.

### What it actually was

A sub-pixel bug in the geometric augmentation, in `data/dataset.py`.

Translation augmentation moved the **image** by cropping, which can only shift
by a whole number of pixels, while it moved the **labels** by the exact
fractional amount. `int()` floors, so the two disagreed by a mean of
**-0.4692 px**, or -0.002444 in normalised units, measured directly over 200,000
draws of the augmentation arithmetic. `_bbox_jitter` had the same defect by
handing PIL a float crop box, which PIL truncates.

On its own that is a harmless registration error. The horizontal flip is what
turns it into a handedness effect: under a mirror, a constant x-offset is
**anti-equivariant** -- it negates -- so the model learns `+d` for one
presentation and `-d` for the other. In y there is no flip, so it stays an
ordinary constant, which is why a constant `+0.0026` y-correction used to
improve accuracy while the same correction in x did not.

The telling measurement needs no labels at all. With `a` the prediction and `f`
the un-mirrored prediction on the mirrored input, `(a - f) / 2` cancels the
target entirely and so is a property of the model alone. It read **-0.0023, the
same sign in all nine checkpoints** across three architectures, two
initialisations and three seeds. At random initialisation it is ~100x smaller
with mixed signs, so the asymmetry was learned, not architectural.

### The explanation that was wrong

The first diagnosis blamed a convention mismatch: labels normalised as `px/W`
against a decoder grid of `(i+0.5)/w`, a half-pixel apart. It was committed, and
then refuted by checking the annotations themselves -- `data/manual` coordinates
are continuous sub-pixel values, mean fractional part 0.2503 with zero integers,
so `px/W` is correct and `1-x` is the exact mirror. The `preprocessed` corpus is
54.3% integers and those rows may sit half a pixel off, but that is their source
annotation and a constant label offset cancels in every handedness statistic
here. Wrong, and retracted.

### Confirming the fix

A matched A/B retrain -- same seed, same epochs, `LEGACY_SUBPIXEL_SHIFT` the
only difference -- then a converged retrain on the full shipped recipe.

| | legacy | fixed |
|---|---|---|
| R-L antisymmetry (paired, audioear2d) | **+17.05%** (p=2.7e-19) | **+1.01%** (p=0.46) |
| right ears | +9.32% (p=5.5e-12) | -0.56% (p=0.60) |
| left ears | -7.73% (p=7.3e-09) | -1.57% (p=0.078) |
| `(a-f)/2` x-component | -0.001618 (t=-11.0) | -0.000431 (t=-2.49) |

A second, independent channel moved as predicted and could have failed: the
*equivariant* y-residual also collapsed, -0.00274 to -0.00039, and the constant
y-correction stopped helping.

Accuracy is a wash. `val_nme` 0.01452 against 0.01487 is one seed on
`data/manual`, the cut that flatters the fix, and the cross-source split nearly
reverses it. The bias removal is the reason to ship it, not accuracy.

### What the fix cost: a retracted conclusion

On the 60-epoch A/B pair, flip fusion's advantage collapsed by 70% once the bug
was fixed, which supported a tidy story -- that test-time fusion had mostly been
a workaround for the augmentation defect. That story is **wrong**, and the
reason is instructive: both arms of that A/B ran with default augmentation
(`perspective_deg=0`) while the shipped recipe uses 65, so neither arm was the
model anyone would ship. On converged models with the real recipe:

| | legacy | fixed | collapse |
|---|---|---|---|
| fusion vs none | -2.95% (p=3.8e-06) | **-1.83% (p=0.009)** | 38% |
| fusion vs replacement | -2.67% (p=1.9e-05) | **-2.35% (p=1.7e-03)** | 12% |

Fusion still pays, significantly, on a clean model. A sibling project consuming
these landmarks in 3D reached the same conclusion from the opposite direction:
on its gross-error gate, fusion's advantage over doing nothing *grew* after the
fix.

### Flip fusion, and why it is not a free lunch

Fusion is now on by default: average the prediction with the un-mirrored
prediction on the mirrored crop. It is batched into a single forward of batch 2,
so the second arm costs +0.3% of pipeline wall clock on CUDA rather than a whole
second pass. It is also *exactly* equivariant, which is a property worth having
and is pinned by tests -- the decoder grid is `(i+0.5)/w`, so the un-mirror is
exactly `1-x`, and training applies the flip as `1.0 - x` with **no index
permutation**, so point `i` is the same anatomical point in both arms.

The honest caveat is in the tail. The gain is in the mean and the body of the
distribution: points over 0.05 improve by -0.57pp (p=1.4e-09). Rows whose worst
point exceeds 0.20 went the **wrong** way, 3.48% to 4.30% -- nominally p=0.039,
and not significant once corrected for the five thresholds tested. Weak on its
own, but the sibling project measured the same sign independently on a >10mm
gross-error gate over renders, and two weak signals in different domains
agreeing on direction is worth more than either p-value. Notably, on the legacy
checkpoint fusion helped at *every* threshold; the tail cost appears only once
the bias is gone, which fits fusion no longer having a systematic offset to
cancel. If your metric is the far tail, measure it and consider
`--no-flip-fusion`.

### Not a strict upgrade for everyone

On a single-handed pipeline -- one that presents every ear in the same form --
an anti-equivariant offset is not a handedness bias at all. It is a constant
whose sign either suits that convention or does not. The sibling project's
corpus is entirely left-form, and there the legacy model's offset was *helping*:
its as-is arm is 0.275pp worse with the fixed checkpoint (p=3.2e-04), while its
mirrored arm improves by about as much. That is what losing an accidental
alignment looks like, and it would reverse on right-presented ears -- so it is a
dependency rather than an asset. Still worth measuring before adopting.

### Found along the way

Checkpoint filenames carry `nme=` rounded to four decimals, and runs here
usually end with two or three checkpoints tied at it. Three selectors ranked by
that string, and both `min()` and a strict `<` keep the first element in readdir
order -- which is epoch order. The result was a systematic preference for the
**earliest** tied checkpoint: `train.py --resume best` would have rolled the
converged run back 45 epochs. Fixed in `checkpoint_utils.py`, which ranks by the
score stored inside the file.

## Where the detector's recall actually goes

Measured on 2000 labelled full frames, because the obvious fixes to the face
stage kept failing and it was worth knowing why. Two-stage recall is 75.65% at
IoU>=0.3. Splitting the 24.35% of misses by asking whether any face crop even
covered the ear:

| | share of all ears | share of misses | fixable by a better FACE model? |
|---|---|---|---|
| ear-stage miss -- a crop covered the ear, BlazeEar still missed it | 14.60% | **60%** | **no** |
| face-stage miss -- no crop covered the ear | 9.75% | 40% | partly |

**These two shares are on the retracted single-ear basis.** On the corrected
human basis, measured within one 3500-image sample rather than across runs, the
face-stage share is **8.70%** (361 of 4151 ears). The 60/40 split is roughly
preserved and the conclusion below is unchanged.

Faces per frame: mean 1.84, median 1, max 8, and only **1.1% of frames detect no
face at all**. So the large majority of face-stage misses happen on frames where
a face *was* found and the crop simply did not reach the ear -- which is crop
geometry, not detection.

**Crop geometry is already optimal, and sharply.** `FACE_CROP_EXPAND` is 1.5;
sweeping it on 1200 frames loses significantly in both directions -- 1.2 gives
-9.25 pp (p=4e-19), 1.9 gives -3.42 pp (p=1e-05), 2.4 gives -8.25 pp, 3.0 gives
-14.42 pp. The mechanism is visible in the paired counts: expanding to 1.9
*gains* 22 ears by covering more, and *loses* 63 by shrinking the ear inside the
256 px crop. Coverage and resolution trade directly, and 1.5 is the optimum.

### What this means for training a new stage-1 detector

Not a custom NMS -- NMS is post-processing, needs no training, and the measured
version costs recall (see the web-demo section). The real version of the idea is
**re-targeting stage 1**: BlazeFace is stock MediaPipe, built for "a single
frontal face in selfie-style images", while this pipeline's subject is a profile
head. A stage-1 model fine-tuned to emit one *ear-inclusive* box per head would
attack the coverage misses **without** the resolution penalty that expanding the
crop pays, and `trainable_blazeface` exists to fine-tune exactly these weights.

Its ceiling is bounded and worth stating before anyone spends a week on it:
- compute -- one crop per frame instead of 1.84 is **-45.8% of stage-2 calls**
- recall -- at most **+8.70 pp**, and only if BlazeEar then finds ears it has
  never had the chance to see

That ceiling has been quoted at three different values and only one survives.
9.75% is the retracted single-ear basis; 6.82% comes from dividing an
ears-total and an ears-delivered figure that were drawn with *different seeds*,
which is the same cross-run arithmetic error retracted elsewhere in this file;
**8.70% is measured within a single 3500-image sample** and is the one to use.

**Beware a numeric coincidence across the two repositories.** BlazeEar's README
also reports an 8.7%, and it is *not* the same quantity as the 8.70% above --
it is a sub-component of it. Decomposed the way BlazeEar decomposes it, on the
same 3500-image sample:

| | this repo | BlazeEar `common_heldout` |
|---|---|---|
| all face-stage misses | **8.70%** (361/4151) | **11.3%** (40/355) |
| of which, no face detected at all | 0.99% | 2.5% |
| of which, face found but ear outside every window | 7.71% | **8.7%** |

So BlazeEar's headline 8.7% corresponds to the **7.71%** row here, not to the
8.70% total. Read side by side without this table, the two documents look like
they agree exactly, and they would be wrong twice -- about the agreement, and
about which quantity is being compared.

The residual differences are definitional and should not be reconciled or
averaged: this counts an ear as delivered when 50% of its *area* falls inside a
window, BlazeEar counts it when its *centre* does, on a different split and
355 ears against 4151. Anyone scoping stage-1 work picks one operationalisation
and states it.

**The larger prize is stage 2.** 60% of misses are ears that a correct crop
delivered and BlazeEar still missed; no amount of face-stage work touches them.

### Getting the ground truth right took three attempts

The numbers below were measured against the YOLO export under
`BlazeEar/data/yolo11_ears/labels/`, which carries exactly one box per image.
That is not the full ground truth, and chasing it produced one wrong correction
before landing on the right basis. All three bases, on the same corpus:

| basis | boxes | per image | what is wrong with it |
|---|---|---|---|
| YOLO export (used below) | 13,482 | 1.000 | misses 1,877 genuine second ears |
| all of `master.csv` | 26,567 | 1.971 | 39.9% are POSE pseudo-labels sitting on eyebrows and hair |
| **human sources + geometry** | **15,359** | **1.181** | **correct** |

`HUMAN_ANNOTATION_SOURCES` is `GT`, `GT+EAR`, `GT+REVIEW`, plus
`finetune_yolov11.py`'s geometry filter (min 6 px side, 100 px area, aspect
0.15-2.0, which drops 613 of 15,972 = 3.8%). The POSE rows are pseudo-labels:
0.0% of them reach IoU 0.30 with any human box in the same image and 97.8% are
exactly zero, so scoring against them makes every ear an automatic miss.

Rescored on the correct basis -- 1500 images, 1746 ears, 15.9% multi-ear:

| | YOLO export | all master.csv | **human + geometry** |
|---|---|---|---|
| recall@0.3 | 75.60% | 54.34% | **79.55%** |
| precision at 0.70 | 61.30% | 84.41% | **73.68%** |
| false positives / image | 0.477 | 0.201 | **0.330** |
| misses proposed below threshold | 90.2% | 56.2% | **93.1%** |
| misses never proposed | 9.8% | 43.8% | **6.9%** |
| ear size, hits vs misses (px) | 67.6 / 40.1 | 68.2 / 56.0 | **68.0 / 45.4** |

79.55% against the independently documented 78.6% is the check that the basis is
now right. The section below therefore stands, slightly strengthened -- 93.1%
rather than 90.2% -- and the middle column is what a contaminated target does to
an otherwise sound measurement.

The never-proposed group is **14 ears**, with median 43.8 px and p25 18.2 px.
At that size BlazeEar's anchor coverage is 0.571 with 3.6% below IoU 0.35, so
anchor geometry is not the mechanism and the under-scoring hypothesis stands
alone.

### Recall against ear scale, and the two mechanisms are different

Binned by ear size inside the 256 px crop, on the human basis, for ears a crop
actually delivered. "Median best score" is the score of the best box reaching
IoU>=0.3 with that ear:

| size (px in 256 crop) | n | recall | median best score |
|---|---|---|---|
| 0-32 | 104 | **25.0%** | 0.370 |
| 32-48 | 177 | 66.7% | 0.820 |
| 48-64 | 461 | 92.8% | 0.927 |
| **64-80** | 620 | **96.0%** | 0.940 |
| 80-96 | 187 | 94.1% | 0.946 |
| 96-120 | 53 | 92.5% | 0.955 |
| 120-160 | 17 | 82.4% | **0.961** |

Peak recall is the 64-80 band, and BlazeEar's training median of 0.253 ear per
frame is **65 px** in a 256 crop. The peak sits on the training mass.

Recall falls away in both directions while the score column rises monotonically
to 0.961. **An earlier version of this section read that as two different
failures -- under-scoring below the mode, localisation above it -- and that was
an inference error.** The median score is taken over every ear in the band,
which is overwhelmingly the hits, so it says nothing about the handful of
misses. Decomposing the misses themselves settles it:

| band | n | recall | misses | never proposed | under-scored |
|---|---|---|---|---|---|
| 0-32 | 203 | 18.2% | 166 | 18 (11%) | 148 (89%) |
| 32-48 | 426 | 58.0% | 179 | 3 (2%) | 176 (98%) |
| 48-64 | 1128 | 92.1% | 89 | 5 (6%) | 84 (94%) |
| **64-80** | 1402 | **95.7%** | 60 | 3 (5%) | 57 (95%) |
| 80-96 | 450 | 94.4% | 25 | 4 (16%) | 21 (84%) |
| 96-120 | 130 | 90.8% | 12 | 4 (33%) | 8 (67%) |
| 120-160 | 34 | 85.3% | 5 | 2 (40%) | 3 (60%) |
| 160+ | 17 | 82.4% | 3 | 0 | 3 (100%) |

Re-run on 3500 images -- 3790 ears delivered inside a crop, 539 misses -- after
two smaller passes disagreed on the rump's size because they were different
samples presented as though comparable.

Under-scoring dominates at **every** band, including above the mode: pooled,
**500 of 539 misses (92.8%) are proposed and discarded**, against 39 (7.2%)
never proposed. That reproduces the 93.1%/6.9% headline on a 2.3x larger sample.
There is one mechanism here, not two.

The never-proposed rump is **39 ears**, median 38.6 px (p25 19.4, p75 80.7),
with 46.2% under 32 px and 33.3% under 24 px. It *straddles* the band where
BlazeEar's anchor coverage degrades -- median best-anchor IoU 0.475 with 14.3%
below 0.35 under 24 px, against 0.8% below 0.35 at the mode -- rather than
sitting inside it. At n=39 that cannot separate weak anchor coverage from weak
features, so anchors are ruled out as an explanation for the 92.8%, which is the
claim that matters, and left open for the rump.

BlazeEar's render measurements agree rather than conflict: at 0.774 ear/frame
(198 px in a 256 crop, beyond anything in this table) 97.6% of renders have a
box at IoU>=0.3, median IoU 0.704, and the full-anchor oracle is 100% at
IoU>=0.5 -- correct boxes, scores under the gate.

Above 80 px the miss counts are 11, 4 and 3, so those rows are indicative only.
The practical consequence is unchanged: 0-48 px holds 281 ears at 25% and 67%
recall against 70 ears above 96 px, so the recall is on the small side whatever
the mechanism above the mode turns out to be.

The size axis is confounded above the mode, and the last column settles it
rather than merely flagging it. Ear size relative to the *detector's* face box
climbs monotonically across the bands: 0.14, 0.24, 0.34, 0.41, 0.50, 0.60,
0.79, **1.04**. In the top band the ear is *larger than the detected face box*.
That is not a face seen at an angle; it is a face mostly out of frame or mostly
occluded. So the upper bands are not a size variable with pose as a nuisance --
they are substantially a pose variable wearing a size label, and no separate
frontality test is needed to say so. (The trend is measured over all 3790 ears;
the individual upper bands are n=130, 34 and 17.)

This also dissolves the one apparent tension with BlazeEar's render data. Never
proposed rises to 16%, 33% and 40% in the top bands here, while their renders
localise 97.6% at 198 px with a full-anchor oracle of 100%. At n=4, 4 and 2
that is noise either way, but the images differ systematically: their renders
are clean, centred, unoccluded ears on a complete head, and an ear at
ear/face-box ~1 is the opposite. Same nominal size, different pictures, no
contradiction to resolve.

**The face stage predicts ear tragions and the pipeline throws them away.**
BlazeFace carries MediaPipe's 6 keypoints -- `num_coords = 16`, separate
`regressor_8_kp` and `regressor_16_kp` heads, all four tensors populated from
the pretrained weights -- in the order right eye, left eye, nose, mouth, right
tragion, left tragion.

They are **decoded and then thrown away**, which is a one-line fix rather than
missing plumbing: `blazedetector.py:339` calls `decode_boxes_with_keypoints`,
and `blazedetector.py:262` slices the result to `[:, :5]` inside NMS filtering.
The face model reports `num_keypoints = 6` and `num_coords = 16`, and
`process()` still returns `(N, 5)`. (An earlier version of this section blamed
`decode_boxes` reading `[..., 0:4]`; that is the *training/debug* decoder in
`utils/box_utils`, not the inference path.)

Two things not to misread. The ear model's keypoints really are untrained --
the ear corpus has no keypoint labels -- while the face model's are MediaPipe's,
unchanged. And in `trainable_blazeface` the keypoint regressors are **frozen**
with no keypoint loss, so the tragion predictions are pure inherited MediaPipe
and have never been fine-tuned on this data. "The tragion prior is weak on
turned heads" is therefore a statement about MediaPipe's training distribution,
not about what the architecture could learn.

It looked like a lever, because **40% of missed ears are face-stage misses**
-- no crop ever delivered them -- and widening the crop is measured dead at
every setting, which leaves placement as the only remaining candidate. A
tragion-centred second window was the one concrete proposal either project had.

**It is dead, and it fails precisely where it would have to work.** Measured on
the ears that actually escape every crop window, against the general population:

| group | n | p25 | median | <=0.5x | <=1.0x | <=2.0x |
|---|---|---|---|---|---|---|
| reachable | 315 | 0.19 | 0.31 | 68.3% | 86.3% | 94.3% |
| **escaping** | **31** | **3.14** | **6.61** | **0.0%** | **9.7%** | **19.4%** |

Distance from the nearer predicted tragion to the ear's centre, as a multiple of
that ear's own size. Not one of the 31 escaping ears has a tragion within half
an ear-size; the median is 6.61x away. The two distributions barely overlap.

The reason is that the failure modes are **correlated, not independent**. An ear
escapes a concentric crop when the head is turned or placed such that
`FACE_CROP_EXPAND` x the face box does not reach it -- and that is exactly the
geometry in which MediaPipe's tragion for that side is occluded or
extrapolated. The prior is strong where the crop already works and absent where
it fails, so there is no union, confidence gate or fallback that recovers the
ears needing recovery.

What it points at instead: a frontal-face square is the wrong primitive for a
laterally-placed organ on a head that may be turned, and no post-hoc placement
rule reconstructs what the face box never localised. So the 8.7% remains the
larger share of the recall ceiling with **no mechanism currently proposed** --
a worse position than having a bad candidate, and the honest one.

The `FACE_CROP_EXPAND` null survived every basis unchanged. On the correct one
(1200 images, 1400 ears): 1.5 optimal at 79.86%, 1.2 at -7.93 pp (p=7e-19), 1.9
at -4.29 pp (p=3e-09), 2.4 at -9.00 pp (p=4e-24), with the paired counts at 1.9
gaining 22 and losing 82. A result that holds across two wrong targets and the
right one is as settled as this corpus can make it.

**The sweep stands; the MECHANISM given for it was tested predictively and
failed.** "Going bigger gains coverage and loses resolution" was used in 2026-10
to predict that a detector emitting boxes 1.0998x larger should have its
constant divided by that ratio to restore the same effective geometry. At the
rescaled 1.364 **both** arms got worse and face-stage misses rose in every band,
so the optimum is not about effective scale relative to the face box. Half the
story survives -- a smaller window does lose coverage -- but the resolution half
does no predictive work, and the 1.5 optimum has no mechanism that has survived
a prediction. See "T4 reopened" below. Do not re-derive a crop constant from
this mechanism; re-fit it against the downstream objective.

### Stage 2 is a scoring problem, not a localisation one

**Every number in this subsection is on the COLLAPSED YOLO-export basis, which
is retracted** (see above). The conclusion survives re-measurement and the
figures do not; the corrected ones follow the table. Retained because the
decomposition method is the reusable part.

For every stage-2 miss, re-running that same crop with `min_score_thresh`
floored to 0.01 and asking whether any box reaches IoU>=0.3 with ground truth:

| | count | share | *corrected basis* |
|---|---|---|---|
| proposed, but below the gate | 202 | 90.2% | **93.1%**, and 92.8% at n=3790 |
| never proposed at any score | 22 | 9.8% | **6.9%** |

Scores of the recoverable ones were given as median 0.404 -- **withdrawn**, same
retracted basis, and never re-measured until 2026-10-15. On the human-source
basis, 3,500 images / 4,136 ears / 3,809 delivered inside a crop, the median is
**0.453 at our 0.70 gate**, 0.599 at 0.80, 0.701 at 0.85. A miss-score median is
gate-dependent by construction -- a miss at 0.70 cannot include a box scoring
0.75 -- so it is meaningless without its gate and its localisation IoU.

The surviving claim: the box regression is working and the classification head
is under-scoring. What it under-scores was given as small ears -- largest side
inside the 256 px crop a median 40.1 px for misses against 67.6 px for hits --
and that reading is now **contested** by BlazeEar's null levers; see "The prize
behind stage-2 calibration is contested".

Sliding the threshold does not fix it, because precision pays for it:

| threshold | recall | precision | FP/frame |
|---|---|---|---|
| 0.80 | 70.20% | 67.80% | 0.333 |
| **0.70** | **75.60%** | **61.30%** | **0.477** |
| 0.60 | 78.40% | 54.67% | 0.650 |
| 0.50 | 80.47% | 46.28% | 0.934 |
| 0.40 | 82.00% | 35.17% | 1.511 |
| 0.30 | 83.33% | 21.39% | 3.062 |

0.60 buys +2.80 pp of recall for -6.6 pp of precision and may be worth taking;
below 0.40 precision collapses. The win is *moving* this curve, which is a
stage-2 training change: 202 ears are already localised and scored around 0.40,
so scoring them like the 68 px ones would be up to **+13.5 pp of pipeline
recall at an unchanged threshold and unchanged compute**.

**Caveat on that precision column -- and it is ANSWERED.** All 13,482 frames in
the collapsed YOLO export carry exactly one ear, 100%, no exceptions, so on that
basis a correct second detection on a frontal face scores as a false positive
and precision is a lower bound. The open question was whether one-ear-per-image
was a labelling convention or a property of the images. **It is neither: it was
an artefact of the export.** BlazeEar measured master.csv's human sources with
the geometry filter -- 15,972 ears over 13,482 images -- and 17.8% of images
carry two or more, holding 30.6% of all ears (82.2% one, 17.2% two, 0.5% three,
0.1% four-five); `data/yolo11_ears_human/` shows the same split over its 13,007
label files. So on the corrected basis precision is **not** systematically
depressed, and every measurement in this file taken on that basis -- including
every matched-precision table below -- stands as measured. Only the figures in
this retracted subsection carry the lower-bound caveat.

### T4 reopened: a better stage 1 improves coverage and delivers nothing

`trainable_blazeface` retracted the evidence T4 was closed on. Their six
negative configurations all used a cell-based anchor encoder that assigned every
ground-truth box to both grids; with a proper per-anchor assignment and an
**ignore band** -- anchors that overlap a face but lose the top-k race leave the
positive set *and* hard-negative mining, instead of training as background that
mining then specifically hunts -- fine-tuning clears the gate this side set
(0.934 at 16-32 px and 0.980 at 32+ px against thresholds of 0.854 and 0.966,
recall 0.437 -> 0.501 at better precision).

Their checkpoint loads into the stage-1 model this pipeline already builds:
78 tensors, strict=True, zero mismatches. So it was scored directly, on all
13,007 images / **15,359 human-source ears**, both arms in one pass so the
denominator is identical and every ear is paired (McNemar on discordant pairs).
Bands are the ground-truth ear box's max side in the *original* image, not
size-in-crop, because an ear no crop covered has no size-in-crop -- and those
are precisely the ears under test.

**Coverage improves, unambiguously and threshold-independently.** Ears that no
crop covered, out of 15,359:

| stage 1 | face-stage misses | vs stock |
|---|---|---|
| stock | 1301 | -- |
| tuned, as dropped in | 908 | -30% |
| **stock crops UNION tuned crops** | **516** | **-60%** |

**Delivered recall at matched precision does not.** Interpolating each arm's
recall onto stock's precision at each of its operating points:

| matched precision | stock recall | union | tuned (matched crop budget) |
|---|---|---|---|
| 87.28% | 60.65% | -2.44 | -- |
| 84.01% | 68.24% | -1.68 | -4.97 |
| 81.30% | 73.01% | -1.57 | -4.95 |
| 78.52% | 76.08% | -0.94 | -3.98 |
| 77.14% | 77.24% | -0.44 | -3.70 |
| 74.52% | 78.96% | **+0.54** | -3.06 |

The union's headline +4.81 pp at the shipped threshold (p=3.3e-188, gaining in
every band, +16.79 pp at 0-32 px) is almost entirely a precision trade. At
matched precision it is negative at every operating point except the two
loosest, where it is +0.23 and +0.54 pp -- one seed, and smaller than the
spread this corpus shows on far cruder changes.

**Two constants were silently denominated in stock's units**, which is the
transferable lesson and came from their side:

* *Geometry.* Their boxes are linearly larger on the same face -- measured
  **1.0998** on 2,486 paired faces on this corpus (bootstrap 95% CI
  1.094-1.105), not the 1.117 they measured on theirs, whose value falls outside
  our interval. `make_face_crops.py:66` sets `side = expand * max(box_w, box_h)`,
  so `FACE_CROP_EXPAND = 1.5` silently becomes an effective 1.65. Their
  prediction was that restoring 1.5/1.0998 = **1.364** would recover the loss.
  It does not: at 1.364 *both* arms lose (stock 79.09% -> 77.52%, tuned 80.26%
  -> 79.35%) because a smaller window loses ear coverage faster than it gains
  in-crop ear size -- face-stage misses rise in every band. So the units
  observation is right and the correction is wrong, and the 1.5 optimum is not
  about effective scale relative to the face box. It did shrink the 80+ px
  regression from -1.35 pp (p=1.1e-07) to -0.66 pp (p=0.033), so the
  *large-ear* part of the loss is partly geometry; the small-ear gain is not.
* *Score.* The same error on the axis neither side checked, and it went the
  other way. At the shared `FACE_CROP_THRESHOLD = 0.2` the tuned model hands
  stage 2 **6.16 crops per image against stock's 1.84**, with an all-emission
  score median of 0.234 against stock's 0.710. Reading that as compression and
  concluding the model needed recalibrating was **wrong**: a median over all
  emissions is a mixture statistic, and split by correctness their true
  positives sit *higher* than stock's (0.683 vs 0.578) with *better* separation
  (0.466 vs 0.330). But their conclusion -- raise the constant -- is also wrong
  here, because sweeping it against delivered ear recall is monotone in the
  opposite direction. Face-stage misses out of 15,359, by tuned face threshold:
  **530** at 0.10, 631 at 0.15, 908 at 0.20, 1723 at 0.28, 1883 at 0.30, 2217
  at 0.35, 2455 at 0.40 -- against stock's 1301. Pooled recall tracks it: +2.82
  pp at 0.10 down to -4.66 pp at 0.40. **Their low-score tail is load-bearing**:
  the detections their own face-precision metric counts as false positives are
  disproportionately the crops containing the ears this pipeline needs, because
  our objective does not care whether a crop is a true face, only whether it
  contains an ear. Both readings were measuring the wrong quantity; the optimum
  for a swapped detector cannot be derived from either model's own metrics and
  has to be swept against the downstream objective.

The best coverage arm is the worst matched-precision arm: tuned at face
threshold 0.10 beats stock in *every* band on raw recall (+2.82 pp pooled,
misses 1301 -> 530, matching the crop union at half the detector compute) and
loses **4-7 pp** once precision is held fixed. **Nine configurations** -- two
expand values, seven face thresholds, the crop union -- and the verdict does not
move.

**The 0-32 px gain is the one config-invariant result**: +11.76 pp as dropped
in, +10.33 pp at expand 1.364, +5.74 pp at a matched crop budget, +16.79 pp for
the union, each at p<1e-08. A better stage 1 really does find small faces this
one misses.

### The low-score tail is an implicit head detector, and both first explanations were wrong

The tail is where the coverage lives, and explaining it took two retractions.

**Mine, refuted by them.** I claimed their face-precision metric was mislabelling
real faces their RetinaFace ground truth had missed. They checked against YuNet,
a third architecture: of 39,822 tuned detections in 0.10-0.20 that miss their
ground truth, YuNet confirms **19** as faces -- **0.0%** (1.6% at 0.20-0.30,
against stock's 1.5% and 6.8%). Those boxes are not unlabelled faces by any
available face criterion. A crop does not have to *be* a face to *contain* one.

**Mine again, caught by my own control.** Their follow-up -- is the tail's worth
*placement*? -- was answered first with a per-crop null: move each crop's centre
at random, keep its size. 12.72% covering an ear against 12.81% null, lift 1.0x,
p=0.34, apparently chance. **That statistic has no power.** On a head-filling
photograph a same-sized box dropped anywhere hits an ear about 13% of the time,
so the rate saturates. It also contradicted the arm's own face-stage-miss count,
and that contradiction was the tell -- the same lesson as every wrong measurement
in this file.

Re-run as controls destroying one thing at a time (3,000 images / 3,550 ears):

| arm | face-stage misses | recall @0.70 |
|---|---|---|
| tuned @ face 0.10 | **110** | 81.80% |
| stock @ face 0.20 | 312 | 78.76% |
| **placement destroyed, per-image box sizes kept** | **1026** | 52.82% |
| placement and scale destroyed | 1417 | 37.15% |

Randomising only *where* the boxes land -- same count, same sizes, same images --
multiplies face-stage misses by nine. So placement is strongly informative, and
"it is just volume" dies with it: 35 random windows per image recover a third of
the ears, the same 35 placed by the model recover 82%.

**Scope limit on all of this, raised by `trainable_blazeface` and it stands.**
On head-filling frames "lands on a head" and "lands on image content" are nearly
the same event, so a tail that merely avoids empty background would ablate like
head-seeking. Their discriminator -- re-run the ablation where heads occupy
least of the frame -- **cannot be run here with power: only 70 of 6,000 frames
have their largest face under 30% of the short side.** This corpus has no
low-head-fraction regime. On those 70 frames / 90 ears the direction holds and
strengthens (tuned 2 face-stage misses against 47 for placement-destroyed), but
at that n it is consistent with head-ness rather than evidence for it. On their
own corpus -- WIDER-style, median face 13.4 px -- the same band is
*anti*-correlated with faces, which they are careful to say cannot test an
ear-labelled claim, since a box on hair or the back of a head is head-ness by
our measure and a miss by theirs. **So "head-ness" versus "frame composition"
is unresolved, and it matters before a training target is written, not after.**

**RE-REPORTED PER FACE-SIZE BAND, and the aggregate was mostly composition.**
BlazeEar ran the discriminator and told me the pooled figure is a mixture
dominated by head-filling frames. They are right, and two things of mine were
wrong.

*The first attempt at banding was defective.* It binned images by the TUNED
model's largest detection at threshold 0.10 -- an arm emitting ~35 boxes per
image, including large spurious ones -- so the max side was inflated and only 70
of 6,000 frames qualified as small-head. Head scale has to come from a sober
estimate (stock at its own 0.20), not from the tail under test. Re-run that way,
this corpus has 433 small-head frames, not 70.

*The "ninefold" was a miss-ratio, and miss-ratios inflate when the base rate is
small.* 110 against 1026 face-stage misses is a ninefold in misses and a **1.36x**
difference in coverage. Both are arithmetically true; the miss framing is the
misleading one to headline.

Coverage of ears by band, head scale = stock's largest face side / frame short
side, null destroys placement only (4,000 images, 4,749 ears):

| head scale | ears | tuned | stock | null | lift | **lift ceiling (1/null)** | **gap** | headroom captured |
|---|---|---|---|---|---|---|---|---|
| 0.00-0.20 | 523 | 89.1% | 70.4% | 41.3% | 2.16x | 2.42x | **+47.8 pp** | 81.4% |
| 0.20-0.30 | 1368 | 95.6% | 87.5% | 57.0% | 1.68x | 1.75x | +38.6 pp | 89.8% |
| 0.30-0.45 | 1340 | 97.6% | 95.2% | 72.5% | 1.35x | 1.38x | +25.1 pp | 91.3% |
| 0.45+ | 1518 | 99.6% | 98.4% | 93.2% | 1.07x | **1.07x** | **+6.4 pp** | 94.1% |
| all | 4749 | 96.7% | 91.3% | 71.2% | 1.36x | 1.40x | +25.5 pp | 88.5% |

**The lift column was a poor headline and BlazeEar is right that a reader cannot
see why.** A coverage lift is bounded above by `1/null`, so the 0.45+ row could
not have exceeded 1.07x whatever the detector did -- its apparent collapse is
the ceiling collapsing, not the detector. The honest effect size is the
**absolute gap**, and it widens monotonically as heads shrink: +6.4 pp on
head-filling frames against **+47.8 pp** on small-head ones. That is the
head-ness reading and it survives the correction.

The headroom share goes the other way, mildly -- the detector captures 81.4% of
the available headroom on small heads against 94.1% on large ones -- which is
what "small heads are harder to detect" predicts and is not evidence for
composition, since composition's contribution *is* the null being divided out.

**Two corrections went into that table and they fixed different things**, which
is worth keeping straight because the weaker one is the more quotable.
`trainable_blazeface` supplied the **axis**: what saturates the null is size
**times count**, not size, so a small-head crowd frame can still carry a 52.5%
null -- re-banded on occupancy, BlazeEar's ordinary-crop lift sharpens from
5.15x to 8.13x against a 9.6% null. BlazeEar supplied the **statistic**: the
ceiling is `1/null`, an exact number printable per row, which supersedes
`trainable_blazeface`'s own rule of thumb that a null above roughly 50% means no
power. Their words, recorded at their request: the refinement improved the
banding axis, the ceiling fixed the statistic, and theirs was the lesser of the
two. **Band on occupancy and print the ceiling** if this is re-measured; the
table above is banded on size alone.

BlazeEar's matched-power figure for the tail population is **2.26x** against
ordinary crops' 8.13x -- about a quarter as well-placed as a confident face box,
measured on *stock's* tail rather than the tuned one, so it does not settle this
version. That number belongs next to the phrase "implicit head detector", which
sounds stronger than a quarter of a confident face box.

**Both conclusions are needed.** On head-filling frames the lift is 1.07x
against a 93.2% null -- there, placement genuinely carries almost nothing beyond
composition, exactly as the confound predicted. But the gap widens monotonically
as heads shrink, and in the small-head band a placement-destroyed window lands
on an ear 41.3% of the time against the detector's 89.1%. **That widening is
head-ness's prediction and not composition's**, so the direction survives in the
regime where the two explanations come apart -- and the aggregate is not the
evidence for it.

BlazeEar measured the same structure independently on val_v2 with the ordinary
stock crop rather than the tuned tail: 5.15x / 2.58x / 1.61x / 1.10x across the
same bands, aggregate 1.69x. Different population, same monotone shape and the
same two conclusions.

**Consequence for a head-ness target:** it would be worth training for
small-head frames and worth almost nothing for head-filling ones, which is also
where this corpus has least data (433 of 4,000 frames). And per BlazeEar, score
it on **delivered ears at matched precision and on midline false positives per
image, not on coverage** -- coverage has now failed to convert twice, and their
midline FP/image went 0.022 to 0.056 when crops were multiplied.

**The tail is neither a shotgun nor a face detector. On this corpus it behaves as
an implicit head detector** -- boxes no face metric will credit, landing reliably
on ear-bearing regions. Which makes head-ness, not face-ness, the stage-1
objective this pipeline actually wants, and their tail an accidental uncalibrated
version of it. That is a different and better-supported direction than T4's
ear-inclusive box, and it is still gated on stage 2, because a better head
detector delivers more crops of exactly the small marginal heads stage 2
under-scores.

### Terminal result: the delivered population is unscoreable, so neither lever reaches it

BlazeEar scored the 767 rescued ears under a **per-ear oracle crop** -- the
nearest stock face box grown to the minimum expand that brings the ear inside,
a deliberate ceiling no real stage 1 can beat. 41 had no detected face at all
and dropped out.

| band | n | localised (IoU>=0.5) | scored >= 0.80 | median score | ear px in crop |
|---|---|---|---|---|---|
| 0-32 | 332 | 43.1% | **0.3%** | 0.484 | 21.6 |
| 32-48 | 145 | 61.4% | 6.9% | 0.547 | 34.1 |
| 48-64 | 95 | 74.7% | 18.9% | 0.651 | 43.4 |
| 64-80 | 58 | 72.4% | 24.1% | 0.699 | 50.0 |
| 80+ | 96 | 87.5% | 31.2% | 0.702 | 60.5 |
| **all** | 726 | 59.1% | **10.1%** | 0.614 | |

**One ear of 332 in the 0-32 band clears the deployed gate under an oracle
crop.** The whole rescue is worth about **73 scoreable ears**, and a real stage 1
delivers fewer.

**And the failure is localisation, not calibration**, which settles the fork this
whole exchange turned on. **56.9% of the 0-32 band is not localised at all** at
IoU >= 0.5 in a crop built to contain the ear -- there is no box to re-score, and
score calibration cannot reach a detection that does not exist. For this
population the bottleneck is neither stage-1 coverage nor stage-2 calibration: a
21.6-pixel ear at a 256 px input does not produce a box.

**Coverage and convertibility are anti-correlated**, and this should govern the
roadmap. 46% of the rescued ears sit in 0-32 px converting at 0.3%; 12.5% sit in
80+ converting at 31.2%. The coverage gain is concentrated exactly where stage 2
cannot use it, and the ears that would convert are large ones the shipped crop
was already finding. All three links of "better stage 1 -> more coverage -> more
delivered ears" are now measured and it fails at the third.

**So "revisit the union after stage-2 scoring is fixed" is withdrawn for the
0-32 band.** It was my sequencing claim and it assumed the delivered ears were
scoreable-in-principle. They are not. What survives is narrow: the 32-48 and
48-64 bands convert at 6.9% and 18.9%, so a coverage gain there is worth
something, and the information-limited floor below ~24 real pixels is an input
resolution question rather than a stage-1 or stage-2 one.

Selection caveat, stated by BlazeEar against their own table: these are ears
stock's crop failed to cover, which correlates with small and awkward. Their
val_v2 band curve puts 0-32 recall near 4.9% against this 0.3%, and the gap is
that selection. So this is a statement about the rescued population, not about
0-32 px ears generally.

**So T4 closes again, for a better reason, and it hands its lever to stage 2.**
The coverage is now there and cannot be converted: 516 remaining face-stage
misses against 1301, and the extra ears arrive as boxes the gate discards --
the same binding constraint measured above. **Revisit the union only after
stage-2 scoring is fixed.** How much that is worth is contested; see "The prize
behind stage-2 calibration is contested" below, where BlazeEar's null results
argue the newly-fed band is information-limited rather than calibration-limited.

**WITHDRAWN: "90.2% of stage-2 misses are localised boxes scored around 0.40".**
Both numbers came from the collapsed YOLO-export basis that was later retracted,
and carrying them forward was an error of mine that reached two other sessions'
notes. The localised *share* was re-measured on the correct basis long ago
(93.1%, then 92.8% at n=3790); the score *median* never was. Re-measured now,
3,500 images / 4,136 ears / 3,809 delivered inside a crop, human-source basis:

| gate | stage-2 misses | localised | share | median | p25 | p75 |
|---|---|---|---|---|---|---|
| **0.70** (our `DETECTOR_CONFIDENCE`) | 553 | 515 | **93.1%** | **0.453** | 0.305 | 0.600 |
| 0.80 (BlazeEar's `DETECTION_SCORE_THRESHOLD`) | 809 | 771 | 95.3% | **0.599** | 0.379 | 0.734 |
| 0.85 (BlazeEar's TTA gate) | 1070 | 1032 | 96.4% | **0.701** | 0.454 | 0.800 |

So the correct pair at our gate is **93.1% and 0.453**, not 90.2% and 0.404.

**And the median is gate-dependent by construction**, which reconciles a figure
that looked like a contradiction. A miss at a 0.70 gate cannot include a box
scoring 0.75; a miss at 0.80 can. So any "median score of misses" is
meaningless without its gate, and comparing 0.404-at-0.70 against BlazeEar's
0.752-at-0.80 was comparing two different populations. At a matched 0.80 gate
the figures are 0.599 here against their 0.752 -- a real but much smaller gap,
and the residual is explained by their scoring localisation at IoU >= 0.5 where
this measures IoU >= 0.3: a tighter localisation test keeps the better-fitting
boxes, which score higher. **Quote a gate and an IoU with any such median.**

### Our stage-2 gate and BlazeEar's have diverged

`inference.py` ships `DETECTOR_CONFIDENCE = 0.70`, inherited when this pipeline
adopted BlazeEar v2 (54026c0). BlazeEar has since re-picked its own on an
end-to-end sweep of val_v2's 2,414 ears and ships **0.80 plain / 0.85 fused**,
reporting F1 argmax 0.7273 at 0.80 against 0.7081 at 0.70, and 0.7497 fused at
0.85 against 0.6886. They also report 0.70 being live for two hours and reverted
because it tripled fused false positives per image, 0.152 to 0.488.

**That makes our 0.70 a candidate defect rather than a choice**, and it is
untested here: their sweep is on their corpus with their gate semantics. Not
changed on their say-so -- it needs the same end-to-end sweep on this side
before moving, and every arms table in this file is internally consistent at
0.70 and therefore unaffected either way.

### The prize behind stage-2 calibration is contested

BlazeEar argues the band a head detector would newly feed is
**information-limited, not calibration-limited**, with the direct tests run:
the median ear inside a crop is 24 real pixels and 70.9% are under 32; widening
the downward scale augmentation to 0.35-1.25 aimed at exactly that band was
null (three ears became six at n=123 while the 64-80 px mode lost 1.2 pp on 758
ears, reverted); a per-prior negative-mining ceiling was also null; and paired
on the same ears both levers produce a uniform downward confidence shift of
0.005-0.009 across **all** ears (p=1.7e-17) with no small-ear-specific
component (+0.0004, p=0.81).

**Accepted as a discount on the prize, not on the direction.** Their reading is
that "stage 2 under-scores small ears" describes where a uniform shift crosses a
gate rather than a small-ear mechanism, and nothing here contradicts that. The
band table from the re-measurement above does show a size gradient in
localised-miss scores -- median 0.415 at 0-32 px against 0.514 at 80+ at the
0.70 gate, 0.493 against 0.717 at 0.80 -- but **that cannot be read as evidence
against them**, because conditioning on *being a miss* selects differently in
each band: at 80+ px misses are rare and so are the hard tail of that band,
while at 0-32 px they are common. It is a question worth a designed test, not a
counter-measurement.



One correction accepted in the other direction: the min-16 training replication
reported from this side *did* inherit their defective assignment -- it ran their
`train_blazeface.py` with their defaults, so their dataloader and encoder. Only
the scoring test, which consumed their weights alone, was unaffected. The
distinction was theirs to insist on, not mine.

## Scope: scale-rebalancing stage 2

The one piece of work this analysis supports. Written so it can be funded or
declined on evidence rather than re-derived.

**The deficit.** Stage-2 recall against ear size in the 256 px crop: 18.2% at
0-32 px, 58.0% at 32-48, 92.1% at 48-64, peaking at 95.7% in 64-80. 92.8% of
its misses are boxes already localised at IoU>=0.3 and discarded by the 0.70
score gate. The peak sits on the training mass -- BlazeEar's training median of
0.253 ear/frame is 65 px in a 256 crop.

**Already ruled out, with the measurement, so nobody repeats them:**

| candidate | verdict |
|---|---|
| anchor coverage | best-anchor IoU 0.571 at 32-48 px, 3.6% below 0.35 -- not a mechanism for the 92.8% |
| NMS | recall and median IoU flat to three decimals from 0.10 to 0.99 |
| crop geometry | `FACE_CROP_EXPAND` 1.5 optimal; 1.2 -7.93 pp, 1.9 -4.29 pp, 2.4 -9.00 pp |
| **size-dependent threshold** | **dominated by the global gate at every config tried** |

That last one is the new result and it is what makes this a training job. The
score separates true from false positives at every band (AUC 0.93-0.98), so the
score is not broken -- but a per-band gate still loses to simply moving the
global one:

| config | recall | FP/image | precision |
|---|---|---|---|
| global 0.70 (current) | 78.93% | 0.317 | 74.7% |
| global 0.60 | 81.37% | 0.458 | 67.8% |
| leave <32 alone, 32-48 at 0.45 | 81.23% | 0.484 | 66.5% |
| <32 at 0.30, 32-48 at 0.40 | 83.06% | 1.788 | 35.5% |

Every size-aware variant sits below the global curve. There is no post-hoc
calibration fix.

**The likely mechanism, and the lever.** The negative-to-positive ratio among
candidate boxes is wildly scale-dependent: at 0-32 px it is 126,363 false to 47
true, roughly 2700:1; at the 64-80 px mode it is 891 to 751, about 1.2:1. A
classifier trained on that is correctly expressing that a small box is a poor
bet, and the fix is to rebalance what it is asked to learn -- scale-aware loss
weighting or focal gamma, scale augmentation to synthesise small-ear mass, or
rebalancing hard-negative mining per scale. Those are `trainable_blazeface`-
and BlazeEar-side changes; none of them live here.

**How to evaluate it**, which is the part this repo can supply: the harness at
`/var/tmp/claude/ear_landmarker/2026-10-15-face-nms/scale_curve.py`, human
ground-truth basis, 3500 images / 3790 ears delivered in crop. Primary metric is
recall@IoU 0.3 **per size band at matched FP/image**, never pooled -- 0-32 and
32-48 reported separately, because 18.2% against 58.0% is a 40-point cliff that
a pooled number would hide. Baseline `BlazeEar_best_256px.pth`.

**What would make it a success, and what would make it a failure.** 32-48 px is
the band to target: 426 ears at 58.0%, where the scores already sit just under
the gate (TP median 0.757 against FP p90 0.298). Moving that band to ~75% at
today's 0.317 FP/image would be worth roughly +2 pp of overall recall. The
0-32 band is a stretch and may be irreducible -- at 20 px an ear may not be
decidable. **The guard that matters: 1402 of 3790 ears sit in the 64-80 mode,
so any rebalancing that trades mode accuracy for tail accuracy can lose overall
recall while improving the thing it was aimed at.** Report the mode band every
time.

**Training cost: 1-3 GPU-hours per run** on this box (BlazeEar's estimate,
inferred from checkpoint mtimes rather than a logged run, so order-of-magnitude
-- but hours rather than days is the safe part). Cost is therefore not the
constraint on the ~+2 pp, and all three candidate fixes can be tried in
sequence.

### The training-side origin of the 2700:1

BlazeEar traced it, and the obvious explanation is not the right one. Small
ears are **not** starved of positive supervision: `ANCHOR_TOP_K = 3` delivers
exactly 3.00 positive anchors per ear at every size band by design, because
only 23.8% of ears reach IoU 0.5 with even fitted priors. Positive *count* is
size-independent.

The budget is misallocated instead. Over 12,595 human training ears:

| prior (px) | slots | positives | pos/slot | share of positives |
|---|---|---|---|---|
| 6.9 x 11.3 | 1024 | 299 | 0.29 | 0.8% |
| 12.5 x 24.3 | 1024 | 2,596 | 2.54 | 6.9% |
| 17.7 x 41.2 | 256 | 2,955 | 11.54 | 7.8% |
| 34.8 x 39.9 | 256 | 7,088 | 27.69 | 18.8% |
| **29.2 x 62.7** | 256 | 17,589 | **68.71** | **46.6%** |
| 58.4 x 101.1 | 256 | 6,760 | 26.41 | 17.9% |
| 129.8 x 178.2 | 256 | 405 | 1.58 | 1.1% |
| 191.5 x 260.4 | 256 | 93 | 0.36 | 0.2% |

**2048 of 3584 slots -- 57% of the anchor budget -- carry the two smallest
priors and receive 7.7% of all positives.** Pos-per-slot is 0.29 and 2.54
against 68.71 for the prior serving the 64-80 px mode, a 27x gap. And
`HARD_NEGATIVE_RATIO = 50` mines the highest-scoring backgrounds *globally*
across all 3584 anchors with no per-level cap, so those 2048 slots absorb the
bulk of mined negatives on every image. That is the 2700:1, upstream of
inference.

Two secondary asymmetries push the same way: a small ear's three positives are
worse matches than the mode's (median IoU 0.425 at 0-32 px against 0.535 at
64-80), and `ANCHOR_IGNORE_IOU = 0.35` protects ~0.94 neighbouring anchors
around a small ear against 8.84 at the mode -- so a small ear's immediate
neighbourhood is almost entirely available for hard-negative mining.

**Three levers, cheapest first.** (1) Focal loss is already implemented and
*off* -- `use_focal_loss=False`, alpha 0.25, gamma 2.0 unused; a flag flip.
(2) Scale augmentation range: the dataloader uses `scale_range=(0.75, 1.25)`
and the training median ear is 64.9 px, so x0.75 = 48.7 px -- augmentation
essentially *cannot* manufacture a sub-32 px ear, and only the natively-small
5.9% populate that band. Widening the low end to ~0.35 synthesises small-ear
mass at zero data cost, one line. (3) A per-level hard-negative cap so the 2048
small slots cannot absorb the global budget.

BlazeEar would try (2) before (1), on the grounds that it attacks the exposure
gap directly rather than reweighting a loss over a distribution that still
barely contains the band -- and that it leaves the mode's mass intact, which is
what the guard above is protecting.

**Strengthened guard:** report mean positives-per-ear and the per-prior positive
share on every run alongside the 64-80 band, because levers (2) and (3) change
the assignment statistics and a run that improved the tail by starving the mode
would show up there before it showed up in recall.

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
