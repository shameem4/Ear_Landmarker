# v2 results

Every number here is test-split (861 held-out real samples, touched once per run)
unless it says otherwise. NME is mean L2 in normalised [0,1] ear-crop
coordinates over visible points; multiply by 192 for pixels at the model's input
resolution. Reproduce with `python scripts/eval_test.py`.

## The noise floor comes first

Three seeds of an identical configuration give **test NME 0.02919 +/- 0.00031**
(persp65: 0.02923 / 0.02948 / 0.02886). So the resolution of this setup is about
**1% relative**, and any claim smaller than that is not a result, it is a seed.
This was measured before the comparisons below were interpreted, and it
retroactively demoted several of them.

## What shipped

| Change | Effect | Verdict |
|--------|--------|---------|
| Rotation label fix | test NME unchanged (+0.8%, inside noise); **photometric jitter -19.5%** | Shipped -- it was a real bug regardless of the metric |
| Soft-argmax (DSNT) head | test NME **0.0301 -> 0.0291 (-4.0%)**; normal error **-11.5%** | Shipped -- clearly above noise |
| Perspective augmentation | **-36% contour error at 50 deg yaw**; +0.0002 on the frontal test set | Shipped -- buys off-axis robustness, costs nothing in-distribution |
| Temporal + bbox smoothing | jitter **-66% to -80%**, error also down | Shipped |
| NMS on intersection-over-minimum | nested duplicate boxes eliminated | Shipped |

### The rotation bug

v1's rotation augmentation transformed labels by the *transposed* rotation
matrix, mislabelling every rotated sample by 12.5px mean / 26.8px p90 at 192px.
It affected 100% of samples. Fixing it did **not** improve accuracy -- the model
had evidently learned to average over the corruption -- but it cut frame-to-frame
jitter by 19.5%. The accuracy story in v2 is the soft-argmax head, not this fix.
`tests/test_augmentation.py` fails on 24 cases if the old matrix is restored.

### Why perspective aug ships despite a flat test number

The test split is near-frontal, so it cannot see what perspective augmentation
is for. Measured on deliberately warped views, contour error at 50 deg yaw drops
36%, plateauing around 65 deg. The shipped web model is `v6_persp65`
(test 0.02928) rather than the marginally better `v2_heatmap` (0.02912), because
that 0.00016 gap is half the seed spread while the yaw robustness is real.

## What did not work

| Change | Result | Why it was rejected |
|--------|--------|---------------------|
| Synthetic renders at 50/50 | test-equivalent val **+1.7% worse**, flat across 200 epochs | Complete seed separation the wrong way: every synthetic seed (val 0.02995 / 0.03012 / 0.02997) worse than every real-only seed (0.02900 / 0.02922 / 0.02895) |
| Contour canonicalization | deformed the contour 2.09px even with PCHIP resampling | Cost exceeded any benefit |
| persp50 vs persp65 | 0.26 sigma apart | Not separable; do not claim either is better |
| Tangential loss weighting | improves `nme_normal`, not total; non-additive with persp50 | Kept as a flag, off by default |
| ROI_EXPAND 1.7 hypothesis | ground truth over 380 matched ears gives ratio 0.982 -> **1.26-1.29** | Refuted. 1.30 was already correct |

### Synthetic data (10,002 AudioEar renders)

Ingested, balanced 50/50 against real, and run for 3 seeds to epoch 257/500
before being stopped. Epoch-matched running-best val NME:

| epoch | real only | +synthetic | delta |
|-------|-----------|------------|-------|
| 20    | 0.0388    | 0.0384     | -0.0004 |
| 60    | 0.0333    | 0.0338     | +0.0004 |
| 140   | 0.0307    | 0.0312     | +0.0005 |
| 220   | 0.0296    | 0.0302     | +0.0005 |
| 257   | 0.0295    | 0.0300     | +0.0005 |

Synthetic leads slightly at epoch 20 then settles into a **fixed +0.0005 deficit**
that does not close. The renders teach useful early structure but do not survive
as a co-equal training signal -- consistent with a domain gap the real val set
correctly penalises.

Reverting is the default: `--synthetic-ratio` defaults to 0.0, the real splits
were never modified, and synthetic never entered val or test. `data/synthetic/`
can be deleted or ignored. Untried and plausible: a lower ratio (~0.25), or
synthetic warm-start followed by real-only fine-tuning.

### ROI_EXPAND -- this conclusion was wrong, and is retracted

The original claim here was that ROI_EXPAND 1.30 is correct and a proposed 1.7
was refuted, on the basis of 380 matched ears giving true-ear-extent /
detector-box-extent = 0.982.

**That measurement compared the detector's boxes to the YOLO dataset's own
ground-truth boxes.** It therefore showed that BlazeEar reproduces its training
convention -- not that its boxes cover the ear. Those GT boxes are themselves
tight. I validated the detector against itself, which is a circularity of the
same family as the one the original hypothesis was criticised for.

Measured against the ear's actual extent on real captures, the ratio is 1.02 to
1.53, implying a required expansion between 1.31 and 1.97. The 1.7 proposal was
closer to right than this file claimed.

No single constant fixes it, since the ratio varies by a factor of 1.5 across
frames. The pipeline now derives the ROI from the landmarks instead; see the
Pipeline section of README.md.

## Where the remaining error is

**72% of squared error is tangential** -- along the contour rather than across
it. Per-source tangential error tracks that source's landmark spacing
coefficient of variation almost exactly:

| Source | spacing CV | tangential error |
|--------|-----------|------------------|
| audioear2d | 0.137 | 2.74px |
| collectionB | 0.250 | 4.96px |
| collectionA | 0.286 | 7.44px |

That is annotation noise, not model error: human annotators place points at
inconsistent arc-length positions along the same contour. It puts a floor under
NME that no architecture change will lift, which is why `nme_normal` is the
better model-selection metric and why this configuration is considered done.

## Corrections found after the fact

Two defects surfaced in review after the experiments above were complete. Neither
changes any number in this file -- the ROI bug is inference-side and the naming
bug is confined to `model/measure.py`, which the training and NME results do not
use -- but both affected shipped behaviour.

**ROI crops were clamped to the frame, not padded.** The intended square ROI was
being clipped at image edges, producing a non-square crop that was then resized
to 192x192, stretching the ear along one axis. The model never saw that
distortion in training. Measured against grey-128 padding (what `data/dataset.py`
uses) on held-out samples with a 20% edge overlap:

| policy | NME |
|--------|-----|
| clip (old) | 0.05295 |
| pad (new) | 0.04373 |

**-17.4%, better on 89% of samples.** 14.8% of ground-truth ears in the local
test set sit close enough to a frame edge to trigger it, losing 19.5% of the
intended crop area on average (p90 35%). Present in both `inference.py` and the
browser pipeline. Fixed; `tests/test_roi_crop.py` covers it.

**Crop width has an asymmetric cost.** Too tight is a hard failure: the ear does
not fit and landmarks cannot reach the rim. Too wide degrades gracefully, then
steeply. Measured on full scenes with real surrounding context, drift from the
converged answer is ~0 at 1.7x, 7% at 2.0x, 18% at 2.5x, 31% at 3.0x and 67% at
4.0x -- driven mainly by resolution, since the 192x192 input spends more of
itself on background as the crop grows. If a single constant must be used,
1.6-2.0 is the safe band; below ~1.4 it fails hard on tight boxes.

**Three of four landmark group names were wrong.** The strips were named
helix / antihelix / concha / tragus; under the iBUG scheme these sources actually
use, 50-54 is the *superior crus*, not the tragus, and the real tragus (35-38)
sits inside the strip that was called "concha". `measure_ear()` consequently
reported `tragus_to_antitragus` computed from the superior crus, and measured the
concha across a span five structures wide. Fixed, with `IBUG_REGIONS` as the
authoritative mapping and `tests/test_landmark_naming.py` locking it.

## Cynic pass: what a review of this file found

Reviewed after the fact, several claims above did not survive contact.

**The adaptive-ROI validation was circular.** It measured error against the
model's own converged fixed point, which the refinement reaches by construction --
that shows convergence, not accuracy. Re-run against real ground truth on the
test split, with the detector box artificially tightened, the fix does hold and
the honest framing is different: adaptive keeps NME at ~0.040 regardless of how
tight the box is, while the fixed policy degrades from 0.0400 to 0.0657 (64%)
across the observed ratio range. Where the box already frames the ear correctly
the two are a wash, within +/-0.6%.

(That re-run was itself repeated later. The first version fed the model [0, 1]
input when it is trained on [-1, 1], which handicapped both arms; the numbers
above are the corrected ones. The conclusion did not change, but the "+1.5% cost
at correct framing" first reported here was an artifact of that error.)

**`scripts/eval_test.py` did not reproduce the logged numbers**, despite this
file telling readers to reproduce with it. It picked the best checkpoint by the
`nme=` field in the filename, which is rounded to four decimals -- and every run
here has two or three checkpoints tied at that precision, so the tie fell to glob
order and a different checkpoint than train.py had tested. Now selected by the
full-precision score stored inside the checkpoint; the numbers match the logged
`test/nme` exactly. The shipped model is 0.0292, not the 0.0293 previously
reported here.

**The corrected tragus measurement broke the module's own premise.** `measure.py`
exists to be index-free, and the first version of the naming fix took a max over
raw point sets -- reintroducing exactly the tangential sensitivity it removes.
The test that was supposed to cover it used degenerate input (all tragus points
identical) and could not have caught it. Both are fixed.

**A cached ROI expansion could leak across sources.** The per-track cache was
keyed by track id, but `EarTracker.reset()` restarts ids at 0, so after switching
webcam to image an unrelated ear would inherit a stale expansion. The cache also
grew for the life of the process. Both fixed, in the Python and JS pipelines.

**The refinement could run away on video.** Each frame seeds its ROI from the
previous frame's cached expansion, which makes the loop self-reinforcing: a
landmarker reporting a saturated extent grows the crop, the grown value is
cached, and the next frame starts larger. Measured at ~1.53x growth per frame,
reaching **38x the detector box within eight frames**, with no recovery -- each
enlargement shrinks the ear in the crop, which keeps the model saturated. The
earlier 30-frame check missed it because every detection in it was good. The
expansion is now clamped to 1.0-2.5x, which leaves the real captures untouched
(they converge to ~1.99x) and keeps the crop inside the measured-cheap band.

**Smaller:** `docs/README.md` documented a `reset()` method that did not exist
(the real one was `resetSmoothing()`); an alias now makes the documented name
real. And the "+26 tests" figure in one commit message was inflated -- 23 of
those are one file-scan guard parametrised across source files, so the genuinely
new assertions number about 25.

Verified and unchanged: the iBUG landmark semantics were re-checked against the
primary source (ibug.doc.ic.ac.uk), not the search summary they were first taken
from, and match.

Not challenged by the review: the rotation bug, the soft-argmax gain, the
smoothing results, the NMS calibration and the synthetic negative result all rest
on held-out or logged measurements that still check out.

## Detector: BlazeEar v2 (two-stage)

The limitation recorded here previously -- "the detector recalls 47.5% and is the
binding constraint" -- has been addressed upstream. BlazeEar v2 restored the
MediaPipe two-stage pattern: BlazeFace on the full frame, then the ear model on a
square crop at 1.5x the face box.

Re-measured here on 500 annotated full scenes, with the same harness that
produced the original 47.5%:

| detector | ear recall (IoU>=0.3) | GT ear / box, median | p10-p90 |
|----------|------------------------|----------------------|---------|
| pre-v2 single-stage | 47.6% | 0.981 | 0.715-1.280 |
| **v2 two-stage** | **78.6%** | 0.984 | **0.827-1.153** |

Recall up 65% relative. The old figure reproducing at 47.6% is a useful check on
the harness. Two things follow for this repo:

- **The box convention did not change** (median 0.981 -> 0.984), so ROI_EXPAND
  1.3 and the adaptive refinement remain correctly calibrated. Predicted ear
  heights on the four capture fixtures are unchanged to within 1%.
- **The boxes are more consistent**, p10-p90 narrowing from 0.72-1.28 to
  0.83-1.15, so the refinement fires less often. It is still load-bearing: the
  spread straddles any single constant.

Caveat: these images come from BlazeEar's own training corpus, so both recall
figures are optimistic. The comparison between them, and the ratio, are what this
measures. BlazeEar's own held-out numbers (mAP@0.5 0.179 -> 0.581 on images no
model in that repo trained on) are the ones to cite for the detector itself.

The update was also a **breaking change**: v2 replaced the folded backbone with a
trainable-BatchNorm one and fitted new ear anchors, so this repo's hand-rolled
`BlazeEar()` construction raised on load. Detection now defers to BlazeEar's own
`load_ear_model` / `blazeear_inference.js` instead of duplicating them.

## Could MediaPipe's FaceMesh have done this job?

No. The question was whether v1 should have adopted MediaPipe's face landmark
architecture and its weights instead of sizing a backbone independently -- and
there is **no recorded rationale** for that original choice anywhere in the
repo, which is why it was worth measuring rather than arguing.

`model/facemesh_ear.py` copies the topology faithfully: MediaPipe's published
weights load into it with `strict=True`, and its 192x192 input happens to match
EarLandmarker's exactly. The only change is forced -- the output conv emits
468 x 3 = 1404 channels and ears need 55 x 2, so that one layer is resized and
left random. It is also the only layer holding face-landmark *identity*, so no
face geometry is reshaped into an ear; what transfers is generic feature
extraction.

Test NME, 3 seeds per arm, scored against a rule fixed before the runs started:

| arm | params | init | test NME | vs control |
|-----|--------|------|----------|------------|
| **control** (EarLandmarker) | 340,167 | scratch | **0.02919 +/- 0.00031** | -- |
| `fm_pre` | 206,942 | 84.6% MediaPipe | 0.03147 +/- 0.00010 | **+7.8%** |
| `fm_scratch` | 206,942 | random | 0.03273 +/- 0.00027 | +12.1% |

**The independent design earns its keep.** A faithful copy loses by 7.8% even
with the pretrained weights -- far outside the 1% seed-noise floor, with no seed
overlap.

**The pretrained weights do help: +3.83%, complete rank separation.** Every
`fm_pre` seed beats every `fm_scratch` seed. Generic face features transfer to
ears despite the face-vertex layer being discarded. But they buy 3.8% against a
12.1% deficit, closing under a third of it.

The transfer advantage decayed as training went on -- 22% at epoch 40, 12% at
80, 4.9% at 195, 3.8% at the end -- the usual shape for pretraining, and it did
not vanish.

### Separating the head from the backbone

`fm_heat_pre` keeps MediaPipe's `backbone1` exactly as published, still 81.6%
pretrained, and swaps its regression head for this project's heatmap decoder.
That drops the model to 141,991 params, because the head it replaces was 81% of
MediaPipe's weights.

| step | test NME | gain | rank sep |
|------|----------|------|----------|
| `fm_scratch` (MediaPipe head, random) | 0.03272 +/- 0.00027 | -- | -- |
| + MediaPipe pretrained weights | 0.03147 +/- 0.00010 | **+3.83%** | yes |
| + soft-argmax head (`fm_heat_pre`) | 0.03087 +/- 0.00033 | **+1.92%** | yes |
| remainder: backbone / capacity | 0.02919 (control) | **+5.75%** | -- |

Both interventions are real and separable, and together they recover 5.75 of the
12.1 points. **The largest single component is the backbone itself.**

Worth recording: the head's advantage is mostly a CONVERGENCE-SPEED effect. It
measured +9.45% at epoch 44, +5.31% at 200, +3.70% at 307 and +1.92% at
completion, while the gap to the control WIDENED over the same span. An early
read of that arm would have concluded the opposite of the final result, which is
the third time in this work a partial measurement inverted.

### Design or just size? Not settled

The control is 2.4x larger than `fm_heat_pre`, so the +5.75% above cannot
distinguish a better architecture from a bigger one. `--width-mult` scales every
channel count to test it; 0.62 puts this project's own design at 142,935 params,
a 0.7% capacity match with the MediaPipe copy.

**The run was stopped at epoch 227 of 500 and has no test number.** Where it
stood, epoch-matched on validation:

| arm | params | init | val NME @227 |
|-----|--------|------|--------------|
| `cap62` (this design) | 142,935 | scratch | 0.03133 +/- 0.00011 |
| `fm_heat_pre` (MediaPipe) | 141,991 | 81.6% pretrained | 0.03133 +/- 0.00028 |
| control | 340,167 | scratch | 0.02965 |

Dead level at matched capacity, with no rank separation -- and `cap62` reached it
from random initialisation against an arm carrying pretrained weights. Subtracting
the +3.83% those weights were measured to be worth would put this design ahead
per-parameter, but that is a subtraction across experiments, not a measurement,
and it is not claimed here.

**So the statement "the independent design earns its keep" is NOT established.**
On the evidence actually in hand, the two designs are indistinguishable at equal
size, and the shipped model's advantage over the MediaPipe copy is substantially
its extra parameters. Finishing `cap62` would settle it;
`scripts/run_capacity_matched.sh` re-runs it.

### How far does more capacity go? Unanswered

A width sweep above 1.0x was started and stopped at epoch 46 of 500 -- 9%
through, which this work has repeatedly shown is too early to read. It is
recorded only so the arm is not mistaken for untried. `w18` (1.03M params) was
cancelled before starting.

The reason for stopping was not the numbers but the diagnosis underneath them:

- the shipped model's train/val gap is **+14.1%** measured with augmentation ON,
  so the true gap is wider. That is a data-limited model, where extra capacity
  costs generalisation rather than buying it.
- **72% of its squared error is tangential**, tracking annotator spacing rather
  than anything the model controls. If that is a floor, the best reachable NME is
  ~0.0233 against today's 0.0292 -- **20% of total headroom**, shared across every
  possible change, of which capacity can claim only a slice.

Scaling the model is pushing on the wrong variable. The binding constraints are
label consistency and real-data volume.

## Known limitation

Detector recall is now 78.6% on this set rather than 47.5%, so the landmarker and
the detector are closer to balanced. The remaining ceiling is faces BlazeFace
misses (1.4% of images), which no second stage can recover.

# v3: replacing the training data

The shipped model is now `manual_occ_s42`, trained on `data/manual` alone. This
section records why, including the parts that argue against it.

## The benchmark was measuring the wrong thing

A model trained only on the commissioned data scores **0.0347** against
`v6_persp65`'s **0.0292** -- 19% worse. That number does not mean what it looks
like. Broken down by source:

| source      |   n | v6_persp65 | manual_occ | gap    |
|-------------|-----|------------|------------|--------|
| collectionB | 472 |    0.03379 |    0.04326 | +28.0% |
| collectionA |  89 |    0.04590 |    0.04863 |  +5.9% |
| audioear2d  | 300 |    0.01710 |    0.01726 |  +0.9% |
| ALL         | 861 |    0.02923 |    0.03475 | +18.9% |

**Excluding collectionB the gap is +3.1%.** collectionB is 55% of the test split
and 54% of `v6_persp65`'s training data, so the benchmark substantially measures
agreement with one source's annotation convention. That source is also the
outlier on every structural axis measured: ear-axis tilt 30 deg off every other
source, the highest and tightest frame occupancy, 16 of the 20 worst samples, the
loosest point spacing (step CV 0.249 against audioear2d's 0.149), and the only
ground truth in the corpus that a model beats on image-gradient alignment.

Two things this does NOT establish. collectionB is not corrupt -- it has zero
self-crossing contours and its random sample tracks ear anatomy fine. And the
sources disagree with each other on ear aspect ratio (medians 1.62 / 2.26 / 2.61)
with no independent arbiter, so "collectionB is wrong" is not proven, only
"collectionB is different and dominant".

## Judging without any ground truth

455 real full frames from BlazeEar's local data, 653 ears, both models through
the identical two-stage pipeline. Contour placement scored by mean image-gradient
response along the outer helix, per-image normalised so blur and exposure cannot
bias it (`scripts/eval_gt_free.py`). Three seeds per side:

| group | seeds                 | mean  | within-group range |
|-------|-----------------------|-------|--------------------|
| v6_persp65    | 3.9285 / 3.8854 / 3.8520 | 3.8886 | 0.0765 |
| manual_occ    | 4.0774 / 4.0461 / 4.0769 | 4.0668 | 0.0313 |

Difference **+0.1782, which is 2.3x the larger within-group range**, with
complete separation: the worst clean seed beats the best shipped seed. Exact
permutation over all 20 splits puts the observed difference first, p = 0.05 --
the floor a 3-vs-3 design can reach.

**The case against this result.** Edge response rewards a contour lying on a real
intensity boundary but cannot tell the helix rim from the jawline, so a contour
that overshoots onto another strong edge is not penalised as it should be. It is
evidence about contour placement, not a substitute for annotated error. The
honest summary is that the shipped model is better on a proxy and competitive on
everything except collectionB -- settling it needs a test set annotated from
scratch, which does not exist yet.

## Two bugs found while chasing this

**Constant-scale crops.** The first ingestion cropped every manual image to
exactly 0.777 occupancy -- sd 0.001 against the old corpus's 0.097. 99.2% of the
old test set lay outside that range. Restoring the spread cut the model's
scale-error sd from 0.0578 to 0.0409 and closed the gap on audioear2d from +22.5%
to +0.9%, while leaving the aggregate NME flat because collectionB moved the
other way. The aggregate hid a large per-source effect in both directions.

**Clamped landmarks.** `preprocess.py` clipped normalised landmarks to [0,1],
pinning out-of-crop points onto the frame border, where `dataset.py`'s
`(lm >= 0) & (lm <= 1)` visibility test passed them as visible and supervised them
at a fabricated position. 714 points, 0.41% of collectionB's against ~0%
elsewhere. Both ends fixed; `tests/test_landmark_clamping.py` locks it.

## What rotation did not explain

The old corpus has a wider and bimodal ear-axis distribution (collectionB median
+47 deg, audioear2d +17 deg). Matching the total rotation sd with wider
augmentation did nothing: +-15 deg 0.03420, +-26.5 deg 0.03469, +-40 deg 0.03491.
The per-tilt breakdown is flat at ~17% across the whole rotation range and both
models correlate with tilt almost identically (Spearman +0.444 vs +0.454).

## Pose behaviour, measured on real 3D ear geometry

Perspective augmentation is a homography. It reproduces foreshortening but not
parallax or self-occlusion, and its measured benefit plateaued at 65 deg. Two
readings fit -- the model saturates, or the warp stops resembling a head turn.
Rendering real ear meshes (`scripts/render3d_ears.py`, AudioEar3D) separates them.

**The warp is a poor match to a real rotation at every angle**, not just past 65
(`scripts/eval_homography_fidelity.py`, 12 ears). There is no cliff. NOTE: these
numbers predate the reconstruction fix below; both sides of each comparison come
from the same mesh so the NCC column largely cancels the defect, but the
silhouette IoU column is contaminated and has not been re-measured:

| yaw | NCC(true 3D, homography) | silhouette IoU |
|-----|--------------------------|----------------|
| 10  | 0.804 | 0.850 |
| 20  | 0.614 | 0.747 |
| 30  | 0.466 | 0.658 |
| 50  | 0.255 | 0.533 |
| 65  | 0.146 | 0.440 |
| 80  | -0.086 | 0.231 |

**And it buys nothing measurable on either axis**
(`scripts/eval_pose_consistency.py`, 28 ears, drift vs each model's own pose-zero
landmarks lifted to 3D):

| comparison, abs(angle)>=20 | yaw | pitch |
|----------------------------|-----|-------|
| v6_persp65 vs v2_heatmap | +6.2% (t=+2.53) | -3.0% (t=-0.90, ns) |
| manual_occ_s42 vs v2_heatmap | **+35.2%** (t=+9.67) | +3.3% (t=+0.72, ns) |

Perspective augmentation shows no pitch benefit and is slightly WORSE on yaw. So
the earlier "-36% contour error at 50 deg yaw" says the model learned to handle
the WARP; under a real head turn no benefit is measurable on either axis. That
sits consistently with the fidelity table above -- the warp is a poor match to a
rotation at every angle, so there is little reason to expect transfer.

CORRECTION. An earlier version of this section reported the pitch comparison as
**-18.0% (t=-4.00)** and explained it mechanically: the homography helps where
self-occlusion is absent (pitch) and cannot help where it dominates (yaw). That
was measured on meshes whose reconstruction left holes, where only 67% of
landmark rays hit the surface -- so a third of each contour was dropped, selected
by where reconstruction failed rather than by anything about pose. On corrected
meshes (96% hit rate) the pitch benefit is -3.0% and not significant. The
mechanism story was clean, plausible and an artifact. The yaw conclusion and the
NCC fidelity measurement both survived.

The shipped model drifting ~35% more under yaw DID survive the fix (+40.2% before,
+35.2% after), and is the one result arguing against it. Treat it as unresolved
rather than as a defect: this metric scores each model against its own pose-zero
output, so sharper landmarks are a harder reference to re-hit, and the
ground-truth-free photo comparison found manual_occ placing contours better.
Separating precision from fragility needs annotated ground truth.

### Two measurement routes that failed, so nobody repeats them

**Edge response on renders.** The metric behind `scripts/eval_gt_free.py` works
on photographs, where the ear sits against a head. An isolated render against
flat grey has an enormous silhouette gradient, so a foreshortened ear rewards any
contour hugging its outline. It reported that rotating an ear IMPROVES quality --
retention above 100% at nearly every angle.

**Lifting AudioEar3D's annotations.** The dataset annotates a photograph and
ships a point cloud of the same ear with no camera relating them. Silhouette
registration reaches 0.897 IoU and still leaves 12px reprojection error, 3.7% of
the crop diagonal, against the 2.9% NME it would judge. Fitting an outline leaves
in-plane rotation and depth slack exactly where landmarks are most sensitive.
Defining the landmarks on our own render instead makes the camera exact: 0.006%
round-trip error, 600x better, no registration at all.

### Can the renders be used as training data?

Not as they are. They are isolated ears on flat grey, with Poisson holes and a
single-sided surface, against real crops where the ear is embedded in hair, skin
and varied lighting. A model would learn "grey surround = ear boundary", a
shortcut absent from real images. Using them would mean compositing onto real
head backgrounds first, and even then they add pose diversity for 112 subjects
rather than new subjects. The narrow target is yaw, since pitch is already
covered.

## Rendering real geometry so a photo-trained model can read it

Renders are only useful here if the landmarker treats them like photographs.
Measured by the model's own mean landmark confidence, on HUTUBS heads against
60 real crops from data/manual:

| render                              | confidence | of real |
|-------------------------------------|------------|---------|
| grey clay                           | 0.336      | 82%     |
| Open3D, baked AO + reddening (`scripts/skin.py`) | 0.346 | 84% |
| **Blender Cycles, Principled BSDF** (`scripts/blender_skin_render.py`) | **0.392** | **96%** |
| real photographs                    | 0.410      | 100%    |

Blender beats the Open3D shader by +0.0456 (t=+3.10, n=10), closing three
quarters of the remaining gap. Real subsurface scattering and specularity are
worth the ~100x render cost for a final measurement; Open3D is still right for
iteration.

### What actually moves it

One factor at a time, 3 ears, Cycles. Those 3 ears are easier than average
(baseline 0.426 against 0.392 over 10), so read the spreads against each other,
not against 100%.

| factor          | spread | note |
|-----------------|--------|------|
| **light azimuth** | **0.041** | -80deg 0.385 -> -40deg 0.426 |
| skin tone       | 0.020  | palest 0.413, mid/deep 0.430-0.433 |
| key energy      | 0.020  | shallow optimum at 5W |
| subsurface      | 0.009  | 0.421 (none) -> 0.430 |
| freckles        | 0.007  | slightly NEGATIVE: 0.426 -> 0.419 |
| roughness       | 0.003  | flat |

**Randomise lighting azimuth per render.** This is the finding that matters, and
it is about measurement validity rather than realism: a fixed light contributes
as much confidence variation as the whole renderer upgrade, and in a pose sweep
it rakes across the geometry as the ear turns, confounding appearance with pose.

### Lighting in the real data is not uniform either

Estimating in-image light direction from the brightness centroid within the ear
(handedness folded), over 1,200 training crops: **89% falls in a contiguous
140-degree arc**, circular concentration R=0.715. The estimator was validated on
renders with known azimuth -- it tracked them monotonically over 77deg of
estimated range -- but it is compressed and carries a shape-dependent bias, so
absolute angles do NOT compare across the render/photo domains. Randomise over a
concentrated arc, and calibrate it by matching this estimator's output
distribution rather than by angle arithmetic.

The deployment fixtures (overhead plus monitor) sit at +173deg against training's
-161deg -- 26deg apart, with matching brightness and contrast. The shipped model
is being used in the lighting it was trained for.

### Specular: the amount is right, the placement is wrong

In 1,500 training crops the brightest 1% of ear pixels fall on the antihelix
(36.7%), concha/tragus (35.6%) and helix rim (31.1%), and avoid the lobe (10.6%).
Specular pixels are 0.33% of ear area; renders already produce 0.39-0.62%, so
there is no deficit of specularity. But by region the renders put ~0% in the
concha against 29% in reality, and 32% on the superior crus against 7%.

Driving roughness from mesh curvature does not fix it and changes confidence not
at all (0.424 at every level). The reason is that the concha is glossy because it
is oily and wet, not because of its curvature -- being concave, curvature-driven
gloss makes it MATTE, which is backwards. Fixing it needs an anatomy-based gloss
mask and a small bright light; whether that is worth doing is unproven, since
appearance effects cancel between models in any paired comparison.

# v4: depth

Labels are geometric, not model-derived: 55 marks placed by hand on the corpus
mean, propagated through 476 template fits, carried back onto the original head
meshes by the transform `cut_ears` records, and projected through a known
camera. 9,123 rendered views, split by SUBJECT -- 21 poses per ear and two ears
per person would otherwise put near-copies on both sides.

Depth is root-relative and in the same units as x and y (fractions of the crop
width), because a crop fixes no absolute distance. Every depth number below is
quoted against the **flat predictor** that outputs z = 0 everywhere and scores
0.07043 on the test split; `skill` is the fraction of that error removed.

## Four heads, 35 held-out subjects

| head | z MAE | skill | NME | mesh skill |
|------|-------|-------|-----|------------|
| **1: depth as an integral over the heatmap** | **0.01233** | **0.825** | **0.01064** | -- |
| 2: shape coefficients + pose, with coefficient supervision | 0.02907 | 0.587 | 0.03284 | **0.154** |
| 3b: lifter + features sampled at the landmarks | 0.01507 | 0.786 | -- | -- |
| 3a: lifter from 55 (x, y) alone, no pixels | 0.02027 | 0.712 | -- | -- |
| the 40-mode shape basis fitted directly to the labels | 0.01194 | 0.827 | -- | -- |
| Depth Anything 3, zero-shot, global affine | 0.06397 | 0.056 | -- | -- |

Head 2 here is the 60-epoch run. Trained for 150 epochs its mesh skill
reaches 0.171; see the head 4 section for the final numbers on both splits.

Head 1 wins, and that survived being challenged. Across three seeds the spread
is 0.00012-0.00032 while the gaps between heads are 0.00168-0.00209, five to
seventeen times larger; paired over resampled test subjects every pair is
separated; and a learning-rate sweep helped only head 1, widening the gap. At
150 epochs and lr 1e-2 it passes the basis ceiling, which only ever bound
head 2.

## Most of the depth is in the 2D configuration

A lifter that never sees a pixel reaches skill 0.712, **88% of head 1's**. The
image is worth about 0.11 of skill on top. The 5x-capacity control earns its
place here: a 964k-parameter lifter scores 0.02002 against 189k's 0.02027, so
head 3b's advantage over 3a is the image features and not the weights.

## The mesh is unobtainable from landmarks, by any route

Head 2 exists for the mesh, and judged on landmark depth alone it is actively
misleading. Trained only through the 55 projected points it predicts
coefficients of the right MAGNITUDE -- 0.838 sd against a true 0.773 -- and
correlation **0.017** with the truth. Its mesh sits further from the subject's
ear than the MEAN ear does: skill **-0.468**, while its z MAE of 0.01742 was
the best of the variants tried.

The cause is a gauge freedom: shape and pose are not uniquely separable, so a
loss on projected points constrains only their product. Any decomposition that
reprojects correctly satisfies it equally.

This is not a quirk of training. Fitting shape and pose OFFLINE to head 1's
predicted landmarks -- no network, 1500 iterations to convergence -- gives mesh
skill **-1.037** and coefficient correlation 0.065. Fifty-five projected points
simply do not determine the shape.

Supervising the coefficients directly is the only thing that works: correlation
0.017 -> 0.304, mesh skill -0.468 -> +0.171, at the cost of NME 0.01064 ->
0.03284. So the two heads are run together or not at all.

What the mesh is worth: about 15% of each ear's deviation from the average.
Mode 1 is recovered at 0.77 correlation and seven modes exceed 0.5, but the
tail is noise, and predictions are shrunk (0.444 against a true 0.773) because
that is what least squares does under uncertainty. It is a mildly personalised
ear-shaped mesh, not a reconstruction.

## Two hypotheses that failed, and one metric that lied

Replacing head 2's global average pool with a head that keeps the 6x6 grid was
the obvious fix -- pose is about position and pooling destroys position. Three
times the parameters bought 0.8% (0.01731 -> 0.01718).

Projecting head 1's output onto the shape manifold degrades it by 17.9%. The
first run of that test reported 77% and was simply under-converged at 300
iterations; the direction was right and the magnitude was nonsense.

Averaging head 1 and head 3b gives 0.01287 against head 1's 0.01315 at equal
weights -- about 2%, for 2.7x the parameters and two forward passes. The best
weight found, 0.70, was selected on the test set and is not a real number.

## An outside opinion on the annotations

Every other check in this project measures our landmarks against other
annotations from the same corpus, and when they agree they can still be jointly
wrong -- which happened when a barycentric bug manufactured kinks in three
successive hand-corrected sets and every statistic endorsed them.

SAM 2.1 knows nothing about this project. Prompted with a padded box, so the
contour is its own, its ear silhouette sits **0.006 of ear size** from the outer
helix across 150 images, while the inner helix -- which should not be on a
silhouette -- sits at 0.050. The worst outer-helix points are 0 and 1, the
attachment end where there is no edge to lie on.

## Caveat on all of the above

These are synthetic renders. The production 2D model scores 4.26% error on them
against 1.47% on real photographs, so the renders are harder than production
and these are relative rankings, not real-photo accuracy.

## Head 4: one network for landmarks, depth and a mesh

The question was whether head 2's mesh could be had without head 2's cost to
the landmarks -- by hanging the shape branch off head 1 rather than replacing
it. Head 4 samples the decoder's features at the predicted landmark positions
(detached, so nothing the mesh wants can move a landmark) and regresses the
coefficients from those local samples plus the template's current fit, fed back
for `refine_steps` iterations.

Scored by `scripts/final_eval.py` on both splits, with the same ground-truth
coefficients and the same mean-ear denominator:

| model | z MAE | z skill | NME | mesh skill (val) | mesh skill (test) | params |
|---|---|---|---|---|---|---|
| head 1 | 0.01233 | **0.825** | **0.01064** | -- | -- | 345,502 |
| head 2, coeff sup | 0.02893 | 0.589 | 0.03289 | **0.178** | **0.171** | 906,817 |
| head 4, refine 1 | 0.01382 | 0.804 | 0.01241 | 0.057 | 0.059 | 879,567 |
| head 4, refine 2 | 0.01362 | 0.807 | 0.01182 | 0.051 | 0.048 | 879,567 |

It loses on both counts. The mesh is a third of head 2's, and the landmarks are
11-17% worse than head 1's despite 2.5x the parameters -- so detaching the
samples did not fully insulate the landmark branch either. Feeding the template
fit back a second time changes nothing: refine 2 is marginally BEHIND refine 1
on mesh on both splits, which is the signature of no effect.

Head 2's own mesh improved with the longer schedule, 0.156 at 60 epochs to
0.171 at 150. That is the only thing this round bought.

So the conclusion from before head 4 existed stands: run head 1 for landmarks
and depth, head 2 with `--coeff-sup 1.0` for the mesh. Two networks or no mesh.

### Why feature locality worked for the lifter and not for shape

Head 3b reads local features and gets 88% of head 1's depth, so sampling at
landmark positions is not the problem. The difference is what is being asked
for. Depth at a point is a local property and a local feature can carry it. A
shape coefficient is a global property of the whole ear -- mode 1 is overall
size and curl -- and 55 local samples do not aggregate into it any better than
the pooled vector head 2 already uses. The gauge-freedom result says the same
thing from the other side: the information is missing from a single view, and
no amount of rearranging where the network reads from puts it back.

### A number I reported that was wrong

Mid-training I quoted head 4's validation mesh skill as 0.103 (refine 1) and
0.122 (refine 2) and said the template-as-input was "showing up positively".
Neither figure is reproducible -- `final_eval.py` gives 0.057 and 0.051 on the
same split and the same checkpoints, and the training logs agree with the
script. The 0.12 came from my own arithmetic against the wrong denominator,
and the ordering it implied was never real.

Correcting it, I then claimed the opposite error -- that mesh skill had
collapsed from validation to test. It had not; the two splits agree to within
0.01. Head 4's mesh is not overfitted, it is simply weak, consistently, on both
splits. `final_eval.py` prints both splits and `mesh_base` beside every skill
so that neither mistake is available again.

## Where the 55-point scheme runs out, measured

The iBUG scheme was designed for FRONTAL PHOTOGRAPHS, so every one of its 55
points sits on a ridge or a border and none samples a surface. With a 476-ear
shape model in dense correspondence, "which features should it gain" stops
being a question of anatomical taste and becomes optimal sensor placement.

Landmarks are L = A b under the linear shape model, so observing them with
isotropic noise sigma gives a Gaussian posterior on the coefficients and every
vertex an expected squared reconstruction error. At sigma = 0.0106 of ear size
-- head 1's measured NME on the renders:

| state | surface uncertainty, % of ear size |
|---|---|
| knowing nothing | 2.67 |
| knowing all 55 points | **1.18** |
| after 10 greedily chosen additions | **0.95** |

So the 55 points remove 56% of the surface uncertainty and the worst remaining
region sits at 2.80% -- as uncertain as knowing nothing at all.

**The gaps are the fossae and the lobe, and that is not a coincidence.** The
TRIANGULAR FOSSA between the crura, the CRUS OF THE HELIX, the INTERTRAGIC
INCISURE, the LOBE surface (only its border is marked) and the CONCHA FLOOR are
all smooth shaded hollows with no crisp edge -- which is exactly why a scheme
built for photographs has no points there.

**A CORRECTION, THEN A CORRECTION OF THE CORRECTION.** This section first
named the SCAPHOID FOSSA as the largest gap. I then called that an artefact of
the template being a CUT-OUT ear, on the grounds that uncertainty correlates
**-0.512** with distance to the cut boundary. That correlation is real but it
was read wrong, and separating the two quantities settles it:

| | correlation with distance to the boundary | within 5 mm | beyond |
|---|---|---|---|
| PRIOR variance, how much the ear really varies | **-0.040** | 2.66% | 2.55% |
| POSTERIOR variance, what the 55 points leave | **-0.512** | 1.30% | 0.99% |

The ear varies just as much near the boundary as anywhere; what is missing
there is LANDMARKS. A boundary vertex is 7.8 mm from the nearest of the 55
against 4.9 mm for an interior one. So the elevated uncertainty is sparse
coverage, which is a reason to put points there, not an artefact to exclude.

Two things followed from the wrong reading and are withdrawn. Excluding a
5 mm band was removing exactly the region that most needs landmarks, so
`validate_marks.py --margin` now defaults to 0. And re-wrapping the template
to the UNCROPPED HEAD, which was supposed to remove the artefact, was measured
over 24 ears and changes nothing: boundary-to-interior variance ratio 1.17
with the cut target against 1.14 with the whole head. The template's free rim
moves because it is free, not because of where the head was cut.

Returns diminish fast: the first two additions buy 8.9% and 9.0% of total
uncertainty, the tenth buys 2.5%. About six points capture most of what is
available.

**The catch is that informative is not the same as annotatable.** A point in
the middle of the scaphoid fossa cannot be placed reliably by eye on a
photograph. The fix is to add CHAINS along the valley lines rather than
isolated points, the way the inner helix is already traced -- a groove's
deepest line is followable even where a single point on it is not.

`scripts/where_to_add_landmarks.py` reports and plots this.

### A stored field that is wrong

`ear_pca_*.npz` records `ear_size` in each dataset's NATIVE units: SONICOM in
mm (median 59.9) and HUTUBS in metres (median 0.372), a factor of 160. The
model itself is unaffected, because the Procrustes step removes scale before
the decomposition -- but anything that uses `ear_size` to put size back will be
wrong for one of the two datasets. The PCA's own mean extent (0.4269) is the
only scale that matches `components`.

## The shape model knows which dataset an ear came from

Re-wrapping the template to UNCROPPED HEADS was done to remove the last
dataset-specific rule from the pipeline -- SONICOM ears were cut with a fixed
(30, 52, 42) mm box and HUTUBS with (0.30, 0.42, 0.50) x tragion separation,
so the template's free rim could only ever land on a boundary defined
differently for the two sources. All 476 ears re-wrapped, zero failures.

**It changed nothing measurable, three times over.** Boundary-to-interior
variance ratio 1.13 against 1.11. Source separability 0.945 against 0.939.
And the fits themselves moved 0.55% of ear size on average, 0.33 mm, which is
below the 2D landmarker's own 1.1% error. The crop was inert: the 14 control
points and the projection limit were already pinning the wrap. Kept anyway,
because one less dataset-specific rule is worth having, but it is hygiene and
not an improvement, and `data/depth3d_v5` was NOT re-rendered for it.

**What the exercise did find is worse than what it was looking for.** From 40
scale-free shape coefficients, a logistic regression tells SONICOM from HUTUBS
on a held-out SUBJECT **93.9%** of the time:

| modes | 1 | 2 | 5 | 10 | 20 | 40 |
|---|---|---|---|---|---|---|
| source accuracy | 0.756 | 0.775 | 0.746 | 0.800 | **0.939** | 0.939 |

It is not one bad mode. Mode 11 separates at d = 1.50 and carries most of it,
but accuracy is already 0.76 from mode 1 alone and climbs steadily -- the
signal is spread through the basis. An earlier estimate of 0.60 came from 24
ears and 12 modes and badly understated it.

This is not the crop and it is not the units, since neither the re-wrap nor
canonicalisation moved it. The likely cause is the source geometry itself:
SONICOM ships BEM ACOUSTIC SIMPLIFICATIONS while HUTUBS ships raw scans, so
the pinna surface really is different before anything here touches it. That
would also explain `ccby_plus`, where HUTUBS and AudioEar3D separate at
d = 2.01 and survived every correction.

**What follows.** A pooled shape model mixes two distinguishable
distributions, and any model trained on the pooled corpus can learn the source
instead of the ear -- the depth corpus is 76% SONICOM and its splits are
subject-disjoint but not source-balanced. This cannot be cleaned away
upstream; it has to be handled where it bites: source-balanced splits, and
per-source metrics reported beside the pooled number so a source shortcut
shows up instead of averaging away.

## The source shortcut is real, and the depth head does not use it

The previous section left a loaded gun: 40 shape coefficients identify a
held-out subject's source 93.9% of the time, the depth corpus is 76% SONICOM,
so some of head 1's skill might be "this is SONICOM and SONICOM ears look like
this" rather than "this is the ear in front of me". This measures it.

**Three recipes, one yardstick.** 14 HUTUBS and 14 SONICOM subjects were
carved out first, on their own seed, before any split was made, and trained on
by nothing. All numbers below are on those same 1171 views.

| | train | differs from A by |
|---|---|---|
| **A** proportional | 6148 views, 148 SONICOM + 30 HUTUBS subjects | — |
| **B** balanced | **the same 6148 rows**, sampler weighting the sources equally | sampling only |
| **C** leave-one-source-out | 6144 views, SONICOM only | no HUTUBS ear at all |

A and C match to 0.07% on training-set size, so C is not merely a smaller run,
and C's checkpoint is chosen on a SONICOM validation set -- it never sees a
HUTUBS ear, at any point, for any purpose.

**Three seeds each for A and C**, because the claim being made here is a
null, and a null needs the same error bar as a difference or it only means
"one run, and I did not look twice".

| | A (HUTUBS in train) | C (never saw HUTUBS) | A − C | ranks separate? |
|---|---|---|---|---|
| **HUTUBS** z-skill | 0.7886  [0.7848, 0.7935] | 0.7815  [0.7764, 0.7849] | +0.0070 (+0.90%) | no, ranges overlap |
| **SONICOM** z-skill | 0.8193  [0.8148, 0.8219] | 0.8261  [0.8237, 0.8279] | −0.0068 (−0.82%) | yes |
| **pooled** z-skill | 0.8044  [0.8015, 0.8082] | 0.8045  [0.8008, 0.8065] | −0.0001 | no |

Seed sd is 0.0045 (HUTUBS) and 0.0032 (SONICOM); the two differences are 1.9
and 2.6 standard errors. This project's pre-registered bar is complete rank
separation across three seeds AND >1% relative, and **neither side clears
it**.

**So: deleting an entire source from training costs 0.7% skill on that
source and buys 0.8% on the one that remains.** A small, symmetric
specialisation, in the direction anyone would guess, under 1% either way, and
invisible in the pooled number. The 93.9% separability is real but IDLE --
the information is sitting there in every render and the depth head is not
leaning on it. Balanced sampling (B, seed 0: 0.7886 HUTUBS / 0.8127 SONICOM)
lands on top of A, which says the same thing from the other direction.

HUTUBS scores about 0.03 below SONICOM in every run **including C**, which
never trained on HUTUBS, so that gap is not training composition. It is partly
that HUTUBS is harder in absolute terms (z MAE 0.0149 against 0.0127) and
partly a smaller skill denominator (z base 0.0689 against 0.0740). Both push
the same way, which is why the per-source evaluator prints the baseline beside
the skill.

**What this does NOT license.** It is a statement about depth, on renders.
See the next section.

### Three bugs, each of which produced believable numbers

Worth recording because none announced itself:

- **Every recipe held out its own test subjects.** No two of the three test
  sets shared a subject that all three had kept out of training, so every
  difference would have been part model and part test set. The first version
  of this experiment was discarded and `--holdout` added, which now refuses to
  run if a second invocation would compute a different yardstick.
- **z was re-centred on the wrong points.** The corpus stores z relative to
  the mean of all 95 landmarks; slicing to the iBUG 55 left a per-view offset
  (mean −0.027, sd 0.015 crop widths) and inflated the skill denominator from
  0.0702 to 0.0747, flattering every number by ~6%. Caught before the compute
  landed.
- **Stale checkpoints won a glob.** `ls runs/*/ckpt/epoch=*.ckpt | head -1`
  picked `epoch=0` files left by an interrupted run, and scored three models
  at epoch 0 into a complete, plausible table (z-skill 0.35–0.47).
  `train_depth.py` now refuses to start in a directory that already holds
  checkpoints.

## Head 1 has never been validated on a photograph

`runs/` is not tracked, so the two figures this section describes are made
with:

    python scripts/demo_3d.py --data data/depth3d_v5 --split common_test.csv \
        --n-points 55 --out runs/demo_3d_renders.png \
        --ckpt runs/sb_a_prop/ckpt/epoch=132-val_z_mae=0.0140.ckpt
    python scripts/demo_3d.py --data data/preprocessed --split manifest.csv \
        --sources audioear2d --n-points 55 --match-framing 0.354 0.677 \
        --out runs/demo_3d_photos.png --ckpt <same>

On held-out renders it does what it was built to do. Turn the predicted point
set 55 degrees about the vertical and it is still a recognisable ear: the
concha floor, the helix standing off the head, the lobe. That survives only if
the depths are jointly right, so it is a stronger statement than the MAE.

On real photographs it does not, and the gap is now measured rather than
eyeballed. **300 held-out AudioEar2D photographs**, scored against the human
annotation, with the SHIPPED 2D ONNX on the same images as the reference --
without a floor, head 1's number has nothing to be bad relative to:

| model, on the same 300 photographs | NME | vs floor |
|---|---|---|
| shipped 2D ONNX, image as stored (the floor) | 0.01726 | 1.00x |
| head 1, image as stored | 0.09286 | 5.38x |
| head 1, re-cropped to the render framing | 0.08842 | 5.12x |

**Head 1 is 5.1x worse than what is achievable on these images.** An earlier
pass through this section quoted 0.044-0.088 from six hand-picked panels and
called the magnitude unknown; over 300 images it is 0.088, and the floor makes
it interpretable. Both of those earlier statements are superseded.

**Framing explains almost none of it, and the control says so twice.** The
two corpora frame the ear very differently -- renders at 0.354 x 0.677 of the
crop, the real-photo pipeline at 0.835 x 0.827 -- so the model was being shown
ears 2.4x too wide. Correcting it moves 5.38x to 5.12x and no further.

The second check is the one worth keeping, because it nearly overturned the
result. The re-crop is not a free operation: AudioEar2D ships ear crops with
no surrounding head, so cropping outward pads with black instead of revealing
pixels. Run that same round trip on RENDERS, where ground truth exists
(`scripts/eval_crop_degradation.py`, 400 held-out views):

| condition | NME | z skill |
|---|---|---|
| as rendered, in domain | 0.01178 | 0.8025 |
| tight-cropped to the photo framing | 0.01447 | 0.6673 |
| then back out to the render framing, black padded | 0.01853 | 0.5837 |

So the crop step IS damaging -- 1.6x on NME and a quarter of the depth skill
-- and the honest baseline for the photo number is 0.0185, not 0.0118. The
domain gap on top of that is 4.8x rather than the 7.5x a naive comparison
gives. Either way it dwarfs the framing. Worth noting separately how brittle
depth is to context: z skill falls 0.80 to 0.58 from cropping alone, on
images that are otherwise perfectly in domain.

**The x, y half is now measured; the DEPTH half on a real photograph still is
not**, because no depth label exists for a photograph anywhere in this
project. Every z number in this document, including the source-balance result
above, is a statement about rendered ears.

### The paired data to fix that is already on disk

AudioEar3D is **CC BY 4.0** and ships 112 ears from 56 subjects as RGB
photographs AND point-cloud scans OF THE SAME EAR. Both are local, and
`lm3d_v5_audioear_mm.npz` already carries all 95 landmarks for all 112 ears
with no missing points. What is absent is the camera: a depth label for a
photograph needs the 3D landmarks expressed in that photograph's frame.

`scripts/pnp_camera_recovery.py` was written for exactly this -- feature
matching plus RANSAC PnP with a focal sweep, exploiting the fact that the
AudioEar3D clouds carry per-point RGB sampled from the very photo being
localised -- but it has never been run to completion, has no saved output and
appears nowhere in this document. Finishing it would turn 112 real
photographs into the project's first real-photo depth benchmark.

Two alternatives were checked and rejected. The **York 3D Ear Dataset** (500
3D ears, York Ear Model) is non-commercial research and education only, the
same category as Aachen and IIT Indore. **HUTUBS** is CC BY 4.0 but ships
meshes and HRTFs, not photographs -- the ear images in the literature on it
are rendered from its meshes, which is what this project already does.

## The real-photo depth benchmark cannot be built from AudioEar3D

The plan was to turn AudioEar3D's 112 photograph-plus-scan pairs into the
project's first real-photo depth labels: recover each photo's camera, project
that ear's own 3D landmarks, read off z. **It does not work, and the reason is
specific and measured.** No benchmark was shipped.

### The method itself is sound, and validated

`scripts/fit_photo_depth.py` solves pose by scaled orthography rather than
full PnP, and that is a choice rather than a shortcut. Full PnP with unknown
focal is badly conditioned on something as shallow as an ear -- a long lens
far away and a short lens close by fit almost equally well. But the corpus
defines depth as `zn = (zc - zbar) * fx / (s_crop * zbar)`, which depends on
fx and zbar ONLY through their ratio, the pixels-per-millimetre scale. That
ratio is exactly what a scaled-orthographic fit observes. The ambiguity
cancels out of the quantity being measured.

Against 1171 renders whose true z is known, the solver recovers depth to
**0.0120 mean, 0.0091 median**, and it degrades gracefully with noise on the
2D input: 0.0144 at noise 0.01, 0.0198 at 0.02. Using v6 landmarks against v5
renders raises the error to 0.0127 and the residual with it, which is a
pleasing independent check that the version pairing is wired correctly.

### Scaled orthography needs the depth sign, and two plausible rules were wrong

Orthography cannot tell a shape from its mirror in depth. Rules tried:

| disambiguator | correct on true render depths |
|---|---|
| the concha is farther than the helix | 76% |
| the canal is the farthest point | 89% |
| the ear's plane normal, oriented by the canal being innermost | inverted 14% of EARS |

The first two are corpus statistics, so enforcing either would have corrupted
one label in ten while appearing to fix things. The third is the instructive
failure: the plane normal was fine, it was the CANAL TEST orienting it that
flipped, and because that flips per ear rather than per view it survived
every per-view quality filter at a constant 17.9% -- residual-based gating
did not move it at all.

The fix is that no per-ear estimate was needed. Canonicalisation has already
put every ear in one frame, so the outward direction is a CONSTANT: over 588
ears the landmark surface normal agrees with a single axis to within 7.6
degrees worst case and 0.6 on average, left and right alike. Only the overall
sign needed settling, one bit, fixed once against the renders. **0.0% flipped
after that**, against 17.9% before.

### The test that killed it

A depth label is only worth having if it describes THIS ear. So: fit each 2D
set to its own 3D ear, and to a stranger's, and compare the residual.

| | own ear | a stranger's | ratio |
|---|---|---|---|
| renders | 0.00660 | 0.01361 | **2.06** |
| AudioEar3D photographs | 0.03364 | 0.03369 | **1.00** |

On renders the individual ear carries the fit and using the wrong one costs
167% more depth error. On the photographs **a stranger's ear fits exactly as
well as the subject's own**. A label built from that is a generic ear under a
recovered pose, and could not tell a good 3D model from one that predicts the
mean ear -- which is the entire thing a depth benchmark exists to do.

### Which of the three suspects it is, and which it is not

- **Not the 3D fits.** The AudioEar3D ears carry 82% of the main corpus'
  shape variance, 11.07 mm of relief against 10.49, and 49 of 55 landmarks
  sit within 2 mm of an actual cloud point (median 0.99 mm). The "40 of 55
  supported" note in my own memory was pessimistic.
- **Not the pairing.** Pairing each photo with the subject's OTHER ear scores
  1.00 as well, so this is not an off-by-one in the name mapping; nothing
  fits better than anything else.
- **Not perspective.** Full perspective PnP swept from 0.8 to 20 crop-widths
  of focal never beats weak perspective (0.0360 at best against 0.0336) and
  never lifts the ratio above 1.04.
- **Mostly the 2D annotations, but not only.** The photo residual of 0.0336
  at ratio 1.00 lands exactly where renders land at ~0.03 of 2D noise (0.0369
  at ratio 1.06), so AudioEar3D's hand annotations carry roughly 3% of crop
  width of error, about 5.7 px on a 190 px image. But substituting the
  shipped 2D model's own predictions -- which disagree with those annotations
  by NME 0.0220 -- only moves the residual to 0.0305 and the ratio to 1.02.
  A better 2D source barely helps, so a floor of about 3% of crop width is
  left that no rigid camera model and no 2D source removes.

### What would be needed

AudioEar3D ships no camera parameters -- the per-ear `.json` is a LabelMe file
of 55 2D points and nothing more -- and the clouds are textured from a
different capture than the photographs (cloud RGB mean [154, 99, 94] against
the photo's [113, 82, 74]), which is why the SIFT route in
`pnp_camera_recovery.py` finds 8 RANSAC inliers of 307 and calls itself WEAK.

So the blocker is not the camera solver. It is that nothing in this dataset
pins a photograph to its own scan accurately enough. Getting a real-photo
depth benchmark needs either 2D landmarks on these 112 photos that are
several times more accurate than either source available, or a dataset that
ships the capture camera. One useful thing did come out of looking: unlike
AudioEar2D, **the AudioEar3D photographs include surrounding head and hair**,
so they are not pre-cropped and would suit the framing question from the
previous section.

### Would better annotations rescue it? No, and the arithmetic says why

The obvious next move is to annotate the 112 photographs properly. It was
costed before doing it, and it does not work.

**The 3D ears are not mute.** They predict an identity signal of 0.0060 of
crop width on the photographs against 0.0095 on renders -- 63% as strong, so
the scans do carry individual shape. The problem is what sits on top of it:
the miss is 0.0340, **5.6x the signal**, against 1.3x on renders.

**Decomposing that miss settles the question.** Two independent 2D sources
are available for these photographs -- AudioEar3D's own hand annotation and
the shipped 2D model -- and they disagree by 0.0220. If they err about
equally that is 0.0156 each, or 3.0 px on a 187 px image. But the miss is
0.0340, so

    systematic part = sqrt(0.0340^2 - 0.0156^2) = 0.0303   (5.8 px)

**The systematic part is twice the annotation part**, and it is 5.0x the
identity signal. A flawless annotator removes the 0.0156 and leaves the
residual ratio at **1.02**, against the ~1.50 a usable label needs and the
1.00 measured now. Annotation is not the binding constraint; something
systematic between an AudioEar3D scan and its own photograph is.

For completeness, the accuracy that WOULD be needed is 0.007 of crop width,
about **1.3 px on a 187x186 image** -- and 187x186 is AudioEar3D's native
resolution, not a crop of something larger. That is 0.7% of image width on a
photograph soft enough to have laplacian variance 63. No annotator reaches
it, and averaging independent annotators to get there would take about 25 of
them.

**Nor does annotating all 95 points instead of 55 help**, which was the
natural follow-up since the 40 extensions were chosen precisely to catch
shape the 55 miss:

| points | 2D noise | own ear | stranger | ratio | identity signal |
|---|---|---|---|---|---|
| 55 | 0 | 0.00660 | 0.01361 | **2.06** | 0.00924 |
| 95 | 0 | 0.00697 | 0.01353 | 1.94 | 0.00884 |
| 55 | 0.0156 | 0.02006 | 0.02353 | 1.17 | 0.00940 |
| 95 | 0.0156 | 0.02050 | 0.02351 | 1.15 | 0.00889 |

Slightly WORSE, and the identity signal slightly lower. The extensions are
derived from the same template surface, so they carry no shape information
independent of the 55 for the purposes of a rigid fit that 55 points already
over-determines many times over. Worth knowing before commissioning
annotation at twice the price. (The photo side is stuck at 55 regardless:
AudioEar3D and AudioEar2D both ship the iBUG 55, and the extensions have
never been annotated on a photograph by anyone.)

### A camera-free metric was tried and is invalid

If the camera is the problem, compare shapes without one: Procrustes-align
head 1's predicted 3D point set to the subject's scan, pose and scale free.
The control kills it. On renders the **TRUE** landmarks score 0.1205 against
0.0754 for simply predicting the mean ear -- perfect ground truth loses to a
constant. Head 1's output lives in a perspective-distorted camera space (x, y
projected, z divided by the crop's world width) and the 3D ear does not, and
a similarity transform cannot absorb the difference. Any photo number from
this metric would have been meaningless, and none is reported.

One comparison inside it does survive, because both sides share the identical
distortion: head 1's prediction against the same prediction with z zeroed.
Depth **helps** on renders (+0.294) and **hurts** on photographs (-0.119).
That is camera-free, needs no annotation, and says the depth head is not
merely weak off-domain but actively wrong there.

## Correction: the photo depth benchmark CAN be built, and head 1 fails it

The section above concluded that AudioEar3D could not yield a real-photo
depth benchmark. **That was wrong, and the error was mine.** It is built, it
works, and it produces the first measurement of depth on real photographs in
this project.

### The mistake

The identity test asks whether the 2D data can tell you WHICH ear it is. A
stranger's ear fit a photograph as well as the subject's own, so I concluded
the label would be a generic ear.

That conflates two different things. **We never needed the 2D to identify the
ear -- the dataset already tells us the pairing.** What is needed is an
accurate POSE; project this ear's own 3D under it and the depth that comes
out is this ear's, whether or not a stranger would also have fitted. A big
residual is grounds for suspecting the pose, not for concluding it is wrong,
and I stopped at the suspicion instead of measuring.

### Measuring it properly

Two measurements, neither of which needs the identity test:

| | |
|---|---|
| label error on renders, driven to the photos' observed fit residual | **0.0242** |
| two independent 2D sources, disagreement of the labels they produce | **0.0204** |

They corroborate. The label is not exact and it is not junk: it carries about
0.024 of error against a 0.0443 signal. That is hopeless for grading a model
already accurate to 0.014, and perfectly serviceable for one in the
0.05-0.09 range -- which is exactly where a render-trained head lands on
photographs. The benchmark was rejected for being imprecise without ever
asking imprecise *relative to what*.

`scripts/fit_photo_depth.py --build` writes it: 112 photographs, 2D taken as
the average of the hand annotation and the shipped model's prediction, which
is genuinely independent (trained on data/manual, never shown an AudioEar3D
photo). Averaging removes the part of their error that is independent; the
part they share it cannot touch, so the floor quoted is the total from the
render measurement, not the flattering half.

**z_skill on this set is capped near 0.45.** A score at or above that would
mean the label noise is being fitted rather than the ear, and anyone quoting
a number from this set should quote the cap beside it.

### What it says

| model | z MAE | z skill | beats flat z |
|---|---|---|---|
| predict z = 0 | 0.0443 | 0.000 | — |
| **head 1, render-trained** | **0.0526** | **−0.188** | 43% |

**Head 1's depth is worse than useless on real photographs.** The same
checkpoint scores +0.80 on renders. Correcting for the label floor, its true
error is sqrt(0.0526^2 - 0.0242^2) = 0.0467 and its true skill about -0.05 --
so the honest statement is that the depth head contributes nothing on a
photograph, rather than that it actively harms.

This independently confirms the one camera-free comparison that survived
earlier, where zeroing the predicted z IMPROVED agreement with the scan on
photographs (-0.119) while helping on renders (+0.294). Two unrelated
methods, built on different assumptions, agree.

**What this benchmark is for.** It is now the target metric for Route C and
anything after it. The 2D side of the domain gap was already measurable
against data/manual; the depth side was not, and that was the gap that
mattered, since depth is the whole reason head 1 exists.

## Route C: making renders look like photographs fixes x, y and not depth

The render-to-photo gap is a domain gap, so the cheapest attack is to close
it at the source. `scripts/render_vs_photo.py` measures which image
statistics separate the two populations and a logistic probe on seven of them
says how separable they are overall.

**Measurement corrected two things I had backwards before any code was
written.** Renders are SIX TIMES LESS sharp than the photographs (laplacian
variance 28.9 against 191.2), not more -- real ears come with hair, pores and
scene edges while a render is smooth skin with Nyquist-limited vertex colour.
And there is no backdrop to replace: background covers 5.9% of the average
crop and 0% of most, so "background sd" is skin being too evenly lit, not a
grey backdrop. Compositing scenery behind the ear, which was the plan, would
have fixed nothing. Everything needed turned out to be photometric, so it is
augmentation (`data/photo_aug.py`) and **no re-render was required**.

| | probe |
|---|---|
| render as-is | 0.934 |
| one strength knob at 1.0 | 0.855 |
| separate knobs, fitted to the photo statistics | 0.753 |
| + 1/f texture spectrum | **0.740** |

One knob could not work: the setting matching photo noise and sharpness left
contrast at 26 against 48, and the setting fixing contrast overshot laplacian
variance **11x**. Skin needs far more shading than grain. White-noise texture
matched sharpness while putting 24% too much energy high, because real skin
and real optics roll off as 1/f.

### The result, and why the depth benchmark earned its keep

| | photo x,y NME | photo z skill | render z skill |
|---|---|---|---|
| no aug (control) | 0.1045 (7.3x floor) | −0.188 | 0.8015 |
| aug, shading 1.0 | **0.0749 (5.2x)** | −0.280 | 0.7984 |
| aug, shading 0.3 | 0.0733 (5.1x) | −0.366 | 0.7964 |
| **aug, shading 0.0** | 0.0768 (5.4x) | **−0.049** | 0.7971 |

x, y on real photographs improves **26-30%** whichever way shading is set,
and in-domain render depth never moves. But full-strength augmentation made
photo DEPTH worse, and only the new benchmark caught it -- without it this
would have been reported as a 28% win.

The suspect was local shading: a random multiplicative brightness field is
precisely the shape-from-shading cue a depth head reads, so randomising it
teaches invariance to the signal depth depends on. Turning it off while
keeping texture, noise, occluders, defocus and JPEG:

| paired bootstrap over the same 112 ears | skill gain | 95% CI | P(better) |
|---|---|---|---|
| shading 0.0 vs no augmentation | **+0.138** | [+0.033, +0.249] | 0.99 |
| shading 0.0 vs shading 1.0 | **+0.229** | [+0.131, +0.323] | 1.00 |

Confirmed. The pairing matters -- per-ear variance dominates the unpaired
intervals, and the 0.3 run scoring below the 1.0 run is inside that noise
rather than a real inversion.

**But read the absolute number.** shading 0.0 sits at −0.049, CI [−0.149,
+0.047], which straddles zero. Route C removed the harm and did not make
depth useful: on a photograph head 1's depth is still indistinguishable from
predicting z = 0. The recipe to keep is augmentation with shading off, worth
26% of x, y error and nothing on depth.

## Regressing depth loses to fitting a model, as MediaPipe already knew

MediaPipe FaceMesh outputs z "with the depth at center of the head being the
origin" and "roughly the same scale as x" -- the same convention this project
uses. The difference is what is done with it: MediaPipe's metric 3D comes
from its Face Transform module fitting a canonical model to the screen-space
landmarks by Procrustes under a perspective camera, **not** from the
network's z. This project ships the network's z as the product.

Tested on the photo benchmark with label and prediction taking DISJOINT 2D
sources, so no pose error is shared:

| method | photo z skill |
|---|---|
| predict z = 0 | 0.000 |
| head 1 network z | **−0.171** |
| fit mean ear to shipped-model 2D, label from human | **+0.217** |
| fit mean ear to human 2D, label from shipped model | **+0.264** |

A mean ear and a pose, with no per-subject 3D and no depth network at all,
beats the depth network by about 0.4 skill on real photographs.

### Where each one's limit is

| | renders | real photos |
|---|---|---|
| predict z = 0 | 0.000 | 0.000 |
| head 1 network z | **0.804** | −0.171 |
| fit mean ear to 2D | 0.589 | +0.217 / +0.264 |
| fit mean ear to GROUND-TRUTH 2D | 0.584 | — |
| fit THIS EAR's true 3D to gt 2D | **0.827** | — |

Two things fall out. **The fit's limit is the mean ear, not the landmarks** --
perfect 2D scores 0.584, no better than head 1's own. And **head 1 nearly
reaches the ceiling in domain**: 0.804 against the 0.827 available from
knowing the subject's actual 3D ear, so it really is reading individual shape
out of pixels. It simply does not transfer.

They fail in opposite directions. head 1 has the capability and no geometric
constraint; the fit has the constraint and no individual shape. The synthesis
is a head that predicts shape coefficients and solves pose geometrically,
which is head 2's existing architecture and MediaPipe's division of labour.
For production on photographs today, the fit wins and needs no depth network.

Cost is not the discriminator: the fit is a 55x3 least squares plus a 2x3 SVD,
**0.053 ms per ear on CPU**, on top of 2D that is already being computed.

### A naming hazard worth fixing

`lm_basis_v6_55_mm.npz` is **not in mm**: its mean ear spans [0.22, 1.00,
0.53] because `build_landmark_basis.py` divides by ear size on purpose and
stores `ear_size` so the scale is recoverable. Nothing is numerically wrong
and the `_mm` means "built from mm inputs". But a file suffixed `_mm` holding
unit-height data is the exact trap the canonicalisation work existed to
remove, and it should be renamed.
