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

### A naming hazard, now closed

`lm_basis_v6_55_mm.npz` was **not in mm**: its mean ear spans [0.22, 1.00,
0.53] because `build_landmark_basis.py` divides by ear size on purpose and
stores `ear_size` so the scale is recoverable. Nothing was numerically wrong
and the `_mm` meant "built from mm inputs" -- which is why it survived. But a
file suffixed `_mm` holding unit-height data is a factor of 65 waiting for
whoever trusts the name, and it is the exact trap the canonicalisation work
existed to remove.

Fixed in four parts, because renaming alone would not stop it recurring:

- the four files are now `lm_basis_v{5,6}_{55,95}_unitheight.npz`. No live
  code or checkpoint referenced them -- all 36 checkpoints carrying a basis
  path point at the already-retired `lm_basis_v4.npz` -- so the rename was
  free. The v6 95-point file was then rebuilt from the recorded command and
  reproduces bit-exactly, so what is on disk is verifiably what the script
  writes.
- `build_landmark_basis.py` **refuses** an `--out` name claiming millimetres.
  Making the wrong name impossible is cheaper than documenting it.
- each file carries a `units` string, so an npz opened on its own answers the
  question.
- `templates/UNITS.md` tabulates every artefact's real units, each one
  measured from the file's own mean geometry rather than assumed.

A second instance of the same hazard turned up while checking:
`ear_pca_v5_canonical.npz` was identical to `ear_pca_v5_all_mm.npz` in every
array and differed only in a note string. Two names for one artefact means a
result gets credited to "the canonical model" with no way to tell which bytes
produced it, so it is retired to `archive_duplicates/`. Moved, not deleted --
a missing path fails loudly where a shadow copy does not.

A stale comment in the builder was also asserting the wrong thing: it said the
source PCA was in "HUTUBS normalised head units, not millimetres". True of v4,
false since the canonical rebuild. The division is still correct, but for a
different reason than the comment gave -- the pose solver already owns scale.

## Route B: the shape head works, does not transfer, and takes my predicted failure mode with it

Route B replaces regressed depth with MediaPipe's split: the network predicts
40 shape coefficients, pose is SOLVED from the 2D landmarks by weak
perspective, and depth is whatever shape and pose imply. The ceiling was
measured before any training (`scripts/shape_fit_ceiling.py`), and it said go:
40 modes reach 0.807 against the 0.817 full-3D oracle, equal to head 1's
0.804, on a concave curve where a quarter of the coefficient magnitude buys
42% of the gain.

Four runs, coefficient supervision swept, one no-augmentation control.

### Supervision is necessary and it costs the landmark metric

| run | render z_skill (end-to-end) | NME | mesh_err / mesh_base |
|---|---|---|---|
| coeff-sup 0 | **0.752** | **0.0160** | **1.384** |
| coeff-sup 1 | 0.576 | 0.0288 | 0.814 |
| coeff-sup 10 | 0.266 | 0.1108 | 0.853 |
| coeff-sup 1, no aug | 0.609 | 0.0293 | 0.814 |

Unsupervised coefficients are not merely uninformative, they are **worse than
predicting the mean ear** (1.384). And that run has the best landmarks and the
best depth of the four. So the head reaches a good fit by using the 40
coefficients as free parameters with no shape meaning, and **no landmark or
depth metric can tell** -- which is exactly what the earlier head 2 attempt
recorded, now with the number attached. Its coefficients fed to the solver
score 0.023 against the mean ear's 0.584.

### In domain the shape prediction is real

Renders, 400 views, fitted to ground-truth 2D, paired bootstrap against the
mean-ear fit at 0.584:

| condition | z_skill | delta | 95% CI | P(better) |
|---|---|---|---|---|
| mean ear + solver | 0.584 | — | — | — |
| sup1: coeffs + solver | **0.696** | **+0.113** | [+0.097, +0.129] | **1.00** |
| sup10: coeffs + solver | 0.676 | +0.092 | [+0.076, +0.109] | 1.00 |
| sup0: coeffs + solver | 0.023 | −0.561 | [−0.654, −0.469] | 0.00 |

sup1 recovers about half the available gain (0.696 of a 0.584-to-0.807 range).
The capability is genuine.

### On photographs it reverses sign

112 photographs. Labels from the hand annotation and the ear's true 3D,
predictions from the shipped model's 2D, so no pose error is shared.

| condition | z_skill | delta vs mean ear | 95% CI | P(better) |
|---|---|---|---|---|
| mean ear + solver | **0.250** | — | — | — |
| sup10: coeffs + solver | 0.150 | −0.100 | [−0.189, −0.015] | 0.01 |
| sup1, no aug: coeffs + solver | 0.144 | −0.106 | [−0.165, −0.047] | 0.00 |
| sup1: coeffs + solver | 0.107 | −0.143 | [−0.233, −0.057] | 0.00 |
| head 1: regressed depth | −0.044 | −0.294 | [−0.398, −0.200] | 0.00 |
| sup0: coeffs + solver | −0.326 | −0.576 | [−0.756, −0.413] | 0.00 |
| sup1: end-to-end, REGRESSED pose | −0.509 | −0.759 | [−0.892, −0.638] | 0.00 |

Every variant loses to the plain mean ear. **Route B fails its pre-registered
bar.**

### My predicted failure mode was wrong, and that is the finding

The argument for Route B was that a domain shift would drive coefficients
toward zero, and zero coefficients ARE the mean ear, so the floor was supposed
to be +0.250. That is not what happens. The predicted coefficient standard
deviation is **0.577 on renders and 0.617 on photographs** -- essentially
unchanged. The magnitude transfers; the DIRECTION does not. The head is
confidently wrong rather than cautiously average, so the geometric constraint
bought a bounded failure (−0.143) instead of a harmless one.

Scaling the predictions by alpha makes the cost explicit, with alpha = 0 being
the mean ear by construction:

| alpha | z_skill | delta vs alpha = 0 | 95% CI | P(better) |
|---|---|---|---|---|
| 0.00 | 0.250 | — | — | — |
| 0.10 | 0.258 | +0.0085 | [−0.0010, +0.0180] | 0.96 |
| 0.25 | **0.262** | +0.0125 | [−0.0111, +0.0364] | 0.85 |
| 0.50 | 0.239 | −0.0109 | [−0.0581, +0.0348] | 0.31 |
| 1.00 | 0.107 | −0.1428 | [−0.2330, −0.0573] | 0.00 |

So there is at most a whisper of transferable shape -- about +0.01 skill, and
neither CI clears zero. The correct policy on a photograph today is alpha = 0,
which is to say: use the mean ear and do not ask the network about shape.

### What did transfer: solve pose, never regress it

This is the half of the MediaPipe thesis that survived, and it is not a small
effect.

| model | solved pose | regressed pose | gain from solving |
|---|---|---|---|
| sup1 | 0.107 | −0.509 | **+0.62** |
| sup1, no aug | 0.144 | −0.717 | **+0.86** |
| sup10 | 0.150 | −0.216 | +0.37 |

A regressed pose has to survive the domain shift; a solved one rides on the 2D
model that already does. In domain the two are a tie for sup1 (0.696 solved
against 0.594 regressed, and sup0 actually prefers regressing), so **nothing
in the render metrics predicts this** -- it only appears on photographs, which
is the argument for keeping the photo benchmark in the loop for every head.

### Standing

The MediaPipe-equivalent pipeline -- shipped 2D, mean ear, solved pose -- is
still the best thing available on a photograph at **+0.250**, beating head 1 by
0.29 and every shape head tried here. Route B's shape prediction is a real
in-domain capability (+0.113, P = 1.00) that does not survive the render-to-
photo gap, which is the same disease as head 1 in a milder and bounded form.
Photo-realism augmentation does not fix it either: the no-aug control is the
BEST of the supervised runs on photographs (0.144 against 0.107), so the
Route C recipe that cut x,y error 26% does nothing for shape transfer and may
mildly hurt it.

The honest read is that per-subject 3D ear shape is not recoverable from a
single in-the-wild photograph with the supervision available -- 588 rendered
ears from two sources. Nothing measured here says it is impossible; it says
that render supervision plus appearance augmentation is not enough, and that
the next thing worth trying is supervision whose APPEARANCE is real, not
synthesised. The coefficient head itself is sound and cheap to retrain once
such a signal exists.

## Freezing the 2D model does not rescue depth: every LEARNED lift fails, the geometric one does not

head 1 is architecturally the shipped 2D model plus a 1x1 conv: same
`EarLandmarkerHeatmap`, same backbone, 348,466 parameters against 353,801,
with `to_z` contributing 5,335 of them (1.51%). What it is NOT is the shipped
WEIGHTS -- `--head z` trains the whole thing from scratch on renders, so its
x,y is a second, render-trained 2D model that scores 0.0768 NME on
photographs against the shipped model's 0.0143.

That suggests an obvious fix: freeze the 2D model that already transfers and
train only the depth on top. Two heads already do exactly this and had never
been scored on the photo benchmark, which did not exist when they were built.
Both are frozen on the real shipped checkpoint, `manual_occ_s42`.

| on the 112-photo benchmark | z_skill | vs mean ear | P(better) |
|---|---|---|---|
| mean ear + solved pose | **+0.250** | -- | -- |
| head 1: image -> x, y, z | −0.044 | −0.294 [−0.398, −0.200] | 0.00 |
| head 3b: frozen 2D + features -> z | −0.536 | −0.786 [−0.943, −0.639] | 0.00 |
| head 3a: x, y -> z, NO image | −0.559 | −0.809 [−0.958, −0.668] | 0.00 |

**Freezing the transferable part makes it WORSE, and head 3a is the clean
experiment that says why.** head 3a consumes nothing but the 2D model's 55
coordinates and their confidences -- no pixels, no decoder features, nothing
that carries a render's appearance. Everything in its input transfers. It
still scores −0.559.

And the comparison is exact rather than approximate. Scoring the mean-ear fit
at head 3a's OWN landmarks gives +0.250, identical to three decimals, with a
paired delta of +0.000 [−0.000, +0.000]. So the two methods receive literally
the same 55 points and differ by **0.81 skill** on how they turn those points
into depth: one learned the mapping from 588 rendered ears, the other solves
it with a camera model and a mean ear.

(That reference row also confirms, incidentally, that
`docs/EarLandmarker_web.onnx` is an exact export of the `manual_occ_s42`
checkpoint -- the two 2D sources produce the same landmarks to the precision
this metric can see.)

So the failure was never the image features. A learned x,y -> z mapping
memorises the geometry of the corpus it was trained on; the distribution of
real-ear landmarks is not that distribution, and the mapping has no physics to
fall back on. A weak-perspective solve has nothing to memorise.

### The same four heads on the head scan

The scan is a RENDER wearing a real texture, so it should and does behave like
the render domain rather than the photo domain:

| | left ear | right ear |
|---|---|---|
| head 1 | **+0.277** | −0.051 |
| head 3b | −0.056 | −0.469 |
| mean ear + solved pose | −0.101 | −0.297 |

head 1 wins both, as it does on the render corpus, and head 3b beats the mean
ear on the left ear and loses on the right. Nothing here contradicts the photo
result; it confirms that this scan sits on the render side of the gap.

### What this settles

Five ways of producing depth have now been scored on real photographs. Every
one that LEARNS the depth mapping loses to predicting z = 0:

    head 1, image -> xyz                     −0.044
    head 2, shape coefficients + solved pose  +0.107 (best of four variants)
    head 2, regressed pose                   −0.509
    head 3a, x,y -> z                        −0.559
    head 3b, frozen 2D + features -> z       −0.536

and the one that does not learn it wins:

    mean ear + weak-perspective solve        +0.250

The common factor is not architecture, not the frozen base, not the image
features and not the depth parameterisation -- head 1's amplitude is correct
in domain to 1.008 and its optimal rescale out of domain is 0.42, so there is
no range pathology to fix either. It is that a learned depth mapping does not
survive the render-to-photo gap in any form tried, while a geometric one needs
no transfer because it was never fitted to a domain.

## A general depth model beats every ear-specific depth head we trained

The route: take x and y from the shipped 2D landmarker (0.0149 NME), read z
off a general monocular depth map at those same pixels, and let the head scan
grade the result. No learned ear depth head anywhere in the path, and nothing
in the path has ever seen an ear label.

The model is MoGe-2 (ViT-L), chosen over a disparity model because it emits a
metric POINT MAP: `points[...,2]` is directly comparable with a renderer's
view-space z, with no inversion and no affine fit to discover first.

Scored on BJ002 through `scan_depth_check.py`, so the locate, the pinna frame
and the corpus framing are identical to every depth head measured here, and
head 1 reproduces its 0.277 on the same dump:

| ear | source | z_skill | corr |
|---|---|---|---|
| left  | shipped 2D + MoGe-2 | **+0.381** | 0.655 |
| left  | head 1 (regressed depth) | +0.277 | 0.641 |
| left  | mean ear + solved pose | −0.101 | 0.237 |
| right | shipped 2D + MoGe-2 | **+0.153** | 0.539 |
| right | head 1 | −0.051 | 0.245 |
| right | mean ear + solved pose | −0.297 | 0.206 |

It wins on both ears, and on the right ear it is the first thing in this
project to score positively at all.

NOTHING IS FITTED TO THE SCAN in those numbers. Corpus z is exactly invariant
to a global scale on the depth map -- the factor cancels in
`(a*z - a*zbar)/(a*zbar)`, verified to 1e-16 -- and a metric point map carries
no unknown shift, so the raw output is usable as it stands. This is the
deployable configuration, not an oracle-assisted one.

### The relief is really there, and the dense test had to be fixed to show it

`moge_lift_check.py` scores the depth map densely against the renderer's own
float32 depth buffer, with the global affine solved on the pixels OUTSIDE the
ear and applied to the ear unchanged -- normalising on the ear would fit the
very relief being measured.

The first version of that test scored the whole landmark BOX and reported
truth's relief as 11.32 mm, against this scan's measured 3.69 mm of pinna
relief. The box also contains scalp and background a full ear-depth behind the
pinna, so the 0.971 correlation it produced was one easy step edge: any model
that knows an ear protrudes scores above 0.95 without resolving anything.
Narrowed to the depth band the on-pinna landmarks occupy, truth reads 3.62 mm
-- which independently reproduces the 3.69 mm measured earlier by a different
route, and is the check that validates the whole chain.

| ear, map | truth relief | MoGe relief | ratio | dense corr |
|---|---|---|---|---|
| left, crop | 3.62 mm | 2.92 mm | 0.807 | +0.886 |
| left, full render | 3.62 mm | 2.83 mm | 0.781 | +0.849 |
| right, crop | 2.93 mm | 2.03 mm | 0.692 | +0.810 |
| right, full render | 2.93 mm | 2.00 mm | 0.681 | +0.705 |

For contrast, HSRD-100's full-body photogrammetry reached 0.49 of this scan's
relief and was rejected for it. A general depth model, with no ear training,
recovers 0.68 to 0.81.

### The crop beats the whole head, which is the opposite of the prediction

A monocular model was expected to want the head for scene context. It does
not: framing the ear large is worth +0.17 z_skill on the left ear (+0.512 vs
+0.339 by the dense scorer) and +0.07 on the right. The model has a fixed
token budget, so context costs resolution, and on something 3 mm deep the
resolution is what matters.

### Two scorers, same conclusion, and the gap is not hidden

`moge_lift_check.py` reads +0.512 / +0.290 (left / right, crop) where
`scan_depth_check.py` reads +0.381 / +0.153 on the same maps. The two differ
in how truth is sampled -- nearest-neighbour on the depth buffer against
back-projection with silhouette snapping -- and in which points each counts
as usable. Both beat head 1 on both ears by a clear margin; the absolute
level is scorer-dependent to about 0.13 and should not be quoted tighter
than that.

### What this does not establish

n = 2 ears, one subject, so there is no interval here and this is not a
benchmark result. The input is a RENDER of the scan graded against that same
scan -- though note the domain shift runs the favourable way for once: MoGe
was trained on real photographs, so a render is its hard case, where for
every head we trained the render was the easy case. The open question is
amplitude, not direction: the relief ratios sit at 0.68-0.81 and the oracle
amplitude row barely improves on the raw one, so MoGe is slightly flattening
the pinna rather than mis-shaping it.

## The wrap's landmarks are ground truth, and they cost two frame bugs to reach

Every 3D number in this project before this was a distance to the nearest point
on the scan, which has no idea *which* point a landmark should be on. The wrap
gives a reference with correspondence: each landmark is a (triangle,
barycentric) address, so point *i* can be compared with point *i*.

**The reference is the ear's own surface.** The wrap fits the cut ear to 0.002
mm median (`corpus/fits2_hutubs.csv`, 0.003% of the ear) and the cut inverts
onto the head's own vertices to 3e-05 mm, so once the frames compose correctly
the landmarks sit **0.058 mm** from the scan. There is no residual to deduct --
a number here is our error, not a budget shared with the reference.

`hutubs pp10_right`, 21 views, all 21 contributing to every landmark:

| estimator | median | mean | p90 | max | beats |
|---|---|---|---|---|---|
| plain mean | **5.98** | 6.54 | 10.95 | 15.03 | 20/21 |
| RANSAC | 6.08 | **6.23** | **9.86** | **14.82** | 19/21 |
| geometric median | 6.22 | 6.67 | 11.41 | 16.87 | 16/21 |
| coordinate median | 6.28 | 6.77 | 11.61 | 17.74 | 14/21 |

Single views run 5.80 to 11.09 mm. **The estimators are a wash within 0.3 mm,
and the plain mean is not beaten on median error.** What robustness buys is the
tail -- RANSAC's p90 is 9.86 against the mean's 10.95 -- not accuracy. The
earlier claim that the robust estimators beat the mean was made on surface
distance and does not survive a reference with correspondence.

By strip the outer helix is worst at 9.06 mm against 5.12 for concha and
tragus, which independently agrees with the inter-view spread already putting
the weak landmarks at 2-5 and 16/17, all outer helix. A best-fit rigid
alignment of reference to estimate still removes 11.9 degrees of rotation and
drops the median to 5.48, so a good part of what is left is systematic
orientation rather than per-point noise -- worth chasing before anything else.

Surface distance and the paired metric rank the 21 views with a correlation of
**+0.349**: weak, so surface distance carries some signal but little. The
qualitative failure is still there -- `y-16_p+0`, flagged by eye as broken, is
the worst of the twenty-one at 11.09 mm while surface distance puts it mid-pack
at 2.33, and `y+24_p+12` is the best by surface distance (0.96) and eighth by
GT. (An earlier version of this section reported +0.012 and called surface
distance uninformative; that was measured against the mis-framed reference
below and is withdrawn.)

**Three traps, all with the same signature:** the landmarks stay *on* the
surface, so a surface-distance check passes while the answer is wrong.

- The barycentrics address the **canonical** mesh. Applied to the `faceform2`
  fit, every point still lands on that fit, reading 3.2 mm and looking like
  ordinary error while scattering the landmarks at random. Only the structure
  gave it away: pairwise-distance correlation 0.04 and 23 consecutive-step
  jumps instead of 3.
- **Each wrap generation pairs with its own cut generation.** `wrapped2_*` sits
  on `ears_hutubs` at 0.002 mm and on `ears4_hutubs` at 0.856 mm; `wrapped3_*`
  is the other way round. Crossing them looks exactly like a 0.9 mm wrap
  residual, which is what I first reported it as.
- **The cut frame is a normalised head.** `cut_ears.py` works on
  `load_head`, which centres the mesh and divides by its largest centred
  coordinate, so the saved matrix maps into *that* frame.
  `corpus/tf_mm_*/matrix` rescales the translation but never adds the centroid
  back -- 5.56 mm adrift on pp10, read as a 1.95 mm reference-to-scan gap, and
  inflated every paired number by about 1.7 mm. `verified_gap: 5.7e-17` in
  those files verifies the matrix inverts the cut, which it does; it says
  nothing about the units it inverts into.

`scripts/gt_landmark_check.py` composes the chain itself from `tf_*` plus the
scan's own centroid and radius, and refuses to return a reference that lands
more than 1 mm from the scan.

## The 11.9 degree rotation is a labelling convention, not a pose error

The best rigid alignment of reference to consensus removed 11.9 degrees, which
looked like a systematic pose problem worth chasing. It is not a rotation at
all.

**Depth is innocent.** Lifting the same 2D landmarks with the renderer's own
depth buffer instead of MoGe's -- exact depth, by construction -- leaves the
rotation at 11.7 degrees (sd 1.9 over 21 views) and the paired error at 6.34 mm
against MoGe's 6.28. Whatever this is, it is in the 2D points or in the
correspondence, not in the depth.

**And it is not a rotation.** Fitted per strip the apparent angles are -11.1,
-9.4, -2.4 and -23.0 degrees. The axis is the pinna normal (mean component
0.88, views agreeing at 0.869), so it lies in the plane of the ear. What
actually happens is that our points sit further along each contour than the
wrap's, and a rigid fit absorbs that as rotation.

Reparameterising each strip by arc length, `t -> alpha*t + beta`:

| strip | as-is | fitted | alpha | beta | pp10_left beta |
|---|---|---|---|---|---|
| outer helix | 9.06 | 4.53 | 0.96 | +0.091 | +0.051 |
| inner helix | 5.38 | 2.93 | 1.00 | +0.046 | +0.055 |
| concha border | 5.12 | 4.59 | 1.15 | -0.074 | -0.028 |
| superior crus | 6.04 | 2.73 | 0.68 | -0.000 | +0.001 |

Both helix strips sit about **0.05 of their arc further along** in our scheme
than in the wrap's hand-marked one -- on the 20-point outer helix that is
exactly one index step, which is also what an integer-shift search finds
(9.06 -> 4.57 at s=+1). Our five superior-crus points span only **0.65-0.75** of the arc the wrap
gives that strip: an endpoint disagreement about where it ends. (Earlier
revisions of this section called 35-49 'concha/tragus' and 50-54 'the lobe';
the scheme's own names are concha_border and superior_crus, and the 55 points
include no lobe at all.)

**Confirmed on a second ear.** pp10_left was rendered separately and lifted
with oracle depth: outer helix beta +0.051 against +0.053 on the right, inner
helix +0.055 against +0.048, superior-crus alpha 0.651 against 0.752. Across 40 views
in total beta's spread between views is 0.03-0.06, against offsets of 0.05, so
this is a property of the two schemes and not something fitted to one ear.

With **one fixed correction applied to every ear** rather than one fitted per
ear, `--reparam` in `gt_landmark_check.py`:

| estimator | as-is | relabelled |
|---|---|---|
| RANSAC | 6.08 | **3.54** |
| coordinate median | 6.28 | 4.02 |
| geometric median | 6.22 | 4.17 |

and the best-rigid rotation falls from 11.9 to 6.8 degrees. So roughly **2.5 mm
of the 6.08 was a labelling convention** and about 3.5 mm is real geometric
error -- which also means our 3D accuracy has been understated by the metric
rather than overstated.

Not acted on: the two schemes could be reconciled by re-marking the template,
but landmark definitions are load-bearing across the shape model and the 2D
corpus, so the offsets are recorded and corrected at measurement time instead.
One loose end, separate from this: pp10_left's outer helix reads 23.4 mm as-is
with beta scattering at sd 0.40 across views, far worse than the right ear's
10.4 and 0.04, which is a real per-view failure rather than a convention.

## The corpus re-propagated onto the new marks, in one frame

A landmark is a (triangle, barycentric) address on the template, so changing
the marks does not require re-solving a single fit. `corpus/canonical` is
already the Faceform Wrap output in one frame at one scale, so the whole
corpus is one evaluation of the new addresses on those meshes --
`propagate_landmarks.py` onto the three canonical directories, no re-wrap.

`templates/lm3d_v7_canonical_mm.npz`, 588 ears x 95 landmarks, from
`mark_mean_v6_spacing`:

- centroid spread across all 588 ears **0.19-0.27 mm sd**, ear size 49.2-79.5
  mm, and the propagator's own units check reports consistent units. One
  frame, one scale, by measurement rather than by assertion.
- **0 of 588 ears** have a contour jump inside a strip, against the 23 that
  the wrong-mesh barycentrics produced on a single ear.
- agrees with the reference computed on the fly through `canon2scan` to
  **9.7e-07 mm**.

Moved from v5, over all 588: median 2.09 mm, outer helix 6.59, superior crus
4.57, inner helix and concha border 0.86-0.87.

**A naming correction that runs through the earlier sections.** Strips 35-49
and 50-54 were written here as "concha/tragus" and "the lobe". The scheme's
own names, in `ear3d.scheme.STRIP_NAMES`, are `concha_border` and
`superior_crus`, and the 55-point scheme contains no lobe point at all --
which the scheme-gaps note already said. The numbers are unaffected; the
anatomy attached to them was wrong, and `gt_landmark_check` now reads the
names from the scheme instead of carrying its own copy.

Still stale and deliberately not rebuilt: `lm_basis_v*_unitheight.npz` is a
PCA of the landmarks and so is downstream of this, and which corpus it is
built from is a licensing choice rather than a mechanical one.

## The landmark basis rebuilt on the CC-BY corpus and the new marks

`build_landmark_basis.py` projects the ear shape PCA onto the marks, so the
CC-BY choice is which PCA it reads: `ear_pca_v6_ccby_mm.npz`, HUTUBS only,
112 ears. Written as `lm_basis_v7_ccby_55_unitheight.npz` and
`..._95_unitheight.npz`, 40 modes each, carrying 96.6% of the retained
variance, mean ear 66.2 mm. Unit-height, as the names say.

Mean per-landmark reconstruction residual at 40 modes, in mm:

| basis | on all 588 | on HUTUBS |
|---|---|---|
| v6 marks, all-588 PCA | 0.453 | 0.443 |
| v7 marks, all-588 PCA | **0.435** | **0.422** |
| v7 marks, ccby-112 PCA | 0.578 | 0.441 |

Two separate effects, which the middle row exists to tell apart. **The
re-marking slightly improves compressibility** -- 0.453 to 0.435 on the same
588 ears -- so nothing was lost by moving the marks. **The CC-BY restriction
costs 0.14 mm when generalising** to SONICOM and AudioEar3D (0.435 to 0.578)
and almost nothing on HUTUBS-like ears (0.422 to 0.441), 112 training ears
against 588 being the whole of that gap. That is the price of the licence,
and it is now a number rather than an assumption.

## Re-marking does not move head 1 at all

Head 1 retrained from scratch on the re-labelled corpus, against an identical
run on the old labels -- same code, same settings, both on CPU, so neither is
being compared with a published number from a different configuration. Each
model scored against its OWN labels, on its own test split:

| labels | nme (xy) | z MAE | flat z | z skill |
|---|---|---|---|---|
| v5 hand marks | 0.01235 | 0.01401 | 0.07043 | 0.801 |
| v7 re-marked | **0.01235** | 0.01391 | 0.06959 | 0.800 |

**Identical to five decimals on xy, and 0.7% on z, which is noise.** The
convention was never head 1's problem: it learns whichever labelling it is
given, to the same precision. The control runs also reproduce the published
head 1 closely (0.01182 nme, 0.808 skill), so the CPU setup is sound.

The cross-evaluation is the part worth keeping, because it sizes the mismatch
rather than assuming it:

| model | labels | nme | z MAE |
|---|---|---|---|
| v5 | v5 | 0.01235 | 0.01401 |
| v5 | v7 | 0.04295 | 0.02039 |
| v7 | v5 | 0.03915 | 0.01919 |
| v7 | v7 | 0.01235 | 0.01391 |

A head trained on one convention scores **3.2-3.5x its own error** against the
other. So the earlier claim stands and is now measured rather than estimated:
any xy comparison across the two conventions is dominated by the convention,
not by accuracy. What this rules out is the other half of that worry -- there
was no accuracy hiding behind the mismatch for head 1 to recover.

It says nothing about the lifter heads, which are the ones that actually ate
mismatched inputs (shipped-convention `pred_xy` against wrap-convention z);
head 1 reads pixels and was always internally consistent. That experiment was
not run.

## The Wrap fits fold, and no single parameter fixes it

Every one of the 588 fits carries edges whose two faces meet past 90 degrees --
a median of 42, and about 290 at full subdivided resolution. They are not
anatomy: only 2 edges fold in 95% of fits while 250 fold in under 10%, so they
move per ear, and the generic template they all start from has none.

Two causes, found separately.

**ProjectGeom.** It runs last, on the dense mesh, snapping every vertex to its
nearest scan point with no smoothness term, so on a thin ridge adjacent
vertices snap to opposite faces. Dropping it takes folds from 42 to 11 across
four ears with scan->fit unchanged. What it buys is real but small: a 5.4x
tighter typical surface fit, 0.087 mm to 0.016 mm, concentrated entirely in
regions that already fit -- the p95 barely moves (1.12 mm to 1.00 mm) and
coverage is identical. It polishes where nothing needed polishing.

**Wrapping itself**, which leaves about 11 folds with projection fully off.

Subdividing before the fit does not help, after two bugs of mine were fixed: a
control point is (triangleIndex, u, v) and subdivision renumbers triangles,
and anchors sampled on subdivision midpoints have no counterpart on the coarse
automatic fit and were silently dropped. Corrected, fitting the subdivided
template still gives 97 folds against 11 and 1.43 mm against 0.09 mm.
Wrapping's freedom comes from its deformation graph, not mesh density: raising
nSubdivisions from 5 to 6 moved the result in the fourth decimal.

**A single parameter does not solve the folding.** On four ears,
globalPoint2PointFittingWeight looked decisive -- 10.8 folds to 0.8 at no
accuracy cost, a 13x reduction. It did not replicate. On twelve fresh ears the
best weight gives 2x, weight 2.0 is WORSE than control, and the per-ear
response is chaotic: pp31_left goes 2, 14, 7, 8, 3, 19 as the weight rises.
The weight is not suppressing a mechanism, it is perturbing a non-convex
optimisation into a different local minimum, and whether that minimum folds is
luck. The 13x figure is withdrawn.

**Per-ear search does solve it.** No single weight clears more than 7 of 12
ears, but 10 of 12 reach zero folds at SOME weight, and accuracy is flat across
the whole ladder (0.122-0.130%), so choosing per ear is free on the other axis.
Walking the ladder and stopping at the first clean fit:

| policy              | mean folds | clean  | scan->fit |
|---------------------|-----------:|-------:|----------:|
| control (shipped)   |       4.50 |   1/12 |   0.1280% |
| best fixed (p2p 64) |       2.25 |   6/12 |   0.1225% |
| adaptive            |       0.33 |  10/12 |   0.1251% |

3.8 fits per ear, about 6.9 h for the corpus. Two ears never reach zero in this
ladder, so a residue remains that either needs a wider search or a de-folding
pass.

### The escalation loop, validated on 20 fresh ears

`scripts/wrap_refit_adaptive.py`. The ladder starts at the SHIPPED DEFAULT and
stops at the first fold-free, untorn fit, so an ear that does not fold today is
refitted with today's parameters and stops there -- integrity for the ears that
are already fine is a property of the ordering, not something measured
afterwards. Later rungs are ordered by clean rate from the twelve-ear sweep
(128, 64, 4, 2, 32).

Screening 60 fresh ears at the default: only 5 are fold-free, the median is 5
folds and the tail reaches 16. The folding is a corpus-wide property, which is
why the mean template inherited it.

Test set of 20, stratified: the 10 worst offenders, the 5 clean, 5 near-clean.

| group                | folds        | clean  |
|----------------------|--------------|--------|
| 10 worst (9-16)      | 11.5 -> 0.8  |  6/10  |
| 5 near-clean (1-2)   |  2.0 -> 0.2  |   4/5  |
| 5 already clean      |  0   -> 0    |   5/5  |
| **all 20**           | **6.25 -> 0.45** | **5/20 -> 15/20** |

Accuracy improves rather than paying for it: scan->fit 0.1355% -> 0.1264%
(-6.7%), p95 flat at +0.1%. Unfolding puts vertices back where they belong, so
surface agreement gets better. The five already-clean ears come out at exactly
0.000000% delta -- the same rung, so the same mesh.

3.0 fits per ear: 1764 fits for the corpus, 5.4 h at one job and 1.3 h at four.

Two caveats. The worst offenders were selected for being bad, so some of their
improvement is regression to the mean -- the claim that best-of-ladder beats
any single setting rests on the twelve-ear sweep (0.33 vs 2.25 folds), not on
this run. And 5 of 20 never reach zero, bottoming at 1-4 folds, so a residue
needs a wider ladder or a de-folding pass.

### The corpus re-fit, and what folding actually is

476 ears re-fitted with the escalation ladder, 3.4 fits each. At the BASE
1188-vertex resolution that base_topology, the PCA and the template consume:

| base resolution     | old faceform2 | new adaptive_v1 |
|---------------------|--------------:|----------------:|
| fold-free           |     0/476 (0%)|    221/476 (46%)|
| median folds        |            40 |               1 |
| mean folds          |         40.45 |            1.27 |
| worst               |            69 |              23 |

31.8x fewer. Note the ladder's own exit condition scored the SUBDIVIDED 4676-
vertex output, where it reports 400/476 (84%) clean; that is the wrong mesh,
and 46% is the honest figure for the one downstream uses.

**A fold is vertex bunching, not a mis-fit.** Measured on the worst ears:
folded vertices sit 0.1-0.5 mm from the scan, the template is 99%+ supported,
face normals still agree with the scan at +0.80, and the scan itself is smooth
there (median dihedral 1.0 deg, nothing past 90). What collapses is the
SPACING: triangle area falls to 0.33-0.53x the mesh median and edge length to
0.64-0.70x. Vertices drift tangentially along a smooth surface, crowd, and the
surplus surface pleats. This is why no distance check ever caught it -- a
folded fit is ON the scan.

**No single parameter fixes it.** Three independent attempts: p2p weight (13x
on four ears, did not replicate), minCos/nNodesToUse (overlap with p2p, add
only accuracy cost), step limits maxDp/minDp (19->4 folds on P0057_left, no
effect on four of six others). All perturb a non-convex solve into a different
local minimum rather than removing the tangential-drift freedom; a real fix
needs an edge-length or as-rigid-as-possible term, which Wrapping does not
expose. Per-ear search is what works.

### scripts/wrap_qa.py

A gate for future data. The checks are INTRINSIC -- computable from the fit
alone -- because the defect is invisible to any distance-to-scan test. It
scores folded edges and near-folds at base resolution, max dihedral, triangle
area and edge-length collapse ratios, torn edges, and with --scan the
discriminating test: a crease in the fit where the scan has no ridge.
--baseline calibrates thresholds to a corpus's own percentiles.

One calibration note: min_area_ratio is a property of the TEMPLATE
(ear_L_template is itself 0.1447), so it must be scored relative to the
template rather than against an absolute, or every fit fails by inheritance.
All 476 fits come out below the template's value (median 0.0575), which is the
bunching signature showing up corpus-wide.

## Back of the ear: delete the control points, keep the geometry

Hair bridges to the head on the posterior surface, so the back is where the
scan is least trustworthy. Two ways to act on that were tested separately on
the same 12 ears (5 worst folders + 7 clean), projection off, 14 anchors.

**Control points off the back — a large win.** Restricting the 14 farthest-point
anchors to front-facing template vertices:

| anchors | base folds | clean | full folds | scan>fit |
|---|---|---|---|---|
| 14 shipped | 5.50 | 1/12 | 6.00 | 0.1326% |
| + inner helix (lm 21,23,25,27,29) | 5.58 | 1/12 | 6.42 | 0.1334% |
| **front-facing only** | **3.33** | **3/12** | **1.50** | 0.1333% |
| front-facing + inner helix | 3.92 | 3/12 | 1.83 | 0.1340% |

Full-resolution folds fall 75% for +0.0007 pp of scan>fit, which is noise.
No geometry is lost -- this only moves where anchors sit.

**Geometry off the back — a large loss.** Deleting the posterior surface from
the template *before* the fit (dot <= -0.6, 90.6% of area kept, same 14
anchors in both arms, verified identical), scored over the same 2145 faces in
both arms so that less mesh cannot flatter the trimmed one:

```
folds>90   full  4.75   trimmed 16.42      (3.5x worse, worse on all 12 ears)
folds>60   full 17.42   trimmed 46.33
scan>fit   full 0.5720%  trimmed 0.5818%
coverage   full 61.0%    trimmed 54.9%
```

The damage is worst at the cut and decays inward, but does not vanish --
removing a collar around the new boundary from the scored region leaves the
trimmed arm 8.0x / 7.2x / 3.4x / 2.1x worse at 0/1/2/3 rings. The full arm is
flat at ~1.25 wherever it is scored. So it propagates into the front; it is
not a free-edge artifact.

The back's *tension* is load-bearing: it is what the deformation graph braces
the front against. The back's *control points* are the part that drags the fit
onto a hair bridge. Drop the points, keep the mesh. If the back is unwanted in
the output, trim it after the fit, not before.

**Dead hypothesis.** The 14 anchors have nothing in landmarks 20-35, and the
manual fixes to the three worst projects added 21 of 23 extra points exactly
there, so the inner helix looked under-constrained. Adding those points does
not help (5.58 vs 5.50) and hurts on top of front-only anchors (3.92 vs 3.33).
What fixed those three projects was the gross anchor misplacement on P0057 and
the hair-bridge exclusion on pp16, not the point count.

Caveat: 12 ears chosen as 5 worst + 7 clean, not a random sample. The
direction is large and consistent, but confirm on fresh ears before adopting
front-only anchors corpus-wide.

## The template artifacts were in the averaged mean

`ear_mean_v2.obj` carries 39 edges past 90 degrees, max 179.2, including 4 in
the intertragic incisure (max 142.8). `ear_L_template.obj` has none.
`ear_lmv_v1` inherited all 39 because it was promoted from v2 -- the landmark
promotion did not create them. This is the artifact complaint that started
the re-fit work, and it was never in the fitting.

**Mean, not median -- measured.** Candidates from the 476 re-fits:

| candidate | >90 | >60 | >40 | max |
|---|---|---|---|---|
| mean | 0 | 0 | 59 | 54.5 |
| geometric median | 0 | 0 | 61 | 55.6 |
| componentwise median | 5 | 37 | 169 | 131.7 |
| ear_mean_v2 | 39 | 100 | 225 | 179.2 |

The componentwise median *creates* folds: adjacent vertices take their values
from different ears, so it manufactures the roughness it is supposed to
resist. It is also not rotation-equivariant. The geometric median is robust
and fold-free but ties the plain mean for more cost. Restricting to the 221
fold-free fits changes nothing, which shows v2's folds were never outlier
ears -- they are inconsistent correspondence, which no robust estimator fixes.

**Two traps on the way, both worth remembering.** `corpus/adaptive_v1` is raw
Wrap output, and HUTUBS fits there are in per-subject normalised units while
SONICOM is in millimetres. An extent filter reads the 112 HUTUBS ears as
"degenerate" at 0.4 mm and silently deletes the entire CC-BY half of the
corpus. Substituting Procrustes-with-scale hides the split but discards the
real size distribution, which is exactly what `canonicalise_corpus.py`
refuses to do. The fix is to canonicalise the re-fit first
(`corpus/canonical_v7`, 476 fits, round-trip 7.8e-14, caliper r = +0.854) and
take a plain mean of that.

**Shipped.** `templates/ear_mean_v3.obj` (0 folds, centroid identical to v2 to
four decimals, extent 61.78 vs 62.00 -- a drop-in in the canonical mm frame)
and `templates/ear_lmv_v2.obj`, built by *replaying* the stored v1 promotion
recipe rather than recomputing it, so connectivity is byte-identical to
`ear_lmv_v1` and face-index masks transfer unchanged. Incisure max 46.7, down
from 142.8.

**Landmark margins and the switch to v3.** The hand-marked exclusion left lm19
0.52 mm from the cut -- the value that got trim 2 rejected, against trim 1's
accepted 2.00 mm. Releasing selected faces within 2.4 mm of any deficient
landmark brings all 55 to >= 2.00 mm (worst lm11 at 2.10) for 29 polygons,
leaving 1936 faces, 72.6% of area, one component, no non-manifold edges.

Every script now defaults to `ear_mean_v3`. This is free: as a Procrustes
target v3 is *identical* to v2 -- 0.000 deg rotation, 0.0000 mm displacement,
centroid difference 1e-08 mm -- because v3 was built in the canonical frame,
so switching `canonicalise_corpus.py` cannot move the canonical frame. The
marks carry over unchanged ((triangle, u, v) on shared topology); they land
1.33 mm from their v2 positions on average, max 3.24 mm at lm15, which is the
new mean's shape, not a re-marking.

Pre-existing and untouched: `validate_marks.py` defaults to
`ear_pca_v4_all.npz`, which does not exist.

## Front-only anchors confirmed out of sample

20 ears drawn at random from the 438 never used in any earlier tuning
(seed 20261011; 17 SONICOM, 3 HUTUBS), control vs `--front-only`, projection
off, default p2p:

```
base folds  4.00 -> 1.10   (-72.5%)
full folds  6.30 -> 1.75   (-72.2%)
clean       2/20 -> 8/20
scan>fit    0.1282% -> 0.1297%   (+0.0016 pp)
per-ear: better 17, worse 0, tied 3    one-sided sign test p < 0.0001
```

Not one ear got worse. The out-of-sample base-fold reduction is larger than
on the 12 tuning ears (-39%), so the original result was not selection
driven. Front-only anchor placement is settled: adopt it.

## The folded template was most of the cloud path's folding

The AudioEar3D cloud fits were written off as intrinsically harder than the
mesh path -- partial support (~0.4), and an adaptive ladder run that returned
0/10 fold-free at a mean of 42.50 folds. That run used `ear_mean_v2` as the
template, which carries 39 folds of its own.

Re-fitting against `ear_mean_v3` (0 folds), same ladder, same clouds: the
first 103 of 113 come back fold-free, with cloud-to-surface and support in
the same range as before (0.08-0.16 and ~0.35-0.39, against 0.11-0.64 and
0.28-0.47). So the folding was inherited from the template, not a property of
clouds, and the "clouds fold badly" conclusion never held.

## A processing confound in the ccby_plus re-test

`regen_mm_artefacts.py` rebuilt on v3 reports mode 1 separating the sources at
Cohen's d = 6.11, up from v4's 2.01, and concludes "still source-driven, keep
the warning". That verdict is not about the data. HUTUBS and SONICOM had been
adaptively re-fit; AudioEar3D was carried over unchanged, so the model was
being asked to separate re-fit from not-re-fit:

```
hutubs  vs sonicom    m1 0.52  m2 0.27  m3 0.85    both adaptively re-fit
hutubs  vs audioear   m1 0.28  m2 5.24             audioear NOT re-fit
sonicom vs audioear   m1 0.22  m2 5.25             audioear NOT re-fit
```

Two sources sharing processing do not separate at all; the one that does not
share it separates from both at d ~ 5.2. The test has to be re-run with
AudioEar3D on equal footing before the "DO NOT USE AS A SHAPE MODEL" warning
can be read as a statement about the data.

**AudioEar3D re-fit, and the confound resolved.** All 113 clouds re-fit
against `ear_mean_v3` with the same adaptive ladder: 113/113 fold-free, mean
0.00 folds, against 0/10 and mean 42.50 on `ear_mean_v2`. Accuracy improved
at the same time -- cloud-to-surface median 0.3100 -> 0.2225, support
0.400 -> 0.438 -- so this is not a slacker fit. `corpus/canonical_v7` now
holds all 589 ears from one pipeline (round trip 8.5e-14; HUTUBS alignment
residual 2.33 -> 1.92 mm now that canonicalisation targets v3).

Re-running the separation with every source on equal footing:

| pair | before (audioear not re-fit) | after |
|---|---|---|
| ccby_plus mode 1 | d = 6.11 | **d = 1.63** |
| hutubs vs audioear | d = 5.24 | d = 1.43 |
| sonicom vs audioear | d = 5.25 | d = 1.16 |
| hutubs vs sonicom | d = 0.85 | d = 1.01 |

About three quarters of the apparent source separation was processing. The
residual 1.63 is below v4's 2.01 and sits in the same range as HUTUBS vs
SONICOM (1.01), two sources nobody disputes are different populations. That
does not clear ccby_plus -- 1.63 on a mode carrying 17.8% of variance is
real -- but the warning rests on much less than it did.

`ear_mean_v3` is left as the mean of the 476 HUTUBS+SONICOM fits. It is a
fixed point: re-canonicalising against itself moves it 0.001 mm. Adding
AudioEar3D would move it 0.155 mm and keep it fold-free, so that is a
judgement call about whether a different capture modality belongs in the
template, not a correctness question.

One trap: `canonicalise_corpus.py --sources` REWRITES manifest.csv with only
the sources named, so running one dataset at a time silently drops the others'
provenance rows. Pass every source in one invocation.

**AudioEar3D folded into the template.** Adding it keeps the mean fold-free,
so it is in: `ear_mean_v3` is now the mean of all 589 canonical fits (0 edges
>90, 0 >60, max 56.1, centroid unchanged, 0.155 mm from the 476-ear version).
`ear_lmv_v2` replayed on it is also fold-free, and the intertragic incisure
improves slightly (max 45.6 from 46.7). Re-canonicalising against the new mean
moves it 0.0001 mm, so it is a fixed point. The hand-marked exclusion still
holds on it unchanged: 1989 faces, 75.4% of area, 55/55 landmarks, no
non-manifold edges, min landmark margin 2.17 mm.

ccby_plus is now 225 ears and still reports d = 1.63 on mode 1.

## Three fixes

**The ladder exited on the wrong resolution.** Wrap returns a subdivided mesh,
but `base_topology`, the canonical corpus, the PCA and the template all read
the first `len(template)` vertices with the template's own faces. The exit
condition scored the subdivided mesh. The two counts disagree and sometimes
anti-correlate, so the ladder could stop on a fit that was still folded in the
only mesh anything downstream uses. Caught in a four-ear run:

```
P0122_left
   default    base   8  full  15
   p2p_128    base   2  full   5     <- now kept as best
   p2p_32     base  10  full   0     <- old rule exited HERE, "fold-free"
```

It would have shipped a 10-fold base mesh as clean. `folds` is now the base
count and drives the exit; `folds_full` keeps the subdivided count for
comparison.

**`canonicalise_corpus.py --sources` destroyed manifest rows.** It rewrote
`manifest.csv` from only the datasets named in the run, dropping the to_mm
factor, mirror and round-trip check for every other dataset -- the only record
that an ear's units were recovered correctly. It now merges: rows for the
datasets rebuilt in this run replace their old ones, the rest carry over.
Verified by re-running one source against a complete manifest: 589 rows in,
589 out, all byte-identical, none lost.

**`validate_marks.py` pointed at a file that does not exist**
(`ear_pca_v4_all.npz`, retired when the corpus went to millimetres). Now
`ear_pca_v5_all_mm.npz`; the script runs to completion.

## Extra anchors do not help, and the lobe gap is missing template

A visual observation -- "part of the lobe was missing" -- caught what fold
count and median scan>fit both missed. Neither metric can see a region the
template never reaches. Per-region coverage over 20 fresh ears:

```
ascending_helix   85.1%        tragus      99.8%
ear_lobe          90.3%        antitragus  99.7%
superior_crus     95.3%        concha      99.4%
```

Adding anchors in the deficient regions, with front-only held fixed and the
minimum detectable change pre-specified from the paired variance:

```
             ear_lobe    ascending_helix   base folds   clean
F_base          --             --             1.10      8/20
+lobe pts    +0.36 [1.39]   +0.61 [0.36]      2.10      7/20
+asc pts     +0.21          -0.23             1.15      7/20
+both        +0.34          +0.33             2.05      7/20
```

The lobe gain is below detection and closes 1/28 of a 10 pp gap. Anchors
placed directly in the ascending helix moved that region -0.23 pp. Folds
roughly double. So extra anchors buy nothing measurable and cost real
quality -- the earlier inner-helix null was right, but for a reason the
earlier test could not have established.

**Why no anchor helps.** Locating the unsupported scan surface relative to
the template's free boundary:

| region | unsupported | distance of that surface to the template edge |
|---|---|---|
| ear_lobe | 12.2% | 0.07% of extent |
| ascending_helix | 17.3% | 0.91% |
| helix | 11.7% | 0.87% |
| descending_helix | 5.9% | 8.77% |
| descending_inner_helix | 4.3% | 9.62% |

Two different faults. The lobe and helix rim are EDGE TRUNCATION: the
template stops short of real ears, and no constraint can pull geometry that
does not exist -- the fix is to extend the template, which is also what the
scheme-extension work needs. The descending helix and descending inner helix
are INTERIOR mis-fit, 8-10% of extent inside the boundary, a genuine fitting
problem and the one front-only slightly worsens (-2.6 and -1.2 pp).

Method note: fold count and median scan>fit are blind to both. Per-region
coverage against the scheme groups is the metric that sees them, and the
minimum detectable change should be fixed from paired variance before the
comparison is run, so that a null result can be distinguished from a test
too weak to see anything.

# v5: auditing the 2D annotations without the model

Everything above judges the landmark model with numbers derived, somewhere
up the chain, from the landmark model or from annotations it seeded. This
section is the independent check: does the image itself agree with the
annotations? Three tools answer it, and they are deliberately not fused --
their value is that they fail for different reasons.

    scripts/qa_silhouette_sam3.py    ear silhouette -> is the outer contour ON the ear
    scripts/qa_ridge_cv.py           traced ridges  -> do landmarks sit on real structure
    scripts/qa_agreement.py          per-image agreement, for triage
    scripts/qa_find_bad_contours.py  ranks rows by misrouted outer points
    scripts/README_qa_2d.md          how to run them and how to read them

## The annotations are good: 81% sit on traced image structure

Per-image agreement over 500 rows with the depth channel, and over all
10,535 rows without it:

    with depth (n=500)   p50 0.48   on structure 81.2%   disagrees 3.0%
    pure CV  (n=10,535)  p50 0.59   on structure 69.0%   disagrees 3.8%

"On structure" means the landmarks sit closer to a traced ridge than a
random point inside the ear, by a factor better than 0.70. The pure-CV run
needs no GPU and takes about 40 s on 24 cores, so it is the default for
triage; the depth channel is for characterising the rim.

**The 3.8% that disagrees is not bad annotation.** Rendered and looked at:
ears occluded by hair, by a finger or by a foreground object, near edge-on
poses, severe motion blur. Agreement tracks the depth channel's own
confidence (rho -0.48) and NOT ear size in pixels (rho +0.01), so it
measures image quality. For a model-seeded corpus the honest reading is
"these rows cannot be verified from pixels", which is also where a seeded
landmark could be a hallucination with nothing in the image to contradict
it.

## Which channel sees which part of the ear

Scored as distance from each landmark to the nearest traced structure,
against random in-hull controls (<1 better, <0.70 traced):

| group | sato base | + CLAHE | DA3 depth |
|---|---|---|---|
| descending_inner_helix | 0.55 | 0.66 | **0.31** |
| inner_helix | 0.79 | 0.76 | **0.39** |
| helix / descending_helix | 0.67 / 0.62 | 0.80 | **0.42 / 0.39** |
| antitragus | 0.74 | **0.54** | 1.48 |
| canal | 0.76 | **0.57** | 1.75 |
| tragus | 0.79 | **0.63** | 2.01 |
| superior_crus | 0.55 | **0.46** | 0.69 |
| concha | 1.05 | 1.03 | 1.04 |

CLAHE before sato is free and strictly better on the interior and worse on
the rim, so both are run. DA3 depth owns the rim and the inner helix and is
the worst channel for everything else; it traces on 67% of images and knows
which -- ear-vs-surround depth separation is 3.45 sd where it works against
1.21 sd where it fails, so gate on that.

**The concha resists every channel** (1.03 of control at best, over five
tried). DA3-SMALL renders the ear as a smooth blob with no internal bowl, so
depth does not see it either. It needs a shape prior, not another filter.

## The bad annotations are real: 5.4%, and the contour is simply too big

Full corpus, all 10,535 rows against the SAM 3 silhouette; 9,843 (93.4%)
returned a usable mask. Counting outer landmarks more than 15% of ear height
off the rim:

    rows with >=2 misrouted points   8.73%  (859 rows)
    rows with >=3 misrouted points   5.40%  (532 rows)   <- the working figure
    rows with >=5 misrouted points   1.57%  (155 rows)
    per-point rate                   1.70%

Evenly spread across the splits, so no split is unusually dirty: train
5.43%, val 5.19%, test 5.40%.

**The annotated contour is systematically too large.** Comparing the
landmark hull's area with the SAM mask's, the implied linear scale of the
annotation against the actual ear is:

    flagged rows   1.18x      clean rows   1.03x

So a flagged annotation is not a different shape, it is the right shape
about 18% too big -- it runs OUTWARD, off the ear.

**And it runs outward at the ENDS of the chain.** The per-landmark rate
across the 20 outer points is a clean U:

    pt  0  4.83%  ascending_helix   ########################
    pt  1  3.03%                    ###############
    pt  3  3.15%                    ###############
    pt  4  2.86%  descending_helix  ##############
    pt  7  0.59%                    ##
    pt 10  0.07%  helix
    pt 12  0.02%  helix                 <- effectively perfect
    pt 14  0.20%  ear_lobe
    pt 17  2.00%                    ##########
    pt 18  3.46%                    #################
    pt 19  6.01%  ear_lobe          ##############################

By group that reads as ascending_helix 3.43%, ear_lobe 2.17%,
descending_helix 1.68%, helix 0.09%.

The middle of the chain is the helix proper, which has sky or background
behind it and is essentially never wrong -- 1 row in 5,000 at point 12. The
two ENDS are where the ear meets hair and the side of the head, and that is
where the contour runs on.

**This was first written up as a SCHEME problem -- that points 0 and 19 have
no well-defined place to terminate -- and that is WRONG.** Rendering twelve
random CLEAN rows cropped to the ear settles it: point 0 sits at the
top-front where the helix meets the temple and point 19 at the bottom-front
where the lobe meets the cheek, in the same place every time, with the chain
an open rim stopping at the ear/face junction on both ends. The convention
exists and 94.6% of the corpus follows it.

What fails is EXECUTION, not definition. Those two spots are where the ear
meets skin and hair at low contrast, so they are the hardest points in the
chain to localise -- and the easiest to let drift outward. That is better
news than a scheme defect, because it means the fix is to snap the drifting
endpoints back to a convention the corpus already demonstrates, rather than
to invent one and re-annotate against it.

At the extreme tail the whole contour goes, not just the ends -- the worst
six rows have a correctly-shaped ear outline sitting well outside the ear.
But those are the tail; the typical flagged row is the ends drifting.

### The endpoint fix was built, and it is withdrawn

The plan was to learn the endpoint convention from the clean majority and
re-derive the drifters against it. It does not work, and the reason
invalidates most of the 5.40% above.

Rendering the SAM masks against the flagged rows shows SAM systematically
UNDER-SEGMENTS the ear, worst at the two ends: on a large share of rows it
drops the LOBE wholesale, cutting across the ear above it. A correctly
placed point 19 then reads as far outside the rim. Signed distances confirm
this is systematic, not random -- 65-92% of ALL twenty points sit outside
the mask, median +0.66% of ear height mid-chain, rising to +3.24% at point 0
and +2.04% at point 19.

So "distance to the rim" cannot separate an annotation that drifted outward
from a mask that stopped short: they are the same measurement. And the
magnitudes fit the mask explanation -- the offending points are a median
17.9-19.4% outside (p90 24-28%), which a dropped lobe produces on its own,
and only 3 rows in 9,843 exceed 40%. There is no population of grossly
misplaced endpoints.

Applied to the ends-only rows, the fix did visible HARM: it dragged correct
lobe points up onto the truncated mask and kinked the chain. A continuity
term barely changed it (0.212 -> 0.208 of ear length), which is what showed
the problem was the target rather than the objective.

**Revised reading.** Of the 532 flagged rows, 151 are ends-only, 296 mixed
and 85 wholesale. Only the WHOLESALE set -- 85 rows, 0.86% of the corpus --
is confirmed bad by eye, with the whole ring sitting off the ear. The
endpoint-flagged remainder is dominated by mask shortfall and cannot be
adjudicated by this method at all. Treat 0.86% as the defensible
bad-annotation figure and 5.40% as an upper bound that includes mask error.

To do better, the silhouette has to agree with the annotation convention
about whether the lobe is part of the ear. That is a segmenter problem, not
an annotation one.

**This defect is invisible to the agreement metric.** On 200 agreement-worst
rows against 200 agreement-best, the misrouted-point rate was the same --
4.84% vs 4.70%. Agreement finds hard IMAGES; this finds wrong LABELS.
Filtering the corpus on agreement would drop the wrong rows entirely.
Contour jaggedness does not find it either: a ring routed along the wrong
curve is perfectly smooth.

## Which segmenter, measured

Same 40 rows, same gates:

| model | gates dropped | interior/outer | what the renders show |
|---|---|---|---|
| SAM 2.1 | 30.8% | 3.05x | noisy, fragments, takes hair |
| **SAM 3** | **12.5%** | **3.52x** | **tight, accurate rim** |
| U-Net (local) | 5.0% | 3.62x | blobby, misses the rim |
| Florence-2 | 60.0% | 2.30x | empty or degenerate |
| jorge-jrzz YOLOv8 | 20.0% | 2.17x | masks cropped to the box, hard rectangular edges |

Over the full 10,535 rows SAM 3 is dropped by the gates on only 6.6%,
against SAM 2.1's 32.9% on a 1,500-row subset, and its masks come out
TIGHTER than the landmark hull on essentially every row, so a flag means
what it should.

**Licences, since this corpus has to stay commercially clean.** SAM 2.1 is
Apache 2.0 for code and weights. SAM 3 is Meta's custom SAM License with
gated weights -- its text carries NO Llama-style restriction on using
outputs to train or distil, the grant is royalty-free with no commercial
bar, and the live restrictions are acceptable-use and no-reverse-engineering.
Florence-2 is MIT. The local U-Net is torchvision (BSD) on the CC BY 4.0
Roboflow set, trained locally. jorge-jrzz has NO LICENSE file at all, which
is the absence of a grant rather than a restrictive licence, plus Ultralytics
AGPL-3.0 -- though AGPL triggers on distribution or network service, so
internal use does not trip it. Its one useful contribution is a pointer: the
dataset bundled in it (mutyi/ear-segmentation v3, 434 images) is MIT.

## Four traps, each of which produced a confident wrong answer

These cost the most time and are the most likely to be repeated.

**1. A dark blob scores like a traced line.** A "shadow" channel -- blurred
negative intensity above the 85th percentile -- appeared to recover the
ascending helix from 0.62 to 0.34 and three groups with it. It had found the
HAIR, which is dark and borders exactly those groups: 83-95% of its hits lay
outside the landmark hull.

**2. Clipping it to the hull does not fix that**, because the outer groups
DEFINE the hull and sit on its edge -- measured at 0.04 of the control
distance. Any broad region touching that edge scores ~0 for them whatever it
contains. qa_ridge_cv.py now prints that edge distance beside every ratio.

**3. A mask that takes the whole crop passes a covers-the-hull gate
perfectly.** That is how a batch of "twenty landmarks off the rim" turned
out to be SAM segmenting the image border. Three gates are needed, not one:
coverage (under-segmentation), bloat vs hull area (over-segmentation), and
largest-component share (confetti). With all three, SAM 2.1 fails on 33% of
images, not the 7% first measured. The headline bad-annotation rate moved
from 12% to 2.4% to 4.9% purely as a function of this gate -- always quote
it with the gate.

**4. Gate pass rate measures "produced a plausible blob", not "found the
rim".** The locally-trained U-Net wins every gate statistic -- 5.0% dropped,
best interior/outer separation of any model -- and loses badly on the
renders, swallowing an eye and a slab of background on one row. Do not pick
a segmenter on these numbers without drawing the masks.

The common thread: every one of the four was caught by RENDERING and
looking, and none of them by any statistic. Several were cases where every
number agreed with every other number and all of them were wrong. The
converse also holds and is worth stating -- when the renders of
exposure-corrected dark crops made the helix ring look plainly inflated onto
the background, the measurement said the opposite and the measurement was
right. A render answers "is this channel tracing hair or anatomy", which is
a blob-vs-line question statistics cannot see. It does not answer "is this
landmark a few pixels out". Pick the tool to the size of the claim.

## Dropping rows: measured before training anything

| filter | train rows | % |
|---|---|---|
| agreement > 1.0 (hard images) | 321 | 3.81% |
| contour jaggedness, top 5% | 435 | 5.16% |
| BOTH of the above | 39 | 0.46% |
| **misrouted contour, SAM 3** | **428** | **5.43%** |

The first three are the wrong criterion: agreement selects hard images whose
labels are fine, and jaggedness cannot see a smooth ring on the wrong curve.
Their intersection, 39 rows, is the only set those two agree is mislabelled,
and 0.46% cannot move a model.

The SAM 3 contour criterion is the right one and is ten times larger. If the
drop experiment is run it needs THREE arms -- full, drop-worst-N, and
drop-RANDOM-N of the same size -- because without the random arm "dropped
bad data" is confounded with "dropped data", and val/test must stay
unfiltered or the improvement is manufactured.

## Infrastructure notes worth keeping

**Letterbox padding.** 45.6% of data/manual images carry >5% constant-valued
padding and 26.8% carry >15%. A flat pad meets the photo at a perfectly
straight, perfectly sharp edge -- the strongest ridge in the frame -- so the
filters traced the LETTERBOX and counted it as structure.
`qa_common.padding_mask` removes it. Correcting it moved the corpus numbers
almost not at all (median 0.47 -> 0.48) because the pad lies outside the
landmark hull; it is kept for being correct, not for this result.

**DA3 in .venv_da3** lacks libnvrtc-builtins.so.13.0, so the second
inference in a process dies with an nvrtc JIT error. Nothing needs
installing: `exrcise_trainer/.venv_cu130/.../nvidia/cu13/lib` already carries
that exact library, so putting it on LD_LIBRARY_PATH runs a whole batch in
one process instead of one image per process.

**Florence-2 loading**, if anyone retries it: use
`florence-community/Florence-2-large`, NOT `microsoft/Florence-2-large`
(whose remote-code layout has no image_token and raises in the native
processor), and the class is `Florence2ForConditionalGeneration`, not
`AutoModelForCausalLM`.


## Can the lobe shortfall be prompted away? No, and it is not SAM's fault

The v5 audit ended by retracting the endpoint fix: SAM 3 under-segments the
lobe, so "distance to the mask rim" cannot tell a drifting annotation from
a short mask, and the bad-contour rate was left as a range, 0.86% confirmed
by eye up to 5.40% by the gate. Four things were tried to close it. Only
the last one worked, and not in the way expected.

**1. Name the lobe in the prompt. No effect.** Over 156 rows, scoring the
signed distance from each annotated point to the mask edge (% of ear
height, positive = outside), and watching point 19 (lobe end) minus point
12 (mid-helix, where the mask is already good):

| prompt | n usable | pt0 | pt12 | pt19 | 19-12 |
|---|---|---|---|---|---|
| `ear` | 156 | 3.63 | 0.84 | 2.52 | **1.68** |
| `ear including the earlobe` | 156 | 4.25 | 0.87 | 2.73 | 1.86 |
| `whole ear with earlobe` | 156 | 3.53 | 0.77 | 2.35 | 1.58 |
| `human outer ear and earlobe` | 156 | 3.92 | 0.83 | 2.48 | 1.65 |
| `auricle` | **0** | | | | |
| `pinna` | **0** | | | | |

The spread brackets the control, so it is noise -- confirmed by re-running
`ear` in a separate process, which reproduced 3.63/0.84/2.52/1.68 to the
last digit. The anatomical terms are not in SAM 3's concept vocabulary at
all: zero masks on every row.

**2. Instruct it instead. Much worse.** SAM 3 matches a CONCEPT, so an
imperative is embedded as a bag of nouns and it goes looking for them:

| prompt | n usable | 19-12 |
|---|---|---|
| `extrapolate helix/lobe shape if occluded` | **0 / 156** | |
| `ear, extrapolate helix and lobe shape if occluded` | 26 / 156 | 3.78 |
| `helix and earlobe` | 2 / 156 | -- |
| `complete ear outline including hidden helix and lobe` | 78 / 156 | 1.83 |

Longer phrasings dilute the concept and lose most of the masks.

**3. Give it a VLM box.** SAM 3 accepts boxes, so the missing prior was
fed in from a VLM that had never seen the landmarks (`qa_ear_box_vlm.py`,
66 rows, then `qa_sam3_boxed.py`):

| arm | n | pt0 | pt12 | pt19 | 19-12 |
|---|---|---|---|---|---|
| text only | 66 | 3.43 | 0.89 | 2.40 | **1.51** |
| box only | 65 | 3.57 | 0.86 | 2.55 | 1.68 |
| text + box | 65 | 3.77 | 0.86 | 2.55 | 1.68 |

The box is not ignored -- rendering the arms side by side shows the masks
differ in their holes and along the rim -- it just does not move the
boundary. **Where SAM 3 stops is a property of its notion of "ear", not of
where it was told to look.** No prompt and no box will fix it.

**4. The VLM box itself is the useful artefact.** Scored against the
annotation's own bounding box, with no mask anywhere in the comparison:

| source | median IoU | IoU>0.7 | annotation past the box bottom |
|---|---|---|---|
| Gemini 3.8 Flash | **0.909** | 92% | -0.2% median, 23% of rows |
| qwen2.5vl:7b (local) | 0.680 | 41% | +7.4% median, 80% of rows |

Gemini agrees with the annotated extent to within 0.3% of ear height on
every edge, and its box contains the annotated lobe point in 59 of 66
rows. Qwen boxes the CARTILAGE -- it cuts the lobe in 45 of 66 and runs
7% short on the right as well, the same failure as SAM 3 from an
independent model. So a lobe-aware prior does exist, and it is strong
enough to serve as the mask-free extent check that the endpoint work
needed. That is the route to narrowing 0.86%-5.40%, and it costs one API
call per row -- which means UPLOADING THE CROPS, so it is the user's call,
not a default.

**Three traps worth keeping:**

- *Never put a usable value in a format example.* The first box prompt
  carried `{"x0":0.0,...,"x1":1.0}` as its format illustration and
  qwen2.5vl echoed it on all 66 rows. Every box came back as the whole
  frame, and the summary statistic (median width 1.000) was the only thing
  that gave it away.
- *Ask each VLM in its own documented convention.* qwen2.5vl answers in a
  0-1000 frame even when asked for pixels of a 448px image. Handed qwen's
  x-first `bbox_2d` key, Gemini answered y-first on some rows and x-first
  on others; switching to its documented `box_2d` / `[ymin,xmin,ymax,xmax]`
  made it consistent. No parser can undo a per-row axis flip.
- *A box prompt arrives as float32 against a half-precision model* and
  the box-prompt Linear refuses the mix, even though the image path is
  cast for you.

**SAM 3 only ever saw the ear crop.** Every number above and in v5 comes
from `data/manual/images/*.jpg`, a tight square crop; the "originals" in
`new_manual_data` are square crops too (127-498 px, zero non-square in 40
sampled), and `data/raw` holds only a `sources.json`. There is no
full-face view of this corpus anywhere on disk, so "would context help?"
cannot be answered directly. What can be tested is the variation that does
exist -- the ear fills 65% to 90% of the frame across the corpus -- and it
predicts nothing:

| crop | n | pt0 | pt12 | pt19 | 19-12 |
|---|---|---|---|---|---|
| loosest (most context) | 2461 | 3.24 | 0.65 | 1.98 | 1.33 |
| loose | 2461 | 3.22 | 0.66 | 2.10 | 1.44 |
| tight | 2461 | 3.23 | 0.66 | 2.00 | 1.34 |
| tightest | 2461 | 3.29 | 0.67 | 2.12 | 1.46 |

Spearman(tightness, pt19 excess) = **+0.015**. Crop RESOLUTION matters a
little (-0.187: bigger crops, slightly tighter masks), which is a
mask-precision effect, not a context one. This rules out mild context
variation only; it does not rule out that a genuine full-face view would
behave differently, and that cannot be tested without re-sourcing the
images.

**And the crops cannot be traced back.** There is no recorded
correspondence between an ear crop and a source image, checked three ways:
the manifest's `source` column is uniformly `manual` for all 10,535 rows
and its `original_path` points only at the square crop; each
`*_annotation.txt` is exactly 55 coordinate lines with no header or
provenance; and `data/raw/sources.json` shows the upstream sets were
themselves distributed as crops ("605 ear crops", "6312 ear crops",
"pre-cropped 224x224"). Only AudioEar2D names a parent corpus -- "2000
synthetic 299x299 ear crops from FFHQ" -- without recording which image or
where in it, and the paths in that file are Windows OneDrive paths that do
not exist on this machine. Recovering full-face views would mean
re-sourcing FFHQ and image-matching 2,000 crops back to their parents,
which covers 19% of the corpus and raises its own licence question. Treat
"SAM 3 sees only a crop" as a fixed property of this corpus.


## Costing the Gemini check, and what shape it can actually return

**Padding does not cure the cutoff either.** If the lobe were being lost
because the ear runs into the crop border, padding would fix it. Over the
same 156 rows, paired against the unpadded run:

| padding | n | pt0 | pt12 | pt19 | 19-12 | lobe |
|---|---|---|---|---|---|---|
| none | 156 | 3.63 | 0.84 | 2.52 | **1.68** | 1.10 |
| grey 25% | 156 | 3.58 | 0.73 | 2.48 | 1.74 | 0.91 |
| replicate 25% | 155 | 3.21 | 0.72 | 2.35 | 1.63 | 0.90 |
| reflect 25% | 155 | 3.94 | 0.77 | 2.59 | 1.82 | 1.12 |
| grey 50% | 156 | 3.52 | 0.67 | 2.23 | **1.56** | **0.79** |
| reflect 50% | 153 | 3.57 | 0.79 | 2.63 | 1.83 | 1.02 |

Grey padding gives a real but UNIFORM improvement -- at 50% the lobe
column moves -0.34 (Wilcoxon p=2.5e-22) -- while the lobe-SPECIFIC effect
is only -0.13, about 8% of the 1.68 gap. Reflect padding does nothing at
all (+0.01, p=0.39): mirroring invents ear texture that is not there,
while grey separates cleanly, which is also the fill `square_roi_crop`
already uses. Worth keeping grey padding for any future mask work; it is
not a fix for the lobe.

**The full-face test could not be run honestly.** Eight public-domain
Wikimedia faces were boxed by Gemini, cropped as `inference.py` crops,
landmarked by the shipped model, and segmented by SAM 3 twice -- full
image and crop -- scored against the same points. The design is sound (the
landmarks are model output, but both arms share them, so landmark bias
cancels and any difference belongs to the mask). The execution is not:
rendering shows two of seven crops contain NO EAR, and on the one clean
row the landmarker runs its chain off across the cheek instead of down the
lobe. The resulting "19-12 = 50" is a landmarker failure on out-of-domain
archive photographs, not a SAM measurement. Reported for the record;
nothing should be concluded from it. A real version needs in-domain
full-face photos, which this corpus cannot supply.

**A box is not a shape, and the conversions mostly fail.** Four routes:
feeding the box back to SAM 3 inherits the very shortfall the box was
meant to escape (measured: no change); cropping and landmarking is
circular, since the landmarker is what is being audited; classical CV
inside the box runs into the concha, which resists every ridge filter
tried in v5. The fourth works. Gemini will emit a **polygon** outline
directly, 8/8 rows, 24-28 points each, and rendering shows it tracks the
rim well -- and in rows 155, 161, 428 and 506 it tracks it visibly BETTER
than the annotation does, which is consistent with those being flagged
rows. It still stops above the lobe where an EARRING covers it (113, 318,
506), which is defensible: the lobe is genuinely not visible there.

Do NOT ask Gemini for `mask`. It answers, but with an undocumented RLE
that is not COCO compressed RLE -- a correct pycocotools-compatible
decoder returns negative run lengths, because the leading `A` sign-extends
to -15 -- and only 4 of 8 rows returned a mask at all. Polygons are
unambiguous, need no decoder, and worked on every row.

**Cost over all 10,535 rows**, measured rather than estimated (1192-1208
input tokens per row; output INCLUDES thinking tokens, which Google bills
at the output rate and which dominate):

| task | model | input | output | total |
|---|---|---|---|---|
| box | 3.8-flash | 12.56M | 3.23M | **$21.55** |
| box | 3.1-flash-lite | 12.56M | 3.23M | $7.99 |
| polygon | 3.8-flash | 12.73M | 12.84M | **$57.70** |
| polygon | 3.1-flash-lite | 12.73M | 12.84M | $22.44 |

At $0.75/$3.75 per 1M (3.8-flash) and $0.25/$1.50 (3.1-flash-lite), the
introductory rate through 2026-12-31; **everything doubles on
2027-01-01**. Thinking is 86% of the box run's output tokens and its
length is wildly unstable on polygons (229, 1103 and 2325 tokens on three
consecutive rows), so capping the thinking budget is the obvious lever --
untested for quality, and worth one 66-row run before committing to a
corpus-wide spend.


## Capping the thinking budget, and the full-image test done properly

**Thinking off is free, and halves the bill.** Gemini 3.x takes
`ThinkingConfig(thinking_budget=0)`. On the same 66 rows:

| run | boxes | out tok/row | median IoU | IoU>0.7 | lobe outside box |
|---|---|---|---|---|---|
| 3.8-flash, thinking on | 66/66 | 307 | 0.909 | 92% | 7/66 |
| 3.8-flash, thinking OFF | 64/66 | 39 | **0.910** | 92% | 5/64 |
| 3.1-flash-lite, thinking off | **15/66** | 30 | -- | -- | -- |

Identical quality -- paired across the 64 shared rows, the largest box edge
moves 1.00% of frame at the median, 2.67% at p90 -- for an eighth of the
output tokens. **$21.55 becomes $10.97** for the corpus. Two rows stop
returning a parseable box, so retry those with thinking on rather than
dropping them. Flash-lite is NOT a cheaper option at any thinking setting:
it failed on 51 of 66 rows. The floor is 3.8-flash with thinking off.

**The full-image test, rerun with the project's own detector.** The first
attempt failed on finding, so this one uses the shipped pipeline end to
end -- BlazeEar two-stage, then EarLandmarker -- and simply does not score
an ear it cannot find. Sources: 7 generated photorealistic profile
portraits plus the photographs from Wikimedia's "Profile portraits"
category, which turns out to be mostly RELIEF SCULPTURES, medals,
silhouettes and engravings; the detector rejecting those is correct
behaviour, not failure. All 7 generated faces were found at confidence
>= 0.96, ears 130-257 px, and render cleanly in both framings.

| arm | pt0 | pt12 | pt19 | 19-12 | lobe |
|---|---|---|---|---|---|
| full image | 0.39 | -1.87 | 1.17 | **2.47** | -0.92 |
| ear crop | 0.39 | -1.32 | 1.37 | **2.50** | -0.53 |

Paired, n=7: full-minus-crop on 19-12 is **+0.16, p=0.58** -- no effect.
**Framing does not cure the lobe shortfall**, which closes the question
left open when this corpus turned out to have no full-face view. The
absolute lobe distance does improve slightly in the full image (-0.74,
p=0.02), exactly the same uniform, non-lobe-specific character as grey
padding.

One observation that needs a caveat rather than a conclusion: on these
clean, well-resolved faces the lobe points sit INSIDE the mask (-0.5 to
-0.9) where the corpus crops put them outside (+1.10), consistent with
Spearman(resolution, pt19 excess) = -0.187. That would suggest the
shortfall is worst on small low-resolution crops. But the landmarks here
are the landmarker's own output, and a model that under-reaches at the
lobe produces negative distances whatever SAM does, so the ABSOLUTE level
is not evidence. Only the full-vs-crop difference is, because the points
cancel. Six of the seven faces are also generated, and a generated ear is
not a photographed ear.

**Detector confidence is not a usability filter.** In the first pass it
fired at 0.92 on a printed political ribbon with two thumbnail portraits
on it, and at 0.95 on a 66px stage-lit ear the landmarker then ballooned
off. Both produced numbers (-44.94 and +28.54) that would have been
reported as findings had the arms not been rendered.


## Seeding the SHAPE, not the landmarks -- and catching the concha

Seeding a VLM with a row's own annotation would destroy the only property
worth having, since a model that anchors on what it is shown blesses a bad
contour. Seeding with the canonical SHAPE avoids that: the model is handed
the median outline of each contour over the 10,003 clean rows, mirrored to
one handedness (the corpus is a 55/45 left-right mix and the outer chain's
signed area separates them exactly), and asked to fit it to the photo. The
output never sees this ear's annotation, so it stays independent -- while
template point j still corresponds to landmark j, which the blind polygon
could not offer. Per-point spread of the template is 0.050 of ear height,
so the shape is well defined. `qa_build_shape_template.py`,
`qa_gemini_refine.py`.

It fits all four strips at once, so the **concha border comes along for
free** -- the structure no classical channel could reach (best of five,
1.03 in v5). Median distance from the annotation, % of ear height, 5 clean
and 5 flagged rows:

| | outer_helix | inner_helix | concha_border | superior_crus |
|---|---|---|---|---|
| thinking OFF, clean | 13.2 | 9.7 | 8.0 | 5.9 |
| thinking OFF, flagged | 14.1 | 17.9 | 9.2 | 11.4 |
| **thinking ON, clean** | **6.1** | **5.1** | **4.9** | 10.9 |
| **thinking ON, flagged** | 12.6 | 7.8 | 9.2 | 15.4 |

**Thinking off does NOT generalise from boxes to this task.** It was
measured free for a box -- identical IoU at an eighth of the tokens -- and
that result does not transfer: here it roughly doubles the fit error and
destroys the separation between clean and flagged rows entirely (13.2 vs
14.1). With thinking on, clean rows fit to ~5-6% and flagged rows sit at
~12%, a 2x separation that the renders confirm: on rows 161, 368 and 385
the annotation sprawls off the ear while the fit stays on it.

Two cautions. Fit precision is ~5% of ear height on clean rows, which is a
screen, not a replacement for an annotation. And the obvious statistic
misleads: median distance-from-annotation mixes fit error with annotation
error, so it showed NO separation at thinking-off while the renders showed
the fit plainly on the ear and the annotation plainly off it. Tail
statistics separate a little better (max 17.2 clean vs 26.0 flagged) but
overlap -- one clean row scores worse than one flagged row.

**Cost, and the design that follows.** The template costs ~1150 input
tokens per call, and thinking on costs 5620 output tokens: **$240.50**
corpus-wide, versus $47.36 with thinking off. That argues for two tiers
rather than one, since the cheap box screen already localises well:

| | |
|---|---|
| box screen, thinking off, all 10,535 rows | $10.97 |
| template fit, thinking on, escalated rows only | $0.023/row |
| **total at the measured 5.4% flag rate** | **$23.94** |
| total if 10% are escalated | $35.01 |

Reserve the expensive fit for rows the cheap screen doubts. Context
caching is the other lever untried: the template prefix is byte-identical
across every row, and Google prices cached input at roughly a twentieth.


## Which seed? Numbers beat pictures, and the GT seed is a flag not a fix

Five ways of seeding the same fit, 5 clean and 5 flagged rows, median
distance from the annotation as % of ear height on CLEAN rows (lower is a
better fit), plus the flagged/clean ratio on the outer helix (above ~2
means flagged rows stand out):

| seed | outer | inner | concha | crus | flag/clean | corpus $ |
|---|---|---|---|---|---|---|
| template numbers, think 0 | 13.2 | 9.7 | 8.0 | 5.9 | 1.06 | 47.36 |
| **template numbers, think 1** | **6.1** | **5.1** | **4.9** | 10.9 | **2.05** | 240.50 |
| two template images, think 1 | 12.7 | 7.5 | 5.2 | 16.9 | 0.64 | 172.91 |
| one template image, think 1 | 9.5 | 10.8 | 6.4 | 15.3 | 0.55 | 211.58 |
| GT seed, think 1 | 0.3 | 0.0 | 0.0 | 0.0 | 7.59 | 406.69 |

**Pictures lose to numbers.** Rendering the template as a labelled image
was meant to save the model the work of building a 55-point shape in its
head. It fits worse on every strip but the concha, and its flagged/clean
ratio drops BELOW 1, meaning it is mildly anti-discriminative -- it fits
flagged rows marginally better than clean ones, which is the wrong
direction for a screen.

**Sending only the correct template costs more, not less.** Handedness is
free and exact from the signed area of the annotated outer chain, so the
model need not decide it. That removes ~1,100 input tokens, and the model
promptly spends more on thinking: $211.58 against $172.91 for sending
both. Worth knowing before optimising prompt length on a thinking model.

**Handedness was never the risk.** Given both templates the model picked
correctly on 10/10, agreeing with the signed-area label every time.

**The GT seed is an echo, except where it disagrees.** Shown the real
annotation and asked to correct it, it returns clean rows essentially
verbatim (0.0-0.8) and moves flagged rows (2.3-22.1), which is why its
ratio is 7.59. That looks like a superb detector and mostly is not one:
agreement is guaranteed by construction, so it cannot vouch for a
contour, only object to one.

The corruption control says how much to trust the objection. Points 8-11,
16-19 and 43-46 were dragged by 10-22% of ear height, in the shape of the
real failures, and the model recovered a median **0.54** of that on clean
rows -- so it IS reading the image, not echoing, but it under-corrects by
half, and on one row it made things worse (-0.45). Drift on the untouched
points stayed at 1-3%, so it leaves correct points alone.

So: GT-seeded DISPLACEMENT is a usable flagging signal, GT-seeded OUTPUT
is not a correction, and neither is independent evidence about an
annotation. The independent configuration remains the canonical-shape
template as numbers, with thinking on. All of this is n=10 and wants a
wider run before anything is decided on it.


## The measurement: the corpus bad-contour rate is ~1.9%, not 5.40%

Five commits of instrument-building left the actual question where
`35a8595` put it: somewhere between 0.86% confirmed by eye and 5.40%
flagged by a screen whose own lobe defect could not be separated from
annotation error. The measurement that closes it cost **$1.11** and had
been affordable since the Gemini box was shown to agree with annotated
extent at IoU 0.909.

**Design.** Gemini boxes (3.8-flash, thinking off) over all 532 flagged
rows AND 532 controls drawn at random from the 9,311 rows SAM 3 screened
and did not flag. 529 and 528 boxes returned. The control arm is the
point: it measures how often a correct annotation pokes outside a Gemini
box anyway, which is the false-positive rate this whole exercise lacked.

Scoring the furthest any outer-chain point lies outside the box, as % of
ear height:

| | median | p75 | p90 | p95 | p99 |
|---|---|---|---|---|---|
| control | 1.1 | 3.1 | 7.4 | 10.7 | 15.6 |
| flagged | 12.0 | 16.1 | 19.3 | 21.5 | 25.7 |

**The threshold was set by eye, not by taste.** Rendering bands showed
8-12% is mostly box error (the annotation follows the ear and the box is
tight), 15%+ is genuinely bad, 12-15% is mixed. Fifteen rows sampled from
the decisive 15-18% band read as 13 bad, 1 earring-occlusion ambiguity
and 1 false positive: **precision 0.87**.

| | |
|---|---|
| extent errors, excursion >15% | 170/529 = 32.1% of flagged |
| x eyeball precision 0.87 | 27.9% genuine |
| shape errors excursion misses | 40/529 = 7.6% |
| union | 35.4% of flagged |
| control false positives | 9/528 = 1.7% |

**Corpus rate ~1.9%, range 1.5-3.5%.** SAM 3's 5.40% **over-counts by
2.8x**: two thirds of what it flagged is mask artefact. The true rate is
also about double the 0.86% confirmed by eye, because that pass counted
only wholesale-wrong contours and missed annotations that merely run past
the ear onto hair.

Note the two criteria overlap less than expected: of the 84 eyeballed
wholesale-bad rows, only 54% exceed 15% excursion, and of the 170 rows
beyond 15% only 45 were in the eyeballed set. Extent error and shape error
are different faults, and neither instrument sees both.

**A second, independent instrument agrees on the scale.** The shipped
landmarker (`qa_model_vs_annotation.py`) disagrees with flagged
annotations 1.7x more than with controls on rows it never trained on
(median 2.9 vs 1.7% of ear height, p90 6.0 vs 3.3; 104 flagged and 97
control held-out rows). At a 10% threshold the excess over control is 18.4
points, putting **0.99%** of the corpus in dispute.

That is a LOWER bound by construction. A model trained on 10,535 rows
containing these errors regresses toward reproducing them: it never
disagreed with any annotation by more than 20% of ear height, and at 30%
not one row in either arm. Its own error floor (control held-out p90 =
3.3%) is the noise. Memorisation inflates agreement only mildly, though --
train 1.3 vs held-out 1.7 on controls -- so the held-out slice is honest
as far as it goes.


## Can classical CV repair SAM 3? Only with a shape prior, and ours is not enough yet

**The diagnostic that decides it.** On 200 clean rows, the gradient along
SAM 3's mask boundary nearest the annotated lobe is **1.18x** the gradient
along the rest of its own boundary, and 2.4x the texture inside the ear
(0.0476 vs 0.0452 vs 0.0198). The mask is already sitting ON an edge. So
every edge-driven repair -- GrabCut, snakes, region growing, matting,
guided-filter matting -- converges to the edge it can see, which is the
one SAM already found. That family cannot help, and it is worth not
spending a day finding out the slow way. Only a shape prior can carry a
contour through locally ambiguous evidence.

Worth noting from the same run: on CLEAN rows the lobe gap is small --
lobe points +0.94% outside versus +0.65% mid-chain. The dramatic lobe
shortfall is concentrated in the harder rows, not uniform.

**Prior art says the same thing, on this exact landmark scheme.** The iBUG
in-the-wild ear work built the 55-point annotation scheme this project
uses and compared Constrained Local Models, Supervised Descent Method and
Active Appearance Models, finding AAMs best. Classical ear SEGMENTATION
has a separate lineage -- Canny plus longest-edge, circular Hough on the
helix (which is roughly three-quarters of a circle), Haar cascades, and
3D shape-model registration to step edges or curvature maps -- all of
which report high numbers on small lab datasets and are documented as
failing on occlusion and poor quality, which is precisely our tail. The
consistent finding across that literature is that the TEXTURE model is
what makes the shape model work.

**What we built.** A point distribution model from the 10,003 clean
shapes (`qa_build_pdm.py`): Procrustes-aligned, one handedness, PCA.
Mode 1 alone is 53.1% of shape variance, **21 modes reach 95%**. The
median template already sent to Gemini is mode 0 of this; the PDM is the
same prior with its variability attached, which is what lets a fit deform
rather than only translate and scale.

Then a Cootes-style ASM (`qa_asm_fit.py`): initialise from SAM 3's mask,
pull the outer chain to the nearest mask-boundary pixel, pull interior
points to the strongest sato ridge along their normal, refit pose and
shape params, clamp b to +-3sd.

| | outer | inner | concha | crus |
|---|---|---|---|---|
| ASM, median | 26.9 | 20.4 | **12.7** | 18.7 |
| ASM, p90 | 68.7 | 39.5 | 26.0 | 38.0 |
| Gemini shape fit, clean rows | **6.1** | **5.1** | 4.9 | 10.9 |

**It is not competitive yet, and the renders say why.** Where
initialisation is right the fit is genuinely good -- 4%, 4%, 7% on the
best rows, tracking the rim closely. Every catastrophic fit is a POSE
failure, not a shape-model failure: the contour lands rotated off the
ear's axis. Sweeping 12 initial rotations instead of 2 improved the tail
(p90 79.3 -> 68.7) and the concha (16.0 -> 12.7) but not the median,
because the data term cannot SCORE pose: a shape collapsed onto the mask
boundary scores well whichever way round it is. That is the appearance
model's job, and this prototype has none -- which is exactly the gap the
iBUG comparison identifies when it finds AAM beating the alternatives.

**Three bugs worth recording**, each of which made the method look worse
than it is, and none of which a statistic revealed:

- the PDM is built in ONE handedness and nothing mirrored it per row, so
  ~44% of ears were fitted with a flipped template;
- the mean shape was rotated by the mask's axis angle without first
  measuring the mean's OWN axis;
- scale came from the square root of the mask covariance eigenvalue,
  which puts a 24px model on a 100px ear. Fixing that alone took the
  outer chain from 46.0% to 23.1%.

**Where this leaves the choice.** Fitting shapes to the disputed rows with
Gemini costs **$12.14** and works today. The ASM is free, offline and
runs at corpus scale, but needs local appearance profiles learned from
the 10,003 shapes -- a real implementation, not a parameter tweak -- to
get from ~23% to anywhere near 6%. Worth doing only if offline
corpus-scale fitting is wanted for its own sake.


## Onboarding economics: screen with the model, escalate the tail

The audit is a one-off; onboarding is a cost on every future ear, so the
question is not "what is the best instrument" but "what is the cheapest
instrument good enough to decide which rows need the expensive one".

**Do not copy iBUG's AAM.** Their 2017 in-the-wild ear work is the right
prior art -- it created this 55-point scheme and found AAM beat CLM and
SDM -- but that was the state of the art for landmark LOCALISATION in
2017. The shipped landmarker is a heatmap CNN at NME 0.0149 and already
supersedes it; the modern successor to AAM is what this project ships.
Recent work (ICCV 2025) refines heatmap decoding rather than replacing the
family. Copying the AAM would be moving backwards.

The literature that matters for this goal is label noise, not landmark
localisation: confident learning, which scores every example against
out-of-fold predictions. No off-the-shelf tool covers keypoints -- the
machinery assumes integer class labels -- so it needs a residual-based
adaptation, which is what follows.

**The screen, measured.** Rank rows by the landmarker's disagreement with
the annotation, escalate the worst to Gemini, judge against the measured
bad set (excursion >15%, precision 0.87), weighting flagged and control
rows back to corpus proportions:

| escalate top | recall, model trained on the row | recall, row held out |
|---|---|---|
| 10% | 42% | 33% |
| 20% | 72% | 62% |
| **30%** | **92%** | 70% |
| 50% | 98% | 76% |

**The contaminated screen wins, which is not a bug.** A row the model
failed to memorise DESPITE training on it is a row inconsistent with
everything else, and that is what a label error is. On a held-out row,
disagreement mixes "bad annotation" with "hard image" and the signal gets
noisier. This is the training-dynamics branch of the noisy-label
literature -- forgetting events, area-under-margin -- rather than the
out-of-fold branch, and it says K-fold is NOT the upgrade it looks like.
Caveat: the held-out curve is n=201 against n=856, so some of the gap is
sample noise.

**The recipe for a new batch**, which follows directly:

1. add the batch to training and fine-tune;
2. score each new row by its training residual;
3. send the worst 30% to Gemini's shape fit;
4. correct or drop what Gemini disputes.

At the measured 92% recall that costs **$6.85 per 1,000 rows** instead of
$22.83 to fit everything, and the 8% missed are the mildest cases. The
screen is free, offline and improves every time the corpus is cleaned,
since each round removes rows that were teaching the model the errors.

**Keep the cost in proportion.** Gemini shape-fitting a row costs $0.023.
If new ears are hand-annotated at anything like a realistic rate, the QA
step is a few percent of the annotation cost and is not the thing to
optimise. It only dominates when the batch is auto-annotated by the
existing model, which is exactly the case where an independent check
matters most.


## The appearance model works; the init was the problem; every independent instrument has a ~5-6% floor

Adding Cootes profile models to the ASM -- a 1-D derivative profile per
landmark along the shape normal, normalised by its absolute sum, scored by
Mahalanobis distance to a mean and inverse covariance learned from 2,500
clean rows (`qa_build_appearance.py`) -- did NOT move the result: 28.4% on
the outer chain against 26.9% shape-only. Profile separability is only
1.41 (median ratio of another landmark's Mahalanobis to its own) and 0.82
at point 7, so the profiles are weakly discriminative.

**But the diagnostic that followed changes the reading entirely.** Started
AT the annotation and allowed to converge, the full ASM drifts only
5.3-5.9% per strip -- which is Gemini's accuracy (4.9-6.1%). The shape
model and the appearance model are therefore good enough; the whole 28%
was initialisation. A mask centroid, scale and principal axis simply do
not land close enough, and no rotation sweep fixes that because the pose
search is over a surface with many near-equal minima.

So the ASM was initialised from the shipped landmarker instead:

| strip | landmarker alone | + ASM refine | Gemini shape fit |
|---|---|---|---|
| outer_helix | **1.5** | 6.8 | 6.1 |
| inner_helix | **1.1** | 6.2 | 5.1 |
| concha_border | **0.9** | 5.9 | 4.9 |
| superior_crus | **1.0** | 5.5 | 10.9 |

**The refinement makes it worse**, because the ASM's ceiling is ~5.8% and
the landmarker is already at ~1%. Refining a good answer with a coarser
instrument drags it down to the coarser instrument's level. (Those rows
are mostly training rows, so 1% is flattered by memorisation; the held-out
figure measured earlier is 1.7%, still three times better.)

**CORRECTION to the first reading of this table.** The column is distance
from the annotation, so for the landmarker it is NOT accuracy -- it is
agreement with its own training target, and no part of it can be read as
"the landmarker is more accurate than Gemini". Comparing the two numbers
side by side was the circularity this whole QA effort exists to avoid. The
honest statements are narrower:

- **5.3-5.9%** is the ASM's own convergence width. Given a perfect start
  it still settles ~5.8% away, so it cannot resolve anything finer than
  that, whatever the annotation says.
- **4.9-6.1%** is Gemini's spread on rows independently believed good. Its
  noise floor, measured without reference to whether we are right.
- **1.0-1.7%** says only that the landmarker memorised its training
  target. It is not evidence about the annotation's correctness.
- The ASM refinement moving 1.5 -> 6.8 therefore does not mean "it made a
  good answer worse". It means the ASM pulls any starting point to its own
  5.8% basin -- including one that happened to sit on the annotation.

What follows is about the INSTRUMENTS, not about who is right:

| instrument | noise floor | independent of our annotations? |
|---|---|---|
| Gemini shape fit | 4.9-6.1% | yes |
| ASM, perfect start | 5.3-5.9% | prior is built from them |
| ASM, mask start | 26.9-28.4% | prior is built from them |
| landmarker residual | n/a -- not an accuracy measure | no |

Both usable independent instruments have a ~5-6% floor. That is the number
that matters, and it has two consequences:

**They can flag, at 15%+.** A disagreement of 15% of ear height is three
times their floor. That is why the $1.11 measurement holds, and note WHY
it holds: it was never "is Gemini right". It was a control arm -- 532
SAM-flagged rows against 532 random rows, 12.0% median excursion against
1.1%. An 11x separation between two arms is valid whatever Gemini's
absolute accuracy is, because the same noisy instrument measured both.
That is the design to copy for any future check.

**They cannot correct, and they cannot see fine error.** An error of 2-3%
of ear height is inside their floor and invisible to them. Replacing an
annotation with a 5-6%-noise estimate would inject more error than the
~1.9% of rows that are actually bad. The GT-seeded experiment said the
same from the other side: Gemini recovered only 54% of a deliberate
corruption.

**Where the landmarker legitimately sits.** Its residual is a RANKER, not
a judge. It never decides whether a row is right; it orders rows so the
independent check is spent on the worst. Its 92% recall was scored against
the Gemini-confirmed bad set -- an external label -- which is why the
measurement is sound and why the contaminated screen beat the held-out
one. A row the model failed to memorise despite training on it is
anomalous, and anomalous is what a label error looks like. That is a
training-dynamics signal, not an accuracy claim.

So the loop has three distinct roles and no circularity:
cheap circular RANKER (landmarker residual) -> independent ADJUDICATOR
with a control arm and a 15% threshold (Gemini) -> human for the right
answer, on the ~1.9% that fail. Nothing in it uses the landmarker to grade
its own training data.

What is still NOT established, and would need a different instrument: the
rate of errors between 2% and 15% of ear height. Every method tried this
session is blind there.


## Web demo refreshed onto the 256 px detector

`docs/` now carries BlazeEar's 2026-10 export: `BlazeEar_web.onnx` at
256x256 (was 128) and the matching `blazeear_inference.js`.
`BlazeFace_web.onnx` was not copied -- it is byte-identical (md5
ba452736) because stage one is still MediaPipe's 128 px front model.

**The two files must move together.** The old JS hardcoded
`numAnchors = 896`, the anchor count at 128 px; the 256 px graph emits
3584. Swapping only the ONNX leaves the loop reading the top fourteen
rows of the anchor grid, returning low-confidence boxes from anchors that
do not cover the subject. It throws nothing and all 467 Python tests pass,
so only running the real graph in a browser shows it.

Measured in-browser (wasm EP, 4 full-face images, old and new on the same
page so warmup is shared):

| arm | detections | confidence | ms/frame |
|---|---|---|---|
| old 128 px | 1 / 1 / 2 / 1 | 0.983 0.968 0.907/0.888 0.978 | 121-201 |
| new 256 px, 1 view | 1 / 1 / 2 / 1 | **0.999 0.998 0.942/0.918** 0.974 | 121-227 |
| new 256 px, 4 views | 1 / 1 / 2 / 1 | 0.999 0.998 0.951/0.935 0.974 | 265-698 |

Same ears, higher confidence, **no latency cost** -- 4x the input pixels
is free here because stage two runs on a face crop, not the frame. A first
pass that showed the new graph as *faster* was warmup: `old` had run
first and absorbed the WASM compile. Reversing the order removed the
effect, which is the only reason the table says "no cost" rather than "a
gain".

**TTA is off in the demo.** The refreshed detector defaults to
`views: 4`, four ear-model passes per face, worth +0.047 mAP@0.5 upstream.
Here it bought +0.01 confidence and not one extra detection for 1.7-3.5x
the latency, which is the wrong trade in a `requestAnimationFrame` loop,
so `EarLandmarkerPipeline` asks for `views: 1` and exposes
`detectorViews` for callers doing still images.

Two things deliberately NOT changed: the demo's 0.70 confidence threshold
(upstream's single-view default is now 0.80, and moving it is a
detection-quality change that wants measuring, not a refresh), and the
Python path, which already reads the 256 px checkpoint through
`DETECTOR_WEIGHTS` and needs no copy step.


## The Gemini BOX is not a valid annotation check -- withdrawing the 1.9% rate

Rendering the disputed rows killed the drop experiment before it finished
training. On rows where an annotated landmark sits >15% of ear height
outside Gemini's whole-ear box, the annotation and the landmarker's own
prediction **agree with each other and both trace the ear**; the box's top
edge slices through the upper helix where it meets hair.

Three measurements, in the order that forced the conclusion:

| | IoU(box, annotation) | IoU(box, model) |
|---|---|---|
| control rows (healthy yardstick) | **0.926** | -- |
| disputed, train | 0.676 | 0.698 |
| disputed, held-out | 0.675 | 0.734 |

The box agrees with NEITHER. If the annotation were simply wrong, the
model -- which never saw the held-out labels -- should have matched the
box at ~0.93. It reaches 0.734.

| model vs ANNOTATION, outer chain | n | median | p90 |
|---|---|---|---|
| control, held-out | 97 | 1.72% | 3.72% |
| disputed, held-out | 36 | **3.09%** | 5.25% |

That is the decisive number. On rows the box calls 17% wrong, an
independent model agrees with the annotation to 3.09%, barely worse than
on control rows. **The box is the outlier, not the annotation.**

Direction confirms a constant bias rather than a disputed-row artefact:
the worst excursion is at the TOP in 37% of disputed rows and 42% of
control rows, at median landmark index 5-6 (upper helix) in both. Gemini
under-covers the top of the ear everywhere; on hair-adjacent ears it does
so badly, and those are exactly the ears SAM 3's mask also falls short on.
Two instruments failing on the same images is not corroboration.

**What this withdraws:**

- the 15%-excursion dispute criterion, and the 133-row drop set built from
  it (all three training arms were stopped mid-run)
- **the ~1.9% bad-contour rate.** The $1.11 measurement's control-arm
  DESIGN was sound and its separation is real -- 12.0% vs 1.1% median
  excursion -- but what separates is "rows where both SAM and Gemini
  under-cover the ear", which tracks hair occlusion, not label error. The
  measurement was right; the name on it was wrong.
- the 92% escalation recall, which was scored against that same
  box-confirmed bad set

**What stands:** the ~5-6% noise floors, flaggers-not-correctors, and the
box as a *detection* aid (control IoU 0.926 is genuinely good). What we no
longer have is a trustworthy bad-label set.

**The caveat that keeps this honest.** The model corroborating the
annotation rules out "this individual row was mislabelled". It does NOT
rule out "the corpus convention places the superior helix slightly too
high", because the model learned that convention from the corpus and would
reproduce it on held-out rows. Distinguishing those needs an instrument
outside the corpus, and the only one with sub-5% resolution so far is a
human looking at the image.

### SAM 3 on the same rows: better than the box, and wrong in the same direction

SAM 3's mask is a far better instrument than Gemini's box -- on several
disputed rows it traces the rim to within the annotation's own line. But
it falls short in the same place:

| | SAM 3 coverage of the annotated ear |
|---|---|
| control rows | 90% |
| disputed rows | 67% |

On the disputed rows the annotation AND the model both extend above the
mask, into hair. Both segmenters stop at the hairline; the 55-point
convention continues the helix through the occlusion, and the landmarker
reproduces that on held-out rows.

**So the two instruments are not failing -- they are measuring a different
quantity.** A segmenter delineates visible ear. The annotation describes
ear anatomy including the occluded part. Where boundaries are clean the
two coincide (90% coverage, box IoU 0.926) and every QA signal looks
healthy; where hair crosses the rim they diverge by construction, and no
prompt, box, padding, threshold or CV post-process can close it. The lobe
cutoff chased earlier in this session was the same effect at the other end
of the same chain.

The practical rule: use segmentation for detection and gross framing,
never as the arbiter of a rim landmark.

### SAM 3 DOES carve around occluders -- and the first metric for it was wrong

A first attempt said it did not: mask-area / convex-hull-area came out at
0.935 on disputed rows and 0.934 on controls, and almost no annotated
point fell inside the hull deficit. Both numbers were artefacts.

- **An ear is not convex.** The intertragic notch alone puts a perfectly
  good mask at ~0.93, so solidity scored both sets identically. The metric
  had no resolution and its agreement meant nothing.
- **The 20 outer points sit ON the rim**, and SAM under-reaches, so they
  are outside the mask already and can never land "inside a bite". The
  bites are in the ear's INTERIOR, where that test had no points at all.

Rendering the masks showed obvious carving the numbers denied. Measuring
the right quantity -- deficit depth from the annotated rim, in ear heights
-- reverses the result:

| | disputed | control |
|---|---|---|
| coverage of annotated ear | 68% | 91% |
| total deficit | 31.8% | 8.7% |
| **of which deeper than 10% of ear height** | **5.6%** | **0.0%** |
| deepest point of the deficit | 19.0% | 8.0% |
| rows with intrusion > 5% of the ear | **57%** | **2%** |

On control rows the deficit is a pure rind: zero deep intrusion. On
disputed rows SAM reaches 19% of ear height inside the rim, on 57% of rows
against 2%.

**Where the intrusion lands is the point.** Rendered, it sits on hair
across the top of the ear and on hands (rows 1743 and 7000); row 3191 is
an unambiguous carve, a blob taken out of the mid-ear where hair crosses,
nowhere near the rim. SAM is excluding the occluder and sacrificing ear
shape to do it.

**So the occlusion-map idea is live, not dead.** An earlier paragraph
retracted it on the strength of the bad solidity metric; that retraction
is itself withdrawn. SAM's deep deficit marks where something non-ear
crosses the annotated ear, it separates 57% from 2%, and it costs nothing
beyond a forward pass we already run. It gives a per-row occlusion score,
a way to scope any segmenter-based check to rim segments that are not
adjacent to an intrusion, and a direct explanation of the disputed set.

It also corroborates the central finding by a third route: the annotation
runs through the occluder, SAM stops at it.

**The methodological lesson, for the third time this session:** the
standing directive is render first and let statistics support. Both wrong
turns here came from trusting a metric whose resolution was never checked
against a picture.

## Scoping SAM 3 to the ears it is entitled to judge: a ~1% instrument

SAM cannot audit this corpus because it delineates visible ear while the
annotation runs through hair. That is a statement about WHICH ROWS it can
be asked about, not about the instrument. On rows with no occluder its
deficit is a pure rind -- 0.0% deep intrusion, 91% coverage, measured --
so there its boundary IS the rim.

`qa_gate.py` selects those rows from IMAGE and INSTRUMENT properties only,
never from the GT, since gating on GT irregularity and then judging the GT
would be circular:

| gate | rows kept |
|---|---|
| has a SAM mask | 9,843 |
| covered >= 0.85 (occlusion-free) | 6,617 |
| bloat in [0.7, 1.6] | 9,249 |
| ear >= 48px | 9,843 |
| in focus (top 90% of scale-free Laplacian variance) | 8,858 |
| **all of them: "proper ears"** | **6,107 (58% of corpus)** |

The occlusion gate is calibrated against the fresh intrusion measurement
on 348 rows: `covered` correlates with deep intrusion at **r = -0.852**,
and `covered >= 0.85` excludes **100%** of rows whose intrusion exceeds 5%
of the ear. So the whole corpus can be gated from the existing run at no
GPU cost.

**On proper ears SAM becomes a ~1% instrument:**

| GT distance to the SAM rim, outer chain | median | p90 | p99 |
|---|---|---|---|
| proper ears | **0.93%** | 1.81% | 3.07% |
| everything else | 1.89% | 4.81% | 8.41% |

A p99 of 3.07% is five times sharper than Gemini's 5-6% floor and in the
landmarker's league. That is the first instrument this session with the
resolution to actually audit an annotation.

**The tail is small.** Of 6,107 proper ears, 68 exceed 3% median, **10
exceed 4%, and none exceed 6%**. An independent second opinion -- GT
irregularity (worst turning angle, spacing coefficient of variation),
which shares no input with SAM -- flags 6 of those 10 against 1 expected
by chance, a **6.1x enrichment**.

**Rendered, 5 or 6 of the 10 are genuine**: the annotation is too big and
runs off the ear into hair (720, 3179, 3198, 6266, and 1315 borderline).
One, row 9698, is not an ear at all -- it appears to be a rock or shell on
gravel carrying an ear-shaped trace, a corpus data defect rather than an
annotation one. Two are SAM failures with a sound annotation (824, 10384,
both fragmented masks in the concha). The rest are dark or blurred.

So precision at this threshold is 50-60%, and the bad-GT rate **on proper
ears** is about 0.1%. The defects share a direction -- the contour is too
big -- which matches the independent earlier finding.

**What this does and does not cover.** It audits 58% of the corpus at ~1%
resolution and says that slice is very clean. It says nothing about the
other 42%, which is excluded precisely because no segmenter can judge it;
that part still needs the landmarker's residual, or a human. The honest
summary is that the corpus is cleaner than any earlier estimate wherever
we can actually measure it.

### Assessing GT correctness on the gated ears: what each region admits

With 6,107 proper ears selected, the question becomes which of the 55
points can actually be checked. Each strip is judged by the instrument
that can see it, and the three sources share no inputs: SAM's silhouette,
sato ridge agreement (`qa_agreement.py`, pure-CV channels, no GPU), and
GT-intrinsic irregularity (kink and spacing CV, which uses no image).

| region | instrument | verdict |
|---|---|---|
| outer chain 0-19 | SAM silhouette | **usable, ~1% resolution** |
| interior 20-34, 50-54 | sato ridge ratio | aggregate only |
| concha border 35-49 | **nothing** | **15 of 55 points unchecked** |

**The outer chain is solved.** Median 0.93% of ear height, p99 3.07%.
Thresholds: 68 rows above 3% (1.11%), 10 above 4% (0.16%), 64 rows with
any point past 15% (1.05%). Rendered, 5-6 of the 10 strictest are genuine,
so precision is 50-60% and the real defect rate is roughly 0.1%.

**The interior is NOT usable per row.** Group medians are 0.18-0.56,
comfortably below the 0.70 "sits on structure" line, which confirms the
annotations are good on average. But the fraction above 1.0 is 12-20% per
group, far too high to be a defect rate. The tool's own docstring says why:
above 1.0 means a bad annotation OR a channel that failed on that image,
and at 15% it is dominated by channel failure. It is an aggregate
instrument, not a detector.

**A combiner that was the wrong shape.** Requiring 2 of 3 signals to agree
gave 29 rows at 2.4x chance, but inspection showed only ~25-30% precision
-- worse than the silhouette alone at a strict threshold. The design error
is that the three signals cover DIFFERENT LANDMARKS: the silhouette sees
points 0-19, the ridge score sees the interior. They cannot corroborate
each other and would co-fire only through a row-level common cause. Voting
across instruments that look at different things is not independence, it
is a category error.

**The honest position.** On proper ears the outer chain can be audited at
~1% and is ~0.1% bad. The interior can be checked only in aggregate. The
concha cannot be checked at all, which leaves 15 of 55 points with no
instrument -- a standing gap, not an oversight.

**Row 9698 is a statue**, not a rock as first reported: enlarged, it is
speckled granite with a carved helix and chiselled hair. Sculpture is a
recurring contaminant in this corpus -- a batch of Wikimedia "profile
portraits" rejected by the detector earlier were relief sculptures. A
statue has correct ear geometry and wrong material statistics, so it is a
data-validity question distinct from annotation correctness, and nothing
in the current screen detects it.

## Correcting the GT from the silhouette: deform the curve, normal component only

Correction was ruled out all session under "flaggers, not correctors", but
that rule came from instruments with a ~5-6% noise floor -- coarser than
the annotation being corrected. Inside the gate the silhouette resolves to
0.93% median, so the rule no longer binds. It was never about the word
"instrument"; it was about the floor.

**What a silhouette can and cannot fix.** It is a CURVE, not a set of
correspondences: it knows where the rim runs, not which of the 20 outer
points belongs where along it. So the error splits into a NORMAL part
(distance to the rim -- visible to it) and a TANGENTIAL part (spacing
along the rim -- invisible to it, and 72% of squared error). Only the
normal part is applied. Snapping each point to its nearest boundary pixel
would silently overwrite the tangential component with whatever the
projection picked, collapsing neighbours at high curvature and reordering
them round the lobe.

`qa_sam_correct.py` carries three guards, one per failure of the naive
version:
- correspondence forced MONOTONE in boundary arc length, so the corrected
  chain cannot cross itself or double back;
- normal offsets smoothed along the chain, since real contour error is
  spatially smooth and an isolated spike is a projection artefact;
- a tolerance, so an already-good row is left alone rather than jittered
  onto the mask's pixel staircase.

**Measured both ways that matter**, on gated rows:

| tol | recover: 12% inflation, before -> after | made worse | no-harm: good rows touched |
|---|---|---|---|
| 2.0% | 5.12% -> **2.68%**, 100% improved | **0%** | **2%** |
| 1.0% | 5.10% -> 2.52%, 100% improved | 0% | 23% |

tol = 2.0% is the operating point: recovery is the same within noise and
the no-harm property is what protects the 99.9% of rows that are already
correct.

**It recovers about half the error, not all of it.** 2.68% residual
against a 0.93% corpus norm. The remainder is the tolerance, the mask's
pixel staircase, and the tangential component it cannot see. So this is a
repair for grossly misplaced contours, not a polish.

**A real limit found by rendering, not by a statistic.** Where SAM's mask
is fragmented (rows 824, 10384, 7288, and the statue 9698), the guards
mostly hold the curve on the GT, but not always -- row 824's lobe follows
the broken mask. The gate therefore needs the largest-component-share
filter that the original three SAM gates had and `qa_gate.py` does not:
`covered`, `bloat` and focus do not catch a mask that is the right size
and in several pieces. That is the next change, and it needs a corpus
re-run because `sam2_silhouette.json` does not record component counts.

**Scope.** This corrects 20 of 55 points. The interior has only an
aggregate instrument and the concha has none, so neither can be corrected
this way.

## Can the interior be fixed from the outer chain? Predict no, propagate partly

Two different operations hide behind the question, and they get different
answers.

**PREDICT -- reconstruct 20-54 from 0-19 through the PDM. No.**

| reconstruction of points 20-54 | median | p90 |
|---|---|---|
| conditional PDM fit, 12 modes, solved from 0-19 | 3.22% | 5.54 |
| mean shape, similarity-fitted to 0-19 | **3.14%** | 5.89 |

Conditioning on the outer chain's SHAPE is no better than fitting the mean
shape's pose and scale -- very slightly worse. The outer chain therefore
carries essentially no information about the interior beyond where and how
big the ear is. Per strip the conditional fit gives inner_helix 2.60%,
superior_crus 2.94%, concha_border 4.36%, against an annotation that sits
0.93% from the SAM rim and a landmarker 1.0-1.7% from the annotation. So a
predicted interior would be about three times worse than the annotation it
replaced. These figures are IN-SAMPLE -- the PDM was built on these rows --
so they are an upper bound, which only strengthens the negative.

That is also a useful fact in its own right: the interior annotation is NOT
redundant with the outer contour. It carries independent information, which
is why it has to be annotated and cannot be synthesised.

**PROPAGATE -- carry the interior along with an outer correction, via a
thin-plate spline driven by the outer chain's own displacement. Partly.**

| | before -> after | better | worse |
|---|---|---|---|
| outer chain, corrected directly | 5.12% -> 2.68% | 100% | **0%** |
| interior, propagated by TPS | 2.69% -> **1.89%** | 70% | **26%** |

A 30% reduction, and it never discards the interior annotation -- it only
moves it the way its neighbourhood moved. But it harms a quarter of rows,
where the direct outer correction harmed none. The asymmetry is
structural: the outer correction is driven by evidence AT the points being
moved, while the interior is moved by extrapolation from a boundary whose
own correction carries noise.

**So the rule is not symmetric with the outer chain.** Apply the
silhouette correction to 0-19 wherever the gate passes, because it is free
of downside. Propagate inward only where the outer correction was large
enough to matter -- on a row whose contour barely moved, the warp is near
identity and the 26% is all that is left. Propagation corpus-wide is not
justified on these numbers.

### Correction: use the same transform, but only when it IS one

The TPS above was the wrong model. A thin-plate spline has roughly as many
degrees of freedom as control points, so it fits the NOISE in the
per-point silhouette corrections and carries that noise inward. Estimating
ONE transform from all 20 points averages the noise down instead.

Interior error under a contour that inflated as a whole:

| warp applied to the interior | before -> after | better | worse |
|---|---|---|---|
| similarity, 4 DOF | 2.69% -> **1.15%** | 100% | **0%** |
| affine, 6 DOF | 2.69% -> **1.08%** | 100% | **0%** |
| thin-plate spline, ~20 DOF | 2.69% -> 1.89% | 70% | 26% |

Halves the error and harms nothing -- far better than the TPS result
reported above, which is superseded.

**But that test is partly rigged, and the honest one reverses it.** The
corruption there is a uniform scaling, which a similarity can represent
exactly. Repeat it with a defect that is NOT a global transform -- one
stretch of the contour wanders off, the interior untouched:

| warp | before -> after | worse |
|---|---|---|
| similarity | 0.00% -> 2.83% | **100%** |
| affine | 0.00% -> 2.69% | **100%** |
| TPS | 0.00% -> 2.29% | **100%** |

Every warp damages a correct interior, because there was nothing wrong
with it and the fix belonged to one stretch of the contour alone.

**So the question is whether the correction IS a transformation**, and
that is measurable before applying it. Fit a similarity to the outer
chain's own displacement and look at the residual:

| defect | similarity-fit residual | p90 |
|---|---|---|
| global (contour inflated) | 2.33% | 2.95% |
| local (one stretch wandered) | 4.23% | 4.68% |

A threshold near 3.5% of ear height separates them. The rule:

1. correct points 0-19 from the silhouette wherever the gate passes --
   always safe, 0% of rows harmed;
2. fit a similarity to that correction and take the residual;
3. below ~3.5%, the defect was global: apply the same similarity to points
   20-54, halving their error with no downside;
4. above it, the defect was local: correct the contour and LEAVE THE
   INTERIOR ALONE, because propagating would damage a correct annotation
   in 100% of rows.

Prefer the similarity to the affine despite its slightly worse number
(1.15 vs 1.08): four parameters cannot introduce shear into an ear, and
the gap is inside the noise.

### The recipe rendered, and the gate that is still missing

Applied to the silhouette's worst 12 proper ears: correct 0-19 by normal
projection, fit a similarity, propagate to 20-34 only if its residual is
below 3.5% of ear height.

**It behaves as designed on 9 of 12.** The outer contour pulls off the hair
onto the rim and the inner helix follows proportionally (rows 720, 3179,
3198, 1315, 3333, 6266, 7288, 3710). Row 1060 is the residual gate working
as intended: at 4.4% it refused to propagate, fixing the contour while
leaving a correct interior alone.

**It is damaged on 3 of 12** -- 824, 9698 (the statue) and 10384 -- where
SAM's mask boundary wanders over texture instead of the ear outline. The
projection follows the wandering boundary, and because the residual still
looked small the inner helix was dragged along too. On 10384 the corrected
contour is WORSE than the original along the right side.

**Two candidate gates for this, both proposed and both measured dead:**

| gate | broken rows | 40 random proper ears | verdict |
|---|---|---|---|
| largest-component share | 0.999, 0.999, 1.000 | median 1.000, min 0.995 | **no separation** |
| boundary raggedness P^2/4piA | 2.50, 1.59, 1.97 | median **1.93**, p90 2.27 | **no separation** |

Component share fails because the stray pieces are specks and the share is
dominated by the one large component -- the earlier claim that this filter
was "the next change" is withdrawn, it would not have helped. Raggedness
fails worse than that: random proper ears are RAGGEDER (1.93) than most of
the flagged rows (1.24-1.72), because the flagged rows tend to be large
high-contrast ears where SAM produces a smooth mask and the disagreement
is a genuine annotation issue. The metric mostly measures ear size and
resolution.

**So the honest state: the correction is sound where the mask is sound,
and there is no validated detector for when the mask is not.** Until there
is, this runs with a human looking at the renders, not unsupervised. The
next thing to TEST -- untested, stated as a hypothesis -- is whether the
landmarker's own prediction can serve as the veto: a correction that moves
the contour away from both the original annotation AND an independent
model is more likely to be the mask's fault than the annotation's.

### Withdrawing the "damaged on 3 of 12" claim

Only row 824 is visibly broken. 9698 and 10384 were called damaged on
inspection and that call does not survive measurement: using the
landmarker -- the one instrument not involved in making the correction --
to measure how far the correction moved the contour from an independent
prediction,

| row | GT -> model | corrected -> model | change |
|---|---|---|---|
| 10384 (called damaged) | 1.90 | 3.71 | +1.81 |
| 1315 (called working) | 2.93 | 5.14 | +2.21 |
| 1060 (called working) | 1.66 | 4.60 | +2.94 |
| 9698 (called damaged) | 4.49 | 4.13 | **-0.36** |

There is no separation between the rows called damaged and those called
working, and 9698 is the ONLY row that moved toward the model. The
classification was not supported and is withdrawn; the damage rate on
these 12 is 1, not 3.

**The arbitration itself is inconclusive, and it is worth being explicit
about why.** The correction moves away from the model on 10 of 12 rows,
median +1.37%. That is what BOTH hypotheses predict: the landmarker was
trained on these annotations, so if the contours really are too big it
learned that convention, and a genuine correction necessarily moves away
from it. This is the same circularity that invalidated the Gemini box
analysis -- a model cannot referee a correction to its own training
target.

**So the correction is validated only against SYNTHETIC corruption**,
where the truth is known and it recovered 100% of rows with none harmed.
On real rows there is currently no referee at all: the silhouette made the
correction, the landmarker shares the convention, and Gemini and the ASM
sit at a 5-6% floor, coarser than the effect being judged. Establishing
whether these corrections are right needs an instrument outside the
corpus -- the 3D wrap, or a human.

## The corrected outer chain, and the clash review that found it damaged

The silhouette correction was applied to all 6,107 gated rows and written
as a separate array, `data/manual/landmarks_sil_corrected.npy`. Nothing in
`data/manual` was modified: `train.py` already takes `--landmarks`, so a
corrected run differs from a baseline run only in points 0-19 on those
rows. Points 20-54 and all 4,428 non-gated rows are asserted identical.

**12.8% of outer points moved; 2,243 rows (37%) were left completely
alone**, which is the corrector doing what it was built to do -- the
2% tolerance means an already-correct row is not jittered.

### The review: inner-helix clearance

The outer correction moves the rim and leaves the inner helix where it
was, so the inner helix is an **untouched witness** to whether the new rim
is plausible. Clearance = signed distance from each inner-helix point
(20-34) to the outer polyline, % of ear height, with a reference
distribution built per index from the original annotations.

**First version of this metric was wrong and the render caught it.** It
closed the outer chain into a polygon. The chain is OPEN -- 0 at the
tragus round the rim to 19 at the lobe -- so closing it draws a chord
across the front of the ear, and point 20 (the crus) sits on that chord.
It scored 14.8% of the ORIGINAL annotations as "inner helix outside the
ear". That was the chord. With an open polyline and a sign taken from the
local inward normal, the original corpus has **3 crossings, not 904**.

| | crossing | too close (z<=-3) | too far (z>=+3) |
|---|---|---|---|
| original annotation | 3 (0.0%) | 18 (0.3%) | 459 (7.5%) |
| after correction | 90 (1.5%) | 616 (10.1%) | 299 (4.9%) |

144 rows had an anomaly the correction FIXED. **669 rows have one the
correction CAUSED** -- 11% of the gated set, against 0% harmed in the
synthetic test. The synthetic test corrupted the annotation and left the
mask perfect; on real rows the mask is the thing that is wrong.

### What the renders show, by category

**crossing (90)** -- correction blow-ups, not GT defects. The corrected
chain detours across the ear or bulges past the lobe. Cause: the
correspondence maps an OPEN 20-point chain onto the CLOSED mask boundary
loop, and the monotone unwrap can traverse the wrong way round, including
across the mask's front edge where the chain has no business being.

**too close (616)** -- the known systematic, inside the gate. The rim at
the top of the helix is under hair; SAM stops at the visible rim; the
correction pulls the rim down onto the inner helix. `covered >= 0.85` does
not exclude this, because the occlusion is a thin band along one arc and
costs little coverage.

**too far (299)** -- mostly near-frontal, foreshortened ears where the
inner helix genuinely sits far from the rim, plus statues and high
contrast art. Partly a pose detector, partly the contamination with no
other detector.

### What this changes

The clash metric is the referee the previous section said did not exist.
It is not out-of-corpus truth, but it is an independent CONSISTENCY
constraint: the inner helix was not touched by the correction, so it can
witness the corrected rim without sharing its evidence. On that witness
the corrector is **not safe unsupervised at the current settings**.

The corrected array stands as built, flagged, and not yet trained on.

## Fixing the corrector: the guards are local, the gate threshold is not

**Raising `covered` is not the fix.** Measured on the 6,107 gated rows,
against the 669 the v1 corrector damaged:

| covered >= | rows kept | damage rate | good rows lost |
|---|---|---|---|
| 0.85 (shipped gate) | 6,107 | 11.0% | 0% |
| 0.93 | 3,112 | 3.6% | 45% |
| 0.97 | 1,019 | 0.5% | 81% |
| 1.00 | **2** | -- | 100% |

It works directionally and costs four good rows per damaged row removed.
**A 100% threshold admits 2 rows out of 6,107** -- SAM's lobe cutoff is
conceptual, so coverage is never complete. Damaged rows sit at covered
0.893 against 0.937 for clean ones: overlapping distributions, which is
what a thin hairline band that costs almost no AREA has to look like.

So both guards are local (`scripts/qa_correct_offline.py`):

**clamp** -- offsets above 10% of ear height are not corrections, they are
teleports; chain length may not change by more than 25%.

**clearance guard** -- no point may move if it drives an inner-helix
clearance below that index's 1st percentile in the original corpus. The
offsets of the outer points the violated inner point leans on are zeroed,
and it re-checks, up to six times.

| | crossing | too close | too far | caused by the correction |
|---|---|---|---|---|
| original annotation | 3 | 18 | 459 | -- |
| v1 | 90 | 616 | 299 | **669** |
| v2 (guarded) | 3 | 15 | 338 | **16** |

Damage 669 -> 16 rows (0.26%), while still moving 8.3% of points on 47%
of rows. Renders confirm it: on every row v1 clipped the top of the helix
down into the ear, v2 holds the original.

**Read the first two columns with suspicion and the third without.** The
guard is built from clearance, so crossing and too-close improve partly by
construction. "Too far" is the column the guard does not control -- it
only vetoes moves that REDUCE clearance -- and it fell 459 -> 338. That is
121 rows improved on a dimension the constraint cannot reach.

`data/manual/landmarks_sil_corrected.npy` is rebuilt from v2.

## Re-marking the whole corpus with the landmarker, and where to put the line

All 10,535 rows re-marked with the shipped checkpoint
(`manual_occ_s42/EarLandmarker_292_nme=0.0149.ckpt`), disagreement scored
as median distance from prediction to annotation over the outer chain,
% of ear height.

**This is a RANKER, not a judge.** The model trained on these rows, so its
residual is partly memorisation and cannot adjudicate them -- that
circularity already invalidated one analysis here. It is used the way the
noisy-label literature uses it: a row the model failed to memorise DESPITE
training on it is inconsistent with everything else in the corpus.

| strip | median | p90 | p99 |
|---|---|---|---|
| outer helix | 1.44 | 3.00 | 5.25 |
| inner helix | 1.18 | 2.36 | 4.57 |
| concha border | 0.96 | 1.87 | 3.63 |
| superior crus | 1.09 | 2.49 | 5.03 |

| split | n | median | p90 | p99 |
|---|---|---|---|---|
| train | 8,427 | 1.39 | 2.78 | 4.70 |
| val | 1,054 | 1.73 | 3.72 | 7.00 |
| test | 1,054 | 1.68 | 3.77 | 6.56 |

### The threshold

Not a round number pulled from the air. **The held-out distribution is the
reference**: a row the model fits WORSE than rows it has never seen is
anomalous whatever the reason, and on a training row it is doubly so,
because the model had every advantage and still could not reproduce it.

| line | value | rows over | % of corpus | of which train |
|---|---|---|---|---|
| held-out p95 | 4.69% | 191 | 1.8% | 85 |
| held-out p99 | 6.91% | 32 | 0.3% | 10 |
| held-out p99.5 | 8.35% | 14 | 0.1% | 3 |

**Recommended: 4.7% to review (191 rows), 6.9% to act on (32 rows).**

Corroborated by an instrument that shares nothing with the landmarker:
against the silhouette residual on the 6,107 gated rows, Spearman 0.433,
and flagging enriches silhouette-suspect rows 4.7x at 2% rising to 33x at
8%. Two instruments with no common evidence agreeing to that degree is
what makes the ranking trustworthy even though neither is a judge.

**Absolute precision against the silhouette is low (5-13%) and that is
expected, not damning**: the silhouette sees only the normal component of
the outer rim, the landmarker sees all 55 points in both components, so
they are detecting different defects. The agreement is in the RANKING, not
the sets.

**A size signature was looked for and is NOT there.** The worst rows look
visibly wrong, and several have a visibly truncated contour, but
annotation area over landmarker area is 0.96 on the worst 32 against 1.00
on controls, and 59% vs 48% smaller -- no separation. The defects are
placement and shape, and they vary. The earlier "contour too big"
signature from the silhouette does not reappear here.

Rows written to `runs/remark/disagree_review.txt` (>3%) and
`runs/remark/disagree_hi.txt` (>held-out p99).

## Rows that are not an ear, and how to remove them

Row 7397 is a blurred shoulder and a red-striped shirt. There is no ear in
it and the 55 points trace nothing. Nothing in the QA stack could say so:
the silhouette assumes an ear and segments the nearest blob, the
landmarker assumes an ear and places 55 points on whatever is there, and
the gate and clash review both measure the SHAPE of an annotation, not
whether its subject exists.

**BlazeEar can, and it is independent.** It was trained on Open Images and
Roboflow ear boxes, not on this corpus, so unlike the landmarker its
opinion is not partly memorisation of the thing being audited. "Is there
an ear here" is also a GLOBAL question, which is where a detector
confidence is strong -- and is not the occlusion question, which is local
and where it would be weak.

Run single-stage on each crop pasted into a 2.2x edge-padded canvas, which
puts the ear back at the face-crop scale the detector was trained on.
Confidence tracks agreement monotonically, which is the sanity check:

| confidence | n | median IoU(box, annotation bbox) |
|---|---|---|
| 0.0-0.3 | 9 | 0.039 |
| 0.3-0.5 | 70 | 0.430 |
| 0.5-0.7 | 224 | 0.593 |
| 0.7-0.9 | 1,681 | 0.679 |
| 0.9-1.0 | 8,551 | 0.772 |

**Row 7397 ranks 15th lowest of 10,535** (confidence 0.324, IoU 0.031).
The instrument finds it.

**But low confidence is NOT a drop rule, and the render is why.** Of the
24 lowest, only two have no ear in them (7397, and row 10, a neck and
hair). The rest are hard and valid: an ear under red light, ears behind
white hair, high-contrast black and white, a heavily degraded frame. My
first read of those called several of them animal ears; zooming in showed
every one is human. Dropping on a confidence threshold would delete
exactly the rows the model needs, which is the same mistake already
measured here -- filtering low agreement removed hard IMAGES, not bad
LABELS.

So the detector proposes and a human disposes. 108 rows (confidence < 0.5,
or < 0.7 with IoU < 0.3) in `runs/earness/review_queue.txt`.

**The removal mechanism** (`scripts/qa_make_clean_splits.py`): rewrite the
split CSVs without the dropped rows. NOT deleting images and NOT editing
landmarks.npy -- every QA artefact here indexes rows by position in the
manifest, so removing a row would silently shift all of them, and the
images are the only copy. The corpus stays complete, training never sees
the row, reverting is pointing `--train-csv` back. The drop list lives at
`qa_lists/drop_rows.txt`, tracked, one row per line with the reason.

Currently 2 rows. The non-ear rate in this corpus is on the order of a
handful, not hundreds.

### The review came back: 84 dropped, 24 kept

All 108 flagged rows were reviewed by hand in the Non-Ear Triage artifact
and the verdicts were read back out of the artifact's database, never
retyped. Splits: train 8,427 -> 8,355, val 1,054 -> 1,049,
test 1,054 -> 1,047. 0.8% of the corpus.

**The user dropped 84 where I had argued for about 2.** My read of the
lowest-confidence renders was that only two frames contained no ear and
the rest were hard-but-valid -- ears under red light, behind white hair,
in black and white. The call went the other way on most of them, and the
call belongs to whoever knows what these annotations are for: a frame
where the ear is technically present but unusable is still unusable, and
that is a judgement about the training target, not about the pixels.

Worth stating plainly for whoever reads this later: this is the one place
in the session where a decision was made on judgement rather than
measurement, and the earlier measurement -- filtering low-agreement rows
removed hard IMAGES rather than bad LABELS -- argues for keeping rows like
these. If a model trained on `train_clean.csv` loses to one trained on
`train.csv`, that measurement is the reason, and the drop list is where to
look. Reverting is one flag: `--train-csv train.csv`.

24 kept rows are in `qa_lists/review_keep.txt`.

## Where the cleaning programme stands after reading the literature

Accepted without re-measuring, 2026-10-13: **zero-mean annotation noise does
not need cleaning.** An L2-trained network converges to the conditional mean
of its targets, so symmetric jitter is absorbed. Xiang et al. show models
trained with injected Gaussian noise at every level converging to the same
NME, and their 300-VW model trained on NOISY data beating the corrected one
(5.55/4.54/9.87 against 5.62/4.63/9.97); they report noise injection working
as regularisation.

So the taxonomy that matters is not right/wrong but:

| noise | treatment |
|---|---|
| random jitter | ignore; it regularises |
| systematic bias (convention, semantic ambiguity) | co-estimate, do not delete |
| gross outliers | change the LOSS, not the dataset |

**The clean-vs-full A/B was proposed and deliberately not run.** The user
accepted the literature's prediction rather than spend GPU hours confirming
it. `train_clean.csv` (8,355 rows) stands as built and unused; it is not
evidence of anything and should not be cited as if a run had happened.

**Two of the three answers are already in this codebase**, which was not
noticed until the literature named them. `model/losses.py` uses Wing loss
(log near zero, linear in the tail) -- the robust-loss answer to outliers.
It also already splits the residual into normal and tangential components
with a `tangential_weight`, which is the right response to ambiguity ALONG a
contour and matches the measured fact that tangential error is 72% of the
squared error. Its default is 1.0, so that mechanism is switched off. The
loss also already accepts a per-point `visible` mask, which `data/dataset.py`
supplies as a (55,) tensor.

**What the literature has that this does not:**

- **Latent ground truth.** Semantic Alignment (CVPR 2019) names this exact
  failure -- contour landmarks have no precise definition, so annotators are
  inconsistent and the inconsistency IS the published GT -- and treats the
  consistent GT as a latent variable, alternating between training the
  detector and re-searching the GT. No external referee required.
- **Consistency from outside the image.** Supervision-by-Registration (CVPR
  2018) adds a registration loss over unlabelled video through a
  differentiable Lucas-Kanade. There is no video here, but the 589-ear wrap
  corpus is the same category of signal. The referee this file said did not
  exist was being looked for in the wrong form: a training CONSTRAINT, not a
  judge of individual rows.
- **Per-keypoint training dynamics.** REMIND and the NeoPose paper (both
  2026) cluster per-keypoint LOSS TRAJECTORIES over epochs rather than
  scoring a final residual, and zero the visibility flag on noisy keypoints
  instead of discarding the image. This supersedes the training-residual
  screen recorded earlier in this file. Both papers state the weakness
  independently: hard-but-correct keypoints have the same dynamics as noisy
  ones -- the failure mode already seen here.

### Plumbing for the loss-trajectory screen

Two changes, both inert until asked for:

`data/dataset.py` now returns `idx`, the MANIFEST row rather than the position
in the split, so a measurement taken during training can be attributed back to
the annotation it came from.

`model/lightning_module.py` takes `point_trajectory_dir` (`--point-trajectory-dir`
on train.py, off by default) and writes one npz per epoch holding, for every
training row, the per-point error. ~1.4 MB per epoch, ~0.4 GB over a 300-epoch
run, no measurable training cost.

**Normal and tangential components are stored APART**, which is the one place
this improves on the published method. REMIND and the NeoPose paper cluster a
single per-keypoint loss; here the residual is already decomposed, and the two
halves mean different things -- tangential is position ALONG the contour, where
annotators disagree by convention, and normal is distance FROM it, where they
are simply wrong. A single number mixes ambiguity with error, and the two want
opposite treatments.

Both papers independently report the method's weakness: hard-but-correct
keypoints have the same training dynamics as wrong ones. The occlusion map from
`qa_sam3_intrusion.py` gives a per-point reason a point is hard, so the two
signals can be crossed -- trajectory says noisy AND occlusion says hidden is
hard-not-wrong; trajectory says noisy and nothing is covering it is an
inconsistent label. That is the one thing this corpus can do that the papers
cannot, and it is a use for the QA instruments that survives the redirection.

### Cynic audit of the trajectory reading -- one claim withdrawn

**WITHDRAWN: "shape carries nothing beyond the endpoint."** The test was
underpowered and the null was uninformative. Per-point shape features were
computed from SINGLE-epoch values, and a single epoch is mostly noise:
epoch-to-epoch sd is 0.71 of the mean, adjacent epochs rank-correlate 0.409,
and split-half reliability of a one-epoch per-point estimate is **0.417**. Any
shape feature built on that was measuring augmentation.

Redone on 21-epoch smoothed curves, drift becomes a real quantity --
split-half reliability **0.556**, against 0.959 for the level -- but it still
does not separate the groups (hand-dropped -0.038, control +0.008, wrong
direction and tiny). The honest claim is weaker than the one withdrawn:
**shape is measurable here and uninformative about these labels**, not absent.

**WITHDRAWN: "the screen needs ~10 epochs, not 214."** Backwards. The
trajectory's value is not that the ranking settles early; it is VARIANCE
REDUCTION over a noisy per-point measurement:

| epochs averaged | split-half reliability of the per-point estimate |
|---|---|
| 1 | 0.417 |
| 5 | 0.770 |
| 10 | 0.862 |
| 25 | 0.937 |
| 50 | 0.966 |

So more epochs is better and 25-50 is where it flattens. This is a different
reason to log trajectories than REMIND and the NeoPose paper give, and it
holds even though their stated reason (shape) does not reproduce here.

**SURVIVES: tangential error exceeds normal, and it is not an augmentation
artefact.** On the UNAUGMENTED val split the ratio is **1.60**, higher than the
1.45 seen on augmented training batches. Normal falls 88.7% from epoch 0,
tangential 81%: the model drives the off-contour component down further and
hits a floor on the along-contour one.

**SURVIVES: the group separation is not a scale confound.** Dropped rows and
controls have near-identical ear size in the crop (0.815 vs 0.782 of crop
height), and the error ratio is 2.00 in raw crop units against 1.96 per ear
height.

**THE UNRESOLVED CORE.** Level separates the groups strongly and reliably, but
nothing here shows level measures LABEL ERROR rather than IMAGE DIFFICULTY --
and the hand-dropped rows were dropped for being hard images, so difficulty
alone predicts their 2x error. This is the literature's own stated failure
mode, reproduced. The per-point screen is validated as "the model finds this
hard", which is not the same claim.
