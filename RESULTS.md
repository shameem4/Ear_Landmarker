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

## Known limitation

Detector recall is now 78.6% on this set rather than 47.5%, so the landmarker and
the detector are closer to balanced. The remaining ceiling is faces BlazeFace
misses (1.4% of images), which no second stage can recover.
