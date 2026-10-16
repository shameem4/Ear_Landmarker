# Candidate fixes, ranked

State of play across `Ear_Landmarker`, `BlazeEar` and `trainable_blazeface` as of
2026-10-15. Everything here is measured; the measurement is cited so each item
can be funded or declined without re-deriving it. Items are grouped by owner,
because almost nothing on this list is fixable in the repository that surfaces it.

The two headline deficits:

| | share of all ears | owner |
|---|---|---|
| stage-2 misses: a crop delivered the ear, BlazeEar missed it | 14.60% | BlazeEar |
| face-stage misses: no crop ever delivered the ear | 8.70% | trainable_blazeface |

---

## Shipped already (this session)

| # | fix | effect |
|---|---|---|
| S1 | Rank checkpoints by stored score, not the rounded filename | `--resume best` no longer rolls a run back 45 epochs |
| S2 | Flip fusion on by default, batched into one forward | -1.83% `nme_normal` (p=0.009), +0.3% wall clock on CUDA |
| S3 | Ship `fixed_conv_s42` | handedness R-L antisymmetry +17.05% (p=2.7e-19) -> +1.01% (p=0.46) |
| S4 | Browser demo fusion parity | six-decimal agreement with Python |
| S5 | `detector_frame_tta` flag, off by default | +3.46 pp detector recall for a second detector pass |

---

## Open: BlazeEar, stage 2

The target is the **32-48 px band** -- 426 ears at 58.0% recall, where the scores
already sit just under the gate (TP median 0.757 against FP p90 0.298). Moving it
to ~75% is worth roughly **+2 pp overall recall**. 0-32 px at 18.2% is a stretch
and may be irreducible. Training cost is **1-3 GPU-hours per run**, so all of
these can be tried in sequence.

| # | fix | cost | why it should work |
|---|---|---|---|
| **B1** | Widen `scale_range` low end from 0.75 to ~0.35 | one line | the training median ear is 64.9 px, so x0.75 = 48.7 px -- augmentation **cannot currently manufacture a sub-32 px ear**. Only the natively-small 5.9% populate that band. Attacks the exposure gap directly and leaves the mode's mass intact |
| ~~B2~~ | ~~Turn focal loss on~~ | **dead** | **Focal loss has been ON for every run in BlazeEar's history.** `train_blazeear.py` calls `set_defaults(use_focal_loss=True)` at alpha 0.35 / gamma 1.5; the README documented it as opt-in at 0.25/2.0 and was wrong on both counts. Retracted by BlazeEar, whose finding it was |
| ~~B3~~ | ~~Per-level hard-negative cap~~ | **misaimed** | the mining imbalance is real but at the **large** end, not the small one -- see below. A cap would help a band that is already at 95.7% |
| **B4** | Lower the score threshold 0.70 -> 0.60 | **no training** | +2.81 pp recall for -6.94 pp precision. An operating-point decision, not a free win |
| **B5** | Expose a calibrated score to the caller | small | better shape than moving a default, since consumers differ on the precision/recall trade |

**The anchor-budget root cause recorded here was wrong, and measuring it
reverses the direction.** It said the 2048 slots holding the two smallest priors
absorb the mined-negative budget because they own 57% of the slots while taking
7.7% of positives. The premise is right and the conclusion does not follow:
hard-negative mining takes a global top-k of background *logits*, so the budget
follows how **confusing** a prior is, not how many slots it owns. BlazeEar
inferred it from the slot table, then measured the mining and retracted it.

Mined-negative share against positive share per prior, shipped weights, 1,280
augmented training images:

| prior (px) | slots | neg share | pos share | ratio |
|---|---|---|---|---|
| 6.9 x 11.3 | 1024 | 9.6% | 1.0% | 9.4x |
| 12.5 x 24.3 | 1024 | 16.8% | 12.6% | **1.3x** |
| 17.7 x 41.2 | 256 | 0.2% | 9.2% | 0.0x |
| 34.8 x 39.9 | 256 | 0.1% | 18.3% | 0.0x |
| 29.2 x 62.7 | 256 | 0.3% | 35.0% | 0.0x |
| 58.4 x 101.1 | 256 | 19.3% | 22.3% | 0.9x |
| **129.8 x 178.2** | 256 | **53.5%** | **1.0%** | **52.4x** |
| 191.5 x 260.4 | 256 | 0.1% | 0.6% | 0.2x |

**The pathology is at the LARGE end.** One prior with 7.1% of slots and 1.0% of
positives takes over half the entire negative budget; the three priors carrying
62.5% of positives take 0.6% between them, which is why the mode reaches 95.7%.
The 32x32 priors a small ear lands on sit at **1.3x, close to balanced** -- so
mining is *not* what holds the small bands back. It is learned rather than
structural: at initialisation that prior takes 2.1% while 58.4x101.1 takes 87%.

It also predicts a result that had looked anomalous: the score head's **worst
discard is at the largest ears, 36.9 pp at 198 px**, and that prior is trained
52:1 against. Two measurements taken for different reasons agreeing is the
reason to quote either.

Still standing from the original account: small ears are **not** starved of
positive supervision -- `ANCHOR_TOP_K = 3` delivers exactly 3.00 per ear at
every band by design.

**Guard on every run:** report the 64-80 px band, mean positives-per-ear, and the
per-prior positive share. 1402 of 3790 ears sit in the mode, so a run that
improved the tail by starving the mode would show there before it showed in
recall. `measure_anchor_budget.py` prints the latter two, so the guard costs one
script invocation per arm.

**What this does to B1.** It survives and is *better* motivated -- it is now the
only lever aimed at where the ears actually are, rather than the cheaper of two
competing rebalances, since the other addresses the opposite end of the scale.

**But one scoping result cuts against part of it.** The 0-32 px band is
**information-limited, not supervision-limited**: across its 741 training ears
the median ear is **24 real pixels**, 70.9% are under 32 real pixels, and the
crop resamples at 0.94x -- essentially 1:1. Those ears are small *in the source*.
No augmentation, loss re-weighting or anchor rebalance manufactures information
nobody captured, so 18.2% recall there may be far nearer its ceiling than it
looks, and **a B1 arm that moves 0-32 px deserves suspicion rather than
celebration**. 32-48 px is the band with headroom -- 1,322 ears, median 38 real
pixels, only 36.5% under 32 real, at 58.0% recall -- and is where B1's result
should be read hardest.

**B1 is built and queued, not started.** `AUGMENT_SCALE_RANGE` is (0.35, 1.25)
with the old range reachable from the environment. At the data level the change
does what it should: the 0-32 px band goes from 5.99% of training boxes to
13.34%, 32-48 from 10.89% to 17.09%, and the p10 ear from 40.0 px to 28.5. It
runs as a **two-arm A/B** -- control is the old range under today's code --
because the shipped checkpoint predates other changes, so a lone retrain
compared against it would have several candidate causes and neither a win nor a
null would be attributable.

**The replication standard is pre-registered in the launcher**, which is the
only time such a standard means anything: *one band improved by a couple of
points with the mode flat, on a single draw and with no interval, is not a
result; the read is "needs a second draw", and the next step is to re-run the
pair rather than write it up.* That is the shape of trainable_blazeface's lone
16-32 px bin at +1.2, which they downgraded themselves.

**One caveat on how any B1 number can be pinned.** `train_blazeear.py` has no
`--seed`, so the two arms control the *variable* but not the *draw* -- every run
is an independent, unreproducible sample. That is adequate for replication (run
the pair again and see whether the effect survives) and inadequate for exact
reproduction, so a result that mattered could not be pinned as things stand.
Adding `--seed` is recorded there as worth doing **if B1 pays** and not worth
touching the trainer for speculatively.

**A trap that sat in B1's path, now fixed.** `_tensors_to_detections` and
`BlazeEarInference` could hand NMS different candidate sets on identical
weights, which mattered because **a partly trained regressor can predict
negative width where a trained one does not** -- so the two paths were unsafe
to compare on a model that is still training, precisely when someone would.
Latent on the shipped model (zero inverted and zero degenerate across 1,075,200
candidates from 300 real crops) and live on random weights (2,191 of 3,584
corner-inverted).

The cause was two masks, not three differences: boxes with no area, and
non-finite scores. (An earlier note here, taken from BlazeEar's first
description, also listed corner-sorting -- `BlazeEarInference` does sort
corners, inside `decode_boxes`, just earlier than `process` does, and decode is
bit-identical once both are sorted. They retracted that themselves.)
`sort_corners` and `usable_box_mask` are now single definitions in `box_utils`
used by both eager paths and the three export graphs, with the graphs zeroing
the score rather than masking rows so their fixed shapes survive. Untrained
weights now agree across seeds 0-4 where seed 0 diverged before, and the
regression test deliberately runs at untrained weights, since that is the only
place the defect was reachable. **Out of B1's path.**

**And a scoping note on "under-scores small ears",** which is half right. Two
different statistics, both measured, not in conflict: the *scoring discard rate*
-- the gap between localised and gated -- is flat at ~20 pp below the mode and
grows to 36.9 pp at 198 px, so the worst discards are at the large end. The
*recall* is lost at the small end: 18.2% at 0-32 px and 58.0% at 32-48 against
95.7% at the mode, with 148 of 203 small ears localised-but-gated against 3 of
34 at 120-160. The large bands hold 17 to 34 ears against 203 and 426. For B1's
targeting the recall statistic is operative, because that is where the mass is.
The accurate phrasing is that the scoring gap is scale-dependent in **both**
directions and the recoverable mass is small.

---

## Open: trainable_blazeface, stage 1

**T1 has run, it did not clear MediaPipe, and the reason reframes T4 rather
than blocking it.** v2 converged flat (val loss 6.70 -> 6.63, mean IoU 0.578 ->
0.582, AP 0.758 -> 0.760) and still destroys precision. The cause is the label
set, not the code: measured at the model's 128x128 input across the complete
splits, their ground-truth faces have a **median of 13.4 px (val) against a
finest anchor stride of 8 px**, with 39.4% under 8 px and only 46.7% clearing
16 px. MediaPipe's recall by size bin is 0.000 / 0.000 / 0.078 / 0.854 / 0.966,
and its measured overall recall is 0.4954 -- **46.7% of boxes being resolvable
and 49.5% being recalled, agreeing to within a point, is the case that it sits
at the architectural ceiling** rather than being a weak baseline. Training on those labels chases unresolvable positives and the damage
lands on faces it could previously see -- 32+ px recall falls 0.966 -> 0.579.
They validated the port against Google's own `blaze_face_short_range.tflite`
(recall 0.4936 vs 0.4954, per-bin identical), so none of it is a porting
artefact.

**The size filter fixed the diagnosed pathology and still did not clear the
gate.** Filtering training boxes under 16 px (and dropping images where over
half the faces would go, so real faces do not become background labels) left
23,746 of 32,325 images at 1.48 boxes each. Scored on the UNFILTERED val split,
2000-image sample, IoU>=0.5:

| | precision | recall | 8-16 px | 16-32 px | 32+ px |
|---|---|---|---|---|---|
| MediaPipe stock | 0.939 | 0.464 | 0.081 | 0.857 | **0.973** |
| v2 unfiltered | 0.199 | 0.487 | 0.264 | 0.841 | 0.920 |
| v2 min-16 | 0.912 | 0.448 | 0.028 | **0.869** | **0.923** |

The filter confirmed the mechanism -- **precision recovers 0.199 -> 0.912, a
4.6x jump**, predicted from the size distribution before the run and far
outside noise. That is the one result from this line of work worth defending
without hedging. But **32+ px lands at 0.923 against the 0.966-0.973 gate, a
5-point miss**, and aggregate recall is still below stock. **No checkpoint in
that repository beats the frozen MediaPipe weights it initialises from.**

The 16-32 px figure of 0.869 against stock's 0.857 is the first time anything
there has beaten its own initialisation, and **it should carry no weight**:
single bin, adjacent to the filter boundary that was chosen, one seed, no
confidence interval, +1.2 points. trainable_blazeface downgraded it themselves
as "exactly the size of thing that does not replicate", and it would need a
second seed before it meant anything.

One caveat in the fine-tuned models' favour, flagged by trainable_blazeface
against their own interest: the fine-tuned checkpoints are calibrated
differently and fall off much faster with threshold (min-16 goes 0.912
precision at 0.5 to 0.985 at 0.75, recall 0.448 to 0.294), so a fixed-threshold
comparison understates them. At matched precision the gap is smaller. It does
not close.

### T2 is dead: the 45.8% compute saving was counting boxes, not people

The claim was that merging overlapping face boxes at label time would yield one
box per head and cut stage-2 calls by 45.8%, since the detector averages 1.84
faces per frame. **It assumed those extra boxes are duplicates. Measured, 75% of
them are distinct heads.** Flagged by trainable_blazeface, who found *zero*
duplicates on their own corpus and correctly guessed mine would differ in detail
but not in direction.

1500 frames, 1.78 boxes/frame, 706 multi-detection frames, 1,969 pairs:

| | pairs | share |
|---|---|---|
| above the 0.30 NMS IoU threshold -- genuine IoU duplicates | **0** | **0.0%** |
| essentially disjoint, IoMin < 0.1 -- distinct heads | 1478 | **75.1%** |
| nested, IoMin >= 0.35 -- one box inside another | 313 | 15.9% |

Pairwise IoU maxes out at exactly 0.300, the suppression threshold -- so
`_weighted_non_max_suppression` is already doing this job and there is nothing
left to merge by IoU. Merging the disjoint 75% would discard real faces and the
ears attached to them, which is a recall loss booked as a compute win.

**And the nested 16% were already tested, by me, with the opposite result.** The
face-stage IoMin experiment suppressed exactly those pairs: it cut stage-2 calls
by 10.6% and cost **-0.50 pp of recall, 10 ears lost and 0 gained, p=0.002**. So
the realistic ceiling for this idea is not 45.8% but 10.6%, and that 10.6% is
already measured as costing recall significantly.

I had both halves of this on the page -- the IoMin result and the 45.8% claim --
and never put them together. The 45.8% came from dividing frames by boxes, which
is a statement about counting, not about redundancy.

### SUPERSEDED: "The face-box test ran. T4 is closed."

**That closure rested on `trainable_blazeface`'s negative result, which they
retracted.** All six of their configurations used a defective anchor assignment;
with a per-anchor assignment plus an ignore band, fine-tuning clears this side's
gate (0.934 at 16-32 px, 0.980 at 32+ px). Their checkpoint was then scored
directly here, on all 13,007 images / 15,359 human-source ears, paired.

**T4 closes again on a different and more useful finding: the coverage gain is
real and cannot be converted.** Face-stage misses fall 1301 -> 908 as dropped
in, and -> 516 for the union of both detectors' crops (-60%). But at matched
precision the union is negative at every operating point except the two
loosest (+0.23, +0.54 pp), and the tuned arm alone is -3 to -5 pp. The
+4.81 pp headline is a precision trade.

Two constants were denominated in stock's units -- `FACE_CROP_EXPAND` (their
boxes are linearly 1.0998x larger on this corpus, CI 1.094-1.105) and
`FACE_CROP_THRESHOLD` (their scores sit at median 0.234 against stock's 0.710,
so 0.2 hands stage 2 6.16 crops/image against 1.84). Recalibrating expand to
1.364 makes *both* arms worse, so the units observation is right and that
correction is wrong.

**The lever moves to stage 2, and is now worth more.** The extra ears arrive as
boxes the 0.70 threshold discards -- the same constraint as the 90.2%
under-scoring result. Revisit the crop union only after stage-2 scoring is
fixed, at which point +16.79 pp of raw 0-32 px coverage (the one
configuration-invariant gain, p<1e-08 in all four configurations) is sitting
unconverted for the price of a second detector pass.

Full measurements and the matched-precision table: README, "T4 reopened".

### Original closure, retained because the gate it set still stands

**Fine-tuning stage 1 does not beat stock MediaPipe even in this corpus's
favourable regime.** Gate was "hold 0.966 at 32+ px at matched precision"; it is
missed by a wide margin.

Setup: YuNet face pseudo-labels (`cv2.FaceDetectorYN`, a different architecture
from BlazeFace, so the labels are independent of the thing under test -- and
RetinaFace was unusable, pinning `tensorflow==2.5.0` and a cv2 downgrade that
would have broken the shared env). 13,482 images at conf 0.8, giving a label set
far easier than trainable_blazeface's: **median 36.0 px at 128 input and 78.3%
over 16 px**, against their 13.4 px and 46.7%. Train/val/heldout split by seeded
draw **before** any filter existed; the min-16 px filter applied to train only
(9,385 boxes / 7,442 images); 20 epochs from MediaPipe init. Scored on the
**unfiltered** hold-out, 2,742 boxes over 1,734 images.

| | threshold | precision | recall | 16-32 px | **32+ px** |
|---|---|---|---|---|---|
| stock | 0.60 | 0.938 | 0.654 | 0.619 | **0.954** |
| stock | 0.50 | 0.899 | 0.671 | 0.676 | **0.963** |
| fine-tuned | 0.50 | 0.816 | 0.581 | 0.475 | 0.882 |
| fine-tuned | 0.80 | 0.873 | 0.133 | 0.071 | **0.218** |
| fine-tuned | 0.30 | **0.310** | 0.642 | 0.495 | 0.986 |

At **matched precision** the fine-tuned model is not close: stock reaches 0.963
at 32+ px with precision 0.899, while the fine-tuned model's best precision is
0.873, where its 32+ px recall is **0.218**. Its only win on 32+ px (0.986) buys
precision of 0.310 -- 3,916 false positives against stock's 207. That is the
same precision collapse trainable_blazeface diagnosed, reproduced on a corpus
three times easier.

**The caveat runs in the conclusion's favour, which is why it is worth stating.**
Recall here means agreement with YuNet, and the fine-tuned model was *trained on
YuNet labels* while stock has never seen them -- so the test is biased toward
the fine-tuned model, and it still loses decisively. One seed, 20 epochs, no
hyperparameter search, so a better-tuned run could differ; nothing about this
suggests it would differ by the margin required.

**So T4 is closed.** The face-stage 8.70 pp stands as a measured ceiling with no
mechanism: concentric expansion is dead, lateral shift has no direction
predictor, post-hoc rules on face output fail as a class, and stage-1
re-targeting is now ruled out on its own precondition in the one regime that
favoured it.

### If the face-box test runs, build these in from the start

The cheap decisive gate for T4 is to fine-tune on a slice of *this* corpus with
ordinary face-box targets and ask whether it holds 0.966 at 32+ px. Two
protocol requirements, both free up front and both costing a whole run if
skipped. They come from trainable_blazeface, who was caught by the second one
twice:

1. **Compare at matched precision, not at a fixed threshold.** A fine-tuned
   checkpoint and stock are not on the same score scale, and 0.5 is stock's
   natural operating point rather than a neutral one. Sweep.
2. **Hold out an evaluation slice that no label-selection rule has touched.**
   Every flattering number in their investigation came from a set narrowed in
   the same direction as the hypothesis -- the sorted head slice, then the
   filtered val, which still reports AP 0.957 beside a detection-level loss. If
   the slice here is selected by anything correlated with ear reachability, the
   same thing happens.

**What this does and does not establish for T4.** It establishes that the
failure mechanism is understood and that removing unresolvable labels clears
the pathology cleanly. It does **not** establish that fine-tuning this
architecture *helps* anywhere -- only that it stops being catastrophic. So T4 is
**not green-lit**, and the honest next step if anyone wants it answered is the
cheap one: fine-tune on a slice of *this* corpus with ordinary face-box targets
and see whether it holds 0.966 at 32+ px, before anyone builds ear-inclusive
ground truth. That is a measurement on this side and nobody has asked for it.

**The corpus difference that motivated T4 still holds, and the gap is not
close.**
Measured here on 2000 images, same units -- original pixels scaled to the 128
letterboxed input:

| | this corpus | their labels (val) |
|---|---|---|
| detected face box, median | **31.6 px** | 13.4 px |
| face >= 16 px | **98.7%** | 46.7% |
| face under 8 px | **0.0%** | 39.4% |
| **hull of face + its ears** (what T4 must predict) | **median 42.7 px** | -- |
| hull >= 16 px | **99.4%** | -- |
| hull at 32+ px | **70.4%** | -- |

(An earlier version of this table cited a 5.4 px median and 12.3% over 16 px.
Those were measured on a sorted slice of `val.csv`, which is ordered by
category, so the head of it is WIDER crowd scenes. Retracted by
trainable_blazeface, whose figures they were. The corrected gap is roughly 2x in
usable-target fraction rather than 8x; the direction and the conclusion are
unchanged.)

Essentially every head here sits in the band where the architecture works --
98.7% over 16 px against their 46.7% -- and 70.4% of the actual T4 targets are
in the 32+ px bin where stock MediaPipe already recalls 0.966. So T1's failure is a property of *their* WIDER-style
crowd labels against a detector Google documents as "optimized for front-facing
phone camera images at short range", and it is not evidence that this
architecture resists fine-tuning on *this* data. Their next step is a
minimum-face-size filter on the training GT, unstarted.

| # | fix | status | notes |
|---|---|---|---|
| ~~T1~~ | ~~retrain to match MediaPipe~~ | **failed, twice** | see below |
| ~~T2~~ | ~~Merge overlapping face boxes at label time~~ | **dead -- the duplicates do not exist** | see below |
| **T3** | ~~Same-size crop shifted laterally~~ | **dead -- no direction predictor, see below** | the expand sweep tested *concentric* scaling, which pays resolution everywhere. A shift pays nothing. On the 31 escaping ears, a shift <=0.1 crop-sides recovers **32.3%** and <=0.5 recovers **58.1%**, all geometrically possible. **Open problem: predicting the direction** |

### T3's gate: answered, and the answer is no

The lateral shift needs a direction. Tested on 3000 images / 3525 ears, using
the shipped `process()` path for boxes and crops so the escaping-ear
determination matches every other measurement here, and a runtime-patched
keypoint path only to read the keypoints:

| predictor | escaping ears (n=290) | delivered ears (n=3235) |
|---|---|---|
| eye-midpoint minus nose | **57.6%** | 76.4% |
| far tragion | **57.6%** | 78.6% |
| eye-midpoint minus crop centre | 41.7% | 24.0% |
| always shift right | 30.3% | 51.4% |
| **majority class** | **69.7%** | 51.4% |

**The keypoints carry real direction information -- and it is absent exactly
where it is needed.** On delivered ears, eye-nose geometry calls the side 76.4%
of the time against a 51.4% baseline (p=3e-188). On escaping ears it falls to
57.6% [51.7, 63.3], barely above chance (p=0.011) and **significantly worse than
simply always shifting the same way** (69.7%, p=1.3e-05).

So a keypoint-derived direction rule is beaten by a constant on the only ears it
would be applied to. And the constant is not a rule either: 69.7% is this
corpus's handedness skew, not geometry, so it would not survive a mixed corpus.

**This is the third time the same structure has appeared.** The tragion as a
position predictor, the tragion-centred crop, and now eye-nose direction: in
each case the signal is strong where the crop already works and absent where it
fails. That is not three coincidences, it is one property of the problem --
whatever makes an ear escape a face-centred crop (a turned or partly out-of-
frame head) is the same thing that degrades every face-derived cue. Face
keypoints cannot rescue a face-box crop, because they fail together with it.

**Consequences.** T3 has no direction predictor and is not actionable as a
post-hoc rule. L3, the keypoint plumbing, loses its only justification and
should not be built. The face-stage 8.70 pp is back to having no proposed
mechanism, and the remaining candidate is the one that does not depend on
face-derived cues at all: **re-targeting stage 1 so the box is correct in the
first place (T4), which is gated on T1**.

**Quote T3's numbers with their shift budget attached.** The 58.1% is 18 of 31
ears on one 279-image split, and the required offsets are a smooth continuum
from 0.52 to 2.34 crop-sides (one outlier at 3.41) rather than two clean
populations -- BlazeEar looked for a gap that would separate "near miss" from
"wrong face" and there is none. So the recoverable share is purely a function of
the shift budget allowed, and "58.1% of escaping ears are recoverable" without
the `<=0.5 crop-sides` qualifier overstates it.
| **T4** | Re-target stage 1 to an ear-inclusive box | blocked on T1 | ground truth is derivable from `master.csv` human ear boxes; the hard part is head-assignment at 1.84 faces/frame. Must come from this side |
| **T5** | Add mirror-agreement to the v2 acceptance criteria | free | see below |

**T5 is the cheapest item on this entire list.** The shipped face weights are
**stock MediaPipe**, so the 6.4%-of-boxes-disagree-below-IoU-0.1 instability is
MediaPipe's, not something the local trainer caused -- its flip augmentation (p=0.5)
never touches those weights. If v2 clears T1 *and* is flip-stable, it absorbs the
entire +3.46 pp that S5 currently buys with a second detector pass, at zero
runtime cost. That makes mirror-agreement a v2 acceptance criterion worth
measuring alongside recall, and it is a query against a harness that already exists.

---

## Open: Ear_Landmarker

| # | item | note |
|---|---|---|
| L1 | Detection is 68-76% of pipeline wall clock | the real latency target; no mechanism proposed |
| L2 | Decide whether `detector_frame_tta` should default on | currently off. Every measurement either project has is on a corpus that is single-presentation or scored on crops. Ear_3D_Lifter can build a mixed-presentation corpus that would settle it |
| L3 | ~~Recover the face keypoints~~ | **not worth building** -- its only justification was T3, which is now dead. Mechanism kept below because the 4-line scope and the parity-test gap are useful if it ever returns |

### L3 in detail: propagating keypoints through NMS

The discard is **not** at `blazedetector.py:262`, which is what an earlier note
here and the `trainable_blazeface` session both said. That `[:, :5]` is a no-op:
the tensor already has exactly 5 columns by then. The real discard is upstream
at **`blazedetector.py:326`**, inside `_tensors_to_detections`:

```python
boxes = detection_boxes[i, mask[i]][:, :4]      # <- keypoints dropped here
scores = detection_scores[i, mask[i]].unsqueeze(dim=-1)
output_detections.append(torch.cat((boxes, scores), dim=-1))
```

`_decode_boxes` really does return keypoints (it calls
`decode_boxes_with_keypoints`), and they are thrown away one line later, before
NMS ever sees them.

The change, in dependency order:

1. **`:326`** keep all coords: `boxes = detection_boxes[i, mask[i]]`, giving the
   layout `[ymin, xmin, ymax, xmax, kp_x0, kp_y0, ..., score]` -- **score last**.
2. **`:260-261`** `nms_indices(image_detections[:, :4], image_detections[:, 4], ...)`
   must take the score from `[:, -1]`. **This is the line that breaks silently if
   missed**: with keypoints at index 4, NMS would sort by `kp_x0`.
3. **`:262`** drop the `[:, :5]`.
4. **`:255-256`** widen the empty-detection `torch.zeros((0, 5))` to match.

**`denormalize_detections` needs no change** -- it already computes
`keypoint_coords = num_coords - 5` and transforms indices `4..kp_end`, which is
exactly the score-last layout above.

**Nothing changes on this side either.** `inference.py:_as_boxes` already
collapses a wider detection to `[box4, last column]`, and its docstring records
that scoring by a keypoint has bitten this project before: *"made NMS sort by a
keypoint and report nonsense scores"*. So the one real risk in step 1-4 is
step 2, and it is a risk the project has already paid for once.

**But those 4 lines are the research cost, not the shipping cost.**
`blazeear_inference.py` is a second, independent path that never mentions
keypoints -- it builds a 4-column tensor from scratch at **four** decode sites
(`:272` for the main class, `:662`, `:803`, `:919` for the TorchScript, ONNX and
web wrappers). So:

- **To answer the T3 direction question**, the 4 lines above are enough:
  `make_face_crops` and `evaluate_two_stage` both go through `process()`.
- **To ship anything keypoint-based**, all four of those sites need the same
  change, consistently.

**And the test that exists to catch exactly this will not.**
`utils/tests/tests/test_inference_parity.py` pins a great deal -- preprocessing
bit-identically, one shared anchor tensor, and the raw model outputs across both
paths at `(1, 3584, 16)`, keypoint channels included, bit for bit. What it never
pins is **what either path emits to a caller**. Both decode tests compare four
columns: one calls `decode_boxes_with_keypoints(..., num_keypoints=0)` then
slices `[..., :4]`, the other pads with twelve zeros and also compares four. Not
even the score is asserted. `BlazeEarInference.predict` returns `(100, 5)` and
`BlazeEar.process` returns `(N, 5)` on the same image, and nothing compares
those two tensors.

So a keypoint widening in one path and not the other **passes silently**, and
the divergence would live precisely in the unguarded decode/discard layer where
the repository's earlier four-paths-disagree defects lived. If this work starts,
widening the parity test is the first commit -- and the widening wanted is not
more raw-tensor comparison, which is already bit-exact, but an assertion on the
**emitted** detections: same column count, same scores, same keypoints after
`denormalize_detections`.

**The exported graphs are covered, but not by the test suite.** No test loads an
exported artefact -- `onnxruntime` appears only in `export_onnx.py` and
`export_two_stage_web.py`, which check the graph against the PyTorch wrapper at
export time. Two consequences: the check runs when someone exports rather than
in CI, and `export_onnx.py` returns success when `onnxruntime` is missing (it
prints a notice first). Worth knowing because the parity file's own docstring
names "the exported graphs themselves" among the things *these tests* pin, and
they do not -- the export scripts do.

---

## Ruled out, with the measurement

Do not re-attempt these without new information.

| candidate | verdict |
|---|---|
| Dynamic-batch ONNX export (browser) | **-2.1%**: `onnxruntime-web` needs WASM threads to exploit a batch, which need `SharedArrayBuffer` and COOP/COEP headers GitHub Pages cannot send |
| WebGPU execution provider | rejected on **correctness**: BlazeFace 0.10x speed with outputs differing by 13400, BlazeEar 0.34x differing by 287. Modelled frame 2.3x slower |
| Size-dependent score threshold | **dominated** by moving the global gate at every config tried, despite TP-vs-FP AUC of 0.93-0.98 per band |
| IoMin NMS on the face stage | **-0.50 pp recall, p=0.002**, 10 ears lost and 0 gained, for -10.6% stage-2 calls |
| Widening `FACE_CROP_EXPAND` | 1.5 optimal on three independent bases; 1.2 -7.93 pp, 1.9 -4.29 pp, 2.4 -9.00 pp |
| Anchor coverage as the stage-2 mechanism | best-anchor IoU 0.571 at 32-48 px with 3.6% below 0.35. Open only for the 39-ear never-proposed rump |
| Detector NMS threshold | recall and median IoU flat to three decimals from 0.10 to 0.99 |
| Tragion-centred crop | **9.7%** of escaping ears have a tragion within 1.0x ear size (median 6.61x) against 86.3% of reachable ones. The failure modes are correlated |
| Ear position near the crop edge | ears are already peripheral in training: median radial offset 0.361, only 3.0% within 0.15 of centre |
| ROI refinement as a latency cost | never fires on photographs (1.000 landmarker calls per ear). Fires on 81% of Ear_3D_Lifter's render views, so it is load-bearing there |

---

## Dependency order

1. **T1** (running) gates T4, and its outcome decides whether the 8.70 pp is a
   ceiling on a hard path or a blocked one.
2. **B1** is the cheapest item with a measurable upside and needs nothing from
   anyone else. It is the one to start.
3. **T2** and **T5** are nearly free and independent of T1.
4. **T3** is the only live mechanism for the face-stage 8.70 pp, and it reduces to
   one question: can direction be predicted where position cannot?
