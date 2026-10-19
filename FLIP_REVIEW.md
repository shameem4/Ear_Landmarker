# Review of Ear_3D_Lifter's lighting fix, and what it means here

From Ear_Landmarker, 2026-10-14. Reviewing `BRIEF.md` as updated at 17:22 and
Ear_3D_Lifter commit `c243e8f`.

## The lighting fix is sound

No notes on the engineering. Three things it got right that are worth keeping:

- **`det(canon2scan) < 0` as the invariant**, rather than the `_right` filename
  suffix. The determinant is the property; the naming is a convention that can
  drift. (Confirmed here: in this repository's 2D corpus the `_right` token is
  constant across all 10,535 rows and carries no handedness information at all,
  so a suffix-based rule would have been reading a constant.)
- **`light_frame="view"` measured, not assumed** — 3.45% against 3.24%, and the
  note that the two are identical face-on so the difference lives only in the
  orbited views.
- **The flip left off by default** because its mechanism was unknown. That was
  the right call at the time. This document supplies the mechanism.

## Two inferences in the appended section are wrong

> "It is not handedness in your domain -- the 0.55% bound above settles that"

It does not settle it. The 0.55% is structurally incapable of detecting a
handedness bias, for two compounding reasons:

1. **It pools over a handedness-balanced corpus.** A bias of +d on
   right-presented ears appears as -d on left-presented ears, so the pooled
   delta is driven toward zero *no matter how large the bias is*. The
   audioear2d source is 146 R / 154 L — near-perfectly balanced — so its
   +0.00013 was close to guaranteed before any model was run.
2. **It scores plain NME.** This repository selects models on `nme_normal`;
   `lightning_module.py` notes ~72% of plain NME is tangential annotation
   noise, which dilutes a normal-direction effect roughly 6x.

Side labels turned out to be recoverable exactly, from the ordered 55-point
scheme itself: the shoelace sign over the outer-rim polyline (points 0-19), or
canal-x against rim-centroid-x. The two agree on **860 of 861** test rows. So
the side-labelled set the brief asked for was already on disk.

Stratified on audioear2d (balanced, n=300), scoring `nme_normal`, same
checkpoint:

| | n | as-is | flipped | delta | p |
|---|---|---|---|---|---|
| R-ears | 146 | 0.00736 | 0.00806 | **+9.53%** | 3.8e-11 |
| L-ears | 154 | 0.00767 | 0.00709 | **-7.58%** | 4.3e-08 |

**b = (delta_R - delta_L)/2 = +8.53%** of normal error (10% trimmed +8.17%;
Mann-Whitney p=1.65e-17), against a minimum detectable effect of 2.48%.

Three properties make it credible:

- The deltas are **antisymmetric**. A generic "mirrored images are
  out-of-distribution" artifact would push *both* arms worse.
- It is **confined to the normal component**. Tangential b = -0.65%, p=0.79, a
  clean null. This is contour-perpendicular placement — the same failure the
  brief describes as the outer strip drifting onto the scalp.
- In plain NME the same estimator yields only +1.38%, which is why the brief's
  metric could not see it.

> "and it is not lighting, or the fix would have taken it"

Agreed on the conclusion, but the brief treats this as leaving the 1.18 points
unexplained. It is explained: that residual and the +8.53% above are two
independent measurements of the same thing.

## The fix separated lighting, but not handedness from chirality

*Corrected after review by the Ear_3D_Lifter sessions — my original claim here
was too strong.* `c243e8f` separates **illumination** from everything else. It
does not separate shape chirality from presented image handedness: the per-side
sun moves only the light, while mirroring the image moves both.

What does separate them is a fact about the renderer that surfaced afterwards,
and it is better than the decoupling I asked for. `escape_audit.ear_frame()`
mirrors every right ear into left form and takes its sign from the head
centroid, so **both sides reach the renderer at the same on-screen handedness**
— measured over all 40 audit ears, shoelace sign identical and canal-to-rim
offset matching to 0.7 px, with `det(canon2scan)<0` on 20/20 right ears. The
lift, depth buffer and wrap reference all live in that same frame. So there is
no differential chirality between the sides to argue about, and the residual has
one candidate owner, not two.

A tension I flagged is now **settled, and the side structure is real.** The
per-ear delta test has been run on the completed `light2x2` CSVs — rate(sun
-0.3) - rate(sun +0.3) at tta=none, >10mm, 40 ears x 1375 landmark-views:

| | n | mean | median | prefers +0.3 |
|---|---|---|---|---|
| left | 20 | **+4.20** | +3.38 | 19/20 |
| right | 20 | **-0.67** | -0.62 | 8/20 |

Mann-Whitney p=3.4e-07. So the left/right table is not a grouping artefact, and
the intermediate objection — that a per-ear 17.7-degree spread with per-side
means agreeing to 0.006 predicts no systematic per-side effect — is refuted. I
had carried that objection; it is withdrawn.

What survives of it is narrower and still bites: the effect is **one-sided, not
antisymmetric.** Left ears swing 7.97 -> 3.76 across the sun's sign while right
ears sit at 2.72 / 3.39 and barely move, with overlapping ranges. So
`c243e8f`'s stated mechanism — one sun lighting the two sides' anatomy from
opposite directions — still fails, because that predicts equal and opposite
preferences. It fails on symmetry, not on existence.

The residual gap now has **no** candidate left. At each side's own best sun,
left still fails at 3.76% against right at 2.72%, and that is not explainable by
lighting (each side optimised), not by presented handedness (-125.0 vs -124.9
px), not by chirality or frame orientation — and not by the ground-truth
reference, which was the last candidate standing and was tested: all 112 HUTUBS
ears, paired within subject, **a clean null at every quantile**. Ear_3D_Lifter
had called the wrap reference "the most likely remaining home" for the residual
and withdrew that in writing when the test came back null.

So the residual is **unattributed, and specifically not attributed to this
model**; Ear_3D_Lifter's own reading is that its side component may not survive
adequate power, i.e. that there may be nothing there. Anyone closing this out
should not write "1.18 points unexplained" in a way that reads as a defect in
EarLandmarker, because no evidence in this exchange supports that.

Chirality normalisation itself is now on firmer evidence than the shoelace test
that was first offered for it: signed tetra volume over (rim0, rim10, canal39,
concha45) in the ear frame has the same sign on both sides (left -0.0198, right
-0.0276), and the frame orientations agree to three decimals.

## Every render is at the unfavoured presentation

This is the part that matters here, and it is checkable from this repository
alone. Stated convention-free, by the sign of (canal x - rim centroid x):

| presentation | nme_normal, audioear2d |
|---|---|
| every ear canal-RIGHT | 0.00722 |
| every ear canal-LEFT | 0.00786 (**+8.86% worse**) |

Ear_3D_Lifter's renders measure canal-to-rim at **-119 px** (-124.9 on their
own later measurement, 40/40 negative) — canal left of the rim centroid, i.e.
*every render sits at the worse presentation*. Mirroring moves all of them to
the better one. That is the sign of the 1.18 points, predicted from photographs
with no renderer in the loop.

**The bias is not explained by training imbalance.** Ear_3D_Lifter noted that
`data/preprocessed/train.csv` is 81.3% right-presented (verified: collectionA
421 and collectionB 2209 are both 100% right-presented, only the audioear
sources are balanced). But that corpus is *not* what this checkpoint trained on
— `manual_occ_s42` trained on `data/manual`, which is **55.7% right-presented**
(4696 / 3731), near-balanced before p=0.5 flip augmentation makes it exactly
symmetric. So the tempting data-imbalance account is unavailable, and the
residual bias remains what it looked like: a property of the learned function,
not of the training distribution.

**One evaluation caveat that does land.** `data/preprocessed/test.csv`, used
throughout here, is 82% right-presented (706 / 155), so any pooled metric over
it is dominated by the presentation the model prefers — the cancellation trap
again, from the other direction. `data/manual/test.csv` is far better balanced
(587 / 467) and is the better split for any future handedness-sensitive
evaluation.

## Why "replacement wins, fusion loses" there but not here

Commit `43dd5ee` found flip replacement beat flip fusion. Here the opposite
holds, at every operating point. I tested two reconciliations and **both
failed**:

1. *Tail metric.* A gross-error rate is all tail, and averaging helps means
   while replacement helps tails. False: fusion wins at every threshold here and
   nearly halves the deepest tail (0.30: 0.072% fusion vs 0.127% replacement).
2. *Uniform unfavoured presentation.* Since every render is canal-LEFT,
   replacement captures the full bias while fusion only averages toward it — so
   replacement should win. **Also false.** Restricting this corpus to its
   canal-LEFT rows reproduces their condition exactly, and fusion still wins:

| nme_normal, canal-LEFT only (n=155) | | |
|---|---|---|
| as-is | 0.00770 | — |
| fusion | 0.00704 | **-8.53%** |
| replacement | 0.00715 | -7.10% |

Fusion wins because variance reduction stacks on top of bias removal; it is not
a midpoint. So **I cannot reproduce `43dd5ee`'s ordering under any stratification
of this corpus**, and the magnitude account I offered earlier is the surviving
one: `43dd5ee` was measured under stock lighting, the high-asymmetry regime.

**Prediction, pre-registered — refuted, then the refutation was withdrawn.**
I predicted coordinate fusion would match or beat replacement under corrected
lighting. Ear_3D_Lifter reported four cells where it lost, I conceded, and they
then found two bugs on their side and withdrew that result.

- **Gate bug.** Their escape rate was `np.mean(d > gate)`, and `d` is NaN when
  the predicted pixel is background. `NaN > 10` is False, so every landmark
  thrown clean off the head scored as a *success*. The arms do not miss
  equally (flip 0.744%, fusion 0.493%, none 0.422%), so the convention handed
  the flip arm roughly 140 free passes per 55,000. Re-cut with a miss counted
  as an escape, paired per ear over 40 ears: their anatomy cell 2.789 vs 2.796
  (tie, p=0.494), opposite 3.455 vs 3.765 (replacement, p=0.045), l2x2 sun
  +0.3 2.991 vs 2.942 (tie, p=0.928), sun -0.3 3.253 vs 3.620 (replacement,
  p=0.007).

So "replacement wins in all four" is dead. What survives is a **gradient**:
replacement wins in the worst-lit cell, weakly in the other poorly-lit one, and
the two best-lit cells are dead heats. That is in the *direction* of the
magnitude account rather than against it — the account I had declared dead on
the strength of the buggy table.

- **Renderer winding bug, which is worse.** Their right ears reach the canonical
  frame through a reflection (det = -1), applied to the head's vertices without
  reversing triangle winding, so vertex normals point *into* the head: +0.720
  on pp1_left against -0.720 on pp1_right, 93% of vertices inverted. The right
  ear renders markedly darker and flatter with the helix relief washed out. The
  depth buffer is unaffected so the lift was never wrong, **but the image the
  model saw was wrong on half their corpus throughout.**

**Everything in this document that rests on their left-versus-right numbers is
therefore provisional**, including the per-ear sun result quoted above and the
claim that presented handedness was excluded. A corrected 112-ear run is in
flight with winding as the only difference.

**Withdrawn:** the conclusion that the fusion-versus-replacement choice "does
not transfer across domains." Half of that contrast was the render side, and it
cannot currently be supported. What stands unaffected is the photograph side:
fusion wins here on every cut — thresholds, tails, and every arm-disagreement
decile, with the margin *growing* as the arms disagree:

| nme_normal, by arm-disagreement decile | fusion - replacement |
|---|---|
| 1 (most agreement) | -0.00012 |
| 5 | -0.00000 |
| 9 | -0.00074 |
| 10 (most disagreement) | **-0.00161** |

There is no renderer in this repository's loop, so nothing in their bug touches
the +8.53% or any measurement in this document taken on photographs.

## On a flip-consistency training penalty

> **EDITOR'S NOTE, 2026-10-19 — this section's conclusion no longer stands on
> its stated evidence.** The decomposition below ("only 0.63 of the 2.95 points
> is bias… the remaining 2.32 is ensembling") is the whole quantitative case
> against the penalty, and it is retracted later in this same document, under
> *Fusion's value collapses once the defect is fixed — and my decomposition was
> wrong*: replacement never removed the anti-equivariant offset, so the
> "ensembling share" was offset-cancellation under another name. That section
> was written later and this one was never revisited.
>
> What survives here is the qualitative objection — the penalty admits a
> degenerate direction, since a constant predictor is perfectly equivariant, and
> it costs a retrain — plus the framing argument in the next paragraph, which
> was and remains correct. What does **not** survive is the claim that an
> equivariant model "forfeits the 2.32": that number is void, and the split
> between bias removal and ensembling is now unmeasured.
>
> So treat the penalty as **open on evidence, not rejected.** The text below is
> left unaltered because this is a dated review; only this note is new.

Ear_3D_Lifter asked whether `||m(x) - mirror(m(mirror(x)))||` during training
would be better than fusion at inference, on the grounds that my own framing —
a soft constraint through samples cannot constrain the learned function — is
equally an argument for a loss term that constrains it directly.

The framing argument is right, and stronger than they allowed: augmentation
failing is weak evidence against a consistency penalty, because the two act by
different mechanisms. Augmentation symmetrises the *distribution*; a consistency
penalty penalises asymmetry of the *function* on each sample. The second is
strictly the stronger constraint, so p=0.5 augmentation failing says little.

But the decomposition settles it against the penalty, for this model:

| nme_normal, whole corpus | | |
|---|---|---|
| as-is | 0.01560 | — |
| best possible bias removal (canonicalise) | 0.01550 | -0.63% |
| fusion | 0.01514 | **-2.95%** |

Only **0.63 of the 2.95 points is bias**. The remaining **2.32 is ensembling** —
two decorrelated views of the same image, which no amount of function symmetry
provides. A perfectly flip-equivariant model, trained at whatever cost, captures
the 0.63 and forfeits the 2.32. It would land *worse* than the biased model with
fusion on top.

So the ranking is: fusion now (exact, measured, one forward pass); a consistency
penalty only if the 2x inference cost ever actually binds, and even then
expecting to give up most of the gain. The penalty also carries a degenerate
direction — a constant predictor is perfectly equivariant — so it trades
accuracy for symmetry unless carefully weighted. Not worth a retrain on this
evidence.

## Which fusion: coordinate, probability, or logit

Ear_3D_Lifter's arm table carries a `hm_avg` row distinct from `coord_avg`.
Chasing that produced one exact result and one small free win.

**Probability-space heatmap fusion is mathematically identical to coordinate
fusion, for this decoder.** Soft-argmax returns the *expectation* of the
spatial softmax, and expectation is linear, so averaging two probability maps
and decoding equals decoding each and averaging. Measured, it agrees to every
digit printed:

| nme_normal, whole corpus | | |
|---|---|---|
| as-is | 0.015599 | — |
| flip (replacement) | 0.015556 | -0.27% |
| coord_avg | 0.015139 | -2.95% |
| prob_avg | 0.015139 | **-2.95%, identical** |
| logit_avg | 0.015065 | **-3.42%** |

So "heatmap fusion" is only a distinct operation when done in **logit** space,
where the softmax nonlinearity makes it a geometric rather than arithmetic mean
of the two distributions. That sharpens: a mode only one arm supports gets
suppressed instead of averaged in. It is the better of the two, modestly and
consistently (-3.42% vs -2.95% whole corpus; -8.60% vs -8.53% on the
canal-LEFT rows).

This also accounts for Ear_3D_Lifter's `hm_avg` beating `coord_avg` in 2D
(26.23% vs 26.62% off-mask) on the same checkpoint: their `hm_avg` must be
logit-space, because probability-space could not have differed from
`coord_avg` at all. Two corpora, same ordering.

**Recommendation, qualified after Ear_3D_Lifter pushed back.** Logit fusion's
advantage is real in the *mean* and does not carry into the tail. Per visible
point on this corpus:

| threshold | coord_avg | logit_avg | |
|---|---|---|---|
| 0.10 | 4.325% | 4.380% | logit **worse** (McNemar 71 vs 45, p=0.02) |
| 0.20 | 0.368% | 0.349% | logit better, n.s. |
| 0.30 | 0.072% | 0.068% | logit better, n.s. |

Ear_3D_Lifter measured the same split more sharply: logit fusion put
significantly more landmarks on ear pixels (26.23% vs 26.62% off-mask,
p=6.3e-15) while being a wash at >10mm and significantly *worse* at >20mm
(p=0.006). Their mechanism is convincing — sharpening commits to one mode
instead of landing between two, which wins a pixel-level on-ear test and loses
by the full inter-mode distance when it commits wrong.

Our deep tail does not reproduce their reversal, but our two arms agree far
more closely (coordinate correlation 0.93) than theirs do under the stock sun,
which is exactly the regime where the choice should matter.

So: **use logit fusion for `nme_normal`, the selection metric, where it is
modestly and consistently better. Do not propagate it to tail-sensitive
downstream use without re-measuring in that regime.** The difference either way
is small; the flip fusion itself is the part that matters, and coordinate
averaging remains the safe default for a consumer that cares about gross
errors.

## Still unresolved: handedness vs illumination on photographs

Ear_3D_Lifter attempted a rim-shading light-direction estimator so the bias
measured here could be tested against estimated illumination direction. **It
failed calibration** — it tracks azimuth within a cell (R up to 0.81) but the
sign flips between ears, and across-cell R is 0.564. No light-direction number
for this corpus.

So the caveat raised at the start stands and is not closed: a mirror test cannot
separate shape handedness from illumination direction, because mirroring flips
both, and no instrument currently available can estimate the latter on
photographs. The +8.53% is robust as a *presented-handedness* effect. Whether
its underlying cause is ear shape or a corpus-wide lighting asymmetry is
unknown, and nothing in this exchange settled it. The fix is unaffected either
way, since flip fusion symmetrises both.

# Origin: a sub-pixel bug in the geometric augmentation

**This chapter replaces an earlier one that blamed the label normalisation
convention. That was wrong** — see "The false lead" below. The real cause is in
the augmentation, is confirmed by direct measurement of the augmentation code,
and matches the observed magnitude to within 6%.

## Initiates

`data/dataset.py` translation augmentation shifted the IMAGE by
`int(border + tx*S) - border` **whole** pixels — a crop can only move an image
by an integer — while moving the LABELS by exactly `tx*S`. `int()` floors, so
the two disagreed by the truncated fraction, always in the same direction.

Measured directly from the augmentation arithmetic over 200,000 draws at the
shipped `translation=0.05`, `S=192`:

```
image-minus-label displacement: mean -0.4692 px, sd 0.317, range [-1.00, +0.60]
                              = -0.002444 in normalised units
```

`_bbox_jitter` had the same defect by a different route: it handed a **float**
crop box to `PIL.Image.crop`, which truncates it, while computing the labels
from the un-truncated float.

## Propagates

A constant image-versus-label offset in the model's own input frame is, under a
horizontal mirror, an **anti-equivariant** offset: mirroring negates its x
component, so the as-is and flipped arms acquire equal and opposite shifts. The
horizontal flip is what converts a harmless constant registration error into a
handedness effect.

This also explains why x and y behave differently. There is no vertical flip, so
in y the offset stays a plain constant — which is why a constant `+0.0026`
y-correction *improves* accuracy on both test and val (`nme_plain` 0.034607 ->
0.034495, `nme_normal` 0.015566 -> 0.015397), while the same correction in x
does not.

## Sustains

Nothing pushes back. The loss is provably mirror-symmetric: `_contour_frames`
rotates the tangent 90 degrees so the normal flips sign under a mirror, but both
loss and metric take `.abs()` of the projection, and the spacing term is
distance-based. At `flip_prob=0.5` the model minimises expected loss by
splitting the difference and keeps the offset permanently.

## The evidence

Measured as `(r_asis - r_flip)/2 = (a - f)/2`, in which the target cancels, so
it is a property of the model alone and not of the labels it is scored against:

| model | dx_anti | t |
|---|---|---|
| manual_occ s1 / s2 / s42 | -0.00274 / -0.00176 / -0.00164 | -13.0 / -7.8 / -11.2 |
| fm_heat_pre s1 / s2 / s42 | -0.00296 / -0.00270 / -0.00261 | -25.2 / -21.8 / -22.3 |
| fm_pre_s42 / fm_scratch_s42 / v2_heatmap | -0.00151 / -0.00287 / -0.00139 | -11.5 / -15.2 / -7.8 |

Mean **-0.0023** against the augmentation's measured **-0.0024** — the same sign
in all nine, across three architectures, both initialisation schemes and three
seeds, which is what a shared data-pipeline defect predicts and what a learned
preference does not.

**Learned, not architectural.** Six untrained random inits of the same
architecture give a mean `dx_anti` of -0.000012 with mixed signs, about 100x
smaller and directionless.

**It is a translation.** As a displacement field between the two presentations
the difference is 94.9% affine and essentially a pure shift: |net| 0.004821
against mean|delta| 0.004900. A single constant x-correction collapses the bias
from **+8.53% to +0.86%**.

And it explains the fix: averaging the two arms cancels the anti-equivariant
component exactly, which is precisely the component at fault.

## The false lead, recorded because it was committed

`data/preprocess.py:192` normalises labels as `px/W` with no `+0.5`, while the
decoder reads pixel centres `(i+0.5)/w`. I took that for the root cause and
committed it in `1b4908c`. It is not, for the pool that matters: **`data/manual`
annotations are continuous sub-pixel coordinates** — mean fractional part of the
pixel value 0.2503, median 0.2503, zero exact integers, which is the signature
of uniform continuous positions. For a continuous `p`, `p/W` is correct and
`x -> 1-x` is the exactly-correct mirror. No mismatch, nothing to fix.

`data/preprocessed` is 54.3% exact integers, from COCO 1x1-keypoint sources, and
those rows may sit half a pixel off. That is a property of those source
annotations rather than of this code; it slightly inflates reported NME on that
eval split, and it **cannot** affect any handedness measurement, because a
constant label offset cancels both in `(a-f)/2` and in the paired flip delta.

## The fix, and its confirmation

Both defects are repaired in `data/dataset.py`: the translation now snaps to
whole pixels and moves the labels by exactly the integer shift the image
received, and `_bbox_jitter` builds its labels from the integer crop box PIL
actually applies. `LEGACY_SUBPIXEL_SHIFT` reproduces the old behaviour for
attribution, following the existing `LEGACY_ROTATION_SIGN` convention.

A controlled A/B retrain is in flight — same seed, same epochs, same data, the
flag the only difference — to confirm `dx_anti` collapses. Results below when it
lands.

## A second anti-equivariance source, in the inference path

Found by Ear_3D_Lifter reading `inference.py`, and confirmed here. This one is
independent of the augmentation defect and lives in deployment rather than
training.

- **`inference.py:302`** — `x1 = int(round(cx - side / 2))`. The as-is arm
  rounds `cx - s/2`; a mirrored frame rounds `W - cx - s/2`. Those agree only
  when the fractional parts happen to cooperate, so the rounding is itself
  **anti-equivariant**, up to half a pixel.
- **`refine_roi` defaults to `True`** and re-derives the ROI from the model's
  own landmarks for `ROI_MAX_REFINE` passes. Any asymmetry feeds back, and the
  two arms can settle on *different crops entirely*.

**This does not contaminate any measurement in this document.** Everything here
runs through `EarLandmarkDataset`, which resizes the stored crop directly and
never calls `square_roi_crop` or the refinement loop.

**It does bear on the recommendation.** Flip fusion is provably exactly
equivariant only when the two arms see exact mirrors. Through the deployed path
they do not. So fusion must mirror the **crop**, not the frame — crop once,
refine once, then mirror that tensor and run both arms — which is what the
measurements here already do by construction. Fusing by re-running the full
detect/refine/crop chain on a mirrored frame would reintroduce precisely the
asymmetry fusion exists to cancel.

**The two defects above are confirmed by reading this repository's own code and
do not depend on any measurement from Ear_3D_Lifter.** That matters, because
the numbers I previously cited here for scale have since been retracted at
source: their render call passed the view angles in a cfg dict that `render()`
ignores — the view is an extrinsic — so every "25-view grid" was the same
face-on render 25 times. Every per-view statistic they sent was therefore
inflated roughly 25-fold in effective n, including the scatter and
constant-vector figures this paragraph used.

What survives from their side is ear-level: taking their crop chain out moved
their mean `dx` from +0.00532 to +0.00131 over 12 ears. That still indicates the
crop chain carried most of what they were measuring, which is the only thing
this section needs it for. Their pixel-constancy figure is withdrawn separately
(see the unit-invariance section) and no longer supports anything here.

## Confirmation: the controlled A/B retrain

Same seed, same 60 epochs, same data, `LEGACY_SUBPIXEL_SHIFT` the only
difference. Measured on `data/preprocessed/test.csv` through
`EarLandmarkDataset`, so none of the inference-path asymmetry above is in the
loop.

| arm | epoch | val_nme | dx_anti | t | dy_anti | t |
|---|---|---|---|---|---|---|
| legacy | 51 | 0.0163 | **-0.002164** | -16.4 | +0.000127 | +1.45 |
| legacy | 54 | 0.0163 | **-0.001641** | -12.4 | +0.000159 | +1.80 |
| legacy | 58 | 0.0163 | **-0.002158** | -17.5 | +0.000135 | +1.52 |
| fixed | 52 | 0.0161 | -0.000837 | -4.9 | -0.000596 | -6.2 |
| fixed | 54 | 0.0161 | **+0.000102** | +0.6 | +0.000035 | +0.4 |
| fixed | 58 | 0.0161 | -0.000595 | -3.3 | -0.000145 | -1.5 |

**`dx_anti` collapses.** From a tight -0.0021 +/- 0.0003 with t around -15, to a
mean near -0.0004 whose **sign is no longer consistent** (positive at epoch 54).
That is what removing a systematic component and leaving noise looks like.

**`dy_anti` was already null and stays null** — +0.00014 (t~1.5) before, ~0
after. This is the mechanism's own prediction and a test it could have failed: a
constant y offset is *equivariant* under a horizontal mirror, so it appears
identically in both arms and cancels, while the x component negates. The defect
had to show in x and not in y, and it does.

**No accuracy cost.** The fixed arm is slightly *better*, 0.0161 against 0.0163.

### What the fix does not remove

`dx_anti` does not reach zero: epochs 52 and 58 are still significant (t -4.9,
-3.3). Stratified by per-sample NME quartile, the two components separate:

| quartile | legacy dx_anti | fixed dx_anti |
|---|---|---|
| Q1 (easiest) | **-0.00274** | -0.00027 |
| Q2 | -0.00226 | +0.00009 |
| Q3 | -0.00190 | +0.00016 |
| Q4 (hardest) | -0.00174 | **-0.00234** |

The bug-driven component is *largest on easy samples*, where a systematic offset
dominates and noise does not — and the fix removes it there almost entirely.
What survives in the fixed model lives almost wholly in the hardest quartile,
which is the signature of incidental non-equivariance on inputs the model
handles poorly, not of a systematic pipeline offset. That is a different effect.

**This no longer borrows any outside support, and an earlier version of this
paragraph did.** It read the Q4 concentration as agreeing with Ear_3D_Lifter's
reading that an out-of-domain input is where a network's incidental asymmetries
stop being suppressed. They have since run that reading's own falsifier and it
fired *against* them: their `dy` is flat across their Q1-Q3 (-0.0035, -0.0039,
-0.0027) and *vanishes* in Q4 -- the opposite of the quartile dependence the
reading predicts. So the external agreement is withdrawn. What remains is this
table alone, and the cynic audit's caveat on it stands: bootstrapped CIs for the
fixed model's Q4 overlap the other quartiles heavily, so the Q4 concentration is
suggestive and not established.

**So: the augmentation defect is confirmed as the origin of the systematic
component, and it is not the whole of the anti-equivariance.** The remainder is
input-dependent and does not have a pipeline explanation on this evidence.

### A unit-invariance check that independently places the origin

If the offset is created by the augmentation it is fixed in **192-space**, so it
should be roughly constant in normalised units and must *grow* in original-crop
pixels as the crop gets bigger. If it came from original-crop arithmetic, the
reverse. Legacy model, by crop-width tercile:

| crop width | dx_anti (normalised) | dx_anti (pixels) |
|---|---|---|
| 45-256 | -0.00166 | -0.196 |
| 256-319 | -0.00189 | -0.489 |
| 319-468 | -0.00275 | -1.034 |

Normalised varies 1.7x across the range; pixels vary **5.3x**, rising
monotonically (r=-0.295, p=9e-19 against width, versus r=-0.108 normalised).
The offset is far closer to constant in normalised units, which is the
augmentation's frame and not the crop's. In the fixed model both correlations
vanish (r=+0.035, p=0.3).

**The comparison with Ear_3D_Lifter needs scoping, and an earlier version of
this paragraph overstated it.** Their roughly-pixel-constant offset (1.21 px on
a 609 px crop, ~1.0 px on a smaller one) was measured through a path that still
ran `square_roi_crop` and a 1.6x re-crop. For *that* path the frame-arithmetic
reading holds and they have since diagnosed it: most of their `dx` vanished when
the crop code came out.

For their **model-only** measurement, with no crop code at all, neither unit
holds still: at 6 ears, pixels and normalised have essentially the same relative
spread (0.789 vs 0.802), and the correlation with crop width is r=+0.202 in
pixels against r=-0.003 normalised — if anything the *normalised* one is flatter,
which would point toward this frame rather than theirs. At n=6 none of that
decides anything. So the honest statement is that their residual's frame is
**undetermined**, not that it differs from this one.

What stands unaffected is the result above, which is measured entirely on this
corpus: the systematic component here is near-constant in the augmentation's
192-space frame and grows 5.3x in crop pixels, and it disappears with the fix.

## Cynic audit

Three claims re-tested. One gained an independent confirmation, two were
overstated.

### Strengthened: the equivariant y-channel also collapses

The mechanism predicts the defect shifts **y as well as x**, but equivariantly —
so it is invisible to `dx_anti` and must instead appear as a plain net y-offset
that the fix removes. Not previously tested. It holds:

| arm | net signed dy residual | nme_plain at cy=0 | at cy=+0.0026 |
|---|---|---|---|
| legacy | **-0.002743** | 0.036023 | **0.035832** (helps) |
| fixed | **-0.000386** | 0.036193 | 0.036219 (no longer helps) |

The legacy net y-offset is -0.00274 against the augmentation's directly measured
-0.002444, and it falls 7x with the fix, after which the y-correction stops
helping. This is a second channel, predicted by the same mechanism, measured
independently of `dx_anti`, and it agrees.

### Overstated: "no accuracy cost — the fixed arm is slightly better"

That used `val_nme` on `data/manual` (0.0161 vs 0.0163). On
`data/preprocessed/test.csv` the ordering reverses: 0.036193 fixed against
0.036023 legacy. Both gaps are small and single-seed. The defensible claim is
**no measurable accuracy cost in either direction**; quoting the split that
favoured the fix was cherry-picking.

### Overstated: the residual is "a different effect" in Q4

Bootstrap, 4000 resamples:

| arm | Q4 dx_anti | 95% CI |
|---|---|---|
| legacy | -0.001742 | [-0.002532, -0.000873] |
| fixed | -0.002399 | [-0.003904, -0.001143] |

The intervals **overlap heavily**: the fixed model's hardest quartile is not
distinguishable from the legacy model's. So the honest reading is that the fix
removes the offset on easy and typical samples and does **nothing measurable on
the hardest quartile** — not that a separate effect was isolated there. The
quartiles are also defined by each model's *own* error, so the two Q4 sets are
different samples, which the comparison does not control for.

### Not tested, and it weakens the quantitative match

The fix changed **two** things: the translation snap and the `_bbox_jitter`
integer crop box. Only translation was measured directly (-0.4692 px).
`_bbox_jitter` runs at p=0.5 and contributes an unmeasured amount in the same
direction, so the predicted total should exceed -0.0024 — yet measured
`dx_anti` is -0.0023. The agreement I called a quantitative match therefore
rests on `_bbox_jitter` contributing near zero, which was never checked. The A/B
confirms the pair of fixes together; it does not apportion between them.

### Also single-seed

One seed, one architecture. The nine-model survey gives the *sign* broad
support, but the collapse itself is demonstrated once.

## Fusion's value collapses once the defect is fixed — and my decomposition was wrong

Ear_3D_Lifter named the deciding measurement: fusion against replacement against
neither, on the **fixed** checkpoint. Matched A/B pair, epoch 58 each.

| | as-is | fusion | vs as-is | replacement |
|---|---|---|---|---|
| **legacy** (dx_anti -0.00216) | 0.016526 | 0.016143 | **-2.32%** | 0.016371 (-0.94%) |
| **fixed** (dx_anti -0.00060) | 0.016371 | 0.016260 | **-0.68%** | 0.016350 (-0.12%) |

`nme_normal`. **Fusion's margin falls by 70%** once the augmentation defect is
gone. It stays statistically significant (p=2.4e-22, better on 65% of samples)
but it is now worth 0.68%, not 2.32%. In the deep tail it reverses: gross-error
rate past 0.2 is 0.622% fused against 0.514% as-is on the fixed model.

**This refutes a prediction I made and corrects an error.** I had decomposed
fusion's gain as "0.63 points bias removal, 2.32 points ensembling", by treating
canonicalisation as the best-possible bias removal and attributing the remainder
to ensembling. That reasoning is wrong: **replacement does not remove the
anti-equivariant offset at all.** It picks one arm, and that arm still carries
`+d` or `-d`. Only fusion cancels it, by averaging `+d` with `-d`. So the
"ensembling share" was mostly offset-cancellation wearing another label, which
is exactly why removing the offset by retraining collapses the margin.

The honest reading is Ear_3D_Lifter's: **fusion was largely a workaround for the
augmentation defect, and the fix supersedes most of it.** What remains —
0.68%, significant, on 65% of samples — is the genuine ensembling residue.

### What this changes about retraining

It strengthens the operational case I previously underweighted. Retraining does
not beat `legacy + fusion + y-correction` on accuracy, but it **removes the need
for fusion**: a fixed model at 1x inference lands within 0.7% of its own fused
version, where the legacy model needs 2x inference to close a 2.3% gap. The
value of retraining is not a better number, it is **a model that does not depend
on an inference-time correction to be unbiased.**

For the shipped `manual_occ_s42` specifically, the earlier recommendation stands
on its own numbers (fusion + y-correction, -4.2% `nme_normal`, bias zero). For
anything trained from here, fusion should be measured rather than assumed — on a
clean model it may not earn its keep.

## The converged fixed model, and a retraction of the 70% figure

The A/B arms in the section above were 60-epoch probes run with default
augmentation, `perspective_deg=0`. The shipped model was not: at matched epochs
50/54/58 it sits at val_nme 0.0175/0.0169/0.0175 against the A/B's
0.0165/0.0163/0.0163, worse early, which is what stronger augmentation does. So
the A/B was internally matched but measured on a weaker-augmentation model than
the one we ship. `fixed_conv_s42` closes that: the shipped recipe
(`data/manual`, heatmap, perspective 65, seed 42), 500-epoch cap, the fix the
only difference. Early stopping fired at epoch 483, best 0.01452 at 433 against
the shipped 0.01487 at ~292.

### The handedness bias is gone, and this is the measurement that shows it

Paired within row -- for each row, flip-arm `nme_normal` minus as-is-arm
`nme_normal`. Pairing is not a refinement, it is the difference between seeing
the effect and not: an unpaired subgroup-against-global-mean test on this same
data returns |t| < 0.7 for every cut, because row-to-row difficulty swamps a
4-8% shift. An earlier version of this measurement was unpaired and was
discarded for that reason.

audioear2d, the only handedness-balanced source (154 R / 146 L):

| cut | legacy paired delta | p | fixed paired delta | p |
|---|---|---|---|---|
| right ears | **+9.32%** | 5.5e-12 | -0.56% | 0.60 |
| left ears | **-7.73%** | 7.3e-09 | -1.57% | 0.078 |
| pooled | +0.57% | 0.57 | -1.08% | 0.12 |
| R-L antisymmetry | **+17.05%** | **2.7e-19** | **+1.01%** | **0.46** |

**CORRECTED: the side labels in an earlier version of this table were swapped.**
They came from the sign of the rim polyline's signed area, whose polarity was
assumed and assumed backwards. The anatomical rule decides it: the canal (point
39) is anterior, so `canal_x > rim_centroid_x` is a right ear; that agrees with
the signed-area test on 860/861 rows and was checked against the images. Every
magnitude and p-value here is unchanged -- an antisymmetry statistic is
invariant to the swap, which is why nothing flagged it -- but the direction
was wrong. The legacy model favoured RIGHT-presented ears. The counts are
146 right / 154 left.

The antisymmetry falls **94%**, from p=2.7e-19 to not significant. The pooled
row is the trap this investigation began with, visible in one line: +0.57% and
ns, while its two halves are +9.3% and -7.7% at p<1e-8.

`dx_anti` agrees but far less sharply: -0.001618 (t=-11.0) to -0.000431
(t=-2.49) on all 861 rows, a 73% fall. The paired side test is the stronger
instrument and should be the one quoted.

### What this retracts

**The "70% collapse" in fusion's margin does not survive at convergence, and
the conclusion built on it is withdrawn.** That figure came from the 60-epoch
weak-augmentation A/B. On converged models with the shipped recipe:

| | legacy (shipped) | fixed (converged) | collapse |
|---|---|---|---|
| fusion vs none | -2.95% (p=3.8e-06) | **-1.83% (p=0.009)** | 38% |
| fusion vs replacement | -2.67% (p=1.9e-05) | **-2.35% (p=1.7e-03)** | 12% |

Fusion still pays on the fixed model, significantly, in this domain. So the
sentence this review previously endorsed -- that "fusion was largely a
workaround for the augmentation defect" -- is **not supported on converged
models** and is withdrawn. It was true of the 60-epoch pair and only of them.

Ear_3D_Lifter's independent tail result pointed the same way before this was
measured: on their >10 mm gate, fusion against doing nothing *grew* on the fixed
arm, +0.342pp to +0.497pp. Their fusion-vs-replacement collapse of 79% was
offered as agreeing with the 70%; with the 70% withdrawn, the counterpart to
that agreement is a 12%, and the two no longer correspond.

### Their flag, which stands

On Ear_3D_Lifter's gross-error gate the fixed checkpoint is **worse as-is** than
the legacy one, +0.275pp at p=3.2e-04, while the flipped arm improves by about
as much and fusion is unchanged. In this domain the fix is an improvement
(audioear2d `nme_normal` 0.00752 to 0.00701), so the two are not in conflict --
but the fixed checkpoint is not a strict upgrade for every consumer, and a
release note should say so.
