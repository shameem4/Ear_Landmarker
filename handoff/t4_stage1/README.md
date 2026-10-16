# T4: a better stage 1 improves coverage and delivers nothing

**Handoff from Ear_Landmarker to BlazeEar, 2026-10-15.** This is a detection-side
result and BlazeEar owns stage 2, so the dissection belongs there. Everything
below was measured on BlazeEar's corpus with BlazeEar's ground truth; the
Ear_Landmarker side only ran the harness.

**The one-line version.** `trainable_blazeface`'s fine-tuned stage 1 is a real
improvement in face coverage -- ears that no crop covered fall by up to 60% --
and converting that coverage into delivered ears at matched precision yields
roughly nothing, because the extra ears arrive as boxes the 0.70 stage-2
threshold discards. **Stage-1 coverage is not the bottleneck. Stage-2
calibration is, and it now gates a larger prize than when it was measured
alone.**

---

## Why T4 was reopened

T4 (re-target stage 1) was closed on `trainable_blazeface`'s report that no
checkpoint beat frozen MediaPipe across six configurations. **They retracted
that.** All six used a cell-based anchor encoder that assigned every
ground-truth box to both grids. With a proper per-anchor assignment plus an
**ignore band** -- anchors that overlap a face but lose the top-k race leave the
positive set *and* hard-negative mining, rather than training as background that
mining then specifically hunts -- fine-tuning clears the gate this side had set:

| | precision | recall | 8-16 px | 16-32 px | 32+ px |
|---|---|---|---|---|---|
| stock @0.50 | 0.9380 | 0.4372 | 0.078 | 0.862 | 0.973 |
| tuned @0.50 | 0.9649 | 0.5011 | 0.402 | **0.934** | **0.980** |

Gate was 0.854 at 16-32 px and 0.966 at 32+ px. Cleared on both, at better
precision. That is **their** measurement on **their** face corpus; the rest of
this document is ours on ours.

Their verification, which is the part that made it worth acting on: rescored
against YuNet, a third architecture neither model trained on, and the gain
*grew* (+0.071 recall vs +0.063 against their own RetinaFace teacher). A
teacher-bias artefact shrinks when the teacher is removed. They also stated the
confound rather than burying it -- the advantage grows at tight IoU, the
signature of reproducing a box convention -- so everything below is scored at
**IoU >= 0.30**, well below that regime.

## Method

Their `BlazeFace_final.pth` loads into the stage-1 model our pipeline already
instantiates: **78 tensors, `strict=True`, 0 missing, 0 unexpected, 0 shape
mismatches.** No porting. The harness asserts the loaded weights actually differ
from stock, because a silent no-op load and "no effect" look identical.

* **Corpus.** All 13,007 images, **15,359 ears**.
* **Ground truth.** `master.csv` human sources only (`GT`, `GT+EAR`,
  `GT+REVIEW`) with the geometry filter. **No POSE rows** -- 39.9% of the file,
  and 0.0% of them reach IoU 0.30 with any human box in the same image.
* **Path resolution.** `data/yolo11_ears/` no longer exists;
  `data/yolo11_ears_human/` does, nested a level deeper by source dataset.
  Resolution is by basename, **asserted** unique rather than assumed (13,007
  files, 13,007 unique names, 0 collisions).
* **Pairing.** Both arms run in ONE pass over the same images, so the
  denominator is bit-identical and every ear is paired. Significance is
  **McNemar on discordant pairs**, not two independent proportions.
* **Bands.** The ground-truth ear box's max side in the **original image**, not
  size-in-crop -- an ear no crop covered *has* no size-in-crop, and those are
  precisely the ears under test, so size-in-crop banding would silently drop the
  population being measured. **0-32 and 32-48 are never pooled.**

## Result 1: coverage improves, unambiguously

Ears that no crop covered, out of 15,359. This is a stage-1 property; no
stage-2 threshold can touch it.

| stage 1 | face-stage misses | vs stock |
|---|---|---|
| stock | 1301 | -- |
| tuned, dropped in at expand 1.5 / face thr 0.2 | 908 | **-30%** |
| tuned at matched crop budget (face thr 0.28) | 1723 | +32% |
| **stock crops UNION tuned crops** | **516** | **-60%** |

## Result 2: delivered recall at matched precision does not

Each arm's recall interpolated onto stock's precision at each of stock's
operating points.

| matched precision | stock recall | union | tuned (matched crop budget) |
|---|---|---|---|
| 87.28% | 60.65% | -2.44 | -- |
| 85.67% | 64.83% | -2.30 | -- |
| 84.01% | 68.24% | -1.68 | -4.97 |
| 82.59% | 70.96% | -1.65 | -4.93 |
| 81.30% | 73.01% | -1.57 | -4.95 |
| 79.99% | 74.71% | -1.36 | -4.58 |
| 78.52% | 76.08% | -0.94 | -3.98 |
| 77.14% | 77.24% | -0.44 | -3.70 |
| 75.78% | 78.12% | **+0.23** | -3.22 |
| 74.52% | 78.96% | **+0.54** | -3.06 |

The union's headline is **+4.81 pp** at the shipped threshold, gaining in every
band, p=3.3e-188. **Almost all of it is a precision trade.** At matched
precision it is negative at every operating point except the two loosest, where
it is +0.23 and +0.54 pp -- one seed, and smaller than the spread this corpus
shows on far cruder changes.

## Result 3: the one configuration-invariant gain is 0-32 px

Delta in band recall, tuned (or union) minus stock, at the shipped 0.70:

| band (px, source image) | n | drop-in | expand 1.364 | matched crops | union |
|---|---|---|---|---|---|
| **0-32** | 2091 | **+11.76** | **+10.33** | **+5.74** | **+16.79** |
| 32-48 | 2295 | +0.87 | +3.14 | -3.05 | +6.23 |
| 48-64 | 2259 | +0.75 | +1.37 | -2.48 | +4.16 |
| 64-80 | 1902 | -0.63 | +0.37 | -2.89 | +3.21 |
| 80+ | 6812 | -1.35 | -0.66 | -2.99 | +1.32 |
| **pooled** | 15359 | +1.17 | +1.83 | -1.73 | +4.81 |

0-32 px is positive in all four configurations, each at **p < 1e-08**
(6.8e-31, 7.5e-25, 3.6e-09, 2.2e-98). A better stage 1 genuinely finds small
faces stock misses, and that survives every way the constants are denominated.

## Two constants were silently denominated in stock's units

This is the transferable lesson and the insight came from
`trainable_blazeface`.

### Geometry: `FACE_CROP_EXPAND`

Their boxes are linearly larger on the same face. Measured on **our** corpus:
**1.0998**, bootstrap 95% CI **[1.0936, 1.1048]** over 2,486 faces both
detectors find at IoU >= 0.5. (They measured 1.117 on theirs, which falls
*outside* our interval -- the convention ratio does not transfer exactly between
image distributions.)

`make_face_crops.py:66` sets `side = expand * max(box_w, box_h)`, so
`FACE_CROP_EXPAND = 1.5` silently becomes an effective **1.65**.

Their prediction was that restoring `1.5 / 1.0998 = 1.364` would recover the
loss. **It does not -- it makes both arms worse:**

| expand | stock | tuned |
|---|---|---|
| 1.5 | 79.09% | 80.26% |
| 1.364 | 77.52% | 79.35% |

A smaller window loses ear coverage faster than it gains in-crop ear size:
face-stage misses rise in every band (stock 0-32: 600 -> 662; 32-48: 245 -> 318;
80+: 203 -> 395). **So the units observation is right and the correction is
wrong, and the 1.5 optimum is not about effective scale relative to the face
box** -- which means the mechanism previously given for that sweep
("gains coverage, shrinks the ear in the crop") is at best incomplete.

One part does survive where it was not claimed: the 80+ px regression shrinks
from -1.35 pp (p=1.1e-07) to -0.66 pp (p=0.033) at 1.364. **The large-ear part
of the loss is partly geometry; the small-ear gain is not.**

### Score: `FACE_CROP_THRESHOLD` -- and this one went the other way

The same error on the axis neither side checked. At the shared
`FACE_CROP_THRESHOLD = 0.2` their model hands stage 2 **6.16 crops per image
against stock's 1.84**, and their all-emission score median is 0.234 against
stock's 0.710.

**An initial reading of that -- that their scores are compressed against the
floor, so no threshold separates their true extra faces from their marginal
ones -- is wrong, and was wrong twice over.** It is worth recording how,
because the correct version is the most useful thing in this document.

`trainable_blazeface` pointed out that a median over all emissions is a
**mixture statistic**, dominated by a long low-score tail; split by
correctness, their true positives sit *higher* than stock's (0.683 vs 0.578)
and their true/false separation is *better* (0.466 vs 0.330). On their face
corpus at a matched crop budget they dominate stock on both axes
simultaneously -- +7.3 pp face recall at +9.6 pp face precision -- at every
shared threshold from 0.30 up. Their conclusion was that the constant needs
moving, not the model.

**So the threshold was swept here, on delivered ear recall. It is monotone, and
it runs the opposite way from their proposal.** Raising the face threshold
destroys ear coverage; lowering it improves coverage monotonically. Ears no
crop covered, out of 15,359:

| tuned face threshold | 0-32 | 32-48 | 48-64 | 64-80 | 80+ | **total** |
|---|---|---|---|---|---|---|
| 0.10 | 233 | 93 | 71 | 41 | 92 | **530** |
| 0.15 | 246 | 96 | 80 | 48 | 161 | **631** |
| 0.20 (shipped) | 317 | 144 | 119 | 72 | 256 | **908** |
| *stock @0.20, for reference* | *600* | *245* | *163* | *90* | *203* | ***1301*** |
| 0.28 (matched crop budget) | 627 | 305 | 231 | 138 | 422 | **1723** |
| 0.30 (their suggestion) | 689 | 334 | 255 | 151 | 454 | **1883** |
| 0.35 | 828 | 401 | 303 | 174 | 511 | **2217** |
| 0.40 | 918 | 447 | 337 | 193 | 560 | **2455** |

And delivered pooled recall against stock's 79.09% tracks it exactly: **+2.82
pp** at 0.10, +2.34 at 0.15, +1.17 at 0.20, -1.73 at 0.28, -2.35 at 0.30,
-3.72 at 0.35, -4.66 at 0.40. At 0.10 their detector alone beats stock in
**every** band and roughly matches the crop union (530 misses against 516) at
half the detector compute.

**Which resolves it: their low-score tail is load-bearing for this objective.**
The detections their own face-precision metric counts as false positives are
disproportionately the crops that contain the ears this pipeline needs. The two
metrics disagree **by construction**: our cost function does not care whether a
crop is a true face, only whether it contains an ear, so a profile head, a
partial face, or anything their face ground truth misses can be a worthless
detection by their metric and a necessary crop by ours.

**Both readings were measuring the wrong quantity.** Theirs (clean separation,
so move the constant up) is right about face detection and does not transfer.
The initial one here (compressed scores, so the model needs recalibrating) was
wrong about the distribution *and* would have prescribed the same harmful
direction. Neither survives contact with delivered ear recall, which is the only
metric that decides this.

**The useful form of the units lesson is therefore the strong one:** every
constant fitted against stock -- `FACE_CROP_EXPAND`, `FACE_CROP_THRESHOLD`, and
by extension the stage-2 threshold itself -- is denominated in stock's units,
and **the optimum for a swapped detector cannot be derived from either model's
own metrics.** It has to be swept against the downstream objective. Both
attempts to shortcut that by reasoning from a score distribution failed, in
opposite directions.

## It still does not convert at matched precision

With the face threshold swept to its best setting, each arm interpolated onto
stock's precision at each of stock's operating points:

| matched precision | stock recall | tuned @ face 0.10 | union |
|---|---|---|---|
| 90.83% | 46.90% | -4.35 | -- |
| 87.28% | 60.65% | -7.07 | -2.44 |
| 84.01% | 68.24% | -6.36 | -1.68 |
| 81.30% | 73.01% | -5.98 | -1.57 |
| 78.52% | 76.08% | -5.22 | -0.94 |
| 77.14% | 77.24% | -4.76 | -0.44 |
| 75.78% | 78.12% | -4.23 | **+0.23** |
| 74.52% | 78.96% | -3.85 | **+0.54** |

The best coverage arm is the **worst** matched-precision arm: tuned @0.10 wins
every band on raw recall and loses 4-7 pp once precision is held fixed, because
the crops that carry its extra ears also carry extra non-ear scene. The union
remains the least-bad and is still only neutral.

**Nine configurations** -- two expand values, seven face thresholds, and the
crop union -- and the verdict does not move. That is as settled as one seed can
make it.

## What the low-score tail actually is: an implicit head detector

The tail matters because it is where the coverage lives, so both sides tried to
explain it and **both first explanations were wrong.** This section is the
corrected account, and it is the part most likely to be worth acting on.

**My explanation, refuted by them.** I claimed their face-precision metric was
mislabelling real faces -- that their tail was full of faces their RetinaFace
ground truth missed. They tested it against YuNet, a third architecture neither
model trained on: of **39,822** tuned detections between 0.10 and 0.20 that miss
their ground truth, YuNet confirms **19** as faces. **0.0%.** At 0.20-0.30 it is
1.6%, against stock's 1.5% and 6.8% in the same bands. Those boxes are not
unlabelled faces by any face criterion available. **A crop does not have to BE a
face to CONTAIN one.**

**Their follow-up question, which only an ear ground truth can answer:** if the
tail's worth is not facefulness, is it placement -- do those boxes cluster on
heads, or scatter?

**My first answer to it was also wrong, and the error is instructive.** Scoring
each tail crop against a null that moved its centre uniformly at random while
keeping its size gave 12.72% covering an ear against a null of 12.81% -- lift
1.0x, p=0.34, apparently pure chance. That contradicted the arm's own
face-stage-miss count, and the contradiction was the tell. **The per-crop
statistic has no power:** on a head-filling photograph a same-sized box dropped
anywhere hits an ear about 13% of the time by default, so the rate saturates and
cannot distinguish a detector from a dart. The statistic with power is *unique
ear coverage*, which is what the face-stage-miss count measures.

**Re-run as controls that destroy one thing at a time** (3,000 images, 3,550
ears, stage-2 threshold 0.70, expand 1.5):

| arm | face-stage misses | recall @0.70 |
|---|---|---|
| tuned @ face 0.10 | **110** | 81.80% |
| stock @ face 0.20 | 312 | 78.76% |
| **placement destroyed, each image's own box sizes kept** | **1026** | 52.82% |
| placement *and* scale destroyed (sizes from a global pool) | 1417 | 37.15% |

Randomising only *where* the boxes land, while keeping exactly how many and how
big they are per image, takes face-stage misses from 110 to **1026** -- a
factor of nine. Destroying scale as well adds the rest.

**So the tail's placement is strongly informative, and "it is just volume" is
dead too.** 35 windows per image placed at random recover a third of the ears;
the same 35 placed by the model recover 82%. The tail is not a shotgun and it is
not a face detector. On this corpus **it behaves as an implicit head detector**:
boxes that no face metric will credit, landing reliably on ear-bearing regions.

**Which makes their proposed lever the live one, and it is a different stage 1
from the one T4 was about.** If what the pipeline needs is head-ness rather than
face-ness, then the objective to train is head-ness -- and their tuned model's
low-score tail is already an accidental, uncalibrated version of it. That is
worth more than re-targeting stage 1 at an ear-inclusive box, because it is
supported by a measurement rather than by an assumption about what a face
detector ought to find. It is **also still gated on stage 2**: a better head
detector delivers more crops of exactly the small marginal heads stage 2
under-scores.

## What this hands to BlazeEar

**The lever moves to stage 2 and is now worth more than when it was measured
alone.** The coverage exists and cannot be converted, because the extra ears
arrive as boxes the 0.70 threshold discards -- the same binding constraint
already measured independently on this corpus, where **90.2% of stage-2 misses
are correctly localised boxes scored around 0.40** (median 0.404, p25 0.274,
p75 0.570), and what gets under-scored is small ears.

So the two results are the same result seen twice:

1. Stage 2 under-scores small ears.
2. A stage 1 that finds more small faces delivers nothing, because stage 2
   under-scores small ears.

**Recommended order.** Fix stage-2 scoring first. Revisit stage 1 only after,
at which point there is a lot sitting unconverted: **+16.79 pp of raw 0-32 px
coverage** for the union, or face-stage misses down from 1301 to 530 for the
tuned detector alone at face threshold 0.10 -- which costs no extra detector
pass at all, only a constant.

Note what that means for sequencing: the cheapest stage-1 win available is
**not** a new model, it is `FACE_CROP_THRESHOLD`, and it is only worth taking
once stage 2 can score what it delivers. Until then it buys coverage the
threshold throws away.

**What is worth dissecting here, and why it needs BlazeEar's context rather
than ours:**

* `evidence/rescued_union.json` -- 767 ears the union delivers and stock does
  not, with band and source image. **Are these ears worth having?** This side
  cannot judge whether they are the ears the product needs; BlazeEar can.
* Whether stage 2's precision is the right cost function at all. Per the
  existing note in Ear_Landmarker's README: every labelled frame in the YOLO
  export carries exactly one ear, so a correct second detection on a frontal
  face scores as a false positive. **Precision here is a lower bound of unknown
  looseness**, and if it is loose, every "matched precision" row above is
  pessimistic by an unknown amount. That is the single assumption most likely to
  change the verdict, and resolving it needs the labelling convention, which is
  BlazeEar's to know.

  **And the penalty is not symmetric**, which `trainable_blazeface` pointed out
  and I had not: it falls hardest on the arm that delivers more crops containing
  more ears, which is theirs at a low threshold. So the configurations that lose
  on the matched-precision table are precisely the ones a one-ear-per-image
  convention would penalise most. This does not rescue their arm and they did
  not argue that it does -- the bottleneck finding rests on stage 2
  under-scoring correctly localised boxes and does not touch the precision
  column at all. But **do not read the matched-precision rows as settled until
  this is answered.**
* Whether the 80+ px regression matters. It is small and partly geometric, but
  80+ px is 44% of the ears.
* One seed throughout. Nothing here is seed-replicated; the effects are large
  relative to the usual spread on this corpus, but that is an argument, not a
  measurement.

## Re-running

Harnesses are in Ear_Landmarker `scripts/`, not scratch, because they are
reusable. About three minutes per full-corpus pass on this box.

```bash
# paired two-arm scoring; --cascade scores the crop union instead of tuned alone
python scripts/score_face_arms.py --limit 13007 --seed 1 \
    --expand-stock 1.5 --expand-tuned 1.5 \
    --face-thresh-stock 0.2 --face-thresh-tuned 0.2 \
    --thrs 0.80,0.70,0.60,0.50,0.40,0.30

python scripts/scale_ratio.py       --limit 2000 --seed 7   # box convention ratio
python scripts/calib_face_thresh.py --limit 2000 --seed 7   # matched crop budget
```

Point `--ckpt` at any later checkpoint. If B1 or a successor lands, this table
regenerates without ceremony.

`evidence/` holds the raw stdout of every run quoted above:
`full_seed1.log` (drop-in, face thr 0.2), `full_calib.log` (expand 1.364),
`full_fthr.log` (face thr 0.28, matched crop budget), `full_ft0.10.log` /
`full_ft0.15.log` / `full_ft0.30.log` / `full_ft0.35.log` / `full_ft0.40.log`
(the face-threshold sweep), `full_union.log` (crop union), `full_fine.log` and
`full_ft010_fine.log` (fine stage-2 sweeps for the matched-precision
interpolation), `ratio.log` (box-scale ratio), `calibthr.log` (crop budget).

`rescued_union.json` and `rescued_tuned_dropin.json` are per-ear manifests of
what each arm delivers and stock does not, with band and source image path.

## Correction recorded in the other direction

`trainable_blazeface` claimed the min-16 face-box training replication run from
the Ear_Landmarker side had inherited their defective assignment. **They were
right.** That run invoked *their* `train_blazeface.py` with their defaults, so
it went through their dataloader and the cell-based encoder. A separate scoring
test, which consumed their weights alone and never touched the encoder, was
unaffected -- and that is the one the initial denial was about. Both tests
existed; the denial answered the wrong one.
