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
| **B2** | Turn focal loss on | flag flip | already implemented and unused (`use_focal_loss=False`, alpha 0.25, gamma 2.0). Downweights easy negatives, which is exactly the small-anchor background |
| **B3** | Per-level hard-negative cap | small change to `loss_functions.py` | `HARD_NEGATIVE_RATIO = 50` mines globally across all 3584 anchors, so the 2048 slots holding the two smallest priors absorb the bulk of mined negatives every image |
| **B4** | Lower the score threshold 0.70 -> 0.60 | **no training** | +2.81 pp recall for -6.94 pp precision. An operating-point decision, not a free win |
| **B5** | Expose a calibrated score to the caller | small | better shape than moving a default, since consumers differ on the precision/recall trade |

**Root cause for B1-B3**, measured on 12,595 training ears: the anchor budget is
misallocated by scale. 2048 of 3584 slots carry the two smallest priors and
receive 7.7% of all positives -- 0.29 and 2.54 positives per slot against **68.71**
for the prior serving the 64-80 px mode. Small ears are *not* starved of positive
supervision (`ANCHOR_TOP_K = 3` gives exactly 3.00 per ear at every band); the
problem is the negative side.

**Guard on every run:** report the 64-80 px band, mean positives-per-ear, and the
per-prior positive share. 1402 of 3790 ears sit in the mode, so a run that
improved the tail by starving the mode would show there before it showed in recall.

---

## Open: trainable_blazeface, stage 1

**T1 is a precondition for everything else here.** The fine-tuned checkpoint
currently recalls 25.3% against frozen MediaPipe's 43.7% at matched ~95%
precision, so re-targeting would be fine-tuning a pipeline that has not beaten
its own starting point, and a null result would be uninterpretable.

| # | fix | status | notes |
|---|---|---|---|
| **T1** | v2 retrain to match MediaPipe | **running** | v2 code at 21363eb, data held fixed so it isolates the code fixes. Bar: 43.7% recall at ~95% precision on 8,081 val images |
| **T2** | Merge overlapping face boxes at **label** time | not started | a label-definition change, no model work. Yields one box per head -> **-45.8% of stage-2 calls**. Testable on this repo's harness before any training |
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
