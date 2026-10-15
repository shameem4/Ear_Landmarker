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
| **T3** | Same-size crop **shifted** laterally | not started | the expand sweep tested *concentric* scaling, which pays resolution everywhere. A shift pays nothing. On the 31 escaping ears, a shift <=0.1 crop-sides recovers **32.3%** and <=0.5 recovers **58.1%**, all geometrically possible. **Open problem: predicting the direction** |
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
| L3 | Recover the face keypoints | **4 lines in BlazeEar, 0 here** -- see below. Only worth doing if T3 finds a direction predictor |

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
keypoint and report nonsense scores"*. So the one real risk in this change is
step 2, and it is a risk the project has already paid for once.

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
