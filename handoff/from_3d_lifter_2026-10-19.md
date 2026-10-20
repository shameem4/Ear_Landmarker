# Handoff from Ear_3D_Lifter, 2026-10-19

The 3D side (head mesh + this repo's 2D model -> 3D ear landmarks) is in good
shape and is no longer where the error is. **Further accuracy has to come from
the 2D model.** This note gives the evidence, the leads, and how to measure a
new checkpoint end to end without moving the goalposts.

## Why the 2D model is the bottleneck

- **Oracle ablation:** with perfect 2D landmarks, the 3D lift adds only
  **0.36 mm**, and it never misses by 20 mm. Every gross error traced back to
  the 2D prediction.
- **Clean held-out end to end: 1.58 mm median** per ear (55 points). It was run
  on 16 HUTUBS heads that were never used to build the view table or the
  reference landmarks. Each head was given a fresh random rotation and passed
  in unskinned.
- **Re-fitting the 3D corpus bought nothing in accuracy.** Scored on one fixed
  reference, the new 3D pipeline matches the old one (1.574 vs 1.539 mm). The
  work made the measurement trustworthy, not the pipeline better.
- **On a real head scan** (local, private), the result is stable across
  orientations: a median spread of 0.1–0.2 mm. The user judged it improved
  over the 17 October run and counts it as partial validation.

## Leads, in order of payoff

1. **Upper helix, points 0–8: misses ACROSS the rim.** This is the largest
   cluster of points off by more than 5 mm on held-out heads (4.0% of points).
   - The error is perpendicular to the curve, 1.5–2.2 mm at points 2–5. The
     along-curve offsets average about zero and go both ways, so this isn't a
     spacing or numbering slide.
   - Worst ears: pp16, pp40, pp46, pp70, pp78.
   - Part of the excess comes from the new reference landmarks (lm3d_v11),
     which sit differently on the rim there. The rest is the prediction.
2. **Lobe, points 15–19: unstable.**
   - On the real scan, point 16 moves up to 6 mm between head orientations,
     while the other points move a median of 0.1–0.2 mm.
   - The lobe is also where 3D fits clip; the scans often have the lobe torn
     or missing.
3. **The scan's photo texture is ignored.** The 3D pipeline re-skins every
   mesh with the same procedural skin before rendering, so a real scan's photo
   texture has no effect: textured and stripped runs are identical.
   - That suits a model tuned on those renders.
   - If you train on textured or photographic input, this is the place to plug
     it in (`ear3d/pipeline.py`, the skinning step).
   - Note that the render-domain gap has bitten before: blotchy skin, the
     light direction, and a mirrored-winding bug.
4. **Your detector's confidence threshold works as a quality gate.** Raising
   recall tripled gross errors on the 3D side, so keep a precision-first
   operating point.

## What the 3D side relies on (please don't break)

- **55 points, fixed numbering and meaning.** The strips are 0–19, 20–34,
  35–38, the canal at 39, 40–49, and 50–54 (superior crus).
  - Extend the scheme by **appending** points, never by renumbering. The 3D
    reference marks are (triangle, barycentric) addresses tied to these
    indices.
- **The shipped run.** The 3D side uses `SHIPPED_RUN` from `inference.py`
  (currently `fixed_conv_s42`) and does not pin a run. A change to
  `SHIPPED_RUN` is what the 3D side will pick up.
- **Import surface.** The 3D side imports only through `ear3d/twod.py`, which
  pulls from `inference.py`: `LandmarkPredictor`, `EarDetector`,
  `EarLandmarkerPipeline`, `find_best_checkpoint`, `BLAZEEAR_DIR`,
  `DETECTOR_WEIGHTS` and `DETECTOR_INPUT_SIZE`. Renaming any of these breaks the 3D side.
- **Coordinate fusion for flip TTA** was the best option on every checkpoint
  tried; it was recommended to implement it batched on this side.

## How to score a new checkpoint end to end

Use the `mocap` env, from `Ear_3D_Lifter`:

    "$CODE_UTILITIES/guard_run.sh" 6G 7200 LOG \
      /home/shameem/.conda/envs/mocap/bin/python scripts/validate_pipeline.py \
      runs/e2e_<name> 2 16 <worker> <n_workers>
    # prefix with EAR3D_CKPT=<run or .ckpt path> to score a candidate

- Seed `2` gives the 16 held-out heads at the same rotations as the baseline.
- **Choose the checkpoint with `EAR3D_CKPT`**, set to a `.ckpt` path or a
  run name. No edit to this repo is needed. Without it, the 3D side follows
  `SHIPPED_RUN`.
- Every result records the checkpoint it used, as `ckpt` in each npz.
- Each head writes `pred_{L,R}` and `gt_{L,R}_lm3d_v11`, in the rotated input
  frame.

**Baseline to beat, `runs/e2e_heldout_lmw`** (`fixed_conv_s42/EarLandmarker_433`):
- median per-ear 1.574 mm;
- p95 2.46 mm;
- helix (0–34) 1.88 mm, the rest 1.14 mm;
- 4.03% of points off by more than 5 mm.

**Compare against the same reference (lm3d_v11) every time.** Switching
references moved the score by 0.13 mm with no change to the model, which is
bigger than most model differences. The cross-scoring and decomposition
scripts are in `/var/tmp/claude/ear3d/2026-10-17-blazeear/heldout/`
(`score.py`), which is scratch: move one into `scripts/` before relying on it.

Report the >5 mm rate and the points 0–8 / 15–19 numbers alongside the
median. A median hides the tail, and the tail is where this model loses.

## Constraints

- `supersecret/` is a real person's head scan. It stays on this machine: never
  commit it, send it anywhere, or attach its renders.
- Licences:
  - anything built on the U-Net or its outputs credits SaifulULL (CC BY 4.0);
  - the non-commercial datasets (Aachen, IIT Indore, York, Mendeley) stay
    discarded;
  - accepting licence terms is the user's action.
