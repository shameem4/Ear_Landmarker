#!/usr/bin/env bash
# Can MediaPipe's face landmark architecture, copied exactly, match the
# independently-designed EarLandmarker?
#
# PRE-REGISTERED. Read before looking at any number.
#
# THREE ARMS, one of which is already paid for:
#   fm_scratch   FaceMesh topology, random init      -- is the ARCHITECTURE enough?
#   fm_pre       FaceMesh topology, MediaPipe weights -- do the free WEIGHTS pay?
#   control      v6_persp65, already run, 3 seeds: 0.02923 / 0.02948 / 0.02886
#                mean 0.02919 +/- 0.00031
#
# WHAT IS BEING COMPARED
#   206,942 params against 340,167, at the same 192x192 input. 84.6% of the
#   FaceMesh arm's weights arrive pretrained on millions of faces.
#
#   The copy inherits MediaPipe's head: direct regression from a 3x3 conv at 1x1
#   spatial. This project already measured that family (GAP+FC) at 0.0301
#   against the soft-argmax heatmap's 0.0291. So the architecture starts about
#   4% behind on head design alone, and the pretrained features have to win that
#   back. fm_scratch isolates the architecture; fm_pre adds the weights.
#
# DECISION RULE -- fixed in advance, test NME, 3 seeds per arm
#   "Matches EarLandmarker":  |mean - 0.02919| within 2 sigma (0.00062), i.e.
#                             roughly 0.0286 to 0.0298. Report as a match, not a
#                             win, in either direction.
#   "Beats it":               mean below 0.02857 AND all 3 seeds below the
#                             control's best (0.02886).
#   "Loses":                  mean above 0.02981.
#   Pretraining helped:       fm_pre beats fm_scratch by the same rank-separation
#                             + 1% rule used in the other experiment.
#
#   A loss is a real result: it says the independent design earns its keep, and
#   closes the "should we have started from MediaPipe" question.
#
# COST: 6 runs x 500 epochs; 207K params, lighter than the 340K control.

set -euo pipefail
cd "$(dirname "$0")/.."

PY=/home/shameem/.conda/envs/mocap/bin/python
MP=/mnt/14BE47C2BE479ADE/Code/ear_stuff/trainable_blazeface/model_weights/blazeface_landmark.pth

[ -f "$MP" ] || { echo "MediaPipe weights not found: $MP" >&2; exit 1; }

COMMON=(--arch facemesh --perspective-deg 65 --epochs 500 --num-workers 4)

mkdir -p runs/logs
PIDS=()
for SEED in 42 1 2; do
  $PY train.py "${COMMON[@]}" --seed "$SEED" \
      --run-name "fm_scratch_s${SEED}" > "runs/logs/fm_scratch_s${SEED}.out" 2>&1 &
  PIDS+=($!)
  $PY train.py "${COMMON[@]}" --seed "$SEED" --mediapipe-ckpt "$MP" \
      --run-name "fm_pre_s${SEED}" > "runs/logs/fm_pre_s${SEED}.out" 2>&1 &
  PIDS+=($!)
done

echo "launched ${#PIDS[@]} runs: ${PIDS[*]}"
echo "kill with: kill ${PIDS[*]}"
echo "watch:     python scripts/report_facemesh_experiment.py"
wait "${PIDS[@]}"
echo
$PY scripts/report_facemesh_experiment.py
