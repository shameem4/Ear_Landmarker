#!/usr/bin/env bash
# Third arm: MediaPipe's feature extractor under this project's heatmap decoder.
#
# WHY. fm_scratch and fm_pre both lost to EarLandmarker, but confounded two
# things -- 207K params against 340K, AND MediaPipe's direct-regression head
# against soft-argmax. This project measured that head family at 0.0301 against
# the heatmap's 0.0291, so part of the deficit was predicted to be head design.
#
# Keeps backbone1 exactly as published (the pretrained weights still load) and
# swaps backbone2a for the heatmap decoder. 141,991 params, 81.6% pretrained.
#
# READING IT, against results already in hand:
#   control      0.02919 +/- 0.00031   (340K, heatmap)
#   fm_pre       0.03147 +/- 0.00010   (207K, MediaPipe head, pretrained)
#   fm_scratch   0.03273 +/- 0.00027   (207K, MediaPipe head, random)
#
#   vs fm_pre      -> what the head alone was worth
#   vs the control -> what remains attributable to the backbone
#
# PRE-REGISTERED. Same bars as before: a difference counts only with complete
# rank separation across 3 seeds AND >1% relative. Both arms here are
# pretrained, so this isolates the head, not the weights.
#
# COST: 3 runs x 500 epochs. Smaller than the last set; expect well under 1.5 h.

set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/shameem/.conda/envs/mocap/bin/python
MP=/mnt/14BE47C2BE479ADE/Code/ear_stuff/trainable_blazeface/model_weights/blazeface_landmark.pth
[ -f "$MP" ] || { echo "MediaPipe weights not found: $MP" >&2; exit 1; }

PIDS=()
for SEED in 42 1 2; do
  $PY train.py --arch facemesh_heatmap --perspective-deg 65 --epochs 500 \
      --num-workers 4 --seed "$SEED" --mediapipe-ckpt "$MP" \
      --run-name "fm_heat_pre_s${SEED}" > "runs/logs/fm_heat_pre_s${SEED}.out" 2>&1 &
  PIDS+=($!)
done
echo "launched ${#PIDS[@]} runs: ${PIDS[*]}"
echo "kill with: kill ${PIDS[*]}"
wait "${PIDS[@]}"
echo "done"
