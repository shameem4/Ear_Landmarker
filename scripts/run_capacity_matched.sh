#!/usr/bin/env bash
# Is EarLandmarker's backbone better by DESIGN, or just bigger?
#
# The FaceMesh copy loses by 5.75% even with its head replaced and 81.6% of its
# weights pretrained -- but it is also 2.4x smaller, so nothing so far separates
# the design from the size.
#
# This runs EarLandmarker's own design at --width-mult 0.62, which is 142,935
# params against the copy's 141,991: a 0.7% capacity match. Everything else is
# the shipped recipe.
#
#   if it still beats 0.03087  -> the DESIGN wins on its merits
#   if it lands near 0.03087   -> it was capacity all along, and the honest
#                                 claim becomes "the extra parameters earn their
#                                 keep", not "the architecture does"
#   if it loses to 0.03087     -> MediaPipe's backbone is better per-parameter
#                                 and the shipped model wins only by being bigger
#
# PRE-REGISTERED, same bars: a difference counts only with complete rank
# separation across 3 seeds AND >1% relative.
#
# Reference points, test NME:
#   control 340K    0.02919 +/- 0.00031
#   fm_heat_pre     0.03087 +/- 0.00033   (142K, MediaPipe backbone, same head)
#
# COST: 3 runs x 500 epochs, smallest config yet.

set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/shameem/.conda/envs/mocap/bin/python
PIDS=()
for SEED in 42 1 2; do
  $PY train.py --arch heatmap --width-mult 0.62 --perspective-deg 65 \
      --epochs 500 --num-workers 4 --seed "$SEED" \
      --run-name "cap62_s${SEED}" > "runs/logs/cap62_s${SEED}.out" 2>&1 &
  PIDS+=($!)
done
echo "launched ${#PIDS[@]} runs: ${PIDS[*]}"
echo "kill with: kill ${PIDS[*]}"
wait "${PIDS[@]}"
echo done
