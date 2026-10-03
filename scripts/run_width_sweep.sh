#!/usr/bin/env bash
# How far does scaling the feature extractor keep paying?
#
# PRE-REGISTERED, and with an expectation on record so the result can contradict
# it: I expect the curve to have ALREADY turned at width 1.0, because
#
#   - the shipped model's train/val gap is +14.1% with augmentation ON, so the
#     true gap is wider. That is a data-limited model, and extra capacity should
#     cost generalisation rather than buy it.
#   - 72% of its squared error is tangential, tracking annotator spacing rather
#     than anything the model controls. If that is a floor, the best reachable
#     NME is ~0.0233 -- only 20% below where the model already sits, and every
#     remaining change competes for that same budget.
#
# Widths 0.62 and 1.00 are already measured elsewhere; this adds the upward arm.
#
#   width  ladder                     params      note
#    0.62  16-32-56-80-120           142,935     run separately (cap62)
#    1.00  24-48-96-128-192          340,167     the shipped model, 0.02919
#    1.40  32-64-136-176-272         633,703     this script
#    1.80  40-88-176-232-344       1,031,031     this script
#
# DECISION RULE: a width counts as better only with complete rank separation
# across 3 seeds AND >1% relative, the same bar used throughout. Flat is the
# expected and perfectly informative outcome -- it locates the knee.
#
# Run in waves of three to stay inside GPU memory alongside anything else.

set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/shameem/.conda/envs/mocap/bin/python

for W in 1.4 1.8; do
  TAG="w$(echo "$W" | tr -d .)"
  echo "=== width $W ==="
  PIDS=()
  for SEED in 42 1 2; do
    $PY train.py --arch heatmap --width-mult "$W" --perspective-deg 65 \
        --epochs 500 --num-workers 4 --seed "$SEED" \
        --run-name "${TAG}_s${SEED}" > "runs/logs/${TAG}_s${SEED}.out" 2>&1 &
    PIDS+=($!)
  done
  echo "  launched: ${PIDS[*]}"
  wait "${PIDS[@]}"
  echo "  width $W done"
done
echo "sweep complete"
