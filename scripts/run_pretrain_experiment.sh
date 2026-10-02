#!/usr/bin/env bash
# Does bootstrapping the backbone from the BlazeEar detector help?
#
# PRE-REGISTERED. Read this before looking at any number.
#
# WHY THIS SHAPE
# The shipped backbone (5x5 depthwise, 24-48-96-128-192) shares just 0.9% of its
# parameters with any donor on disk -- BlazeEar's detector or MediaPipe's face
# landmark weights. Transfer into it is a null by construction. So both arms use
# --backbone blazeear, which mirrors BlazeEar v2's backbone1 block for block and
# lets 59% of the backbone (41,162 params) actually carry over.
#
# The two arms differ ONLY in initialisation. Same architecture, same data, same
# schedule, same seeds. Any difference is the transfer.
#
# WHAT IT CANNOT TELL US
# The mirrored backbone is 82K params against the shipped 340K, because the
# donor's ladder tops out at 88 channels. Absolute NME will be worse than 0.0292
# and that is expected -- this measures whether pretraining helps, not whether
# this architecture should ship. A positive result argues for porting the idea
# to the 340K backbone (which needs a donor trained at that width, i.e. real
# work); a negative result closes the question cheaply.
#
# DECISION RULE -- fixed in advance
#   Primary metric: test NME, reported once per run by trainer.test().
#   Call it a win only if BOTH hold:
#     1. complete rank separation -- all 3 pretrained seeds beat all 3 control
#        seeds (exact p = 1/20 = 0.05, the bar the synthetic experiment met);
#     2. mean improvement > 1% relative, which is the seed-noise floor measured
#        on the shipped architecture (+/-0.00031 on 0.0292).
#   Anything less is a seed. Record it as "not separable" and move on.
#   A result in the WRONG direction at the same strength closes the question.
#
# COST: 6 runs x 500 epochs. The 82K model is smaller than the 340K one, so
# expect under the ~2.4h that three 340K runs took. Launching all six at once
# cost only ~3% per-run slowdown when this was measured; drop to three at a time
# if the GPU is shared.

set -euo pipefail
cd "$(dirname "$0")/.."

PY=/home/shameem/.conda/envs/mocap/bin/python
BLAZEEAR_CKPT=/mnt/14BE47C2BE479ADE/Code/ear_stuff/BlazeEar/runs/checkpoints_crop/BlazeEar_best.pth

if [ ! -f "$BLAZEEAR_CKPT" ]; then
  echo "donor checkpoint not found: $BLAZEEAR_CKPT" >&2; exit 1
fi

COMMON=(--arch heatmap --backbone blazeear --perspective-deg 65
        --epochs 500 --num-workers 4)

mkdir -p runs/logs
PIDS=()
for SEED in 42 1 2; do
  # control: identical architecture, random init
  $PY train.py "${COMMON[@]}" --seed "$SEED" \
      --run-name "pre_control_s${SEED}" > "runs/logs/pre_control_s${SEED}.out" 2>&1 &
  PIDS+=($!)
  # treatment: same, but backbone initialised from the detector
  $PY train.py "${COMMON[@]}" --seed "$SEED" --blazeear-ckpt "$BLAZEEAR_CKPT" \
      --run-name "pre_transfer_s${SEED}" > "runs/logs/pre_transfer_s${SEED}.out" 2>&1 &
  PIDS+=($!)
done

echo "launched ${#PIDS[@]} runs: ${PIDS[*]}"
echo "kill with: kill ${PIDS[*]}"
echo "watch:     python scripts/report_pretrain_experiment.py"
wait "${PIDS[@]}"
echo
echo "all runs finished -- scoring against the pre-registered rule"
$PY scripts/report_pretrain_experiment.py
