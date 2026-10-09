#!/usr/bin/env bash
# Route C: does making renders look like photographs close the 5.1x gap?
#
# Head 1 trained on renders scores 0.0884 NME on real photographs against the
# 0.0173 a photo-trained model achieves, and the cause is measured to be a
# domain gap rather than framing or annotation. A logistic probe on seven
# plain image statistics tells a render from a photograph 93.4% of the time;
# data/photo_aug closes the statistics to 0.740, every marginal within
# d = 0.41.
#
# Whether that TRANSLATES is the question. The probe is a proxy and a proxy
# can be satisfied without the model caring, so the readings are:
#
#   data/manual/test.csv NME   the target. 1,054 clean held-out photographs,
#                              images vetted permissive and annotations
#                              commissioned, and a VALID judge for head 1
#                              because head 1 is render-trained and not a
#                              v6_persp65 descendant. Baseline 0.1045,
#                              floor 0.0143.
#   render z_skill             the thing not to break. Validation runs on
#                              UNAUGMENTED renders throughout, so in-domain
#                              depth is still measured honestly. Baseline
#                              0.8015 on common_test.
#
# Two strengths, because 1.0 is fitted to the measured photo statistics and
# there is no reason to assume the best training signal sits exactly there --
# augmentation that matches the target domain can still be too aggressive to
# learn from. Everything else is identical to runs/sb_a_prop, which IS the
# photo-aug 0 control, so a difference is this stage and nothing else.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-/home/shameem/.conda/envs/mocap/bin/python}"
COMMON="--data data/depth3d_v5 --head z --n-points 55 --epochs 150 --batch 64 --workers 10"

PIDS=()
for S in 0.5 1.0; do
  rm -rf "runs/pa_${S}"
  $PY scripts/train_depth.py $COMMON --photo-aug $S --out "runs/pa_${S}" \
      > "runs/pa_${S}.log" 2>&1 &
  PIDS+=($!)
done
echo "pids: ${PIDS[*]}"
wait "${PIDS[@]}"
echo "done"
grep -h "^best val_z_mae" runs/pa_0.5.log runs/pa_1.0.log
