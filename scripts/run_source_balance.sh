#!/usr/bin/env bash
# Does head 1's depth skill survive when the source cannot be assumed?
#
# A logistic regression on 40 shape coefficients names the source of a HELD-OUT
# subject 93.9% of the time, and the corpus is 76% SONICOM. So the shipped
# 0.828 z_skill could in principle be part "knows this ear" and part "knows
# this is SONICOM, and SONICOM ears look like this".
#
# ALL THREE ARE SCORED ON common_test.csv: 14 HUTUBS and 14 SONICOM subjects
# carved out before any split was made, on their own seed, and trained on by
# none of the runs. That file is what makes the three numbers comparable. The
# first version of this experiment let each recipe hold out its own test
# subjects, so every difference was part model and part test set; it was
# discarded rather than reported.
#
#   A  proportional   the control. train.csv as it comes: 148 SONICOM and 30
#                     HUTUBS subjects, 6148 views, shuffled.
#   B  balanced       THE SAME 6148 ROWS, drawn by a sampler that gives the
#                     two sources equal weight per epoch. Nothing else moves,
#                     so a difference is the sampling and not the data.
#   C  loso           SONICOM only, 6144 views -- within 0.07% of A's count,
#                     so C is not simply a smaller run. No HUTUBS ear is in
#                     training at all, and the checkpoint is chosen on a
#                     SONICOM val set, never peeking at the target source.
#
# PRE-REGISTERED readings, so the result cannot be narrated after the fact:
#
#   If C's HUTUBS skill lands near A's HUTUBS skill, depth skill is about ear
#   geometry and transfers across sources. The 93.9% separability is then real
#   but idle -- available to the model and not used -- and the pooled number
#   can be trusted.
#
#   If C's HUTUBS skill collapses while its SONICOM skill holds, the shipped
#   skill is partly source-specific, and 0.828 is optimistic for any ear that
#   is not SONICOM. The number to quote from then on is the cross-source one.
#
#   B is the diagnostic in between. If equal-weight sampling moves nothing,
#   the imbalance was not being exploited. B cannot remove the shortcut by
#   itself -- the source cue is in every render either way -- it only stops
#   the prior from rewarding it.
#
# The same 55 landmarks (points 0..54, the original iBUG scheme) and the same
# 150-epoch recipe throughout, so these are comparable to earlier figures.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-/home/shameem/.conda/envs/mocap/bin/python}"
COMMON="--data data/depth3d_v5 --head z --n-points 55 --epochs 150 --batch 64 --workers 10"

mkdir -p runs
$PY scripts/train_depth.py $COMMON --split-prefix "" \
    --out runs/sb_a_prop > runs/sb_a_prop.log 2>&1 &
A=$!
$PY scripts/train_depth.py $COMMON --split-prefix "" --balance-sampler \
    --out runs/sb_b_bal > runs/sb_b_bal.log 2>&1 &
B=$!
$PY scripts/train_depth.py $COMMON --split-prefix son_ \
    --out runs/sb_c_loso > runs/sb_c_loso.log 2>&1 &
C=$!
echo "A=$A proportional   B=$B balanced-sampler   C=$C loso(sonicom-only)"
wait $A $B $C
echo "all three finished"
grep -h "^best val_z_mae" runs/sb_a_prop.log runs/sb_b_bal.log runs/sb_c_loso.log
