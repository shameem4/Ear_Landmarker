#!/usr/bin/env bash
# The noise floor for the A-vs-C comparison.
#
# On seed 0, a model that never saw a HUTUBS ear (C) scored 0.7833 z_skill on
# held-out HUTUBS against 0.7874 for one trained with 30 HUTUBS subjects (A).
# A gap of 0.0041, or 0.5% relative, is below this project's pre-registered
# bar of complete rank separation across three seeds AND >1% relative -- but
# that bar was written for claiming a difference, and the claim here is that
# there ISN'T one. A null needs the same error bar, or "no effect" just means
# "one run, and I did not look twice".
#
# Seeds 1 and 2 for A and C only. B is the secondary question and seed 0
# already put it on top of A to within 0.001.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-/home/shameem/.conda/envs/mocap/bin/python}"
COMMON="--data data/depth3d_v5 --head z --n-points 55 --epochs 150 --batch 64 --workers 8"

PIDS=()
for S in 1 2; do
  $PY scripts/train_depth.py $COMMON --seed $S --split-prefix "" \
      --out runs/sb_a_prop_s$S > runs/sb_a_prop_s$S.log 2>&1 &
  PIDS+=($!)
  $PY scripts/train_depth.py $COMMON --seed $S --split-prefix son_ \
      --out runs/sb_c_loso_s$S > runs/sb_c_loso_s$S.log 2>&1 &
  PIDS+=($!)
done
echo "pids: ${PIDS[*]}"
wait "${PIDS[@]}"
echo "all four finished"
grep -h "^best val_z_mae" runs/sb_a_prop_s{1,2}.log runs/sb_c_loso_s{1,2}.log
