#!/usr/bin/env bash
# Route B: predict SHAPE, solve pose, let depth fall out -- MediaPipe's split.
#
# WHY, in one paragraph. head 1 regresses 55 depths with no constraint; it
# reaches 0.804 z_skill on renders and -0.171 on real photographs, because a
# domain shift can move each of the 165 outputs anywhere. MediaPipe FaceMesh
# does not do this: its network gives screen-space landmarks and its metric 3D
# comes from Procrustes-fitting a canonical model. Fitting our MEAN ear that
# way already scores +0.217 on the photo benchmark -- better than head 1 by
# 0.39 skill, with no per-subject 3D at all. Route B adds the missing piece,
# per-subject shape, while keeping the geometry.
#
# MEASURED BEFORE SPENDING ANY COMPUTE (scripts/shape_fit_ceiling.py, renders,
# fitted to ground-truth 2D, 400 views, bootstrap 95% CI):
#
#   mean ear, coeffs = 0            0.584 [0.557, 0.608]
#   0.25 x gt_coeffs                0.678 [0.655, 0.699]
#   0.50 x gt_coeffs                0.753 [0.734, 0.770]
#   0.75 x gt_coeffs                0.798 [0.782, 0.812]
#   1.00 x gt_coeffs                0.807 [0.793, 0.821]   <- B2's ceiling
#   1.25 x gt_coeffs                0.788 [0.772, 0.802]
#   full true 3D, oracle            0.817 [0.803, 0.830]
#
# Three things that justify the run. The ceiling 0.807 is indistinguishable
# from the 0.817 oracle, so 40 modes cost nothing. It also EQUALS head 1's
# 0.804, so this is not a capability sacrifice. And the curve is CONCAVE:
# recovering a quarter of the coefficient magnitude buys 42% of the gain, and
# under-prediction is nearly free while overshoot is penalised. The failure
# mode under domain shift is coefficients -> 0, which is the mean-ear fit at
# +0.217, i.e. the best thing we have on photographs. head 1's failure mode is
# garbage. That asymmetry is the whole bet.
#
# POSE IS DISCARDED AT INFERENCE. This head regresses shape AND pose, but a
# regressed pose has to survive the domain shift, while a SOLVED pose rides on
# the 2D model that already does (0.0143 on photographs). So evaluation takes
# the coefficients only and hands them to the weak-perspective solver together
# with the shipped model's 2D. The head's own end-to-end landmarks are scored
# too, which makes solved-vs-regressed pose a measurement rather than a belief.
#
# Four runs. coeff-sup is swept because the open question from the last attempt
# was how much direct coefficient supervision is needed -- the landmark metric
# cannot tell a real mesh from a plausible one, so mesh_err is the thing to
# watch. The no-aug run is the control for Route C's recipe, which fixed x,y
# and had to have shading switched off to stop it destroying depth.
#
# PRE-REGISTERED READINGS:
#  1. mesh_err/mesh_base on render val. coeff-sup 0 should sit near 1.0 (no
#     better than predicting the mean ear). If it does not, supervision is
#     unnecessary and that is the finding.
#  2. coefficients + solver + shipped 2D on the 112-photo benchmark, against
#     the mean-ear fit's +0.217. To call Route B a win it must beat +0.217 on
#     a PAIRED bootstrap over the same ears, not merely score higher.
#  3. render z_skill end-to-end, against head 1's 0.804 and the 0.807 ceiling.
# A run that improves renders and not photographs is head 1 again and does not
# count.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-/home/shameem/.conda/envs/mocap/bin/python}"
TPL=/mnt/14BE47C2BE479ADE/Code/ear_stuff/wrap_work/templates
COMMON="--data data/depth3d_v5 --head shape --n-points 55 --epochs 150 \
  --batch 64 --workers 8 --basis $TPL/lm_basis_v8_all_55_unitheight.npz \
  --coeffs $TPL/gt_coeffs_v7_mm.npz"
AUG="--photo-aug 1.0 --photo-aug-shading 0.0"
PIDS=()
launch() {  # name, extra flags...
  local n="$1"; shift
  rm -rf "runs/$n"
  $PY scripts/train_depth.py $COMMON "$@" --out "runs/$n" > "runs/$n.log" 2>&1 &
  PIDS+=($!); echo "  $n -> pid ${PIDS[-1]}"
}
launch b2_sup0      $AUG --coeff-sup 0
launch b2_sup1      $AUG --coeff-sup 1.0
launch b2_sup10     $AUG --coeff-sup 10.0
launch b2_sup1_noaug     --coeff-sup 1.0
echo "pids: ${PIDS[*]}"
wait "${PIDS[@]}"
echo done
