#!/usr/bin/env bash
# Is the shading component of photo_aug buying x,y at the cost of depth?
#
# photo-aug 1.0 cut x,y error on real photographs 28% (0.1045 -> 0.0749) and
# made real-photo depth WORSE (z_skill -0.188 -> -0.280), with render depth
# unchanged. The suspect is local shading: a random multiplicative brightness
# field is precisely the shape-from-shading cue a depth head uses, so
# randomising it teaches the model to ignore the signal depth depends on.
#
# Two runs at full strength with shading dialled down, everything else held:
#   shading 0.0   texture, noise, occluders, defocus, JPEG only
#   shading 0.3   a little, in case some lighting variety still helps
#
# PRE-REGISTERED: if shading is the culprit, photo depth should recover
# toward or past the -0.188 control while x,y keeps most of its 28% gain. If
# photo depth stays bad with shading off, the cause is elsewhere and the
# honest read is that appearance augmentation fixes 2D and cannot fix depth.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-/home/shameem/.conda/envs/mocap/bin/python}"
COMMON="--data data/depth3d_v5 --head z --n-points 55 --epochs 150 --batch 64 --workers 10 --photo-aug 1.0"
PIDS=()
for S in 0.0 0.3; do
  rm -rf "runs/pash_${S}"
  $PY scripts/train_depth.py $COMMON --photo-aug-shading $S --out "runs/pash_${S}" \
      > "runs/pash_${S}.log" 2>&1 &
  PIDS+=($!)
done
echo "pids: ${PIDS[*]}"
wait "${PIDS[@]}"
echo done
