#!/bin/bash
# Rebuild the whole 3D corpus from the Wrap fits up.
#
# Resumable: each stage is skipped if its output already exists, so an
# interrupted run can be restarted with the same command. Set FORCE=1 to
# redo everything.
#
#   bash scripts/rebuild_corpus.sh            # run it
#   bash scripts/rebuild_corpus.sh --dry-run  # print the plan, touch nothing
#
# --jobs 4, not more: WrapCmd uses ~7.3 cores per fit (14s wall / 105s user),
# so 4 jobs already occupies ~29 of 32 cores and 8 jobs measured SLOWER
# (0.09 vs 0.12 fits/s). There is no GPU path -- Wrap links libOpenCL but
# never appears as a GPU compute process during a fit.
set -euo pipefail
P=/home/shameem/.conda/envs/mocap/bin/python
R=/mnt/14BE47C2BE479ADE/Code/ear_stuff/Ear_Landmarker
W=/mnt/14BE47C2BE479ADE/Code/ear_stuff/wrap_work
C=$W/corpus
A=$C/adaptive_v2                # mesh fits (HUTUBS + SONICOM)
E=$C/ae3d_adaptive_v2           # cloud fits (AudioEar3D)
K=$C/canonical_v8               # canonical mm, one frame
LOG=${LOG:-$W/rebuild_$(date +%Y%m%d_%H%M).log}
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
run(){ if [ $DRY = 1 ]; then echo "    would run: $*"; else "$@"; fi; }
have(){ [ "${FORCE:-0}" = 0 ] && [ -e "$1" ]; }

cd $R
say "=== rebuild_corpus: log $LOG"
say "stage 1/6  mesh fits -> $A   (front-only anchors, 11-rung ladder)"
if have "$A/adaptive.json"; then say "  skip: $A/adaptive.json exists"; else
  run env OMP_NUM_THREADS=4 $P scripts/wrap_refit_adaptive.py --all --front-only \
      --out $A --jobs 4
fi

say "stage 2/6  cloud fits -> $E   (AudioEar3D, min-cos ladder, on ear_mean_v3)"
if have "$E/adaptive.json"; then say "  skip: exists"; else
  run env OMP_NUM_THREADS=4 $P scripts/wrap_cloud_adaptive.py \
      --root /mnt/14BE47C2BE479ADE/Code/ear_stuff/clean_3d_data/AudioEar3D \
      --out $E --jobs 4
fi

say "stage 3/6  split the mesh fits by dataset at base resolution"
if have "$C/adaptive_v2_hutubs"; then say "  skip: exists"; else
  run $P scripts/split_fits_by_dataset.py --src $A --out-prefix $C/adaptive_v2
fi

say "stage 4/6  canonicalise all three sources together -> $K"
# ALL sources in ONE invocation: --sources rewrites manifest.csv for the
# datasets it is given, and a per-source run would drop the others' rows.
if have "$K/manifest.csv"; then say "  skip: exists"; else
  run env OMP_NUM_THREADS=4 $P scripts/canonicalise_corpus.py \
      --sources hutubs=adaptive_v2_hutubs sonicom=adaptive_v2_sonicom \
                audioear=ae3d_adaptive_v2 --out-root $K
fi

say "stage 5/6  mean -> ear_mean_v3, promotion replay -> ear_lmv_v2"
run $P scripts/build_mean_template.py --canonical $K \
    --out $W/templates/ear_mean_v3.obj \
    --recipe $W/templates/ear_lmv_v1.npz \
    --lmv-out $W/templates/ear_lmv_v2.obj

say "stage 6/6  regenerate the derived mm artefacts"
run $P scripts/regen_mm_artefacts.py --canonical $K

say "=== done. QA:"
run $P scripts/wrap_qa.py --fits $A --template $W/templates/ear_L_template.obj || true
