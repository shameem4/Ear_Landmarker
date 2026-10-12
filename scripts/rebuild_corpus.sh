#!/bin/bash
# Rebuild the whole 3D corpus and every artefact derived from it.
#
# Resumable: each stage is skipped if its output exists, so an interrupted
# run restarts with the same command. FORCE=1 redoes everything.
#
#   bash scripts/rebuild_corpus.sh            # run it
#   bash scripts/rebuild_corpus.sh --dry-run  # print the plan, touch nothing
#
# NO CONTROL POINTS FOR THE FIRST PASS. The 14 spread control points the
# old chain used were sampled from a previous generation of fits, so the
# corpus bootstrapped off itself. Dropping them is better on every axis
# measured: 1.89 folds per ear against 70.78, 34.5% of ears fold-free
# against 0%, and the whole corpus in 22 minutes. Control points buy
# initialisation, not correspondence.
#
# THE LADDER IS A RESCUE, NOT A DEFAULT. It clears 77 of the 92 ears that
# still fold, but costs about 0.03pp of surface accuracy, so stage 4 takes
# its fit only where it strictly reduces folds.
#
# --jobs 4, not more: WrapCmd uses ~7.3 cores per fit (14s wall / 105s
# user), so 4 jobs already occupy ~29 of 32 cores and 8 measured SLOWER
# (0.09 vs 0.12 fits/s). There is no GPU path.
#
# Everything heavy runs under tools/guard_run.sh: a systemd scope with a
# hard MemoryMax. A units bug once put 21.9 GB per worker into a proximity
# query and took the machine down; RLIMIT_AS cannot catch it, because
# imports alone reserve ~16 GB of virtual address space.
set -euo pipefail
P=/home/shameem/.conda/envs/mocap/bin/python
R=/mnt/14BE47C2BE479ADE/Code/ear_stuff/Ear_Landmarker
W=/mnt/14BE47C2BE479ADE/Code/ear_stuff/wrap_work
C=$W/corpus
G=$R/scripts/guard_run.sh
T=$W/templates
N=$C/nocp_v1                    # pass 1: no control points, default rung
LAD=$C/ladder92                 # pass 2: ladder, only on the ears that fold
E=$C/escalated_v1               # per-ear pick of the two
AE=$C/ae3d_v2                   # AudioEar3D cloud fits
K=$C/canonical_esc              # canonical mm, one frame, all three sources
LOG=${LOG:-$W/rebuild_$(date +%Y%m%d_%H%M).log}
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
run(){ if [ $DRY = 1 ]; then echo "    would run: $*"; else "$@"; fi; }
have(){ [ "${FORCE:-0}" = 0 ] && [ -e "$1" ]; }

cd $R
say "=== rebuild_corpus: log $LOG"

say "stage 1/8  no-control-point fits -> $N"
if have "$N/adaptive.json"; then say "  skip: exists"; else
  run env OMP_NUM_THREADS=4 $G 24G 14400 $W/tools/s1.log $P \
      scripts/wrap_refit_adaptive.py --all --front-only --points 0 \
      --max-rungs 1 --out $N --jobs 4
fi

say "stage 2/8  split pass 1 by dataset"
if have "${N}_hutubs"; then say "  skip: exists"; else
  run $P scripts/split_fits_by_dataset.py --src $N --out-prefix $N
fi

say "stage 3/8  ladder on the ears that still fold (canal ring masked)"
if have "$LAD"; then say "  skip: exists"; else
  # scan for folded ears first: escalate_corpus with no ladder to pick from
  # still writes dirty.txt, which is the list the ladder has to run on.
  run $P scripts/escalate_corpus.py --nocp $C --ladder /nonexistent \
      --out $C/fold_scan --datasets nocp_v1_hutubs nocp_v1_sonicom
  # UNQUOTED on purpose: --ears takes nargs="+", so the list must word-split.
  # Quoting it passes all 92 names as one argument and the run fits nothing.
  run env OMP_NUM_THREADS=2 $G 24G 14400 $W/tools/s3.log $P \
      scripts/wrap_refit_adaptive.py --front-only --jobs 4 --out $LAD \
      --ears $(cut -f1 $C/fold_scan/dirty.txt | grep -v '^#' | tr '\n' ' ')
fi

say "stage 4/8  pick per ear: ladder only where it strictly helps -> $E"
if have "$E/manifest.json"; then say "  skip: exists"; else
  run $P scripts/escalate_corpus.py --nocp $C --ladder $LAD --out $E \
      --datasets nocp_v1_hutubs nocp_v1_sonicom
fi

say "stage 5/8  AudioEar3D cloud fits -> $AE"
if have "$AE/adaptive.json"; then say "  skip: exists"; else
  run env OMP_NUM_THREADS=4 $G 24G 14400 $W/tools/s5.log $P \
      scripts/wrap_cloud_adaptive.py \
      --root /mnt/14BE47C2BE479ADE/Code/ear_stuff/clean_3d_data/AudioEar3D \
      --template $T/ear_mean_v4.obj --out $AE --jobs 4
fi

say "stage 6/8  canonicalise all three sources together -> $K"
# ONE invocation. --sources rewrites the rows of the datasets it is given
# and carries the rest; a per-source run is safe but re-reads the manifest,
# and carried rows come back from CSV as strings.
if have "$K/manifest.csv"; then say "  skip: exists"; else
  run env OMP_NUM_THREADS=4 $G 24G 7200 $W/tools/s6.log $P \
      scripts/canonicalise_corpus.py --out-root $K \
      --sources hutubs=escalated_v1/hutubs sonicom=escalated_v1/sonicom \
                audioear=ae3d_v2
fi

say "stage 7/8  mean -> ear_mean_v4, promotion replay -> ear_lmv_v3"
# The replay keeps the lmv connectivity identical to ear_lmv_v1/v2. Building
# the lmv with landmark_template.py instead re-derives the insertion and
# CHANGES connectivity, which invalidates any face-index mask.
run $G 24G 3600 $W/tools/s7.log $P scripts/build_mean_template.py \
    --canonical $K --sources hutubs sonicom audioear \
    --out $T/ear_mean_v4.obj --recipe $T/ear_lmv_v1.npz \
    --lmv-out $T/ear_lmv_v3.obj

say "stage 8/8  PCA, landmark bases, ground-truth coefficients"
run $G 24G 3600 $W/tools/s8a.log $P scripts/build_ear_pca.py \
    --fits $K/hutubs $K/sonicom $K/audioear --template $T/ear_mean_v4.obj \
    --out $T/ear_pca_v7_all_mm.npz --note "canonical mm; 589 ears; NOT redistributable"
run $G 24G 3600 $W/tools/s8b.log $P scripts/build_ear_pca.py \
    --fits $K/hutubs --template $T/ear_mean_v4.obj \
    --out $T/ear_pca_v7_ccby_mm.npz --note "canonical mm; HUTUBS only, CC BY 4.0"
for v in all ccby; do
  for n in 55 95; do
    mk=$T/mark_mean_v6_spacing.npz
    [ $n = 55 ] && mk=$T/mark_mean_v6_spacing_55.npz
    run $G 8G 1800 $W/tools/s8_${v}_${n}.log $P scripts/build_landmark_basis.py \
        --pca $T/ear_pca_v7_${v}_mm.npz --marks $mk \
        --template $T/ear_mean_v4.obj \
        --out $T/lm_basis_v8_${v}_${n}_unitheight.npz
  done
done
# gt_coeffs is solved against ear_pca_v7_ALL, so every consumer must pair it
# with lm_basis_v8_all_*, never the ccby basis.
run $G 24G 3600 $W/tools/s8c.log $P scripts/build_gt_coeffs.py \
    --pca $T/ear_pca_v7_all_mm.npz --fits $K/hutubs $K/sonicom $K/audioear \
    --template $T/ear_mean_v4.obj --out $T/gt_coeffs_v7_mm.npz

say "=== done."
