#!/bin/bash
#SBATCH -J cnv_rna_strat
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 72
#SBATCH -t 48:00:00
#SBATCH -A MCB26031
#
# One acceptance x one block: the five CNV / RNA strategies in sequence, in one job.
# Submitted by submit_cnv_rna_strategies.sh. One job instead of one per strategy, because
# array tasks count one by one against the 40-job cap, and because step 2 of a two-step
# strategy has to start after its step 1 and read the budget tag step 1 actually wrote.
#
# Each strategy is the preprocessing factorial's own per-label run
# (tacc_preprocessing_factorial.slurm, called as a script with one label and one worker),
# so manifests, QC (recorded, filters nothing), seeds, solver and null are the factorial's.
# Order: rna, cnv, rna_cnv (single); then cnv_then_rna (RNA on what cnv retained) and
# rna_then_cnv (CNV on what rna retained). A strategy that fails some pairs does not stop
# the others; the job exits 1 at the end if any strategy reported failed pairs, and each
# strategy's DONE/FAILED line is in STRATEGIES_<block>.log under the acceptance root.
#
# Environment (all set by the submit script): STRAT_ACCEPTANCE, STRAT_BLOCK, STRAT_OUT (the
# acceptance root), STRAT_CNV_ROOT, STRAT_MANIFEST, STRAT_MANIFEST_DS, STRAT_DEPTH,
# STRAT_ANNOTATIONS (empty for the uniform call), CONFIDENCEOT_REPO, CANCER_COT_ROOT.
#
# No set -u, as everywhere in this repo.
set -eo pipefail

repo=${CONFIDENCEOT_REPO:?}
acceptance=${STRAT_ACCEPTANCE:?}
block=${STRAT_BLOCK:?}
out=${STRAT_OUT:?}
cnv_root=${STRAT_CNV_ROOT:?}
stage=${STRAT_STAGE:?}   # single: rna, cnv, rna_cnv; two_step: cnv_then_rna, rna_then_cnv; all: both
[[ "$stage" == single || "$stage" == two_step || "$stage" == all ]] || { echo "STRAT_STAGE single, two_step or all" >&2; exit 2; }
label=ranknm256_noscale_ds_cos
factorial=$repo/cancer_metastasis/tacc_preprocessing_factorial.slurm
log=$out/STRATEGIES_${block}.log
mkdir -p "$out"
echo "== $(date -Iseconds) job ${SLURM_JOB_ID:-local} stage $stage acceptance $acceptance block $block commit $(git -C "$repo" rev-parse --short HEAD)" >> "$log"

common="--within-side-acceptance-minimum $acceptance --save-coupling"
status=0

run_strategy() {  # $1 = strategy, $2 = 02_run_pair.py extra arguments
    local strategy=$1 extra=$2
    echo "== $(date -Iseconds) START $strategy: $extra" | tee -a "$log"
    # SLURM_ARRAY_JOB_ID names the factorial's per-pair log directory; outside an array it
    # is empty, and every strategy would write the same logs/fact_local_0/<label>_<i>.log.
    if SLURM_ARRAY_TASK_ID=0 SLURM_ARRAY_JOB_ID="${SLURM_JOB_ID:-local}_${block}_acc${acceptance}_${strategy}" \
       CONFIDENCEOT_FACTORIAL_MANIFEST="${STRAT_MANIFEST:?}" \
       CONFIDENCEOT_FACTORIAL_MANIFEST_DS="${STRAT_MANIFEST_DS:?}" \
       CONFIDENCEOT_PREDOWNSAMPLE_DEPTH="${STRAT_DEPTH:?}" \
       CONFIDENCEOT_FACTORIAL_ROOT="$out/$strategy/arm_A" \
       CONFIDENCEOT_FACTORIAL_LABELS="$label" \
       CONFIDENCEOT_FACTORIAL_WORKERS=1 \
       CONFIDENCEOT_ANALYSIS_SCOPE=malignant CONFIDENCEOT_MALIGNANT_COLUMN=malignant \
       CONFIDENCEOT_DEVICE=cpu CONFIDENCEOT_FACTORIAL_FORCE=0 \
       CONFIDENCEOT_FACTORIAL_BLOCK="$block" \
       CONFIDENCEOT_INCLUDE_ANNOTATIONS="${STRAT_ANNOTATIONS:-}" \
       CONFIDENCEOT_MINIMUM_SCOPE_CELLS=4 \
       CONFIDENCEOT_FACTORIAL_EXTRA_ARGS="$extra" \
       bash "$factorial"; then
        echo "== $(date -Iseconds) DONE $strategy" | tee -a "$log"
    else
        echo "== $(date -Iseconds) FAILED $strategy (some pairs failed; see the factorial lines above)" | tee -a "$log"
        status=1
    fi
}

step1_tag() {  # the one budget tag a step-1 label directory was written under
    local tags
    tags=$(find "$1" -mindepth 4 -maxdepth 4 -name SUCCESS -path "*/scope_malignant/*" -printf '%h\n' 2> /dev/null \
           | xargs -r -n1 basename | sort -u)
    [[ $(printf '%s\n' "$tags" | grep -c .) -eq 1 ]] || return 1
    echo "$tags"
}

if [[ "$stage" != two_step ]]; then
    run_strategy rna "$common"
    run_strategy cnv "$common --representation-source cnv --cnv-root $cnv_root"
    run_strategy rna_cnv "$common --representation-source rna_cnv --cnv-root $cnv_root"
fi

[[ "$stage" == single ]] && { echo "== $(date -Iseconds) finished (single), status $status" | tee -a "$log"; exit $status; }
for entry in "cnv_then_rna:cnv:" "rna_then_cnv:rna:--representation-source cnv --cnv-root $cnv_root"; do
    strategy=${entry%%:*}; rest=${entry#*:}; first=${rest%%:*}; own=${rest#*:}
    step1=$out/$first/arm_A/$block/$label
    if ! tag=$(step1_tag "$step1"); then
        echo "== $(date -Iseconds) FAILED $strategy: step 1 ($step1) has no single budget tag; not run" | tee -a "$log"
        status=1
        continue
    fi
    done_n=$(find "$step1" -mindepth 4 -maxdepth 4 -name SUCCESS -path "*/scope_malignant/*" | wc -l)
    echo "== $strategy: step 1 $first has $done_n completed pairs, tag $tag" | tee -a "$log"
    run_strategy "$strategy" "$common $own --input-gate-root $step1 --input-gate-state retained --input-gate-method M4-E --input-gate-budget-tag $tag"
done

echo "== $(date -Iseconds) finished, status $status" | tee -a "$log"
exit $status
