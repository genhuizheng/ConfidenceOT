#!/bin/bash
# Region-sharded inferCNV BayesNet for one unit, in three phases.
#
#   bash submit_bayesnet_shards.sh test   [--dry-run]
#       The whole workflow on inferCNV's bundled example (one gg job); prints
#       PASS or FAIL in test/test.log.
#   bash submit_bayesnet_shards.sh freeze [--dry-run]
#       Steps 1-17 of the unit once, with the unchanged run script (save_rds
#       copy), stopped at step 18 with the BayesNet inputs frozen; then the shard
#       plan (gg). Read <unit>/infercnv_out/bayesnet_shards/shard_plan.tsv.
#   bash submit_bayesnet_shards.sh run    [--dry-run]
#       One job per shard (one process each, one node each), then one job that
#       merges them by region index into the step-18 object and reruns the
#       unchanged run script, which resumes at step 18 and finishes, followed
#       by the unit diagnostics. A rerun submits only shards without output.
#
# Settings (environment): BAYESNET_UNIT (default Patient5's pilot directory),
# BAYESNET_RUN_SCRIPT (default the save_rds copy of run_infercnv_no_inspect.R),
# BAYESNET_THREADS for steps 1-17 and the resume (default 2), BAYESNET_SHARDS
# (default 16).
#
# No set -u, as everywhere in this repo.
set -eo pipefail

phase=${1:-}
[[ $# -gt 0 ]] && shift
dry_run=0
[[ "${1:-}" == --dry-run ]] && dry_run=1

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
tools=$repo/cancer_metastasis/tools/infercnv_bayesnet
checks=/scratch/10119/ghzheng/checks/prostate
unit=${BAYESNET_UNIT:-$checks/pilot/Patient5}
run_script=${BAYESNET_RUN_SCRIPT:-$checks/run_infercnv_no_inspect_saverds.R}
threads=${BAYESNET_THREADS:-2}
n_shards=${BAYESNET_SHARDS:-16}
env_r=/scratch/10119/ghzheng/conda_envs/infercnv_r
out=$unit/infercnv_out
frozen=$out/bayesnet_frozen_inputs.rds
shards=$out/bayesnet_shards
name=$(basename "$unit")
activate="source /home1/10119/ghzheng/.bashrc; conda activate $env_r"

submit() {
    local what=$1; shift
    if (( dry_run )); then
        echo "DRY RUN  $what" >&2; echo "         sbatch $*" >&2; echo "DRYRUN"; return 0
    fi
    local id
    id=$(sbatch --parsable "$@" | tail -n 1)
    if [[ ! "$id" =~ ^[0-9]+$ ]]; then
        echo "sbatch returned no job id for $what; stopping" >&2; exit 1
    fi
    echo "submitted $what as $id" >&2
    echo "$id"
}

case "$phase" in
    test)
        mkdir -p "$checks/bayesnet_test"
        submit "BayesNet shard test on the inferCNV example" -p gg -N 1 -t 04:00:00 -A MCB26031 -J bn_test \
            -o "$checks/bayesnet_test/test_%j.log" \
            --wrap "$activate && Rscript $tools/test_example.R $checks/bayesnet_test 3" > /dev/null
        ;;
    freeze)
        if [[ ! -f "$run_script" ]]; then echo "missing run script: $run_script" >&2; exit 2; fi
        if [[ -e "$out" ]]; then
            echo "$out exists; move it aside first, e.g. mv $out $out.before_shards" >&2
            exit 2
        fi
        submit "freeze steps 1-17 of $name and plan $n_shards shards" -p gg -N 1 -t 12:00:00 -A MCB26031 -J bn_freeze \
            -o "$checks/bayesnet_freeze_${name}_%j.log" \
            --wrap "$activate && cd $(dirname "$run_script") && BAYESNET_RUN_SCRIPT=$run_script Rscript $tools/freeze_step17.R --in $unit --threads $threads && Rscript $tools/shard.R --frozen $frozen --plan --n-shards $n_shards --shards-dir $shards" > /dev/null
        ;;
    run)
        if [[ ! -f "$shards/shard_plan.tsv" ]]; then echo "no shard plan: run the freeze phase first" >&2; exit 2; fi
        planned=$(awk -F'\t' 'NR==1{for(i=1;i<=NF;i++)c[$i]=i; next}{print $c["shard"]}' "$shards/shard_plan.tsv" | sort -n | uniq)
        ids=()
        for k in $planned; do
            if [[ -f $(printf '%s/shard_%03d.rds' "$shards" "$k") ]]; then
                echo "shard $k already done" >&2; continue
            fi
            ids+=("$(submit "shard $k of $name" -p gg -N 1 -t 24:00:00 -A MCB26031 -J bn_shard \
                -o "$shards/shard_${k}_%j.log" \
                --wrap "$activate && Rscript $tools/shard.R --frozen $frozen --shard $k --shards-dir $shards")")
        done
        dependency=()
        if (( ${#ids[@]} )); then dependency=(--dependency="afterok:$(IFS=:; echo "${ids[*]}")"); fi
        submit "merge by region index and resume $name at step 18" -p gg -N 1 -t 12:00:00 -A MCB26031 -J bn_merge \
            "${dependency[@]}" -o "$checks/bayesnet_merge_${name}_%j.log" \
            --wrap "$activate && Rscript $tools/merge.R --frozen $frozen --shards-dir $shards && cd $(dirname "$run_script") && Rscript $run_script --in $unit --threads $threads && Rscript $repo/cancer_metastasis/tools/infercnv_unit_diagnostics.R $unit" > /dev/null
        ;;
    *)
        echo "usage: $0 test|freeze|run [--dry-run]" >&2; exit 2 ;;
esac
