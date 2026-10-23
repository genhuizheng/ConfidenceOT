#!/bin/bash
# The native simulation on five log1p representations, each with two baselines: ConfidenceOT
# (M4-E / M4-R gate, within-side acceptance 0.90, 0.95, 0.99) and Vanilla UOT (asked 2026-10-22).
# log1p is the code normalisation `log1p`: log1p of the raw counts, no library-size scaling, genes
# not scaled (noscale), the same as the log1p + Pearson arm already run:
#
#   1  log1p_noscale                 2,000 HVGs, PCA 30           squared Euclidean
#   2  log1p_allgenes_noscale        every gene, PCA 30           squared Euclidean
#   3  log1p_nopca_noscale           2,000 HVGs, no PCA           squared Euclidean
#   4  log1p_allgenes_nopca_noscale  every gene, no PCA           squared Euclidean
#   5  log1p_noscale_pcc             every gene, no PCA           2 - 2r (Pearson); OT already run
#
# Every cost is divided by the median sampled cross-side pair, as in every other arm. The pairs
# are the 180 of the native benchmark (3 sizes x 5 replicates x 3 levels x 4 cases) on the
# un-equalised counts: each arm's work list is the raw rows of the 32-arm work list, relabelled,
# dealt round robin by size into 9 blocks of 20, as submit_log1p_pcc.sh does.
#
#   bash benchmark/tacc/submit_log1p_dimensions.sh status
#   bash benchmark/tacc/submit_log1p_dimensions.sh ot --acceptance 0.90 [--arms "a b"] [--dry-run]
#   bash benchmark/tacc/submit_log1p_dimensions.sh uot [--arms "a b"] [--dry-run]
#   bash benchmark/tacc/submit_log1p_dimensions.sh score [--dry-run]
#
# ot     ConfidenceOT on arms 1-4 (default) at one acceptance, 9 tasks per arm. 0.90 writes runs/<arm>
#        (the grid's convention, which the UOT stage reads), 0.95 / 0.99 runs_acceptance_<a>/<arm>.
#        A pair already run is skipped, so a resubmission finishes what failed.
# uot    Vanilla UOT alone (solvers.slurm CONFIDENCEOT_SOLVER_UOT_ONLY=1: no Traditional OT, no
#        Partial OT) on arms 1-5, 4 tasks of 45 pairs each, into solvers_uot_only/<arm>; refuses an
#        arm whose acceptance-0.90 ConfidenceOT runs are not all done (the cost is checked against
#        them). Arm 5's are in runs_acceptance_0.90/.
# score  once everything is done: 56_comparator_metrics.py on the five arms with
#        --solvers-dir solvers_uot_only -> comparator_metrics_log1p_dims/, then a tarball and md5.
# Each stage refuses over the 40-task cap; pass --arms to submit fewer arms at a time.
#
# No set -u, as everywhere in this repo.
set -eo pipefail

stage=${1:-}
shift || true
acceptance=""
arms=""
dry_run=0
while (( $# )); do
    case "$1" in
        --acceptance) acceptance=$2; shift 2 ;;
        --arms) arms=$2; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
root=${CONFIDENCEOT_BENCH_ROOT:-/scratch/10119/ghzheng/OT_project/benchmark_native}
env_path=${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}
source_list=$root/worklist_1000_5000_10000.csv
logs=$root/logs
new_arms="log1p_noscale log1p_allgenes_noscale log1p_nopca_noscale log1p_allgenes_nopca_noscale"
pcc=log1p_noscale_pcc
all_arms="$new_arms $pcc"
out=$root/comparator_metrics_log1p_dims
blocks=9
chunk=20

# the 0.90 runs: runs/ for arms 1-4, runs_acceptance_0.90/ for the pcc arm
runs_090() { [[ "$1" == "$pcc" ]] && echo runs_acceptance_0.90 || echo runs; }
done_runs() { ls "$root/$1/$2"/*/scope_all/budget_*/SUCCESS 2> /dev/null | wc -l; }
done_uot() { ls "$root/solvers_uot_only/$1"/*/solver_comparison.csv 2> /dev/null | wc -l; }

if [[ "$stage" == status ]]; then
    printf '%-30s %8s %8s %8s %8s\n' arm "OT 0.90" "OT 0.95" "OT 0.99" UOT
    for arm in $all_arms; do
        printf '%-30s %8s %8s %8s %8s\n' "$arm" "$(done_runs "$(runs_090 "$arm")" "$arm")" \
            "$(done_runs runs_acceptance_0.95 "$arm")" "$(done_runs runs_acceptance_0.99 "$arm")" "$(done_uot "$arm")"
    done
    echo "(of 180 pairs each)"
    exit 0
fi
[[ "$stage" == ot || "$stage" == uot || "$stage" == score ]] || { echo "stage: status, ot, uot or score" >&2; exit 2; }
test -f "$source_list" || { echo "no $source_list" >&2; exit 2; }

submit() {
    if (( dry_run )); then echo "DRY RUN  sbatch $*" | cut -c1-300 >&2; echo 0; return 0; fi
    local id
    id=$(sbatch --parsable "$@" | tail -n 1 | tr -dc '0-9')
    [[ -n "$id" ]] || { echo "sbatch returned no job id; stopping (anything above was submitted)" >&2; exit 1; }
    echo "$id"
}
cap() {  # $1 = tasks this stage adds
    local queued
    queued=$( { squeue -u "$USER" -h -r 2> /dev/null || true; } | wc -l )
    if (( ! dry_run && queued + $1 > 40 )); then
        echo "$queued tasks queued and this adds $1, over the cap of 40; pass fewer --arms" >&2; exit 3
    fi
}

# arm's work list: the raw rows relabelled, sorted by size, dealt round robin into the blocks
make_worklist() {
    local arm=$1 list=$root/worklist_$1.csv rows count
    rows=$(awk -F',' -v OFS=',' -v arm="$arm" -v blocks="$blocks" '
        NR == 1 { next }
        $4 == "raw" {
            size = $3; sub(/^N/, "", size); sub(/_.*/, "", size)
            $4 = arm
            n[size]++; row[size, n[size]] = $0
        }
        END {
            split("1000 5000 10000", order, " ")
            total = 0
            for (k = 1; k <= 3; k++) for (i = 1; i <= n[order[k]]; i++) all[++total] = row[order[k], i]
            for (b = 0; b < blocks; b++) for (i = b + 1; i <= total; i += blocks) print all[i]
        }' "$source_list")
    count=$(printf '%s\n' "$rows" | grep -c . || true)
    [[ "$count" -eq 180 ]] || { echo "expected 180 raw rows in $source_list, found $count" >&2; exit 2; }
    (( dry_run )) && { echo "$list"; return; }
    { head -n 1 "$source_list"; printf '%s\n' "$rows"; } > "$list.tmp"
    if [[ -f "$list" ]] && ! cmp -s "$list" "$list.tmp"; then
        echo "$list exists and differs; not overwritten" >&2; rm -f "$list.tmp"; exit 2
    fi
    mv "$list.tmp" "$list"
    echo "$list"
}

(( dry_run )) || mkdir -p "$logs"

if [[ "$stage" == ot ]]; then
    [[ "$acceptance" == 0.90 || "$acceptance" == 0.95 || "$acceptance" == 0.99 ]] || { echo "--acceptance 0.90, 0.95 or 0.99" >&2; exit 2; }
    arms=${arms:-$new_arms}
    for arm in $arms; do [[ " $new_arms " == *" $arm "* ]] || { echo "$arm is not one of arms 1-4 (the pcc arm's OT is already run)" >&2; exit 2; }; done
    cap $(( $(wc -w <<< "$arms") * blocks ))
    acc_env=""
    [[ "$acceptance" != 0.90 ]] && acc_env=",CONFIDENCEOT_BENCH_WITHIN_SIDE_ACCEPTANCE=$acceptance"
    for arm in $arms; do
        list=$(make_worklist "$arm")
        id=$(submit --array=0-$(( blocks - 1 )) -J ld_ot_${acceptance} \
            --export=ALL,CONFIDENCEOT_REPO=$repo,CONFIDENCEOT_BENCH_ROOT=$root,CONFIDENCEOT_BENCH_WORKLIST=$list,CONFIDENCEOT_BENCH_CHUNK=$chunk,CONFIDENCEOT_BENCH_SAVE_EDGES=0$acc_env \
            -o "$logs/ld_ot_${arm}_${acceptance}_%A_%a.out" -e "$logs/ld_ot_${arm}_${acceptance}_%A_%a.err" \
            "$repo/benchmark/tacc/ot_array.slurm")
        echo "ConfidenceOT $arm at $acceptance: $id ($blocks tasks)"
    done
fi

if [[ "$stage" == uot ]]; then
    arms=${arms:-$all_arms}
    for arm in $arms; do
        n=$(done_runs "$(runs_090 "$arm")" "$arm")
        (( n == 180 )) || { echo "$arm: $n of 180 acceptance-0.90 ConfidenceOT runs; run the ot stage at 0.90 first" >&2; (( dry_run )) || exit 2; }
    done
    cap $(( $(wc -w <<< "$arms") * 4 ))
    for arm in $arms; do
        if [[ "$arm" == "$pcc" ]]; then
            list=$root/worklist_$pcc.csv  # submit_log1p_pcc.sh's
            test -f "$list" || { echo "no $list" >&2; exit 2; }
        else
            list=$(make_worklist "$arm")
        fi
        id=$(submit --array=0-3 -J ld_uot \
            --export=ALL,CONFIDENCEOT_REPO=$repo,CONFIDENCEOT_BENCH_ROOT=$root,CONFIDENCEOT_BENCH_WORKLIST=$list,CONFIDENCEOT_SOLVER_LABEL=$arm,CONFIDENCEOT_SOLVER_CHUNK=45,CONFIDENCEOT_SOLVER_UOT_ONLY=1,CONFIDENCEOT_SOLVER_RUNS_DIR=$(runs_090 "$arm") \
            -o "$logs/ld_uot_${arm}_%A_%a.out" -e "$logs/ld_uot_${arm}_%A_%a.err" \
            "$repo/benchmark/tacc/solvers.slurm")
        echo "Vanilla UOT on $arm: $id (4 tasks)"
    done
fi

if [[ "$stage" == score ]]; then
    [[ -e "$out" ]] && { echo "$out exists; not rewritten" >&2; exit 2; }
    missing=0
    for arm in $all_arms; do
        for where in "$(runs_090 "$arm")" runs_acceptance_0.95 runs_acceptance_0.99; do
            n=$(done_runs "$where" "$arm"); (( n == 180 )) || { echo "$where/$arm: $n of 180" >&2; missing=1; }
        done
        n=$(done_uot "$arm"); (( n == 180 )) || { echo "solvers_uot_only/$arm: $n of 180" >&2; missing=1; }
    done
    (( missing && ! dry_run )) && { echo "not everything is done; see status" >&2; exit 2; }
    cap 1
    arm_args=""
    for arm in $all_arms; do arm_args="$arm_args --arm $arm"; done
    job="source /home1/10119/ghzheng/.bashrc; conda activate $env_path; cd $repo; python benchmark/56_comparator_metrics.py $root $out --solvers-dir solvers_uot_only$arm_args && cd $root && tar czf comparator_metrics_log1p_dims.tgz comparator_metrics_log1p_dims && md5sum comparator_metrics_log1p_dims.tgz > comparator_metrics_log1p_dims.tgz.md5 && cat comparator_metrics_log1p_dims.tgz.md5"
    id=$(submit -p gg -N 1 -n 1 -c 16 -t 02:00:00 -A MCB26031 -J ld_score \
        -o "$logs/ld_score_%j.out" -e "$logs/ld_score_%j.err" --wrap "$job")
    echo "scoring: $id -> $out (and $root/comparator_metrics_log1p_dims.tgz)"
fi
echo "commit $(git -C "$repo" rev-parse --short HEAD 2> /dev/null || echo unknown)"
