#!/bin/bash
# Splatter N = 5,000 benchmark, levels L0-L5 + ablations, on TACC. The user submits; nothing runs by itself.
#
#   bash realdata_benchmark/splatter_n5000/tacc/submit_n5000.sh [--dry-run] cleanup-audit
#   bash realdata_benchmark/splatter_n5000/tacc/submit_n5000.sh [--dry-run] [--pilot] generate
#   bash realdata_benchmark/splatter_n5000/tacc/submit_n5000.sh [--dry-run] [--pilot] worklist
#   bash realdata_benchmark/splatter_n5000/tacc/submit_n5000.sh [--dry-run] [--pilot] [--nodes N] [--hours H] run
#   bash realdata_benchmark/splatter_n5000/tacc/submit_n5000.sh status | collect
#
# cleanup-audit  read-only: TACC_CLEANUP_MANIFEST.tsv of the old Splatter directories (cleanup_audit.sh); deletes nothing
# generate       one gg job per replicate (generate_n5000.slurm): 8 levels generated, L0-L2 checked against the native
#                generation, conditions + truth built. --pilot: replicate 1 only
# worklist       one short job: registry (from the code) + 31_build_n5000_worklist.py; --pilot: the 18-row pilot worklist
#                (rep 1, Case 2 source-unmatched, L2 reused + L4 new, 8 configurations + condition rows)
# run            N nodes (default 4, 6 h; pilot 1 node, 3 h) of realdata_benchmark/tacc/benchmark_workers.slurm
# collect        one short job: 32_collect_n5000.py -> the deliverable tables
# No set -e / -u / pipefail (TACC rules). Root is versioned; frozen roots are read only.

dry_run=0; pilot=0; nodes=""; hours=""; stage=""
while (( $# )); do
    case "$1" in
        --dry-run) dry_run=1; shift ;;
        --pilot) pilot=1; shift ;;
        --nodes) nodes=$2; shift 2 ;;
        --hours) hours=$2; shift 2 ;;
        cleanup-audit|generate|worklist|run|status|collect) stage=$1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$stage" ]] || { echo "usage: bash realdata_benchmark/splatter_n5000/tacc/submit_n5000.sh [--dry-run] [--pilot] [--nodes N] [--hours H] cleanup-audit|generate|worklist|run|status|collect" >&2; exit 2; }
repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
root=${RB5_ROOT:-/scratch/10119/ghzheng/OT_project/splatter_n5000_benchmark_20261101}
native=${RB5_NATIVE_ROOT:-/scratch/10119/ghzheng/OT_project/benchmark_native}
frozen_fits=${RB5_FROZEN_FITS:-/scratch/10119/ghzheng/OT_project/realdata_benchmark_20261031/splatter/acc0.99}
env_path=${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}
cap=${CONFIDENCEOT_JOB_CAP:-40}
activate="source /home1/10119/ghzheng/.bashrc; conda activate $env_path; cd $repo; export PYTHONPATH=$repo/src:$repo/cancer_metastasis:$repo/realdata_benchmark:$repo"

submit() {
    if (( dry_run )); then echo "DRY RUN  sbatch $*" | cut -c1-500 >&2; echo DRYRUN; return 0; fi
    local id; id=$(sbatch --parsable "$@" | tail -n 1)
    [[ "$id" =~ ^[0-9]+$ ]] || { echo "sbatch returned no job id" >&2; return 1; }
    echo "$id"
}
(( dry_run )) || mkdir -p "$root/logs" || exit 1

case $stage in
cleanup-audit)
    bash "$repo/realdata_benchmark/splatter_n5000/tacc/cleanup_audit.sh" "$root/TACC_CLEANUP_MANIFEST.tsv"
    ;;
generate)
    [[ -d "$native/generated" ]] || { echo "missing $native/generated" >&2; exit 1; }
    reps=(1 2 3); (( pilot )) && reps=(1)
    for rep in "${reps[@]}"; do
        id=$(submit -t 06:00:00 -J rb5_gen -o "$root/logs/generate_rep${rep}_%j.out" \
            --export=ALL,CONFIDENCEOT_REPO="$repo",CONFIDENCEOT_ENV="$env_path",RB5_ROOT="$root",RB5_REP="$rep",RB5_NATIVE_ROOT="$native" \
            "$repo/realdata_benchmark/splatter_n5000/tacc/generate_n5000.slurm") || exit 1
        echo "generate rep $rep: job $id"
    done
    ;;
worklist)
    wl=$root/TACC_N5000_WORKLIST.tsv; pilot_flag=""
    (( pilot )) && { wl=$root/TACC_N5000_PILOT_WORKLIST.tsv; pilot_flag="--pilot"; }
    [[ -f "$wl" ]] && { echo "$wl exists; not rebuilt" >&2; exit 2; }
    cmd="$activate && { [[ -f $root/PREPROCESSING_CONFIG_REGISTRY.tsv ]] || python realdata_benchmark/01_config_registry.py --out $root; } && python realdata_benchmark/splatter_n5000/31_build_n5000_worklist.py --root $root --registry $root/PREPROCESSING_CONFIG_REGISTRY.tsv --frozen-fits $frozen_fits $pilot_flag"
    id=$(submit -p gg -N 1 -n 1 -t 00:30:00 -A MCB26031 -J rb5_work -o "$root/logs/worklist_%j.out" --wrap "$cmd") || exit 1
    echo "worklist job $id"
    ;;
run)
    wl=$root/TACC_N5000_WORKLIST.tsv; (( pilot )) && wl=$root/TACC_N5000_PILOT_WORKLIST.tsv
    [[ -f "$wl" ]] || { echo "missing $wl" >&2; exit 1; }
    if (( pilot )); then nodes=${nodes:-1}; hours=${hours:-3}; else nodes=${nodes:-4}; hours=${hours:-6}; fi
    queued=$(squeue -u "$USER" -h -r 2> /dev/null | wc -l)
    (( dry_run || queued + nodes <= cap )) || { echo "$queued queued; $nodes more would pass the cap of $cap" >&2; exit 3; }
    (( dry_run )) || mkdir -p "$root/logs/run" || exit 1
    for (( n = 0; n < nodes; n++ )); do
        id=$(submit -t "$(printf '%02d:00:00' "$hours")" -J rb5_run -o "$root/logs/run/node_%j.out" \
            --export=ALL,CONFIDENCEOT_REPO="$repo",CONFIDENCEOT_ENV="$env_path",RB_WORKLIST="$wl",RB_REGISTRY="$root/PREPROCESSING_CONFIG_REGISTRY.tsv",RB_LOGDIR="$root/logs/run",RB_NODE="$n",RB_NODES="$nodes" \
            "$repo/realdata_benchmark/tacc/benchmark_workers.slurm") || exit 1
        echo "run node $n of $nodes: job $id"
    done
    ;;
status)
    f=$root/TACC_N5000_WORKLIST.tsv
    [[ -f "$f" ]] || { echo "no worklist"; exit 0; }
    rows=0; done_n=0; fail_n=0
    while IFS=$'\t' read -r -a field; do
        out=${field[15]}; rows=$(( rows + 1 ))
        [[ -f "$out/SUCCESS" ]] && done_n=$(( done_n + 1 )); [[ -f "$out/FAILED" ]] && fail_n=$(( fail_n + 1 ))
    done < <(tail -n +2 "$f")
    echo "n5000: $rows rows, $done_n done, $fail_n failed, $(( rows - done_n - fail_n )) pending"
    ;;
collect)
    cmd="$activate && python realdata_benchmark/splatter_n5000/32_collect_n5000.py --root $root"
    id=$(submit -p gg -N 1 -n 1 -t 01:00:00 -A MCB26031 -J rb5_coll -o "$root/logs/collect_%j.out" --wrap "$cmd") || exit 1
    echo "collect job $id"
    ;;
esac
(( dry_run )) || [[ $stage == status ]] || echo "$stage submitted $(date -Iseconds), commit $(git -C "$repo" rev-parse --short HEAD 2> /dev/null)" >> "$root/SUBMITTED.txt"
