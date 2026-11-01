#!/bin/bash
# Joint Splatter + LTL331 benchmark on TACC (PROPOSED_TACC_JOBS.md). The user submits; nothing here runs by itself.
#
#   bash realdata_benchmark/tacc/submit_benchmark.sh [--dry-run] worklist
#   bash realdata_benchmark/tacc/submit_benchmark.sh [--dry-run] [--nodes N] [--hours H] ltl331|splatter
#   bash realdata_benchmark/tacc/submit_benchmark.sh status
#
# worklist  one short gg job: 01_config_registry.py -> ROOT/PREPROCESSING_CONFIG_REGISTRY.tsv (from the code at this
#           commit; kept if it exists), then 11_build_worklists.py for whichever of ROOT/worklist_{ltl331,splatter}.tsv
#           is missing (an existing worklist is never rebuilt). Refused if both exist.
# ltl331    N nodes (default 2, 6 h) of benchmark_workers.slurm on worklist_ltl331.tsv
# splatter  N nodes (default 6, 8 h) on worklist_splatter.tsv
# status    rows done / failed / pending per worklist, read from the SUCCESS / FAILED flags
# Resubmitting ltl331 or splatter continues: finished rows are skipped. No set -e / -u / pipefail (TACC rules).

dry_run=0; nodes=""; hours=""; stage=""
while (( $# )); do
    case "$1" in
        --dry-run) dry_run=1; shift ;;
        --nodes) nodes=$2; shift 2 ;;
        --hours) hours=$2; shift 2 ;;
        worklist|ltl331|splatter|status) stage=$1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$stage" ]] || { echo "usage: bash realdata_benchmark/tacc/submit_benchmark.sh [--dry-run] [--nodes N] [--hours H] worklist|ltl331|splatter|status" >&2; exit 2; }
repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
root=${RB_ROOT:-/scratch/10119/ghzheng/OT_project/realdata_benchmark_20261031}
env_path=${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}
# benchmark_native = the original Splatter benchmark (ARCHIVE_MANIFEST.md); benchmark_technical is the superseded
# thinning benchmark (3 cases, no populations_disjoint) and must not be used.
splatter_root=${RB_SPLATTER_ROOT:-/scratch/10119/ghzheng/OT_project/benchmark_native}
ltl_root=${RB_LTL331_ROOT:-/scratch/10119/ghzheng/OT_project/ltl331_confidenceot}
ltl_converted=${RB_LTL331_CONVERTED:-/scratch/10119/ghzheng/OT_project/data/ltl331_GSE297328/converted}
cap=${CONFIDENCEOT_JOB_CAP:-40}
registry=$root/PREPROCESSING_CONFIG_REGISTRY.tsv
activate="source /home1/10119/ghzheng/.bashrc; conda activate $env_path; cd $repo; export PYTHONPATH=$repo/src:$repo/cancer_metastasis:$repo/realdata_benchmark:$repo"

submit() {
    if (( dry_run )); then echo "DRY RUN  sbatch $*" | cut -c1-600 >&2; echo DRYRUN; return 0; fi
    local id
    id=$(sbatch --parsable "$@" | tail -n 1)
    [[ "$id" =~ ^[0-9]+$ ]] || { echo "sbatch returned no job id" >&2; return 1; }
    echo "$id"
}

if [[ $stage == status ]]; then
    for w in ltl331 splatter; do
        f=$root/worklist_$w.tsv
        [[ -f "$f" ]] || { echo "$w: no worklist"; continue; }
        rows=$(( $(wc -l < "$f") - 1 ))
        done_n=$(find "$root/$w" -name SUCCESS 2> /dev/null | wc -l)
        fail_n=$(find "$root/$w" -name FAILED 2> /dev/null | wc -l)
        echo "$w: $rows rows, $done_n done, $fail_n failed, $(( rows - done_n - fail_n )) pending"
    done
    exit 0
fi

(( dry_run )) || mkdir -p "$root/logs" || exit 1
if [[ $stage == worklist ]]; then
    [[ -f "$root/worklist_ltl331.tsv" && -f "$root/worklist_splatter.tsv" ]] && { echo "both worklists exist in $root; not rebuilt" >&2; exit 2; }
    [[ -f "$ltl_root/inputs/rawqc/pair_manifest_rawqc.csv" ]] || { echo "missing LTL331 inputs under $ltl_root" >&2; exit 1; }
    [[ -d "$splatter_root/conditions" ]] || { echo "missing Splatter conditions under $splatter_root" >&2; exit 1; }
    cmd="$activate && { [[ -f $registry ]] || python realdata_benchmark/01_config_registry.py --out $root; } && python realdata_benchmark/11_build_worklists.py --registry $registry --out-root $root --splatter-root $splatter_root --ltl331-run-root $ltl_root --ltl331-converted $ltl_converted"
    id=$(submit -p gg -N 1 -n 1 -t 00:30:00 -A MCB26031 -J rb_work -o "$root/logs/worklist_%j.out" --wrap "$cmd") || exit 1
    echo "worklist job $id -> $root (log $root/logs/worklist_$id.out)"
    exit 0
fi

worklist=$root/worklist_$stage.tsv
[[ -f "$worklist" ]] || { echo "missing $worklist (run the worklist stage first)" >&2; exit 1; }
[[ -f "$registry" ]] || { echo "missing $registry" >&2; exit 1; }
if [[ $stage == ltl331 ]]; then nodes=${nodes:-2}; hours=${hours:-6}; else nodes=${nodes:-6}; hours=${hours:-8}; fi
queued=$(squeue -u "$USER" -h -r 2> /dev/null | wc -l)
if (( ! dry_run && queued + nodes > cap )); then echo "$queued queued; $nodes more would pass the cap of $cap" >&2; exit 3; fi
logdir=$root/logs/$stage
(( dry_run )) || mkdir -p "$logdir" || exit 1
for (( n = 0; n < nodes; n++ )); do
    id=$(submit -t "$(printf '%02d:00:00' "$hours")" -J "rb_$stage" -o "$logdir/node_%j.out" \
        --export=ALL,CONFIDENCEOT_REPO="$repo",CONFIDENCEOT_ENV="$env_path",RB_WORKLIST="$worklist",RB_REGISTRY="$registry",RB_LOGDIR="$logdir",RB_NODE="$n",RB_NODES="$nodes" \
        "$repo/realdata_benchmark/tacc/benchmark_workers.slurm") || exit 1
    echo "$stage node $n of $nodes: job $id"
done
(( dry_run )) || echo "$stage submitted $(date -Iseconds), $nodes nodes x $hours h, commit $(git -C "$repo" rev-parse --short HEAD 2> /dev/null)" >> "$root/SUBMITTED.txt"
