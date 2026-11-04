#!/bin/bash
# Frozen-calibration-v3 preprocessing grid on TACC (296 configurations, DS none, within_side_split+mnncovlocal) for
# Splatter N = 5,000, LTL331 (PDX) and KP-Tracer (CRISPR lineage). The user submits; nothing runs by itself.
#
#   bash realdata_benchmark/v3_grid/tacc/submit_v3.sh [--dry-run] [--nodes N] [--hours H] [--procs P] [--fit-workers W] DATASET chain
#   bash realdata_benchmark/v3_grid/tacc/submit_v3.sh [--dry-run] DATASET worklist
#   bash realdata_benchmark/v3_grid/tacc/submit_v3.sh [--dry-run] [--after ID] DATASET pilot
#   bash realdata_benchmark/v3_grid/tacc/submit_v3.sh [--dry-run] [--after ID] [--nodes N] [--hours H] DATASET run
#   bash realdata_benchmark/v3_grid/tacc/submit_v3.sh [--dry-run] [--after ID] DATASET collect
#   bash realdata_benchmark/v3_grid/tacc/submit_v3.sh DATASET status | reclaim
#
# DATASET  splatter | ltl331 | kp
# chain    worklist (short job) -> pilot (1 node, 2 h, PILOT_WORKLIST.tsv; afterok worklist) -> run (N nodes; afterany the
#          pilot, so a failed pilot row does not hold the run) -> collect (afterany every run task)
# worklist build_worklist.py: WORKLIST.tsv, PILOT_WORKLIST.tsv, FROZEN_CALIBRATION_V3.json (refuses a different one)
# pilot    one node on PILOT_WORKLIST.tsv (time per fit, memory); its rows are not fitted again by run
# run      N array tasks on WORKLIST.tsv; finished rows are skipped, so a resubmission continues
# collect  collect.py (STATUS, FIT_METRICS, splatter RESULTS_grid_v3tacc.tsv + summarize_local.py grid summary,
#          ltl331 / kp CELL_GATES per unit, ltl331 COUPLING_CLUSTERS)
# status   rows done / failed / claimed / open (read only)
# reclaim  only when none of this dataset's jobs is queued or running: removes the claim of rows with neither SUCCESS
#          nor FAILED (rows whose node stopped), so the next run takes them again
# Inputs (read only): splatter  $V3_SPLATTER_ROOT (default the paired splatter_default root, conditions/N5000, reps 1-3)
#                     ltl331    $V3_LTL331_RUN_ROOT (default ltl331_confidenceot, inputs/rawqc)
#                     kp        $V3_KP_UNITS (default the lineage factorial's UNITS.tsv)
# Output roots are new and versioned: $V3_ROOT (default /scratch/10119/ghzheng/OT_project/v3_grid_20261104)/<dataset>.
# Array tasks count one by one against the 40-job cap. No set -e / -u / pipefail (TACC rules).

dry_run=0; after=""; nodes=""; hours=""; procs=""; fitw=""; dataset=""; stage=""
while (( $# )); do
    case "$1" in
        --dry-run) dry_run=1; shift ;;
        --after) after=$2; shift 2 ;;
        --nodes) nodes=$2; shift 2 ;;
        --hours) hours=$2; shift 2 ;;
        --procs) procs=$2; shift 2 ;;
        --fit-workers) fitw=$2; shift 2 ;;
        splatter|ltl331|kp) dataset=$1; shift ;;
        chain|worklist|pilot|run|collect|status|reclaim) stage=$1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$dataset" && -n "$stage" ]] || { echo "usage: bash realdata_benchmark/v3_grid/tacc/submit_v3.sh [--dry-run] [--after ID] [--nodes N] [--hours H] [--procs P] [--fit-workers W] splatter|ltl331|kp chain|worklist|pilot|run|collect|status|reclaim" >&2; exit 2; }
repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
env_path=${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}
base=${V3_ROOT:-/scratch/10119/ghzheng/OT_project/v3_grid_20261104}
root=$base/$dataset
cap=${CONFIDENCEOT_JOB_CAP:-40}
here=realdata_benchmark/v3_grid
# per dataset: worklist inputs, default nodes x hours, workers per node x 02_run_pair workers per fit (the pilot measures)
case $dataset in
    splatter) src=${V3_SPLATTER_ROOT:-/scratch/10119/ghzheng/OT_project/splatter_n5000_default_paired_20261103}
              wl_args="--splatter-root $src --replicates 1 2 3 --pilot 36"; d_nodes=10; d_hours=24; d_procs=36; d_fitw=4; max_large=36; large_size=1e9 ;;
    ltl331)   src=${V3_LTL331_RUN_ROOT:-/scratch/10119/ghzheng/OT_project/ltl331_confidenceot}
              wl_args="--ltl331-run-root $src --pilot 14"; d_nodes=4; d_hours=24; d_procs=12; d_fitw=8; max_large=6; large_size=2e7 ;;
    kp)       src=${V3_KP_UNITS:-/scratch/10119/ghzheng/OT_project/lineage_preprocessing_factorial_20261103/UNITS.tsv}
              wl_args="--kp-units $src --pilot 36"; d_nodes=10; d_hours=24; d_procs=36; d_fitw=4; max_large=4; large_size=2e7 ;;
esac
nodes=${nodes:-$d_nodes}; hours=${hours:-$d_hours}; procs=${procs:-$d_procs}; fitw=${fitw:-$d_fitw}
activate="source /home1/10119/ghzheng/.bashrc; conda activate $env_path; cd $repo; export PYTHONPATH=$repo/src:$repo/cancer_metastasis:$repo/realdata_benchmark:$repo"

submit() {
    if (( dry_run )); then echo "DRY RUN  sbatch $*" | cut -c1-400 >&2; echo "DRY$RANDOM"; return 0; fi
    local id; id=$(sbatch --parsable "$@" | tail -n 1); id=${id%%;*}
    [[ "$id" =~ ^[0-9]+$ ]] || { echo "sbatch returned no job id" >&2; return 1; }
    echo "$id"
}
dep() { [[ -n "$2" ]] && echo "--dependency=$1:$2"; }
(( dry_run )) || mkdir -p "$root/logs/workers" || exit 1

do_worklist() {
    [[ -e "$src" ]] || (( dry_run )) || { echo "input $src does not exist" >&2; return 1; }
    local id; id=$(submit -p gg -N 1 -n 1 -t 01:00:00 -A MCB26031 -J "v3${dataset:0:2}_wl" -o "$root/logs/worklist_%j.out" \
        --wrap "$activate && python $here/build_worklist.py --dataset $dataset --root $root $wl_args") || return 1
    echo "worklist job $id" >&2; echo "$id"
}
do_array() {  # $1 dependency (type:ids or empty), $2 worklist file, $3 tasks, $4 hours, $5 name
    local queued; queued=$(squeue -u "$USER" -h -r 2> /dev/null | wc -l)
    (( dry_run || queued + $3 <= cap )) || { echo "$queued queued; $3 more would pass the cap of $cap" >&2; return 3; }
    local id; id=$(submit ${1:+--dependency=$1} -p gg -c 144 -t "$(printf '%02d:00:00' "$4")" -A MCB26031 -J "$5" --array=0-$(( $3 - 1 )) \
        -o "$root/logs/$5_%A_%a.out" \
        --export=ALL,CONFIDENCEOT_REPO="$repo",CONFIDENCEOT_ENV="$env_path",V3_WORKLIST="$root/$2",V3_PROCS="$procs",V3_FIT_WORKERS="$fitw",V3_LOGS="$root/logs/workers",V3_MAX_LARGE="$max_large",V3_LARGE_SIZE="$large_size" \
        "$repo/$here/tacc/v3_workers.slurm") || return 1
    echo "$5 array $id: $3 node(s) x $4 h, $procs workers x $fitw fit workers, on $2${1:+ after $1}" >&2; echo "$id"
}
do_collect() {  # $1 dependency
    local summary=""
    [[ $dataset == splatter ]] && summary=" && python realdata_benchmark/splatter_local/summarize_local.py --root $root --stage grid --tag v3tacc --frozen FROZEN_CALIBRATION_V3.json"
    local id; id=$(submit ${1:+--dependency=$1} -p gg -N 1 -n 1 -t 04:00:00 -A MCB26031 -J "v3${dataset:0:2}_col" -o "$root/logs/collect_%j.out" \
        --wrap "$activate && python $here/collect.py --root $root --tag v3tacc$summary") || return 1
    echo "collect job $id${1:+ after $1}" >&2; echo "$id"
}
name=v3${dataset:0:2}

case $stage in
chain)
    w=$(do_worklist) || exit 1
    p=$(do_array "afterok:$w" PILOT_WORKLIST.tsv 1 2 "${name}_pilot") || exit 1
    r=$(do_array "afterany:$p" WORKLIST.tsv "$nodes" "$hours" "${name}_run") || exit 1
    do_collect "afterany:$r" > /dev/null || exit 1
    ;;
worklist) do_worklist > /dev/null || exit 1 ;;
pilot) do_array "${after:+afterok:$after}" PILOT_WORKLIST.tsv 1 2 "${name}_pilot" > /dev/null || exit 1 ;;
run)
    [[ -f "$root/WORKLIST.tsv" ]] || (( dry_run )) || [[ -n "$after" ]] || { echo "no $root/WORKLIST.tsv (worklist stage)" >&2; exit 1; }
    do_array "${after:+afterany:$after}" WORKLIST.tsv "$nodes" "$hours" "${name}_run" > /dev/null || exit 1
    ;;
collect) do_collect "${after:+afterany:$after}" > /dev/null || exit 1 ;;
status)
    for wl in PILOT_WORKLIST.tsv WORKLIST.tsv; do
        [[ -f "$root/$wl" ]] || continue
        awk -F'\t' 'NR == 1 { for (i = 1; i <= NF; i++) if ($i == "out") c = i; next } { print $c }' "$root/$wl" | {
            n=0; ok=0; bad=0; run=0
            while IFS= read -r out; do
                n=$(( n + 1 ))
                if [[ -f "$out/SUCCESS" ]]; then ok=$(( ok + 1 )); elif [[ -f "$out/FAILED" ]]; then bad=$(( bad + 1 )); elif [[ -d "$out/.claim" ]]; then run=$(( run + 1 )); fi
            done
            echo "$dataset $wl: $n rows (8 fits each), $ok done, $bad failed, $run claimed (running or stopped), $(( n - ok - bad - run )) open"
        }
    done
    ;;
reclaim)
    active=$(squeue -u "$USER" -h -o "%j" 2> /dev/null | grep -c "^${name}_")
    (( active == 0 )) || { echo "$active ${name}_ jobs are queued or running; reclaim only when none is" >&2; exit 2; }
    n=0
    while IFS= read -r out; do
        if [[ -d "$out/.claim" && ! -f "$out/SUCCESS" && ! -f "$out/FAILED" ]]; then rm -rf "$out/.claim"; n=$(( n + 1 )); fi
    done < <(awk -F'\t' 'NR == 1 { for (i = 1; i <= NF; i++) if ($i == "out") c = i; next } { print $c }' "$root/WORKLIST.tsv")
    echo "reclaimed $n rows"
    ;;
esac
(( dry_run )) || [[ $stage == status || $stage == reclaim ]] || echo "v3 grid $dataset $stage submitted $(date -Iseconds), commit $(git -C "$repo" rev-parse --short HEAD 2> /dev/null)" >> "$root/SUBMITTED.txt"
