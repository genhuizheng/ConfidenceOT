#!/bin/bash
# Lineage preprocessing factorial on TACC (PLAN.md). The user submits; nothing runs by itself.
#
#   bash lineage_validation/preprocessing_factorial_20261103/tacc/submit_factorial.sh [--dry-run] prep
#   bash lineage_validation/preprocessing_factorial_20261103/tacc/submit_factorial.sh [--dry-run] [--after ID] [--procs P] pilot
#   bash lineage_validation/preprocessing_factorial_20261103/tacc/submit_factorial.sh [--dry-run] [--nodes N] [--hours H] [--procs P] run
#   bash lineage_validation/preprocessing_factorial_20261103/tacc/submit_factorial.sh status
#   bash lineage_validation/preprocessing_factorial_20261103/tacc/submit_factorial.sh reclaim
#
# prep     one chain: units (01_build_units.py: UNITS.tsv, DS_TARGETS.json) -> worklist (02_build_worklist.py: registry,
#          WORKLIST.tsv, PILOT_WORKLIST.tsv) -> pilot (one node on PILOT_WORKLIST.tsv, 3 h)
# pilot    the pilot node alone (after --after ID if given)
# run      N array tasks (default 20 nodes x 24 h) on WORKLIST.tsv; rows the pilot finished are skipped
# status   rows done / failed / claimed-and-unfinished / open, from the row directories (read only)
# reclaim  only when none of this factorial's jobs is queued or running: removes the claim of every row with neither
#          SUCCESS nor FAILED (a row whose node was stopped), so the next run takes it again
# Array tasks count one by one against the 40-job cap. No set -e / -u / pipefail (TACC rules).

dry_run=0; after=""; nodes=20; hours=24; procs=36; stage=""
while (( $# )); do
    case "$1" in
        --dry-run) dry_run=1; shift ;;
        --after) after=$2; shift 2 ;;
        --nodes) nodes=$2; shift 2 ;;
        --hours) hours=$2; shift 2 ;;
        --procs) procs=$2; shift 2 ;;
        prep|pilot|run|status|reclaim) stage=$1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$stage" ]] || { echo "usage: bash lineage_validation/preprocessing_factorial_20261103/tacc/submit_factorial.sh [--dry-run] [--after ID] [--nodes N] [--hours H] [--procs P] prep|pilot|run|status|reclaim" >&2; exit 2; }
repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
root=${LF_ROOT:-/scratch/10119/ghzheng/OT_project/lineage_preprocessing_factorial_20261103}
env_path=${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}
cap=${CONFIDENCEOT_JOB_CAP:-40}
here=lineage_validation/preprocessing_factorial_20261103
activate="source /home1/10119/ghzheng/.bashrc; conda activate $env_path; cd $repo; export PYTHONPATH=$repo/src:$repo/cancer_metastasis:$repo/realdata_benchmark:$repo"

submit() {
    if (( dry_run )); then echo "DRY RUN  sbatch $*" | cut -c1-400 >&2; echo "DRY$RANDOM"; return 0; fi
    local id; id=$(sbatch --parsable "$@" | tail -n 1); id=${id%%;*}
    [[ "$id" =~ ^[0-9]+$ ]] || { echo "sbatch returned no job id" >&2; return 1; }
    echo "$id"
}
dep() { [[ -n "$2" ]] && echo "--dependency=$1:$2"; }
(( dry_run )) || mkdir -p "$root/logs" || exit 1

do_units() {
    [[ -f "$root/UNITS.tsv" ]] && { echo "$root/UNITS.tsv exists; not rebuilt" >&2; return 2; }
    local id; id=$(submit -p gg -N 1 -n 1 -t 02:00:00 -A MCB26031 -J lf_units -o "$root/logs/units_%j.out" \
        --wrap "$activate && python $here/01_build_units.py --root $root") || return 1
    echo "units job $id" >&2; echo "$id"
}
do_worklist() {  # $1 job to wait for
    local id; id=$(submit $(dep afterok "$1") -p gg -N 1 -n 1 -t 00:30:00 -A MCB26031 -J lf_work -o "$root/logs/worklist_%j.out" \
        --wrap "$activate && python $here/02_build_worklist.py --root $root") || return 1
    echo "worklist job $id${1:+ after $1}" >&2; echo "$id"
}
do_array() {  # $1 job to wait for, $2 worklist file, $3 tasks, $4 hours, $5 name
    local queued; queued=$(squeue -u "$USER" -h -r 2> /dev/null | wc -l)
    (( dry_run || queued + $3 <= cap )) || { echo "$queued queued; $3 more would pass the cap of $cap" >&2; return 3; }
    local id; id=$(submit $(dep afterok "$1") -p gg -c 144 -t "$(printf '%02d:00:00' "$4")" -A MCB26031 -J "$5" --array=0-$(( $3 - 1 )) \
        -o "$root/logs/$5_%A_%a.out" \
        --export=ALL,CONFIDENCEOT_REPO="$repo",CONFIDENCEOT_ENV="$env_path",LF_ROOT="$root",LF_WORKLIST="$root/$2",LF_PROCS="$procs" \
        "$repo/$here/tacc/factorial_workers.slurm") || return 1
    echo "$5 array $id: $3 node(s) x $4 h, $procs workers each, on $2${1:+ after $1}" >&2; echo "$id"
}

case $stage in
prep)
    u=$(do_units) || exit 1
    w=$(do_worklist "$u") || exit 1
    do_array "$w" PILOT_WORKLIST.tsv 1 3 lf_pilot > /dev/null || exit 1
    ;;
pilot) do_array "$after" PILOT_WORKLIST.tsv 1 3 lf_pilot > /dev/null || exit 1 ;;
run)
    [[ -f "$root/WORKLIST.tsv" ]] || (( dry_run )) || { echo "no $root/WORKLIST.tsv (prep stage)" >&2; exit 1; }
    do_array "$after" WORKLIST.tsv "$nodes" "$hours" lf_run > /dev/null || exit 1
    ;;
status)
    for wl in PILOT_WORKLIST.tsv WORKLIST.tsv; do
        [[ -f "$root/$wl" ]] || continue
        awk -F'\t' 'NR == 1 { for (i = 1; i <= NF; i++) if ($i == "out") c = i; next } { print $c }' "$root/$wl" | {
            n=0; ok=0; bad=0; run=0
            while IFS= read -r out; do
                n=$(( n + 1 ))
                if [[ -f "$out/SUCCESS" ]]; then ok=$(( ok + 1 )); elif [[ -f "$out/FAILED" ]]; then bad=$(( bad + 1 )); elif [[ -d "$out/.claim" ]]; then run=$(( run + 1 )); fi
            done
            echo "$wl: $n rows, $ok done, $bad failed, $run claimed (running or stopped), $(( n - ok - bad - run )) open"
        }
    done
    ;;
reclaim)
    active=$(squeue -u "$USER" -h -o "%j" 2> /dev/null | grep -c "^lf_")
    (( active == 0 )) || { echo "$active lf_ jobs are queued or running; reclaim only when none is" >&2; exit 2; }
    n=0
    while IFS= read -r out; do
        if [[ -d "$out/.claim" && ! -f "$out/SUCCESS" && ! -f "$out/FAILED" ]]; then rm -rf "$out/.claim"; n=$(( n + 1 )); fi
    done < <(awk -F'\t' 'NR == 1 { for (i = 1; i <= NF; i++) if ($i == "out") c = i; next } { print $c }' "$root/WORKLIST.tsv")
    echo "reclaimed $n rows"
    ;;
esac
(( dry_run )) || [[ $stage == status || $stage == reclaim ]] || echo "lineage factorial $stage submitted $(date -Iseconds), commit $(git -C "$repo" rev-parse --short HEAD 2> /dev/null)" >> "$root/SUBMITTED.txt"
