#!/bin/bash
# LTL331 author-provided Harmony branch on TACC, beside the Non-Harmony branch (the 64 configurations already fitted).
# The user submits; nothing runs by itself. Every stage is one job; outputs never overwrite (each script refuses).
#
#   bash ltl331_nepc/submit_author_harmony.sh [--dry-run] audit
#   bash ltl331_nepc/submit_author_harmony.sh [--dry-run] --key KEY worklist
#   bash ltl331_nepc/submit_author_harmony.sh [--dry-run] [--hours H] run
#   bash ltl331_nepc/submit_author_harmony.sh [--dry-run] collect
#   bash ltl331_nepc/submit_author_harmony.sh [--dry-run] [--hours H] --key KEY pipeline
#   bash ltl331_nepc/submit_author_harmony.sh [--dry-run] nonharmony-export
#   bash ltl331_nepc/submit_author_harmony.sh status
#
# audit              09_author_harmony_audit.py: the authors' Harmony object (obs / obsm / uns, integration variables,
#                    cell-id alignment) -> $root/AUTHOR_HARMONY_AUDIT/ (HARMONY_AUDIT.md / .json,
#                    AUTHOR_EMBEDDINGS_LTL331.npz). Read HARMONY_AUDIT.md before choosing --key.
# worklist --key K   19_build_author_harmony_worklist.py: 7 pairs x {cos, sqeuc} on obsm key K -> worklist_ltl331_author_harmony.tsv
# run                one gg node, 14 workers (one per row), benchmark_workers.slurm; 05_ fits each row exactly as the
#                    64-configuration arm (same cells, solver, calibration, balanced OT, metrics)
# collect            12_ (PDX_GATE_COUPLING_METRICS_worklist_ltl331_author_harmony.tsv), 13_ (cell gates), 16_
#                    (calibration causes), 18_ (state x state coupling) for the Harmony worklist, then a tarball of the
#                    small tables for download
# pipeline --key K   worklist -> run (afterok the worklist job) -> collect (afterany the run job): the three stages
#                    above as one dependent chain; a failed worklist leaves run / collect in DependencyNeverSatisfied
# nonharmony-export  16_ and 18_ for the existing 64-configuration worklist (calibration causes; state x state coupling
#                    of every fit; reads the saved couplings, writes small tables only), then a tarball
# No set -e / -u / pipefail (TACC rules).

dry_run=0; key=""; hours=4; stage=""
while (( $# )); do
    case "$1" in
        --dry-run) dry_run=1; shift ;;
        --key) key=$2; shift 2 ;;
        --hours) hours=$2; shift 2 ;;
        audit|worklist|run|collect|pipeline|nonharmony-export|status) stage=$1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$stage" ]] || { echo "usage: bash ltl331_nepc/submit_author_harmony.sh [--dry-run] [--key KEY] [--hours H] audit|worklist|run|collect|pipeline|nonharmony-export|status" >&2; exit 2; }
repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
root=${RB_ROOT:-/scratch/10119/ghzheng/OT_project/realdata_benchmark_20261031}
data=${LTL331_DATA:-/scratch/10119/ghzheng/OT_project/data/ltl331_GSE297328}
converted=${LTL331_CONVERTED:-/scratch/10119/ghzheng/OT_project/data/ltl331_GSE297328/converted}
env_path=${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}
activate="source /home1/10119/ghzheng/.bashrc; conda activate $env_path; cd $repo; export PYTHONPATH=$repo/src:$repo/cancer_metastasis:$repo/realdata_benchmark:$repo"
audit=$root/AUTHOR_HARMONY_AUDIT
wl=worklist_ltl331_author_harmony.tsv

submit() {
    if (( dry_run )); then echo "DRY RUN  sbatch $*" | cut -c1-500 >&2; echo "DRY$RANDOM"; return 0; fi
    local id; id=$(sbatch --parsable "$@" | tail -n 1)
    [[ "$id" =~ ^[0-9]+$ ]] || { echo "sbatch returned no job id" >&2; return 1; }
    echo "$id"
}
dep() { [[ -n "$2" ]] && echo "--dependency=$1:$2"; }  # $1 afterok / afterany, $2 job id (empty: none)
(( dry_run )) || mkdir -p "$root/logs/author_harmony" || exit 1

do_worklist() {
    [[ -n "$key" ]] || { echo "--key KEY required (an obsm key listed in $audit/HARMONY_AUDIT.md)" >&2; return 2; }
    [[ -f "$audit/AUTHOR_EMBEDDINGS_LTL331.npz" ]] || { echo "run the audit stage first" >&2; return 1; }
    [[ -f "$root/$wl" ]] && { echo "$root/$wl exists; not rebuilt" >&2; return 2; }
    local cmd="$activate && python realdata_benchmark/19_build_author_harmony_worklist.py --root $root --embedding $audit/AUTHOR_EMBEDDINGS_LTL331.npz --key $key"
    local id; id=$(submit -p gg -N 1 -n 1 -t 00:30:00 -A MCB26031 -J ah_work -o "$root/logs/author_harmony/worklist_%j.out" --wrap "$cmd") || return 1
    echo "worklist job $id -> $root/$wl" >&2; echo "$id"
}
do_run() {  # $1 job id to wait for (afterok); without one the worklist must exist
    [[ -n "$1" || -f "$root/$wl" ]] || { echo "missing $root/$wl (worklist stage)" >&2; return 1; }
    local id; id=$(submit $(dep afterok "$1") -p gg -c 144 -t "$(printf '%02d:00:00' "$hours")" -J ah_run -o "$root/logs/author_harmony/node_%j.out" \
        --export=ALL,CONFIDENCEOT_REPO="$repo",CONFIDENCEOT_ENV="$env_path",RB_WORKLIST="$root/$wl",RB_REGISTRY="$root/PREPROCESSING_CONFIG_REGISTRY.tsv",RB_LOGDIR="$root/logs/author_harmony",RB_NODE=0,RB_NODES=1,RB_PROCS=14 \
        "$repo/realdata_benchmark/tacc/benchmark_workers.slurm") || return 1
    echo "run job $id (14 rows, 1 node, $hours h)${1:+ after $1}" >&2; echo "$id"
}
do_collect() {  # $1 job id to wait for (afterany)
    local cmd="$activate && python realdata_benchmark/12_collect_tacc.py --root $root --ltl331-worklist $wl && python realdata_benchmark/13_export_pdx_cell_gates.py --root $root --worklist $wl && python realdata_benchmark/16_export_calibration_flags.py --root $root --worklist $wl && python realdata_benchmark/18_export_state_coupling.py --root $root --worklist $wl --converted $converted --out $root/PDX_STATE_COUPLING_author_harmony.tsv.gz && cd $root && tar -cf author_harmony_tables_\$(date +%Y%m%d).tar AUTHOR_HARMONY_AUDIT/HARMONY_AUDIT.md AUTHOR_HARMONY_AUDIT/HARMONY_AUDIT.json AUTHOR_HARMONY_AUDIT/AUTHOR_EMBEDDINGS_LTL331.npz PDX_GATE_COUPLING_METRICS_worklist_ltl331_author_harmony.tsv COLLECT_STATUS_worklist_ltl331_author_harmony.tsv PDX_CELL_GATES_worklist_ltl331_author_harmony.tsv.gz PDX_CALIBRATION_FLAGS_worklist_ltl331_author_harmony.tsv PDX_STATE_COUPLING_author_harmony.tsv.gz $wl"
    local id; id=$(submit $(dep afterany "$1") -p gg -N 1 -n 1 -t 02:00:00 -A MCB26031 -J ah_coll -o "$root/logs/author_harmony/collect_%j.out" --wrap "$cmd") || return 1
    echo "collect job $id${1:+ after $1} -> $root/author_harmony_tables_<date>.tar" >&2; echo "$id"
}

case $stage in
audit)
    [[ -e "$audit" ]] && { echo "$audit exists; not redone" >&2; exit 2; }
    cmd="$activate && python ltl331_nepc/09_author_harmony_audit.py --data $data --converted $converted --out $audit"
    id=$(submit -p gg -N 1 -n 1 -t 01:00:00 -A MCB26031 -J ah_audit -o "$root/logs/author_harmony/audit_%j.out" --wrap "$cmd") || exit 1
    echo "audit job $id -> $audit/HARMONY_AUDIT.md"
    ;;
worklist) do_worklist > /dev/null || exit 1 ;;
run) do_run "" > /dev/null || exit 1 ;;
collect) do_collect "" > /dev/null || exit 1 ;;
pipeline)
    w=$(do_worklist) || exit 1
    r=$(do_run "$w") || exit 1
    do_collect "$r" > /dev/null || exit 1
    ;;
nonharmony-export)
    cmd="$activate && python realdata_benchmark/16_export_calibration_flags.py --root $root && python realdata_benchmark/18_export_state_coupling.py --root $root --converted $converted --out $root/PDX_STATE_COUPLING.tsv.gz && cd $root && tar -cf nonharmony_state_tables_\$(date +%Y%m%d).tar PDX_CALIBRATION_FLAGS.tsv PDX_STATE_COUPLING.tsv.gz PDX_STATE_COUPLING_STATUS.tsv"
    id=$(submit -p gg -N 1 -n 1 -t 04:00:00 -A MCB26031 -J nh_export -o "$root/logs/author_harmony/nonharmony_export_%j.out" --wrap "$cmd") || exit 1
    echo "non-Harmony export job $id -> $root/nonharmony_state_tables_<date>.tar"
    ;;
status)
    [[ -f "$root/$wl" ]] || { echo "no $wl"; exit 0; }
    rows=0; ok=0; bad=0
    # the out column by header name: read -a with a tab IFS merges empty fields and shifts the index
    while IFS= read -r out; do
        rows=$(( rows + 1 ))
        [[ -f "$out/SUCCESS" ]] && ok=$(( ok + 1 )); [[ -f "$out/FAILED" ]] && bad=$(( bad + 1 ))
    done < <(awk -F'\t' 'NR == 1 { for (i = 1; i <= NF; i++) if ($i == "out") c = i; next } { print $c }' "$root/$wl")
    echo "$wl: $rows rows, $ok done, $bad failed, $(( rows - ok - bad )) pending"
    ;;
esac
(( dry_run )) || [[ $stage == status ]] || echo "author_harmony $stage submitted $(date -Iseconds), commit $(git -C "$repo" rev-parse --short HEAD 2> /dev/null)" >> "$root/SUBMITTED.txt"
