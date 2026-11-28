#!/bin/bash
# OT on the prepared application units (PLAN_TACC.md step 3; user 2026-11-23): the two configurations
#   best preprocessing = ranknm256-midrank|none|scale|cos|pca50 (Splatter N = 5,000 L5 best, the report's main configuration)
#   logcpm only        = logcpm|none|noscale|sqeuc|pca30         (logCPM + PCA only, no cosine, no scaling)
# frozen calibration v3 (within_side_split + mnncovlocal, acceptance 0.99), M4-E and M4-R, UOT backbone, couplings saved;
# the v3 worker's manifest route (build_worklist --dataset larry). Inputs read only: <dataset>/manifest.csv and
# <dataset>/inputs/*.h5ad from 06_prepare_tacc.py. Output root new: <dataset>/ot_20261123.
#
#   bash submit_app_ot.sh [--dry-run] DATASET worklist|run|status
#   DATASET  kang | sciplex2 | schiebinger | celegans | pancreas | larry_invivo | splintr | zebrafish
#   (splintr / zebrafish added 2026-11-27, same two configurations; the root keeps the name ot_20261123 so that
#   every application is scored by the same code)
#
# worklist  build_worklist.py, units x 2 encoding keys (one fit per row with --configs); login node, one thread, timeout
# run       1 gg node per dataset (workers x fit workers per node set by unit size), worker.py --configs = the 2 configurations
# status    rows done / failed / open (read only)
# Python: the module-python3 venv $V3_VENV (default otdb_otbench, as submit_rna_cnv.sh). The user submits.
# No set -e / -u / pipefail (TACC rules).

dry_run=0; ds=""; stage=""
while (( $# )); do
    case "$1" in
        --dry-run) dry_run=1; shift ;;
        kang|sciplex2|schiebinger|celegans|pancreas|larry_invivo|splintr|zebrafish) ds=$1; shift ;;
        worklist|run|status) stage=$1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$ds" && -n "$stage" ]] || { echo "usage: bash $0 [--dry-run] kang|sciplex2|schiebinger|celegans|pancreas|larry_invivo|splintr|zebrafish worklist|run|status" >&2; exit 2; }
repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
venv=${V3_VENV:-/scratch/10119/ghzheng/OT_project/envs/otdb_otbench}
app=/scratch/10119/ghzheng/OT_project/app_search_20261117
root=$app/$ds/ot_20261123
manifest=$app/$ds/manifest.csv
configs="ranknm256-midrank|none|scale|cos|pca50;logcpm|none|noscale|sqeuc|pca30"
# workers x fit workers per node (144 cores): many small units -> many workers; 10,000 x 10,000 units -> fewer, wider fits
case $ds in
    sciplex2)                        procs=36; fitw=4;  hours=12 ;;
    schiebinger)                     procs=12; fitw=12; hours=24 ;;
    kang|celegans|splintr)           procs=8;  fitw=16; hours=24 ;;
    pancreas|larry_invivo|zebrafish) procs=6;  fitw=24; hours=24 ;;
esac
export PYTHONPATH=$repo/src:$repo/cancer_metastasis:$repo/realdata_benchmark:$repo
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

run() { if (( dry_run )); then echo "DRY RUN  $*" >&2; else "$@"; fi; }
[[ -x "$venv/bin/python" ]] || { echo "missing $venv/bin/python" >&2; exit 1; }

case $stage in
worklist)
    [[ -f "$manifest" ]] || { echo "missing $manifest" >&2; exit 1; }
    cd "$repo" || exit 1
    run timeout 600 "$venv/bin/python" realdata_benchmark/v3_grid/build_worklist.py --dataset larry --root "$root" \
        --larry-manifest "$manifest" --keys 'ranknm256-midrank|none' 'logcpm|none'
    ;;
run)
    [[ -f "$root/WORKLIST.tsv" ]] || { echo "missing $root/WORKLIST.tsv (worklist first)" >&2; exit 1; }
    queued=$(squeue -u "$USER" -h -r 2> /dev/null | wc -l)
    (( dry_run || queued + 1 <= 40 )) || { echo "$queued queued; 1 more would pass the cap of 40" >&2; exit 3; }
    run mkdir -p "$root/logs/workers" || exit 1
    run sbatch --parsable --array=0-0 -t "$hours:00:00" -J "app_${ds}" -o "$root/logs/run_%A_%a.out" \
        --export=ALL,CONFIDENCEOT_REPO="$repo",V3_VENV="$venv",V3_WORKLIST="$root/WORKLIST.tsv",V3_PROCS="$procs",V3_FIT_WORKERS="$fitw",V3_LOGS="$root/logs/workers",V3_MAX_LARGE=4,V3_LARGE_SIZE=5e7,V3_CONFIGS="$configs" \
        "$repo/realdata_benchmark/v3_grid/tacc/v3_workers.slurm"
    (( dry_run )) || echo "app OT $ds run submitted $(date -Iseconds), commit $(git -C "$repo" rev-parse --short HEAD)" >> "$root/SUBMITTED.txt"
    ;;
status)
    n=$(tail -n +2 "$root/WORKLIST.tsv" | wc -l)
    ok=$(ls "$root"/rows/*/*/SUCCESS 2> /dev/null | wc -l); bad=$(ls "$root"/rows/*/*/FAILED 2> /dev/null | wc -l)
    echo "$ds rows $n: SUCCESS $ok, FAILED $bad, open $((n - ok - bad))"
    for f in $(ls "$root"/rows/*/*/FAILED 2> /dev/null | head -5); do echo "-- $f"; cat "$f"; echo; done
    ;;
esac
