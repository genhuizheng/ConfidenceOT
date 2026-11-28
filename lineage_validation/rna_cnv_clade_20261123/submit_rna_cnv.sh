#!/bin/bash
# RNA + CNV on the KP-Tracer / Quinn lineage units (user 2026-11-23): the frozen-calibration-v3 fits of the two RNA
# configurations used by the clade analysis, with the existing infercnvpy profiles added as a second block, and the
# RNA-only fits of the same configurations in the same environment beside them. Nothing else changes: same units (the
# lineage factorial's UNITS.tsv, 113), same cells, same RNA coordinates (encoding -> PCA -> unit rows / none ->
# mnncovlocal correction, v3_common.representations), same calibration (within_side_split, acceptance 0.99), same
# solver. rna_cnv = 02_run_pair --representation-source rna_cnv: [RNA / sqrt(median pair cost) | CNV PCA /
# sqrt(median pair cost)], CNV PCA = infercnvpy profile - 1, PCA with the configuration's n_pcs.
#
#   bash lineage_validation/rna_cnv_clade_20261123/submit_rna_cnv.sh [--dry-run] [--tool TOOL] ARM cnvroot|worklist|run|status
#
# ARM         rna        RNA only, the same environment as every CNV arm (the 2026-11-04 fits ran in the conda
#                        environment the scratch purge broke; compared with the 2026-11-04 gates first)
#             rna_cnv    c = c_rna / median + c_cnv / median (beta 1; the concatenated block-scaled coordinates)
#             rna_cnv_bB c = c_rna / median + B * c_cnv / median (02_run_pair --cnv-weight B), e.g. rna_cnv_b0.25
#             cnv        CNV only
# --tool      infercnvpy (default: the existing exports) | infercnv | copykat | scevan | conicsmat (the exports of
#             cnv_tools/, $base/cnv_exports/<tool>/<dataset_id>/<patient_id>/)
# root        $base/kp_<ARM> for infercnvpy, $base/kp_<ARM>__<tool> otherwise
# cnvroot   $root/cnv_root/{KPTracer,Quinn2021} -> the tool's export trees (symbolic links; nothing copied or
#           recomputed), so one --cnv-root serves both datasets (<root>/<dataset_id>/<patient_id>/)
# worklist  build_worklist.py --dataset kp --keys 'ranknm256-midrank|none' 'logcpm|none' (113 units x 2 keys = 226 rows)
# run       N gg nodes on WORKLIST.tsv, worker.py --configs = the two configurations only (one fit per row)
# status    rows done / failed / open (read only)
# Python: the module-python3 venv $V3_VENV (default the OT depth benchmark's otdb_otbench), login-node steps with one
# thread. Output roots are new. The user submits. No set -e / -u / pipefail (TACC rules).

dry_run=0; arm=""; stage=""; tool=infercnvpy
while (( $# )); do
    case "$1" in
        --dry-run) dry_run=1; shift ;;
        --tool) tool=$2; shift 2 ;;
        rna|rna_cnv|cnv|rna_cnv_b*) arm=$1; shift ;;
        cnvroot|worklist|run|status) stage=$1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$arm" && -n "$stage" ]] || { echo "usage: bash $0 [--dry-run] [--tool TOOL] rna|rna_cnv|rna_cnv_bB|cnv cnvroot|worklist|run|status" >&2; exit 2; }
case $tool in infercnvpy|infercnv|copykat|scevan|conicsmat) ;; *) echo "unknown tool $tool" >&2; exit 2 ;; esac
beta=""; source_=$arm
if [[ $arm == rna_cnv_b* ]]; then
    beta=${arm#rna_cnv_b}; source_=rna_cnv
    [[ "$beta" =~ ^[0-9]+(\.[0-9]+)?$ ]] || { echo "bad beta in $arm" >&2; exit 2; }
fi
[[ $arm == rna && $tool != infercnvpy ]] && { echo "rna takes no --tool" >&2; exit 2; }
repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
venv=${V3_VENV:-/scratch/10119/ghzheng/OT_project/envs/otdb_otbench}
base=${RNACNV_BASE:-/scratch/10119/ghzheng/OT_project/v3_grid_rnacnv_20261123}
root=$base/kp_$arm; [[ $tool != infercnvpy ]] && root=${root}__$tool
units=${V3_KP_UNITS:-/scratch/10119/ghzheng/OT_project/lineage_preprocessing_factorial_20261103/UNITS.tsv}
kp_cnv=${KP_CNV_ROOT:-/scratch/10119/ghzheng/OT_project/kp_tracer_confidenceot/cnv_profiles/infercnvpy_mouse}
quinn_cnv=${QUINN_CNV_ROOT:-/scratch/10119/ghzheng/OT_project/quinn2021_confidenceot/cnv_profiles/infercnvpy_experiment}
if [[ $tool != infercnvpy ]]; then kp_cnv=$base/cnv_exports/$tool; quinn_cnv=$base/cnv_exports/$tool; fi
nodes=${RNACNV_NODES:-2}; hours=${RNACNV_HOURS:-24}; procs=${RNACNV_PROCS:-36}; fitw=${RNACNV_FIT_WORKERS:-4}
configs="ranknm256-midrank|none|scale|cos|pca50;logcpm|none|noscale|sqeuc|pca30"
export PYTHONPATH=$repo/src:$repo/cancer_metastasis:$repo/realdata_benchmark:$repo
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

run() { if (( dry_run )); then echo "DRY RUN  $*" >&2; else "$@"; fi; }
[[ -x "$venv/bin/python" ]] || { echo "missing $venv/bin/python" >&2; exit 1; }

case $stage in
cnvroot)
    [[ $arm != rna ]] || { echo "cnvroot is for the CNV arms" >&2; exit 2; }
    for d in "$kp_cnv/KPTracer" "$quinn_cnv/Quinn2021"; do
        [[ -d "$d" ]] || { echo "missing $d (expected <export root>/<dataset_id>/<patient_id>/)" >&2; exit 1; }
    done
    [[ -e "$root/cnv_root" ]] && { echo "$root/cnv_root exists; left as it is" >&2; ls -l "$root/cnv_root"; exit 0; }
    run mkdir -p "$root/cnv_root" || exit 1
    run ln -s "$kp_cnv/KPTracer" "$root/cnv_root/KPTracer" || exit 1
    run ln -s "$quinn_cnv/Quinn2021" "$root/cnv_root/Quinn2021" || exit 1
    (( dry_run )) || ls -l "$root/cnv_root"
    ;;
worklist)
    [[ -f "$units" ]] || { echo "missing $units" >&2; exit 1; }
    cd "$repo" || exit 1
    run timeout 600 "$venv/bin/python" realdata_benchmark/v3_grid/build_worklist.py --dataset kp --root "$root" --kp-units "$units" \
        --keys 'ranknm256-midrank|none' 'logcpm|none'
    ;;
run)
    need=("$root/WORKLIST.tsv")
    [[ $arm != rna ]] && need+=("$root/cnv_root/KPTracer" "$root/cnv_root/Quinn2021")
    for f in "${need[@]}"; do [[ -e "$f" ]] || { echo "missing $f (cnvroot / worklist first)" >&2; exit 1; }; done
    queued=$(squeue -u "$USER" -h -r 2> /dev/null | wc -l)
    (( dry_run || queued + nodes <= 40 )) || { echo "$queued queued; $nodes more would pass the cap of 40" >&2; exit 3; }
    run mkdir -p "$root/logs/workers" || exit 1
    extra=""
    [[ $arm != rna ]] && extra=",V3_REPRESENTATION_SOURCE=$source_,V3_CNV_ROOT=$root/cnv_root,V3_CNV_NPCS=30"
    [[ -n "$beta" ]] && extra="$extra,V3_CNV_WEIGHT=$beta"
    run sbatch --parsable --array=0-$((nodes - 1)) -t "$hours:00:00" -J "lv_${arm}${tool:+_$tool}" -o "$root/logs/run_%A_%a.out" \
        --export=ALL,CONFIDENCEOT_REPO="$repo",V3_VENV="$venv",V3_WORKLIST="$root/WORKLIST.tsv",V3_PROCS="$procs",V3_FIT_WORKERS="$fitw",V3_LOGS="$root/logs/workers",V3_MAX_LARGE=4,V3_LARGE_SIZE=2e7,V3_CONFIGS="$configs"$extra \
        "$repo/realdata_benchmark/v3_grid/tacc/v3_workers.slurm"
    ;;
status)
    n=$(tail -n +2 "$root/WORKLIST.tsv" | wc -l)
    ok=$(ls "$root"/rows/*/*/SUCCESS 2> /dev/null | wc -l); bad=$(ls "$root"/rows/*/*/FAILED 2> /dev/null | wc -l)
    echo "$arm rows $n: SUCCESS $ok, FAILED $bad, open $((n - ok - bad))"
    for f in $(ls "$root"/rows/*/*/FAILED 2> /dev/null | head -5); do echo "-- $f"; cat "$f"; echo; done
    ;;
esac
