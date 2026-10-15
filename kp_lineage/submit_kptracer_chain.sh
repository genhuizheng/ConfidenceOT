#!/bin/bash
# KP-Tracer through the project's own chain: depth equalisation, then the OT
# gate at the fixed configuration.
#
#   bash kp_lineage/submit_kptracer_chain.sh --max-cells N [--dry-run]
#
# 1. 27_downsample_counts.py on the prepared manifest (01_prepare_kptracer_pairs.py
#    without --plan). The malignant compartment is obs['malignant'] == malignant,
#    the authors' is_normal call; QC is the default 1000 counts, 500 genes, 20%
#    mitochondrial; every analysed cell is subsampled to the 10th percentile of
#    the pooled depth.
# 2. cancer_metastasis/tacc_preprocessing_factorial.slurm with the one label
#    ranknm256_noscale_ds_cos: M4-E gate, within-side split null, acceptance
#    0.99, cell QC recorded but not re-applied, CPU on gg. Afterok on step 1.
#
# --max-cells is required on purpose: it is 02_run_pair.py's per-side cap, whose
# default 10000 subsamples 3724_NT_T1 (14,775 malignant cells), and 0 uses
# every cell. It is a decision, so the script will not take a default for it.
#
# No set -u, as everywhere in this repo.
set -eo pipefail

max_cells=""
dry_run=0
while (( $# )); do
    case "$1" in
        --max-cells) max_cells=$2; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done
if [[ ! "$max_cells" =~ ^[0-9]+$ ]]; then
    echo "pass --max-cells 0 (every cell) or --max-cells 10000 (the runner's default cap)" >&2
    exit 2
fi

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
root=${KPTRACER_ROOT:-/scratch/10119/ghzheng/OT_project/kp_tracer_confidenceot}
env_path=${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}
prepared=$root/prepared
downsampled=$root/downsampled
ot=$root/ot
logs=$root/logs
label=ranknm256_noscale_ds_cos
workers=${KPTRACER_WORKERS:-6}

if [[ ! -f "$prepared/pair_manifest_eligible.csv" ]]; then
    echo "no prepared manifest at $prepared; run 01_prepare_kptracer_pairs.py without --plan" >&2
    exit 2
fi
written=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['h5ads_written'])" \
    "$prepared/preparation_report.json")
if [[ "$written" == "0" ]]; then
    echo "$prepared/preparation_report.json says no h5ad was written (a --plan run)" >&2
    exit 2
fi
mkdir -p "$logs"

submit() {
    local what=$1; shift
    if (( dry_run )); then
        echo "DRY RUN  $what" >&2; echo "         sbatch $*" >&2; echo 0; return 0
    fi
    local id
    id=$(sbatch --parsable "$@" | tail -n 1 | tr -dc '0-9')
    if [[ -z "$id" ]]; then echo "sbatch returned no job id for $what" >&2; exit 1; fi
    echo "submitted $what as $id" >&2
    echo "$id"
}

ds_job=$(submit "depth equalisation" -p gg -N 1 -n 1 -t 04:00:00 -A MCB26031 -J kp_ds \
    -o "$logs/downsample_%j.out" -e "$logs/downsample_%j.err" \
    --wrap "source /home1/10119/ghzheng/.bashrc; conda activate $env_path; cd $repo; export PYTHONPATH=$repo/src:$repo/cancer_metastasis:$repo; python cancer_metastasis/27_downsample_counts.py $prepared/pair_manifest_eligible.csv $downsampled --malignant-column malignant --malignant-value malignant --target-quantile 0.10")

dependency=()
(( dry_run )) || dependency=(--dependency=afterok:$ds_job)
export CONFIDENCEOT_REPO="$repo"
export CANCER_COT_ROOT="$root"
export CONFIDENCEOT_FACTORIAL_MANIFEST="$downsampled/pair_manifest_downsampled.csv"
export CONFIDENCEOT_FACTORIAL_MANIFEST_DS="$downsampled/pair_manifest_downsampled.csv"
export CONFIDENCEOT_PREDOWNSAMPLE_DEPTH="$downsampled/predownsample_depth.csv.gz"
export CONFIDENCEOT_FACTORIAL_ROOT="$ot"
export CONFIDENCEOT_FACTORIAL_BLOCK=kptracer
export CONFIDENCEOT_FACTORIAL_LABELS="$label"
export CONFIDENCEOT_FACTORIAL_WORKERS="$workers"
export CONFIDENCEOT_ANALYSIS_SCOPE=malignant
export CONFIDENCEOT_MALIGNANT_COLUMN=malignant
export CONFIDENCEOT_MALIGNANT_VALUE=malignant
unset CONFIDENCEOT_INCLUDE_ANNOTATIONS
export CONFIDENCEOT_MAX_OBSERVED_CELLS_PER_SIDE="$max_cells"
export CONFIDENCEOT_MINIMUM_SCOPE_CELLS=4
export CONFIDENCEOT_DEVICE=cpu
export CONFIDENCEOT_FACTORIAL_FORCE=0
export CONFIDENCEOT_FACTORIAL_EXTRA_ARGS="--within-side-acceptance-minimum 0.99"

ot_job=$(submit "OT gate, $label, $workers workers" "${dependency[@]}" \
    --array=0-$(( workers - 1 )) -t 24:00:00 -J kp_ot \
    -o "$logs/ot_%A_%a.out" -e "$logs/ot_%A_%a.err" \
    --export=ALL "$repo/cancer_metastasis/tacc_preprocessing_factorial.slurm")

echo
echo "commit      $(git -C "$repo" rev-parse --short HEAD 2> /dev/null || echo unknown)"
echo "prepared    $prepared"
echo "downsampled $downsampled"
echo "ot          $ot/kptracer/$label"
echo "max cells   $max_cells per side"
echo "jobs        downsample $ds_job, OT array $ot_job"
(( dry_run )) && echo "nothing was submitted (--dry-run)"
exit 0
