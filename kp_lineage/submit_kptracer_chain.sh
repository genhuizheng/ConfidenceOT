#!/bin/bash
# KP-Tracer through the project's own chain, in one or both of two arms.
#
#   bash kp_lineage/submit_kptracer_chain.sh --max-cells N [--arm ds|raw|both] [--reuse-downsampled] [--dry-run]
#
# --reuse-downsampled skips step 1 of the ds arm and gates the equalised counts
# an earlier, completed 27_ run left in downsampled/, for resubmitting the OT alone.
#
# Arm ds (the default; the project's chosen preprocessing):
# 1. 27_downsample_counts.py on the prepared manifest (01_prepare_kptracer_pairs.py
#    without --plan). The malignant compartment is obs['malignant'] == malignant,
#    the authors' is_normal call; QC is the default 1000 counts, 500 genes, 20%
#    mitochondrial; every analysed cell is subsampled to the 10th percentile of
#    the pooled depth.
# 2. cancer_metastasis/tacc_preprocessing_factorial.slurm with the one label
#    ranknm256_noscale_ds_cos: M4-E gate, within-side split null, acceptance
#    0.99, cell QC recorded but not re-applied, CPU on gg. Afterok on step 1.
#
# Arm raw (the comparison): the same slurm with the label raw on the prepared,
# unequalised counts. Everything else is held fixed -- manifest pairs, malignant
# call, gate, null, acceptance, cap, seed -- and the cell set is too: the QC that
# 27_ applied before equalisation (1000 / 500 / 20, through the same
# cell_qc_table) is applied here inside 02_run_pair.py instead, so both arms gate
# exactly the same cells and differ only in the representation. It needs no
# equalised input, so it does not wait for the ds arm.
#
# Both arms write under ot/kptracer/<label>, so neither overwrites the other.
#
# --max-cells is required on purpose: it is 02_run_pair.py's per-side cap, whose
# default 10000 subsamples 3724_NT_T1 (14,775 malignant cells), and 0 uses
# every cell. It is a decision, so the script will not take a default for it.
#
# No set -u, as everywhere in this repo.
set -eo pipefail

max_cells=""
arm=ds
dry_run=0
reuse_downsampled=0
while (( $# )); do
    case "$1" in
        --max-cells) max_cells=$2; shift 2 ;;
        --arm) arm=$2; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        --reuse-downsampled) reuse_downsampled=1; shift ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done
if [[ ! "$max_cells" =~ ^[0-9]+$ ]]; then
    echo "pass --max-cells 0 (every cell) or --max-cells 10000 (the runner's default cap)" >&2
    exit 2
fi
case "$arm" in ds|raw|both) ;; *) echo "--arm is ds, raw or both, not '$arm'" >&2; exit 2 ;; esac

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
root=${KPTRACER_ROOT:-/scratch/10119/ghzheng/OT_project/kp_tracer_confidenceot}
env_path=${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}
prepared=$root/prepared
downsampled=$root/downsampled
ot=$root/ot
logs=$root/logs
workers=${KPTRACER_WORKERS:-6}
# The QC 27_downsample_counts.py applies by default, restated for the raw arm.
qc_counts=1000
qc_genes=500
qc_mito=20

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
        echo "DRY RUN  $what" >&2; echo "         sbatch $*" >&2
        if [[ -n "${CONFIDENCEOT_FACTORIAL_LABELS:-}" ]]; then
            echo "         label ${CONFIDENCEOT_FACTORIAL_LABELS}; manifest ${CONFIDENCEOT_FACTORIAL_MANIFEST:-}; equalised ${CONFIDENCEOT_FACTORIAL_MANIFEST_DS:-none}; qc ${CONFIDENCEOT_MINIMUM_TOTAL_COUNTS:-}/${CONFIDENCEOT_MINIMUM_DETECTED_GENES:-}/${CONFIDENCEOT_MAXIMUM_MITOCHONDRIAL_PERCENT:-}; cap ${CONFIDENCEOT_MAX_OBSERVED_CELLS_PER_SIDE:-}; extra ${CONFIDENCEOT_FACTORIAL_EXTRA_ARGS:-}" >&2
        fi
        echo 0; return 0
    fi
    local id
    id=$(sbatch --parsable "$@" | tail -n 1 | tr -dc '0-9')
    if [[ -z "$id" ]]; then echo "sbatch returned no job id for $what" >&2; exit 1; fi
    echo "submitted $what as $id" >&2
    echo "$id"
}

# Settings shared by both arms.
export CONFIDENCEOT_REPO="$repo"
export CANCER_COT_ROOT="$root"
export CONFIDENCEOT_FACTORIAL_ROOT="$ot"
export CONFIDENCEOT_FACTORIAL_BLOCK=kptracer
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

ds_job=-; ot_job=-; raw_job=-
if [[ "$arm" == ds || "$arm" == both ]]; then
    label=ranknm256_noscale_ds_cos
    dependency=()
    if (( reuse_downsampled )); then
        # The equalised counts of an earlier, completed 27_ run, used as they are.
        for needed in pair_manifest_downsampled.csv predownsample_depth.csv.gz; do
            if [[ ! -f "$downsampled/$needed" ]]; then
                echo "--reuse-downsampled, but $downsampled/$needed does not exist" >&2
                exit 2
            fi
        done
        ds_job=reused
    else
        ds_job=$(submit "depth equalisation" -p gg -N 1 -n 1 -t 04:00:00 -A MCB26031 -J kp_ds \
            -o "$logs/downsample_%j.out" -e "$logs/downsample_%j.err" \
            --wrap "source /home1/10119/ghzheng/.bashrc; conda activate $env_path; cd $repo; export PYTHONPATH=$repo/src:$repo/cancer_metastasis:$repo; python cancer_metastasis/27_downsample_counts.py $prepared/pair_manifest_eligible.csv $downsampled --malignant-column malignant --malignant-value malignant --target-quantile 0.10")
        (( dry_run )) || dependency=(--dependency=afterok:$ds_job)
    fi
    # In a subshell, so these settings reach this sbatch (--export=ALL) and no other.
    ot_job=$(
        export CONFIDENCEOT_FACTORIAL_MANIFEST="$downsampled/pair_manifest_downsampled.csv"
        export CONFIDENCEOT_FACTORIAL_MANIFEST_DS="$downsampled/pair_manifest_downsampled.csv"
        export CONFIDENCEOT_PREDOWNSAMPLE_DEPTH="$downsampled/predownsample_depth.csv.gz"
        export CONFIDENCEOT_FACTORIAL_LABELS="$label"
        export CONFIDENCEOT_MINIMUM_TOTAL_COUNTS=0
        export CONFIDENCEOT_MINIMUM_DETECTED_GENES=0
        export CONFIDENCEOT_MAXIMUM_MITOCHONDRIAL_PERCENT=100
        submit "OT gate, $label, $workers workers" "${dependency[@]}" \
            --array=0-$(( workers - 1 )) -t 24:00:00 -J kp_ot \
            -o "$logs/ot_%A_%a.out" -e "$logs/ot_%A_%a.err" \
            --export=ALL "$repo/cancer_metastasis/tacc_preprocessing_factorial.slurm"
    )
fi
if [[ "$arm" == raw || "$arm" == both ]]; then
    raw_job=$(
        export CONFIDENCEOT_FACTORIAL_MANIFEST="$prepared/pair_manifest_eligible.csv"
        unset CONFIDENCEOT_FACTORIAL_MANIFEST_DS CONFIDENCEOT_PREDOWNSAMPLE_DEPTH
        export CONFIDENCEOT_FACTORIAL_LABELS=raw
        export CONFIDENCEOT_MINIMUM_TOTAL_COUNTS=$qc_counts
        export CONFIDENCEOT_MINIMUM_DETECTED_GENES=$qc_genes
        export CONFIDENCEOT_MAXIMUM_MITOCHONDRIAL_PERCENT=$qc_mito
        submit "OT gate, raw, $workers workers" \
            --array=0-$(( workers - 1 )) -t 24:00:00 -J kp_raw \
            -o "$logs/raw_%A_%a.out" -e "$logs/raw_%A_%a.err" \
            --export=ALL "$repo/cancer_metastasis/tacc_preprocessing_factorial.slurm"
    )
fi

echo
echo "commit      $(git -C "$repo" rev-parse --short HEAD 2> /dev/null || echo unknown)"
echo "arm         $arm"
echo "prepared    $prepared"
echo "downsampled $downsampled"
echo "ot          $ot/kptracer/<label>"
echo "max cells   $max_cells per side"
echo "jobs        downsample $ds_job, ds OT array $ot_job, raw OT array $raw_job"
(( dry_run )) && echo "nothing was submitted (--dry-run)"
exit 0
