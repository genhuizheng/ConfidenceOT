#!/bin/bash
# Run one dataset under the configuration the depth screen selected:
#
#   rank256_ds_cos
#
# That string is the configuration, not a description of one. It is the name
# confidenceot.Preprocessing prints for rank encoding at 256 genes per cell,
# read equalisation, and the cosine cost -- the joint PCA representation
# L2-normalised before the cost and before the median scale -- and the same
# string is parsed back by the pair runner and by the rank-cut audit. One name
# in one place, instead of three variables that can disagree: a rank cut set
# while the transform was left at its default produces a run with an ignored
# cut and no name.
#
# The equalisation itself happens in 27_downsample_counts.py, at the file
# level, so that the OT, the pseudobulk, the DEG and the scoring all see one
# matrix. The `_ds` in the label is therefore recorded by the run rather than
# applied by it, which is what lets the run carry its real name.
#
# Why this configuration. Across two simulators and two sizes the rate of
# rejection on populations containing nothing to reject separated perfectly by
# the cost: every Euclidean configuration sat at 0.071-0.098, every cosine one
# at 0.000-0.057, no overlap, at no cost in power. Equalisation's own effect
# reversed between the two simulators, so it is carried here because it is
# cheap and decoupled, not because the screen established it. See
# cancer_metastasis/SEQUENCING_DEPTH_RESOLUTION.md.
#
# The chain, each stage depending on the previous one with afterok so a failure
# leaves the rest PENDING rather than running on absent input:
#
#   J1 equalise   27_downsample_counts.py, skipped when its manifest exists
#   J2 audit      tools/audit_rank_top_n.py -- a gate, not a report. The rank
#                 cut has to stay below the shallowest cell's detected-gene
#                 count, or the encoding manufactures the low-content artefact
#                 it was chosen to remove, and equalised counts are where that
#                 is least safe. The screen ran at 256 on simulated cells with
#                 4,000 genes; that is not evidence about these cells. Exits 3
#                 when more than 1% of any pair's cells fall below the cut,
#                 which leaves the OT unsubmitted instead of producing a run
#                 that silently differs from the screened configuration.
#   J3 ot         tacc_full_array.slurm, given the configuration by name
#   J4 diagnose   25_diagnose_gate_covariates.py against each cell's *original*
#                 depth, plus 26_diagnose_pairing_quality.py. This is the
#                 acceptance test: auc_predownsample_total_counts near 0.5 and
#                 spearman near 0. Without it the run produces a gate nobody
#                 has checked.
#
# Writes to its own output root, so every existing result stays intact and the
# two configurations can be read side by side.
#
# No `set -u`. It fires on things that are fine here, such as an empty array
# expansion, while permitting the failure that has actually cost this project
# job numbers: a path variable that is defined but empty. The explicit
# non-empty absolute-path check below is the protection that matters.
#
# Usage:
#   bash cancer_metastasis/submit_rank_cosine_chain.sh ovarian
#   bash cancer_metastasis/submit_rank_cosine_chain.sh colorectal
#   PREPROCESSING=rank128_ds_cos bash cancer_metastasis/submit_rank_cosine_chain.sh headneck
#   SOURCE_MANIFEST=/path/to/manifest.csv bash cancer_metastasis/submit_rank_cosine_chain.sh colorectal
#   DRY_RUN=1 bash cancer_metastasis/submit_rank_cosine_chain.sh prostate
set -eo pipefail

dataset=${1:-}
case "$dataset" in
  ovarian|prostate|colorectal|headneck) ;;
  *)
    echo "Usage: $0 {ovarian|prostate|colorectal|headneck}" >&2
    exit 2
    ;;
esac

export CONFIDENCEOT_REPO=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
export CANCER_COT_ROOT=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
env_path=${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}
repo=$CONFIDENCEOT_REPO
result=$CANCER_COT_ROOT
replication_root=${CONFIDENCEOT_REPLICATION_ROOT:-$result/author_labeled_replication_GSE181919_GSE225857_20260912}
stamp=${RANK_COSINE_STAMP:-20260921}
preprocessing=${PREPROCESSING:-rank256_ds_cos}
target_quantile=${TARGET_QUANTILE:-0.10}
max_fraction_short=${MAX_FRACTION_SHORT:-0.01}

# Per dataset: the accession, the author malignant labels, the manifest the
# equalisation reads, and the equalised root. Where a root already exists it is
# reused rather than rebuilt: re-running the quantile on a different cell set
# changes the target depth too, and the two runs would then differ in two ways
# at once.
case "$dataset" in
  ovarian)
    accession=GSE180661
    annotations=(Ovarian.cancer.cell)
    source_manifest=$result/manifest/pair_manifest_eligible.csv
    equalised=${EQUALISED_ROOT:-$result/downsampled_GSE180661_20260914}
    array=${OT_ARRAY:-0-7}
    ;;
  prostate)
    accession=GSE271675
    annotations=(Epithelial "Basal Epithelial" Neuroendocrine)
    source_manifest=$result/prepared_GSE271675_20260916/pair_manifest_eligible.csv
    equalised=${EQUALISED_ROOT:-$result/downsampled_GSE271675_20260916}
    array=${OT_ARRAY:-0-7}
    ;;
  colorectal)
    accession=GSE225857
    annotations=(Tu01_AREG Tu02_DEFA5 Tu03_SRRM2 Tu04_RGMB Tu05_PCNA
                 Tu06_NKD1 Tu07_MKI67 Tu08_GNG13 Tu09_MUC2 Tu10_COL3A1
                 Tu11_PLA2G2A)
    source_manifest=$replication_root/GSE225857/manifest/pair_manifest_malignant_eligible.csv
    equalised=${EQUALISED_ROOT:-$result/downsampled_GSE225857_$stamp}
    array=${OT_ARRAY:-0-3}
    ;;
  headneck)
    accession=GSE181919
    annotations=(Malignant.cells)
    source_manifest=$replication_root/GSE181919/manifest/pair_manifest_malignant_eligible.csv
    equalised=${EQUALISED_ROOT:-$result/downsampled_GSE181919_$stamp}
    array=${OT_ARRAY:-0-3}
    ;;
esac

ot=$result/ot_${accession}_${preprocessing}_$stamp
diagnostics=$result/gate_diagnostic_${accession}_${preprocessing}_$stamp
manifest=$equalised/pair_manifest_downsampled.csv

for name in repo result env_path equalised ot; do
  value=${!name}
  if [[ -z "$value" || "$value" != /* ]]; then
    echo "$name must be a non-empty absolute path, found: '$value'" >&2
    exit 2
  fi
done

submit() {
  if [[ -n "${DRY_RUN:-}" ]]; then
    printf 'DRY_RUN sbatch %s\n' "$*" >&2
    echo "000000"
    return
  fi
  # Vista's sbatch prints a welcome banner on stdout, so --parsable alone does
  # not yield a job id: the captured value is the whole banner with the id at
  # the end. Take the last line, digits only.
  sbatch --parsable "$@" | tail -n 1 | tr -dc '0-9_'
}

cd "$repo"
mkdir -p "$result/logs"

# J1. Equalisation, only when its output is absent. A present manifest is
# treated as authoritative: it is the thing every later stage reads.
annotation_args=()
for annotation in "${annotations[@]}"; do
  annotation_args+=(--malignant-annotation "$annotation")
done
if [[ -f "$manifest" ]]; then
  echo "J1 equalise    reusing $manifest"
  equalise_job=""
else
  # The replication roots carry a date in their name, so the configured path
  # above is a guess about a stamp. Search for the manifest rather than insist
  # on the guess: one match is used, several are reported for the caller to
  # choose between with SOURCE_MANIFEST, none is a real error.
  if [[ ! -f "$source_manifest" && -n "${SOURCE_MANIFEST:-}" ]]; then
    source_manifest=$SOURCE_MANIFEST
  fi
  if [[ ! -f "$source_manifest" ]]; then
    echo "Configured source manifest is absent: $source_manifest" >&2
    echo "Searching $result for one belonging to $accession ..." >&2
    mapfile -t found < <(find "$result" -maxdepth 4 -type f \
      \( -name 'pair_manifest_malignant_eligible.csv' \
         -o -name 'pair_manifest_eligible.csv' \) \
      -path "*${accession}*" 2>/dev/null | sort)
    if (( ${#found[@]} == 1 )); then
      source_manifest=${found[0]}
      echo "Found: $source_manifest" >&2
    elif (( ${#found[@]} > 1 )); then
      echo "Several candidates; re-run with SOURCE_MANIFEST=<one of these>:" >&2
      printf '  %s\n' "${found[@]}" >&2
      exit 2
    else
      echo "No manifest for $accession under $result." >&2
      echo "Build it first, or pass SOURCE_MANIFEST=<path>." >&2
      exit 2
    fi
  fi
  equalise_job=$(submit \
    -p gg -N 1 -n 1 -t 04:00:00 -A MCB26031 -J "cot_ds_$dataset" \
    -o "$result/logs/ds_${dataset}_%j.out" \
    -e "$result/logs/ds_${dataset}_%j.err" \
    --wrap="source /home1/10119/ghzheng/.bashrc; conda activate $env_path; cd $repo; export PYTHONPATH=$repo/src:$repo/cancer_metastasis:$repo; python cancer_metastasis/27_downsample_counts.py $source_manifest $equalised $(printf '%q ' "${annotation_args[@]}")--target-quantile $target_quantile --minimum-total-counts 0 --minimum-detected-genes 0 --maximum-mitochondrial-percent 100")
  echo "J1 equalise    $equalise_job -> $equalised"
fi

# J2. The rank-cut gate. Runs on the equalised manifest, so it must depend on
# J1 when J1 was submitted.
scope_args=()
for annotation in "${annotations[@]}"; do
  scope_args+=(--include-annotation "$annotation")
done
audit_dependency=()
if [[ -n "$equalise_job" ]]; then
  audit_dependency=(--dependency=afterok:"$equalise_job")
fi
audit_job=$(submit "${audit_dependency[@]}" \
  -p gg -N 1 -n 1 -t 02:00:00 -A MCB26031 -J "cot_rankaudit_$dataset" \
  -o "$result/logs/rankaudit_${dataset}_%j.out" \
  -e "$result/logs/rankaudit_${dataset}_%j.err" \
  --wrap="source /home1/10119/ghzheng/.bashrc; conda activate $env_path; cd $repo; export PYTHONPATH=$repo/src:$repo/cancer_metastasis:$repo; python cancer_metastasis/tools/audit_rank_top_n.py $manifest --preprocessing $preprocessing --analysis-scope malignant $(printf '%q ' "${scope_args[@]}")--cell-qc --minimum-total-counts 0 --minimum-detected-genes 0 --maximum-mitochondrial-percent 100 --max-fraction-short $max_fraction_short --out $ot/rank_top_n_audit.csv")
echo "J2 rank audit  $audit_job  ($preprocessing, tolerance $max_fraction_short)"

# J3. The OT array, under the screened configuration.
export CONFIDENCEOT_MANIFEST="$manifest"
export CONFIDENCEOT_OUTPUT_ROOT="$ot"
export CONFIDENCEOT_ANALYSIS_SCOPE=malignant
printf -v joined '%s|' "${annotations[@]}"
export CONFIDENCEOT_INCLUDE_ANNOTATIONS="${joined%|}"
export CONFIDENCEOT_PREPROCESSING="$preprocessing"
# The array job refuses these alongside a named configuration, so anything left
# in the submitting shell would stop the job rather than silently override it.
unset CONFIDENCEOT_REPRESENTATION CONFIDENCEOT_RANK_TOP_N CONFIDENCEOT_COST
export CONFIDENCEOT_DEVICE=cpu
export CONFIDENCEOT_THREADS=144
export CONFIDENCEOT_MAX_OBSERVED_CELLS_PER_SIDE=10000
export CONFIDENCEOT_SAVE_PAIRING_EDGES=1
# The cells were selected by the equalisation step, and subsampling lowers the
# detected-gene count a second QC pass would test, so the metrics are recorded
# without filtering again. The gate diagnostics read those metrics.
export CONFIDENCEOT_CELL_QC=1
export CONFIDENCEOT_MINIMUM_TOTAL_COUNTS=0
export CONFIDENCEOT_MINIMUM_DETECTED_GENES=0
export CONFIDENCEOT_MAXIMUM_MITOCHONDRIAL_PERCENT=100
export CONFIDENCEOT_SOURCE_REJECTION_BUDGET=0.85
export CONFIDENCEOT_TARGET_REJECTION_BUDGET=0.00
# Resolved from the manifest at run time, which may not exist yet.
unset CONFIDENCEOT_PAIR_COUNT
unset CONFIDENCEOT_SOURCE_REJECTION_BOUNDS CONFIDENCEOT_TARGET_REJECTION_BOUNDS

ot_job=$(submit --dependency=afterok:"$audit_job" \
  -p gg -t 12:00:00 --array="$array" \
  cancer_metastasis/tacc_full_array.slurm)
echo "J3 ot array    $ot_job -> $ot"

# J4. The acceptance test, against each cell's original depth rather than the
# corrected, near-constant one.
diagnose_job=$(submit --dependency=afterok:"${ot_job%%_*}" \
  -p gg -N 1 -n 1 -t 04:00:00 -A MCB26031 -J "cot_diag_$dataset" \
  -o "$result/logs/diag_${dataset}_%j.out" \
  -e "$result/logs/diag_${dataset}_%j.err" \
  --wrap="source /home1/10119/ghzheng/.bashrc; conda activate $env_path; cd $repo; export PYTHONPATH=$repo/src:$repo/cancer_metastasis:$repo; python cancer_metastasis/25_diagnose_gate_covariates.py $diagnostics --dataset $accession=$ot --predownsample-depth $equalised/predownsample_depth.csv.gz; python cancer_metastasis/26_diagnose_pairing_quality.py $diagnostics --dataset $accession=$ot")
echo "J4 diagnose    $diagnose_job -> $diagnostics"

echo
printf 'dataset        %s (%s)\n' "$dataset" "$accession"
printf 'preprocessing  %s\n' "$preprocessing"
printf 'equalised      %s\n' "$equalised"
printf 'manifest       %s\n' "$manifest"
printf 'ot             %s\n' "$ot"
printf 'diagnostics    %s\n' "$diagnostics"
echo
if [[ -z "${DRY_RUN:-}" ]]; then
  squeue -u "$USER" -o "%.12i %.18j %.10T %.30E"
fi
