#!/bin/bash
# Submit the GSE271675 prostate chain: prepare, depth, OT, pseudobulk, DEG.
#
# Each stage depends on the previous one with afterok, so a failure upstream
# leaves the rest PENDING with DependencyNeverSatisfied rather than running on
# absent or half-written input.
#
# Vista's sbatch prints a welcome banner on stdout, so `--parsable` alone does
# not yield a job id: the captured value is the whole banner with the id at the
# end, and passing that to --dependency made sbatch read the banner's rule of
# dashes as command-line options. Every submission here takes the last line.
#
# Kept as a file rather than a pasted block because a terminal paste of this
# length was corrupted three times in one session, each time silently enough to
# look like it had worked.
#
# No `set -u`: the explicit non-empty absolute-path check below catches the
# defined-but-empty path that -u permits, which is the failure that has
# actually cost job numbers here.
#
# Usage:
#   bash cancer_metastasis/submit_prostate_chain.sh
#   bash cancer_metastasis/submit_prostate_chain.sh 999848   # reuse a prepare job
set -eo pipefail

submit() {
  # Last line only; see the banner note above.
  sbatch --parsable "$@" | tail -n 1 | tr -dc '0-9_'
}

repo=${CONFIDENCEOT_REPO:-${REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}}
result=${CANCER_COT_ROOT:-${RESULTS:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}}
env_path=${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}
stamp=${PROSTATE_STAMP:-20260916}

for name in repo result env_path; do
  value=${!name}
  if [[ -z "$value" || "$value" != /* ]]; then
    echo "$name must be a non-empty absolute path, found: '$value'" >&2
    exit 2
  fi
done

prepared=$result/prepared_GSE271675_$stamp
downsampled=$result/downsampled_GSE271675_$stamp
ot=$result/ot_GSE271675_free_$stamp
four_state=$result/four_state_GSE271675_$stamp
deg=$result/deg_GSE271675_$stamp

export CANCER_COT_ROOT="$result"
export CONFIDENCEOT_REPO="$repo"
cd "$repo"
mkdir -p "$result/logs"

prepare_job=${1:-}
if [[ -z "$prepare_job" ]]; then
  prepare_job=$(submit cancer_metastasis/tacc_prepare_prostate.slurm)
  echo "J1 prepare     $prepare_job  (submitted)"
else
  echo "J1 prepare     $prepare_job  (reused)"
fi
if ! [[ "$prepare_job" =~ ^[0-9]+$ ]]; then
  echo "Prepare job id is not numeric: '$prepare_job'" >&2
  exit 2
fi

# Depth. The target is a quantile rather than a fixed count: multiome nuclei
# are not the whole-cell 3' libraries the ovarian target of 3,119 came from,
# and the prepare job reports this object's own distribution for review.
downsample_job=$(submit --dependency=afterok:"$prepare_job" \
  -p gg -N 1 -n 1 -t 03:00:00 -A MCB26031 -J cot_ds_pca \
  -o "$result/logs/ds_pca_%j.out" -e "$result/logs/ds_pca_%j.err" \
  --wrap="source /home1/10119/ghzheng/.bashrc; conda activate $env_path; cd $repo; export PYTHONPATH=$repo/src:$repo/cancer_metastasis:$repo; python cancer_metastasis/27_downsample_counts.py $prepared/pair_manifest_eligible.csv $downsampled --malignant-annotation Epithelial --malignant-annotation 'Basal Epithelial' --malignant-annotation Neuroendocrine --target-quantile ${PROSTATE_TARGET_QUANTILE:-0.10} --minimum-total-counts 0 --minimum-detected-genes 0 --maximum-mitochondrial-percent 100")
echo "J2 downsample  $downsample_job"

# The cells were selected upstream and subsampling lowers the detected-gene
# count a second QC pass would test, so metrics are recorded without filtering
# again. The gate diagnostics read those metrics.
export CONFIDENCEOT_MANIFEST="$downsampled/pair_manifest_downsampled.csv"
export CONFIDENCEOT_OUTPUT_ROOT="$ot"
export CONFIDENCEOT_ANALYSIS_SCOPE=malignant
export CONFIDENCEOT_INCLUDE_ANNOTATIONS="Epithelial|Basal Epithelial|Neuroendocrine"
export CONFIDENCEOT_DEVICE=cpu
export CONFIDENCEOT_THREADS=144
export CONFIDENCEOT_MAX_OBSERVED_CELLS_PER_SIDE=10000
export CONFIDENCEOT_SAVE_PAIRING_EDGES=1
export CONFIDENCEOT_CELL_QC=1
export CONFIDENCEOT_MINIMUM_TOTAL_COUNTS=0
export CONFIDENCEOT_MINIMUM_DETECTED_GENES=0
export CONFIDENCEOT_MAXIMUM_MITOCHONDRIAL_PERCENT=100
export CONFIDENCEOT_SOURCE_REJECTION_BUDGET=0.85
export CONFIDENCEOT_TARGET_REJECTION_BUDGET=0.00
# Resolved from the manifest at run time, which does not exist yet.
unset CONFIDENCEOT_PAIR_COUNT
unset CONFIDENCEOT_SOURCE_REJECTION_BOUNDS CONFIDENCEOT_TARGET_REJECTION_BOUNDS

ot_job=$(submit --dependency=afterok:"$downsample_job" \
  -p gg -t 12:00:00 --array="${PROSTATE_ARRAY:-0-7}" \
  cancer_metastasis/tacc_full_array.slurm)
echo "J3 ot array    $ot_job"

export CONFIDENCEOT_GATE_ROOT="$ot"
export CONFIDENCEOT_FOUR_STATE_ROOT="$four_state"
export CONFIDENCEOT_METASTASIS_SIZE_CSV="$downsampled/downsample_per_sample.csv"
export CONFIDENCEOT_MALIGNANT_ANNOTATION=Epithelial
# Cap-robustness and origin selection stay off: the cap no longer decides the
# gate, and there is no origin ranking for this dataset.
unset CONFIDENCEOT_ROBUSTNESS_CSV CONFIDENCEOT_SENSITIVITY_ROOT
unset CONFIDENCEOT_FOUR_STATE_PATIENT_COUNT

pseudobulk_job=$(submit --dependency=afterok:"${ot_job%%_*}" \
  --array="${PROSTATE_PATIENT_ARRAY:-0-3}" \
  cancer_metastasis/tacc_four_state_malignant_array.slurm)
echo "J4 pseudobulk  $pseudobulk_job"

export CONFIDENCEOT_DEG_ROOT="$four_state"
export CONFIDENCEOT_PYDESEQ2_ROOT="$deg"

deg_job=$(submit --dependency=afterok:"${pseudobulk_job%%_*}" \
  cancer_metastasis/tacc_paired_pydeseq2.slurm)
echo "J5 pydeseq2    $deg_job"

echo
echo "prepared    $prepared"
echo "downsampled $downsampled"
echo "ot          $ot"
echo "four_state  $four_state"
echo "deg         $deg"
echo
squeue -u "$USER" -o "%.12i %.14j %.10T %.26E"
