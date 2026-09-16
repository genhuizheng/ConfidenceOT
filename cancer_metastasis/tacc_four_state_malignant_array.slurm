#!/bin/bash
#SBATCH -J cot_four_state
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/four_state_%A_%a.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/four_state_%A_%a.err
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 06:00:00
#SBATCH -A MCB26031
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=END,FAIL

source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
manifest=${CONFIDENCEOT_MANIFEST:-$result/manifest/GSE180661_malignant/pair_manifest_malignant_eligible.csv}
gate=${CONFIDENCEOT_GATE_ROOT:-${CONFIDENCEOT_BASELINE_ROOT:-$result/malignant_full_GSE180661}}
output=${CONFIDENCEOT_FOUR_STATE_ROOT:-$result/four_state_malignant_pseudobulk_GSE180661}

# Cap-robustness and origin selection are now opt-in. Both defaulted to real
# paths before, which meant the pre-2026-09 rule applied unless it was
# actively disabled; the rule hedged against a cap that no longer decides
# anything, and the origin table was built under the rotation null.
extra=()
if [[ -n "${CONFIDENCEOT_ROBUSTNESS_CSV:-}" ]]; then
  extra+=(--robustness-csv "$CONFIDENCEOT_ROBUSTNESS_CSV")
fi
if [[ -n "${CONFIDENCEOT_SENSITIVITY_ROOT:-}" ]]; then
  extra+=(--sensitivity-root "$CONFIDENCEOT_SENSITIVITY_ROOT")
  extra+=(--sensitivity-label "${CONFIDENCEOT_SENSITIVITY_LABEL:-source090}")
fi
# Names the largest lesion by malignant QC-passing cells rather than by the
# manifest's all-cell-type count.
if [[ -n "${CONFIDENCEOT_METASTASIS_SIZE_CSV:-}" ]]; then
  extra+=(--metastasis-size-csv "$CONFIDENCEOT_METASTASIS_SIZE_CSV")
fi
# The label is dataset-specific, and the default is GSE180661's. Left wrong on
# another dataset the same-compartment background becomes every cell in the
# file, which is dropped from the contrasts but still reported as a count.
if [[ -n "${CONFIDENCEOT_MALIGNANT_ANNOTATION:-}" ]]; then
  extra+=(--malignant-annotation "$CONFIDENCEOT_MALIGNANT_ANNOTATION")
fi

# Two submissions were lost earlier to a path variable arriving empty, so the
# job refuses to start rather than creating directories at the filesystem root.
for name in repo result manifest gate output; do
  value=${!name}
  if [[ -z "$value" || "$value" != /* ]]; then
    echo "$name must be a non-empty absolute path, found: '$value'" >&2
    exit 2
  fi
done
if [[ ! -f "$manifest" ]]; then
  echo "Manifest does not exist: $manifest" >&2
  exit 2
fi

cd "$repo"
export PYTHONPATH="$repo/src:$repo/cancer_metastasis${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
export PYTHONUNBUFFERED=1
mkdir -p "$output" "$result/logs"

patient_count=${CONFIDENCEOT_FOUR_STATE_PATIENT_COUNT:-$(
  python cancer_metastasis/21_prepare_four_state_malignant_pseudobulk.py \
    "$manifest" "$gate" "$output" --print-patient-count "${extra[@]}"
)}

task_id=${SLURM_ARRAY_TASK_ID:-0}
task_count=${SLURM_ARRAY_TASK_COUNT:-1}
failures=0
for ((index=task_id; index<patient_count; index+=task_count)); do
  echo "===== FOUR STATE START index=$index $(date --iso-8601=seconds) ====="
  if python cancer_metastasis/21_prepare_four_state_malignant_pseudobulk.py \
      "$manifest" "$gate" "$output" --index "$index" "${extra[@]}"; then
    echo "===== FOUR STATE COMPLETE index=$index $(date --iso-8601=seconds) ====="
  else
    failures=$((failures + 1))
    echo "===== FOUR STATE FAILED index=$index $(date --iso-8601=seconds) =====" >&2
  fi
done

if (( failures > 0 )); then
  echo "$failures patient(s) failed" >&2
  exit 1
fi
