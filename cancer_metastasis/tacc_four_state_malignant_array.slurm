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
baseline=${CONFIDENCEOT_BASELINE_ROOT:-$result/malignant_full_GSE180661}
sensitivity=${CONFIDENCEOT_SENSITIVITY_ROOT:-$result/malignant_sourcecap_0p90_targetcap_0p95_GSE180661}
robustness=${CONFIDENCEOT_ROBUSTNESS_CSV:-$result/source_cap_sensitivity_GSE180661_comprehensive/origin_group_robustness.csv}
output=${CONFIDENCEOT_FOUR_STATE_ROOT:-$result/four_state_malignant_pseudobulk_GSE180661}

cd "$repo"
export PYTHONPATH="$repo/src:$repo/cancer_metastasis${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
export PYTHONUNBUFFERED=1
mkdir -p "$output" "$result/logs"

patient_count=${CONFIDENCEOT_FOUR_STATE_PATIENT_COUNT:-$(
  python cancer_metastasis/21_prepare_four_state_malignant_pseudobulk.py \
    "$manifest" "$baseline" "$sensitivity" "$robustness" "$output" \
    --print-patient-count --sensitivity-label source090
)}

task_id=${SLURM_ARRAY_TASK_ID:-0}
task_count=${SLURM_ARRAY_TASK_COUNT:-1}
failures=0
for ((index=task_id; index<patient_count; index+=task_count)); do
  echo "===== FOUR STATE START index=$index $(date --iso-8601=seconds) ====="
  if python cancer_metastasis/21_prepare_four_state_malignant_pseudobulk.py \
      "$manifest" "$baseline" "$sensitivity" "$robustness" "$output" \
      --index "$index" --sensitivity-label source090; then
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
