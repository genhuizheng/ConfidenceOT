#!/bin/bash
#SBATCH -J cot_mal_ot
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/mal_ot_%A_%a.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/mal_ot_%A_%a.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 24:00:00
#SBATCH -A MCB26031

source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"
repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
manifest=${CONFIDENCEOT_MANIFEST:?CONFIDENCEOT_MANIFEST is required}
output=${CONFIDENCEOT_MALIGNANT_OT_ROOT:?CONFIDENCEOT_MALIGNANT_OT_ROOT is required}
pair_count=${CONFIDENCEOT_PAIR_COUNT:?CONFIDENCEOT_PAIR_COUNT is required}
source_cap=${CONFIDENCEOT_SOURCE_REJECTION_BUDGET:-0.85}
target_cap=${CONFIDENCEOT_TARGET_REJECTION_BUDGET:-0.95}
input_root=${CONFIDENCEOT_INPUT_GATE_ROOT:-}
input_budget_tag=${CONFIDENCEOT_INPUT_GATE_BUDGET_TAG:-budget_source_0.85_target_0.95}
input_method=${CONFIDENCEOT_INPUT_GATE_METHOD:-M4-E}
input_state=${CONFIDENCEOT_INPUT_GATE_STATE:-}

cd "$repo"
export PYTHONPATH="$repo/src:$repo/cancer_metastasis${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
export PYTHONUNBUFFERED=1
mkdir -p "$output" "$result/logs"
gate_args=()
if [[ -n "$input_root" ]]; then
  gate_args=(--input-gate-root "$input_root" --input-gate-budget-tag "$input_budget_tag" \
    --input-gate-method "$input_method" --input-gate-state "$input_state")
fi
failures=0
for ((index=SLURM_ARRAY_TASK_ID; index<pair_count; index+=SLURM_ARRAY_TASK_COUNT)); do
  if ! python cancer_metastasis/02_run_pair.py "$manifest" "$output" \
      --index "$index" --analysis-scope malignant \
      --include-annotation Ovarian.cancer.cell --cell-qc \
      --minimum-total-counts 1000 --minimum-detected-genes 500 \
      --maximum-mitochondrial-percent 20 \
      --source-rejection-budget "$source_cap" --target-rejection-budget "$target_cap" \
      --max-observed-cells-per-side 10000 \
      --null-calibration-replicates 5 --null-validation-replicates 5 \
      --calibration-grid-size 5 --tolerance 1e-4 --workers 1 --device cuda \
      "${gate_args[@]}" --skip-completed; then
    failures=$((failures + 1))
  fi
done
(( failures == 0 ))
