#!/bin/bash
#SBATCH -J cot_fixed_c_ot
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/fixed_c_ot_%A_%a.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/fixed_c_ot_%A_%a.err
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
output_base=${CONFIDENCEOT_FIXED_COST_OT_BASE:?CONFIDENCEOT_FIXED_COST_OT_BASE is required}
costs=${CONFIDENCEOT_FIXED_COSTS:-"0.4:0.5:0.6:0.7"}
pair_count=${CONFIDENCEOT_PAIR_COUNT:?CONFIDENCEOT_PAIR_COUNT is required}

cd "$repo"
export PYTHONPATH="$repo/src:$repo/cancer_metastasis${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
export PYTHONUNBUFFERED=1
mkdir -p "$output_base" "$result/logs"
read -r -a values <<< "${costs//[:,]/ }"
total=$((pair_count * ${#values[@]}))
failures=0
for ((linear=SLURM_ARRAY_TASK_ID; linear<total; linear+=SLURM_ARRAY_TASK_COUNT)); do
  cost=${values[$((linear / pair_count))]}
  index=$((linear % pair_count))
  slug=$(printf "%.2f" "$cost"); slug=${slug/./p}
  output="$output_base/rejection_cost_$slug"
  if ! python cancer_metastasis/02_run_pair.py "$manifest" "$output" \
      --index "$index" --analysis-scope malignant \
      --include-annotation Ovarian.cancer.cell --cell-qc \
      --minimum-total-counts 1000 --minimum-detected-genes 500 \
      --maximum-mitochondrial-percent 20 \
      --source-rejection-budget 0.85 --target-rejection-budget 0.95 \
      --fixed-rejection-cost "$cost" --max-observed-cells-per-side 10000 \
      --tolerance 1e-4 --workers 1 --device cuda --skip-completed; then
    failures=$((failures + 1))
  fi
done
(( failures == 0 ))
