#!/bin/bash
#SBATCH -J cot_fixed_c_pb
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/fixed_c_pb_%A_%a.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/fixed_c_pb_%A_%a.err
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 04:00:00
#SBATCH -A MCB26031

source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"
repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
manifest=${CONFIDENCEOT_MANIFEST:?CONFIDENCEOT_MANIFEST is required}
ot_base=${CONFIDENCEOT_FIXED_COST_OT_BASE:?CONFIDENCEOT_FIXED_COST_OT_BASE is required}
output_base=${CONFIDENCEOT_FIXED_COST_PB_BASE:?CONFIDENCEOT_FIXED_COST_PB_BASE is required}
costs=${CONFIDENCEOT_FIXED_COSTS:-"0.4:0.5:0.6:0.7"}
pair_count=${CONFIDENCEOT_PAIR_COUNT:?CONFIDENCEOT_PAIR_COUNT is required}
cd "$repo"
export PYTHONPATH="$repo/src:$repo/cancer_metastasis${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
read -r -a values <<< "${costs//[:,]/ }"
total=$((pair_count * ${#values[@]}))
failures=0
for ((linear=SLURM_ARRAY_TASK_ID; linear<total; linear+=SLURM_ARRAY_TASK_COUNT)); do
  cost=${values[$((linear / pair_count))]}; index=$((linear % pair_count))
  slug=$(printf "%.2f" "$cost"); slug=${slug/./p}
  if ! python -m cancer_metastasis.gse180661.primary_pseudobulk \
      "$manifest" "$ot_base/rejection_cost_$slug" "$output_base/rejection_cost_$slug" \
      --index "$index" --budget-tag budget_source_0.85_target_0.95 \
      --minimum-cells-per-state 20; then
    failures=$((failures + 1))
  fi
done
(( failures == 0 ))
