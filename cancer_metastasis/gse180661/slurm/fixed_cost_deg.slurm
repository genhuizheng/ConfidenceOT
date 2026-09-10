#!/bin/bash
#SBATCH -J cot_fixed_c_de
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/fixed_c_de_%A_%a.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/fixed_c_de_%A_%a.err
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 04:00:00
#SBATCH -A MCB26031

source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"
repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
pb_base=${CONFIDENCEOT_FIXED_COST_PB_BASE:?CONFIDENCEOT_FIXED_COST_PB_BASE is required}
deg_base=${CONFIDENCEOT_FIXED_COST_DEG_BASE:?CONFIDENCEOT_FIXED_COST_DEG_BASE is required}
costs=${CONFIDENCEOT_FIXED_COSTS:-"0.4:0.5:0.6:0.7"}
cd "$repo"
export PYTHONPATH="$repo/src:$repo/cancer_metastasis${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
read -r -a values <<< "${costs//[:,]/ }"
for ((i=SLURM_ARRAY_TASK_ID; i<${#values[@]}; i+=SLURM_ARRAY_TASK_COUNT)); do
  slug=$(printf "%.2f" "${values[$i]}"); slug=${slug/./p}
  python -m cancer_metastasis.gse180661.primary_deg \
    "$pb_base/rejection_cost_$slug" "$deg_base/rejection_cost_$slug" \
    --minimum-total-count 10 --n-cpus 16 --absolute-log2-fold-change-thresholds 0.5 1.0
done
