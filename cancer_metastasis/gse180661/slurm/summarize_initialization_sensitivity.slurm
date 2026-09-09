#!/bin/bash
#SBATCH -J cot_init_sum
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/init_summary_%j.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/init_summary_%j.err
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 02:00:00
#SBATCH -A MCB26031

set -euo pipefail
source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
input=${CONFIDENCEOT_INIT_ROOT:?Set CONFIDENCEOT_INIT_ROOT}
summary=${CONFIDENCEOT_INIT_SUMMARY_ROOT:?Set CONFIDENCEOT_INIT_SUMMARY_ROOT}

cd "$repo"
export PYTHONPATH="$repo/src:$repo/cancer_metastasis${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
python -m cancer_metastasis.gse180661.summarize_initialization_sensitivity "$input" "$summary"
