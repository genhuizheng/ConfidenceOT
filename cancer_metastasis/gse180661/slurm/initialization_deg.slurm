#!/bin/bash
#SBATCH -J cot_init_deg
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/init_deg_%j.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/init_deg_%j.err
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 02:00:00
#SBATCH -A MCB26031

set -eo pipefail
source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"
set -u

repo=${CONFIDENCEOT_REPO:?Set CONFIDENCEOT_REPO}
pseudobulk=${CONFIDENCEOT_INIT_PSEUDOBULK_ROOT:?Set CONFIDENCEOT_INIT_PSEUDOBULK_ROOT}
output=${CONFIDENCEOT_INIT_DEG_ROOT:?Set CONFIDENCEOT_INIT_DEG_ROOT}

cd "$repo"
export PYTHONPATH="$repo/src:$repo/cancer_metastasis${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
export PYTHONUNBUFFERED=1
mkdir -p "$output" "$(dirname "$output")/logs"
python -m cancer_metastasis.gse180661.initialization_deg \
  "$pseudobulk" "$output" --n-cpus 16
