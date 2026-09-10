#!/bin/bash
#SBATCH -J cot_init_pb
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/init_pb_%A_%a.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/init_pb_%A_%a.err
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 02:00:00
#SBATCH -A MCB26031

set -eo pipefail
source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"
set -u

repo=${CONFIDENCEOT_REPO:?Set CONFIDENCEOT_REPO}
manifest=${CONFIDENCEOT_MANIFEST:?Set CONFIDENCEOT_MANIFEST}
baseline=${CONFIDENCEOT_BASELINE_OT_ROOT:?Set CONFIDENCEOT_BASELINE_OT_ROOT}
initialization=${CONFIDENCEOT_INIT_ROOT:?Set CONFIDENCEOT_INIT_ROOT}
output=${CONFIDENCEOT_INIT_PSEUDOBULK_ROOT:?Set CONFIDENCEOT_INIT_PSEUDOBULK_ROOT}
pair_count=${CONFIDENCEOT_PAIR_COUNT:?Set CONFIDENCEOT_PAIR_COUNT}

cd "$repo"
export PYTHONPATH="$repo/src:$repo/cancer_metastasis${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export PYTHONUNBUFFERED=1
mkdir -p "$output" "$(dirname "$output")/logs"

task_id=${SLURM_ARRAY_TASK_ID:-0}
task_count=${SLURM_ARRAY_TASK_COUNT:-1}
failures=0
for ((index=task_id; index<pair_count; index+=task_count)); do
  if ! python -m cancer_metastasis.gse180661.initialization_pseudobulk \
      "$manifest" "$baseline" "$initialization" "$output" --index "$index"; then
    failures=$((failures + 1))
  fi
done
if (( failures > 0 )); then
  exit 1
fi
