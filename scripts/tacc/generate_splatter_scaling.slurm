#!/bin/bash
#SBATCH -J cot_splat_data
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 08:00:00
#SBATCH -A MCB26031
#SBATCH -o /scratch/10119/ghzheng/OT_project/benchmark_scaling/logs/data_%A_%a.out
#SBATCH -e /scratch/10119/ghzheng/OT_project/benchmark_scaling/logs/data_%A_%a.err

set -euo pipefail

source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_R_ENV:-/scratch/10119/ghzheng/conda_envs/infercnv_r}"

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
root=${CONFIDENCEOT_SCALING_ROOT:-/scratch/10119/ghzheng/OT_project/benchmark_scaling}
batch_index=${SLURM_ARRAY_TASK_ID:?Submit with --array=0-1}

if [[ "$batch_index" -eq 0 ]]; then
    batch=none
    batch_fac_loc=0
    batch_fac_scale=0
elif [[ "$batch_index" -eq 1 ]]; then
    batch=mild
    batch_fac_loc=0.1
    batch_fac_scale=0.1
else
    echo "Invalid batch index: $batch_index" >&2
    exit 2
fi

output="$root/data/$batch"
if [[ -e "$output" ]]; then
    echo "Refusing to overwrite existing data root: $output" >&2
    exit 3
fi

mkdir -p "$root/data" "$root/logs"
cd "$repo"
Rscript scripts/generate_splatter_population_benchmark.R \
    "$output" 2 1000 7300 "1000,5000,10000,20000" \
    "$batch_fac_loc" "$batch_fac_scale"
