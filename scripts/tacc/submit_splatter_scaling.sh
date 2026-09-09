#!/bin/bash

set -euo pipefail

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
root=${CONFIDENCEOT_SCALING_ROOT:-/scratch/10119/ghzheng/OT_project/benchmark_scaling}
worker_count=${CONFIDENCEOT_WORKER_COUNT:-8}
max_parallel=${CONFIDENCEOT_MAX_PARALLEL:-4}

if [[ -e "$root" ]]; then
    echo "STOP: benchmark root already exists: $root" >&2
    exit 2
fi
mkdir -p "$root/logs"
cd "$repo"

submit_id() {
    local output job_id
    output=$(sbatch --parsable "$@")
    printf '%s\n' "$output" >&2
    job_id=$(printf '%s\n' "$output" | grep -E '^[0-9]+' | tail -n 1 | cut -d';' -f1)
    if [[ -z "$job_id" ]]; then
        echo "Could not parse job ID" >&2
        return 1
    fi
    printf '%s' "$job_id"
}

common_export="ALL,CONFIDENCEOT_REPO=$repo,CONFIDENCEOT_SCALING_ROOT=$root,CONFIDENCEOT_WORKER_COUNT=$worker_count"
data_job=$(submit_id \
    --array=0-1%2 \
    --export="$common_export" \
    scripts/tacc/generate_splatter_scaling.slurm)

dependency="afterok:${data_job}"
fit_jobs=()
for size in 1000 5000 10000 20000; do
    fit_job=$(submit_id \
        --array="0-$((worker_count - 1))%${max_parallel}" \
        --dependency="$dependency" \
        --export="$common_export,CONFIDENCEOT_N_CELLS=$size" \
        scripts/tacc/run_splatter_scaling_array.slurm)
    fit_jobs+=("$fit_job")
    dependency="afterok:${fit_job}"
done

final_job=$(submit_id \
    --dependency="$dependency" \
    --export="$common_export" \
    scripts/tacc/finalize_splatter_scaling.slurm)

echo "Data job:     $data_job"
echo "N=1000:      ${fit_jobs[0]}"
echo "N=5000:      ${fit_jobs[1]}"
echo "N=10000:     ${fit_jobs[2]}"
echo "N=20000:     ${fit_jobs[3]}"
echo "Final report: $final_job"
squeue -j "$data_job,${fit_jobs[0]},${fit_jobs[1]},${fit_jobs[2]},${fit_jobs[3]},$final_job"

