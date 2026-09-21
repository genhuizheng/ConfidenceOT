#!/bin/bash
#SBATCH -J cot_splat_plot
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 04:00:00
#SBATCH -A MCB26031
#SBATCH -o /scratch/10119/ghzheng/OT_project/benchmark_scaling/logs/finalize_%j.out
#SBATCH -e /scratch/10119/ghzheng/OT_project/benchmark_scaling/logs/finalize_%j.err

set -eo pipefail
source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"
set -u

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
root=${CONFIDENCEOT_SCALING_ROOT:-/scratch/10119/ghzheng/OT_project/benchmark_scaling}
m4="$root/raw/m4"
partial="$root/raw/partial"
report="$root/report"
bundle="$root/ConfidenceOT_splatter_scaling_report.tar.gz"

if [[ -e "$report" || -e "$bundle" ]]; then
    echo "Refusing to overwrite an existing report or bundle" >&2
    exit 3
fi

cd "$repo"
python scripts/aggregate_confidenceot_grid.py "$m4"
python scripts/aggregate_confidenceot_grid.py "$partial"
# 20000 is not in the size list: that run was cancelled, so expecting it makes
# the completeness check fail on a benchmark that is in fact complete.
# CONFIDENCEOT_SIZES puts it back if it is ever run. CONFIDENCEOT_PARTIAL_MAX_N
# has to match what the array job used, or the same check fails the other way --
# Partial OT now runs at every size, and the finalizer used to expect it at one.
python scripts/finalize_splatter_scaling_benchmark.py \
    "$m4" "$partial" "$report" \
    --sizes ${CONFIDENCEOT_SIZES:-1000 5000 10000} \
    --partial-max-n "${CONFIDENCEOT_PARTIAL_MAX_N:-10000}" \
    --replicates "${CONFIDENCEOT_REPLICATES:-2}"

cd "$root"
tar -czf "$bundle" report
sha256sum "$bundle"
