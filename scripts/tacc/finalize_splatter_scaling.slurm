#!/bin/bash
#SBATCH -J cot_splat_plot
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -A MCB26031
#SBATCH -o /scratch/10119/ghzheng/OT_project/benchmark_scaling/logs/finalize_%j.out
#SBATCH -e /scratch/10119/ghzheng/OT_project/benchmark_scaling/logs/finalize_%j.err

set -euo pipefail

source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"

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
python scripts/finalize_splatter_scaling_benchmark.py \
    "$m4" "$partial" "$report" \
    --sizes 1000 5000 10000 20000 --replicates 2

cd "$root"
tar -czf "$bundle" report
sha256sum "$bundle"

