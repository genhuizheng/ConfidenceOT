#!/bin/bash
#SBATCH -J crispr_mx
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -A MCB26031
# Matrix-based CRISPR scoring of the v3 KP grid (MATRIX_METRICS.md): one process per unit bundle, largest first,
# PROCS at a time on one node. Inputs are read only; outputs go to a new directory.
#   sbatch --export=ALL,BUNDLES=...,OUT=... lineage_validation/crispr_matrix_benchmark_20261108/tacc/score_matrix.slurm
repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
env_path=${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}
rows=${ROWS:-/scratch/10119/ghzheng/OT_project/v3_grid_20261104/kp/rows}
depth=${DEPTH:-/scratch/10119/ghzheng/OT_project/kp_tracer_confidenceot/downsampled/predownsample_depth.csv.gz}
procs=${PROCS:-12}
[ -n "$BUNDLES" ] && [ -n "$OUT" ] || { echo "set BUNDLES and OUT" >&2; exit 2; }
source /home1/10119/ghzheng/.bashrc
conda activate "$env_path"
cd "$repo" || exit 1
export PYTHONPATH=$repo/src:$repo/cancer_metastasis:$repo/realdata_benchmark:$repo
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p "$OUT/logs"
ls -S "$BUNDLES"/*.pkl | xargs -P "$procs" -I{} bash -c '
  u=$(basename {} .pkl)
  python lineage_validation/crispr_matrix_benchmark_20261108/10_score_unit.py --bundle {} --rows '"$rows"'/$u --out '"$OUT"' --depth '"$depth"' > '"$OUT"'/logs/$u.log 2>&1
  echo "$u exit $?"'
echo "done: $(ls "$OUT"/*/SUCCESS 2>/dev/null | wc -l) units with SUCCESS of $(ls "$BUNDLES"/*.pkl | wc -l)"
