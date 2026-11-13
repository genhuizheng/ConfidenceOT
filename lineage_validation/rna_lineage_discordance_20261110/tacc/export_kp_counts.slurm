#!/bin/bash
#SBATCH -J rld_kpcounts
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A MCB26031
# PLAN_v3_deg.md: raw counts per KP mouse for the v3 DEG (101_export_kp_counts.py).
#   sbatch --export=ALL,OUT=... lineage_validation/rna_lineage_discordance_20261110/tacc/export_kp_counts.slurm
repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
env_path=${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}
h5ad=${H5AD:-/scratch/10119/ghzheng/OT_project/data/kp_tracer/KPTracer-Data/expression/adata_processed.combined.h5ad}
[ -n "$OUT" ] || { echo "set OUT" >&2; exit 2; }
source /home1/10119/ghzheng/.bashrc
conda activate "$env_path"
cd "$repo" || exit 1
export PYTHONPATH=$repo/src:$repo
python lineage_validation/rna_lineage_discordance_20261110/101_export_kp_counts.py --h5ad "$h5ad" \
    --mice 3451 3454 3457 3460 3465 3466 3508 3513 3515 3724 --out "$OUT"
ls -la "$OUT"
