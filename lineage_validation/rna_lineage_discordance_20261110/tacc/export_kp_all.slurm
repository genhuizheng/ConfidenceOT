#!/bin/bash
#SBATCH -J rld_kpall
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
#SBATCH -A MCB26031
# PLAN_v2.md extension E1 / E2: export obs, logCPM PCA-30 and marker tables of every KP cell (85_export_kp_all.py).
#   sbatch --export=ALL,OUT=... lineage_validation/rna_lineage_discordance_20261110/tacc/export_kp_all.slurm
repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
env_path=${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}
h5ad=${H5AD:-/scratch/10119/ghzheng/OT_project/data/kp_tracer/KPTracer-Data/expression/adata_processed.combined.h5ad}
[ -n "$OUT" ] || { echo "set OUT" >&2; exit 2; }
source /home1/10119/ghzheng/.bashrc
conda activate "$env_path"
cd "$repo" || exit 1
export PYTHONPATH=$repo/src:$repo/realdata_benchmark:$repo
python lineage_validation/rna_lineage_discordance_20261110/85_export_kp_all.py --h5ad "$h5ad" --out "$OUT"
ls -la "$OUT"
