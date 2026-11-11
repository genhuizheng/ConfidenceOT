#!/bin/bash
#SBATCH -J rld_counts
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A MCB26031
# Export the raw counts of the DEG contrast cells (58_export_counts.py) for local DEG / GSEA.
#   sbatch --export=ALL,GROUPS=...,OUT=... lineage_validation/rna_lineage_discordance_20261110/tacc/export_counts.slurm
repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
env_path=${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}
worklist=${WORKLIST:-/scratch/10119/ghzheng/OT_project/v3_grid_20261104/kp/WORKLIST.tsv}
[ -n "$GROUPS" ] && [ -n "$OUT" ] || { echo "set GROUPS and OUT" >&2; exit 2; }
source /home1/10119/ghzheng/.bashrc
conda activate "$env_path"
cd "$repo" || exit 1
export PYTHONPATH=$repo/src:$repo/cancer_metastasis:$repo/realdata_benchmark:$repo
python lineage_validation/rna_lineage_discordance_20261110/58_export_counts.py --groups "$GROUPS" --worklist "$worklist" --out "$OUT"
echo "units exported: $(ls "$OUT"/*.h5ad 2>/dev/null | wc -l)"
