#!/bin/bash
#SBATCH -J rld_coords
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00
#SBATCH -A MCB26031
# View A coordinates for the RNA-lineage discordance analysis (20_rna_coordinates.py): no OT is fitted.
#   sbatch --export=ALL,UNITS=...,OUT=... lineage_validation/rna_lineage_discordance_20261110/tacc/rna_coordinates.slurm
repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
env_path=${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}
worklist=${WORKLIST:-/scratch/10119/ghzheng/OT_project/v3_grid_20261104/kp/WORKLIST.tsv}
[ -n "$UNITS" ] && [ -n "$OUT" ] || { echo "set UNITS and OUT" >&2; exit 2; }
source /home1/10119/ghzheng/.bashrc
conda activate "$env_path"
cd "$repo" || exit 1
export PYTHONPATH=$repo/src:$repo/cancer_metastasis:$repo/realdata_benchmark:$repo
mkdir -p "$OUT"
python lineage_validation/rna_lineage_discordance_20261110/20_rna_coordinates.py --worklist "$worklist" --units "$UNITS" --out "$OUT"
echo "units with obs: $(ls "$OUT"/*/obs.tsv.gz 2>/dev/null | wc -l)"
