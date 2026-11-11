#!/bin/bash
#SBATCH -J rld_deg
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH -A MCB26031
# DEG + hallmark GSEA of discordant vs concordant cells (60_deg_gsea.py), raw counts of the v3 RNA-OT inputs.
#   sbatch --export=ALL,GROUPS=...,OUT=...,GMT_DIR=... lineage_validation/rna_lineage_discordance_20261110/tacc/deg_gsea.slurm
repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
env_path=${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}
worklist=${WORKLIST:-/scratch/10119/ghzheng/OT_project/v3_grid_20261104/kp/WORKLIST.tsv}
[ -n "$GROUPS" ] && [ -n "$OUT" ] && [ -n "$GMT_DIR" ] || { echo "set GROUPS, OUT and GMT_DIR" >&2; exit 2; }
source /home1/10119/ghzheng/.bashrc
conda activate "$env_path"
cd "$repo" || exit 1
export PYTHONPATH=$repo/src:$repo/cancer_metastasis:$repo/realdata_benchmark:$repo
python lineage_validation/rna_lineage_discordance_20261110/60_deg_gsea.py --groups "$GROUPS" --worklist "$worklist" \
    --gmt-human "$GMT_DIR/h.all.v2025.1.Hs.symbols.gmt" --gmt-mouse "$GMT_DIR/mh.all.2025.1.Mm.symbols.gmt" --out "$OUT"
echo "contrasts with DEG: $(ls "$OUT"/deg 2>/dev/null | wc -l)"
