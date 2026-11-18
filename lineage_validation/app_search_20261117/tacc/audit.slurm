#!/bin/bash
#SBATCH -J app_audit
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -A MCB26031
#SBATCH -o /scratch/10119/ghzheng/OT_project/app_search_20261117/logs/audit_%j.out
# Format audit of the downloaded application datasets (05_audit_tacc_data.py): headers, metadata, archive listings.
#   sbatch lineage_validation/app_search_20261117/tacc/audit.slurm
# No set -e / -u / pipefail (TACC rules).

source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"
repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
app=/scratch/10119/ghzheng/OT_project/app_search_20261117
cd "$repo" || exit 1
echo "job ${SLURM_JOB_ID:-local} on $(hostname); commit $(git rev-parse --short HEAD); $(date -Iseconds)"
python -I lineage_validation/app_search_20261117/05_audit_tacc_data.py --data "$app/data" --out "$app/audit"
echo "EXIT $? $(date -Iseconds)"
