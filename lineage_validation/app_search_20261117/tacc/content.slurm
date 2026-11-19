#!/bin/bash
#SBATCH -J app_content
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A MCB26031
#SBATCH -o /scratch/10119/ghzheng/OT_project/app_search_20261117/logs/content_%j.out
# 08_content_checks.py on a compute node (reads only), for the parts given, e.g.
#   sbatch lineage_validation/app_search_20261117/tacc/content.slurm splintr zebrafish sciplex2_wells
# No set -e / -u / pipefail (TACC rules). Threads capped.

source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"
app=/scratch/10119/ghzheng/OT_project/app_search_20261117
script=${CONTENT_SCRIPT:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT/lineage_validation/app_search_20261117/08_content_checks.py}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
echo "job ${SLURM_JOB_ID:-local} on $(hostname); script $script; parts: $*; $(date -Iseconds)"
/usr/bin/time -v python -I "$script" --app "$app" --only "$@"
echo "exit $? $(date -Iseconds)"
