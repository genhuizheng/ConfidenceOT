#!/bin/bash
#SBATCH -J app_prep
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
#SBATCH -A MCB26031
#SBATCH -o /scratch/10119/ghzheng/OT_project/app_search_20261117/logs/prepare_%j.out
# PLAN_TACC.md step 1 on a compute node: 06_prepare_tacc.py for each dataset given, one after another, with time and
# peak memory per dataset (logs/prep_<dataset>_<jobid>.log).
#   sbatch lineage_validation/app_search_20261117/tacc/prepare.slurm kang sciplex2 schiebinger pancreas celegans larry_invivo
# No set -e / -u / pipefail (TACC rules).

source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"
repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
app=/scratch/10119/ghzheng/OT_project/app_search_20261117
cd "$repo" || exit 1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
echo "job ${SLURM_JOB_ID:-local} on $(hostname); commit $(git rev-parse --short HEAD); datasets: $*; $(date -Iseconds)"
for ds in "$@"; do
    echo "=== $ds start $(date -Iseconds)"
    /usr/bin/time -v python -I lineage_validation/app_search_20261117/06_prepare_tacc.py --dataset "$ds" --data "$app/data" --out "$app" \
        > "$app/logs/prep_${ds}_${SLURM_JOB_ID:-local}.log" 2>&1
    echo "=== $ds exit $? $(date -Iseconds)"
    grep -E "Elapsed|Maximum resident" "$app/logs/prep_${ds}_${SLURM_JOB_ID:-local}.log"
done
echo "all done $(date -Iseconds)"
