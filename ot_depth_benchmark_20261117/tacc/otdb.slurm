#!/bin/bash
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 144
#SBATCH -A MCB26031
# One stage of the OT depth benchmark (ot_depth_benchmark_20261117/PLAN.md) on one gg node. Submitted by submit_otdb.sh
# with OTDB_STAGE and OTDB_ARGS; logs in the --output given by submit_otdb.sh. One BLAS thread per process.
# No set -e / -u / pipefail (TACC rules).

source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"
repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
cd "$repo" || exit 1
export PYTHONPATH=$repo/src:$repo/cancer_metastasis:$repo/realdata_benchmark:$repo
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
echo "stage $OTDB_STAGE job ${SLURM_JOB_ID:-local} on $(hostname); commit $(git rev-parse --short HEAD); $(date -Iseconds)"
echo "args: $OTDB_ARGS"
case "$OTDB_STAGE" in
    perturb) python -I ot_depth_benchmark_20261117/run_perturb.py $OTDB_ARGS ;;
    splatter) python ot_depth_benchmark_20261117/run_splatter.py $OTDB_ARGS ;;
    learned) conda activate "${OTDB_LEARNED_ENV:-/scratch/10119/ghzheng/conda_envs/otbench}"
             python -I ot_depth_benchmark_20261117/run_learned.py $OTDB_ARGS ;;
    *) echo "unknown stage $OTDB_STAGE"; exit 1 ;;
esac
echo "finished $(date -Iseconds)"
