#!/bin/bash
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 144
#SBATCH -A MCB26031
# One node of the lineage preprocessing factorial (one array task = one node). Submitted by submit_factorial.sh with
#   LF_ROOT  output root    LF_WORKLIST  worklist TSV    LF_PROCS  workers on this node    LF_MAX_LARGE  large-unit slots
# Every worker claims rows from the shared worklist (10_worker.py), so the array tasks need no row split and a
# resubmission continues where the last one stopped. Each fit runs 8 solver threads (fit_many) with one BLAS thread.
# No set -e / -u / pipefail (TACC rules).

source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"
repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
cd "$repo" || exit 1
export PYTHONPATH=$repo/src:$repo/cancer_metastasis:$repo/realdata_benchmark:$repo
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
procs=${LF_PROCS:-36}
slots=/tmp/lineage_factorial_${SLURM_JOB_ID:-local}
logdir=$LF_ROOT/logs/workers
mkdir -p "$logdir"
echo "task ${SLURM_ARRAY_TASK_ID:-0} job ${SLURM_JOB_ID:-local} on $(hostname): $procs workers, large slots ${LF_MAX_LARGE:-4}; worklist $LF_WORKLIST; commit $(git rev-parse --short HEAD); $(date -Iseconds)"
for (( w = 0; w < procs; w++ )); do
    python lineage_validation/preprocessing_factorial_20261103/10_worker.py --root "$LF_ROOT" --worklist "$LF_WORKLIST" \
        --large-slots "$slots" --max-large "${LF_MAX_LARGE:-4}" > "$logdir/worker_${SLURM_JOB_ID:-local}_${w}.log" 2>&1 &
    sleep 2
done
wait
rm -rf "$slots"
echo "task ${SLURM_ARRAY_TASK_ID:-0} finished $(date -Iseconds)"
