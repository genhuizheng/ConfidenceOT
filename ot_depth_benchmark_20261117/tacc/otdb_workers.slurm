#!/bin/bash
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 144
#SBATCH -A MCB26031
# One node of claim-based workers on an otdb worklist (otdb_tacc.py work), as the v3 grid's v3_workers.slurm: every
# worker claims rows from the shared worklist, so array tasks need no row split and a resubmission continues.
#   OTDB_WORKLIST  worklist TSV     OTDB_PROCS  workers on this node     OTDB_LOGS  log directory (made before sbatch)
#   CELLOT_REPO    the CellOT checkout (cellot rows)     OTDB_THREADS  BLAS / torch threads per row (default 1)
# One BLAS thread per process. No set -e / -u / pipefail (TACC rules).

source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"
repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
cd "$repo" || exit 1
export PYTHONPATH=$repo/src:$repo/cancer_metastasis:$repo/realdata_benchmark:$repo
# OTDB_THREADS threads per row (default 1); CellOT rows use 8 (2026-11-23: 1 thread needs > 3 h per row)
t=${OTDB_THREADS:-1}
export OMP_NUM_THREADS=$t OPENBLAS_NUM_THREADS=$t MKL_NUM_THREADS=$t NUMEXPR_NUM_THREADS=$t XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=$t"
export CELLOT_REPO=${CELLOT_REPO:-/scratch/10119/ghzheng/OT_project/code/cellot}
procs=${OTDB_PROCS:-36}
echo "task ${SLURM_ARRAY_TASK_ID:-0} job ${SLURM_JOB_ID:-local} on $(hostname): $procs workers x $t threads; worklist $OTDB_WORKLIST; commit $(git rev-parse --short HEAD); $(date -Iseconds)"
for (( w = 0; w < procs; w++ )); do
    python ot_depth_benchmark_20261117/tacc/otdb_tacc.py work --worklist "$OTDB_WORKLIST" \
        > "$OTDB_LOGS/worker_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}_${w}.log" 2>&1 &
    sleep 2
done
wait
echo "task ${SLURM_ARRAY_TASK_ID:-0} finished $(date -Iseconds)"
