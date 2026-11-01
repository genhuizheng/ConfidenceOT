#!/bin/bash
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 144
#SBATCH -A MCB26031
# One node of joint-benchmark workers (realdata_benchmark/10_tacc_worker.py). Submitted by submit_benchmark.sh with
#   RB_WORKLIST  worklist TSV        RB_REGISTRY  registry TSV       RB_LOGDIR  log directory
#   RB_NODE      this node's index   RB_NODES     nodes in the stage RB_PROCS   workers per node (default 18)
# Each worker takes rows w, w + total, ... of the worklist (total = RB_NODES x RB_PROCS) and skips rows whose
# SUCCESS exists, so a resubmission continues. 02_run_pair.py runs 8 solver threads; with one BLAS thread each,
# 18 workers fill the 144 cores. No set -e / -u / pipefail (TACC rules).

source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"
repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
cd "$repo" || exit 1
export PYTHONPATH=$repo/src:$repo/cancer_metastasis:$repo/realdata_benchmark:$repo
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
procs=${RB_PROCS:-18}
total=$(( RB_NODES * procs ))
echo "node $RB_NODE of $RB_NODES, $procs workers, $total in all; worklist $RB_WORKLIST; commit $(git rev-parse --short HEAD); $(date -Iseconds)"
for (( p = 0; p < procs; p++ )); do
    w=$(( RB_NODE * procs + p ))
    python realdata_benchmark/10_tacc_worker.py --worklist "$RB_WORKLIST" --registry "$RB_REGISTRY" \
        --worker "$w" --workers "$total" > "$RB_LOGDIR/worker_${SLURM_JOB_ID}_${w}.log" 2>&1 &
done
wait
echo "node $RB_NODE finished $(date -Iseconds)"
