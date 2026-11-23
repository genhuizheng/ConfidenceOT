#!/bin/bash
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 144
#SBATCH -A MCB26031
# One node of the frozen-calibration-v3 grid (one array task = one node). Submitted by submit_v3.sh with
#   V3_WORKLIST  worklist TSV    V3_PROCS  workers on this node    V3_FIT_WORKERS  02_run_pair --workers per fit
#   V3_LOGS      log directory (made by submit_v3.sh)
#   V3_MAX_LARGE / V3_LARGE_SIZE  at most this many rows of units above this many source x target cells at once (memory)
# Every worker claims rows from the shared worklist (worker.py), so the array tasks need no row split and a
# resubmission continues where the last one stopped. One BLAS thread per process.
# No set -e / -u / pipefail (TACC rules).

# V3_VENV (optional, 2026-11-23): a module-python3 venv instead of the conda environment (whose interpreter the scratch
# purge removed); its bin/python is the wrapper that puts the module's libpython on the library path
if [[ -n "${V3_VENV:-}" ]]; then
    module load gcc python3/3.11.8 2> /dev/null || module load python3
    export PATH="$V3_VENV/bin:$PATH"
else
    source /home1/10119/ghzheng/.bashrc
    conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"
fi
echo "python $(command -v python) $(python --version 2>&1)"
repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
cd "$repo" || exit 1
export PYTHONPATH=$repo/src:$repo/cancer_metastasis:$repo/realdata_benchmark:$repo
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
procs=${V3_PROCS:-36}
scratch=/tmp/v3_grid_${SLURM_JOB_ID:-local}
mkdir -p "$scratch"
# V3_CONFIGS (optional): ';'-separated configuration ids, passed to worker.py --configs (fit only these of each key)
configs=()
if [[ -n "${V3_CONFIGS:-}" ]]; then IFS=';' read -r -a configs <<< "$V3_CONFIGS"; configs=(--configs "${configs[@]}"); fi
echo "task ${SLURM_ARRAY_TASK_ID:-0} job ${SLURM_JOB_ID:-local} on $(hostname): $procs workers x ${V3_FIT_WORKERS:-4} fit workers; worklist $V3_WORKLIST; representation ${V3_REPRESENTATION_SOURCE:-rna}; configs ${V3_CONFIGS:-all}; commit $(git rev-parse --short HEAD); $(date -Iseconds)"
for (( w = 0; w < procs; w++ )); do
    python realdata_benchmark/v3_grid/worker.py --worklist "$V3_WORKLIST" --fit-workers "${V3_FIT_WORKERS:-4}" --scratch "$scratch" \
        --large-slots "$scratch/slots" --max-large "${V3_MAX_LARGE:-4}" --large-size "${V3_LARGE_SIZE:-2e7}" "${configs[@]}" \
        > "$V3_LOGS/worker_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}_${w}.log" 2>&1 &
    sleep 2
done
wait
rm -rf "$scratch"
echo "task ${SLURM_ARRAY_TASK_ID:-0} finished $(date -Iseconds)"
