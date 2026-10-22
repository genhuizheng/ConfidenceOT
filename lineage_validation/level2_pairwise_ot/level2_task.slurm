#!/bin/bash
#SBATCH -J l2_ab
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A MCB26031
# One Level 2 case per array task: SLURM_ARRAY_TASK_ID is the row index of inputs/cases.tsv, and
# run_pairwise.py makes exactly the two OT runs (A_standard_OT, B_our_OT) in it. Submitted by submit_full.sh,
# which sets the time limit per resource class, the log paths and L2_OUT. No set -u, as everywhere in this repo.
set -eo pipefail

source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"
repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
cd "$repo"
export PYTHONPATH=$repo/src:$repo/cancer_metastasis:$repo
index=${SLURM_ARRAY_TASK_ID:?not an array task}
echo "== $(date -Iseconds) job ${SLURM_ARRAY_JOB_ID}_${index} host $(hostname) commit $(git rev-parse --short HEAD) case $index"
python lineage_validation/level2_pairwise_ot/run_pairwise.py --index "$index" --out "${L2_OUT:?}"
echo "== $(date -Iseconds) done case $index"
