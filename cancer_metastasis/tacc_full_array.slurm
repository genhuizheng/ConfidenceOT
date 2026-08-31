#!/bin/bash
#SBATCH -J cot_cancer_full
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/full_%A_%a.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/full_%A_%a.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 24:00:00
#SBATCH -A MCB26031
# Supply --array=0-(eligible_rows-1)%4 at submission; see the launch commands.
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=END,FAIL

source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
eligible="$result/manifest/pair_manifest_eligible.csv"
budget=${CONFIDENCEOT_REJECTION_BUDGET:?Set CONFIDENCEOT_REJECTION_BUDGET after reviewing the pilot}
mkdir -p "$result/logs" "$result/full"
cd "$repo"
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export PYTHONUNBUFFERED=1

python cancer_metastasis/02_run_pair.py "$eligible" "$result/full" \
  --index "$SLURM_ARRAY_TASK_ID" \
  --rejection-budget "$budget" \
  --null-calibration-replicates 5 \
  --null-validation-replicates 5 \
  --calibration-grid-size 5 \
  --tolerance 1e-4 \
  --workers 1 \
  --device cuda
