#!/bin/bash
#SBATCH -J cot_gate_ucell
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/gate_ucell_%A_%a.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/gate_ucell_%A_%a.err
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 06:00:00
#SBATCH -A MCB26031
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=END,FAIL

set -eo pipefail
source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"
set -u

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
source_only=${CONFIDENCEOT_SOURCE_ONLY_ROOT:?CONFIDENCEOT_SOURCE_ONLY_ROOT is required}
output=${CONFIDENCEOT_GATE_UCELL_ROOT:?CONFIDENCEOT_GATE_UCELL_ROOT is required}
stage=${CONFIDENCEOT_GATE_UCELL_STAGE:-prepare}
manifest=${CONFIDENCEOT_GSE180661_MANIFEST:-$result/manifest/GSE180661_malignant/pair_manifest_malignant_eligible.csv}
datasets=(GSE180661 GSE181919 GSE225857)
dataset=${datasets[${SLURM_ARRAY_TASK_ID:-0}]}

cd "$repo"
export PYTHONPATH="$repo/src:$repo/cancer_metastasis${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
export PYTHONUNBUFFERED=1
mkdir -p "$output" "$result/logs"

python -m cancer_metastasis.replication.validate_primary_gate_ucell \
  "$source_only" "$manifest" "$output" \
  --dataset "$dataset" \
  --stage "$stage" \
  --budget-tag budget_source_0.85_target_0.00 \
  --signature-sizes 50 100 200 \
  --n-cpus 16 \
  --ucell-chunk-size 1000
