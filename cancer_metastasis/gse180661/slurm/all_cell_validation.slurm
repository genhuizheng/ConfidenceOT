#!/bin/bash
#SBATCH -J cot_gse180661_allcell
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/gse180661_allcell_%j.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/gse180661_allcell_%j.err
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 04:00:00
#SBATCH -A MCB26031
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=END,FAIL

source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"
set -euo pipefail

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
manifest=${CONFIDENCEOT_MANIFEST:-$result/manifest/GSE180661_all/pair_manifest_all_cell_eligible.csv}
ot_root=${CONFIDENCEOT_ALL_CELL_OT_ROOT:-$result/gse180661_all_cell_ot}
output=${CONFIDENCEOT_ALL_CELL_VALIDATION_ROOT:-$result/gse180661_all_cell_validation}

cd "$repo"
export PYTHONPATH="$repo/src:$repo/cancer_metastasis${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export PYTHONUNBUFFERED=1
mkdir -p "$output" "$result/logs"

python -m cancer_metastasis.gse180661.all_cell_validation \
  "$manifest" "$ot_root" "$output" \
  --dataset GSE180661 \
  --method M4-E \
  --plot-each-pair

touch "$output/SUCCESS"
