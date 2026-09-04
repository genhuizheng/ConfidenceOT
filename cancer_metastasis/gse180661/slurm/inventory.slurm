#!/bin/bash
#SBATCH -J cot_gse180661_inventory
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/gse180661_inventory_%j.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/gse180661_inventory_%j.err
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 02:00:00
#SBATCH -A MCB26031
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=END,FAIL

source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"
set -euo pipefail

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
converted=${CANCER_CONVERTED_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/converted}
output=${CONFIDENCEOT_INVENTORY_ROOT:-$result/gse180661_inventory}

cd "$repo"
export PYTHONPATH="$repo/src:$repo/cancer_metastasis${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export PYTHONUNBUFFERED=1
mkdir -p "$output" "$result/logs"

python -m cancer_metastasis.gse180661.inventory \
  "$converted" "$output" \
  --dataset GSE180661 \
  --malignant-annotation Ovarian.cancer.cell \
  --minimum-malignant-cells 20

touch "$output/SUCCESS"
