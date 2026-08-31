#!/bin/bash
#SBATCH -J cot_cancer_manifest
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/manifest_%j.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/manifest_%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 02:00:00
#SBATCH -A MCB26031
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=END,FAIL

source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
converted=${CANCER_CONVERTED_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/converted}
result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
mkdir -p "$result/logs" "$result/manifest"
cd "$repo"
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"

python cancer_metastasis/01_build_pair_manifest.py "$converted" "$result/manifest"
python cancer_metastasis/03_make_budget_pilot.py \
  "$result/manifest/pair_manifest_eligible.csv" \
  "$result/manifest/budget_pilot_manifest.csv"
