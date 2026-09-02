#!/bin/bash
#SBATCH -J cot_deg_meta
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/deg_meta_%j.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/deg_meta_%j.err
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 02:00:00
#SBATCH -A MCB26031
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=END,FAIL

source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
groups=${CONFIDENCEOT_DEG_ROOT:-$result/robust_target_malignant_deg_all_pairs_GSE180661}
output=${CONFIDENCEOT_DEG_META_ROOT:-$result/robust_target_malignant_deg_all_pairs_GSE180661_meta}

cd "$repo"
export PYTHONPATH="$repo/src:$repo/cancer_metastasis${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

mkdir -p "$output" "$result/logs"
python cancer_metastasis/12_meta_analyze_robust_target_deg.py \
  "$groups" "$output" \
  --minimum-patients 5 \
  --minimum-site-patients 3 \
  --individual-fdr 0.05 \
  --individual-min-abs-log2fc 0.25 \
  --individual-min-expression-fraction 0.05 \
  --individual-overlap-top-n 100
