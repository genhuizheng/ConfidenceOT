#!/bin/bash
#SBATCH -J cot_pydeseq2
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/pydeseq2_%j.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/pydeseq2_%j.err
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 04:00:00
#SBATCH -A MCB26031
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=END,FAIL

source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
groups=${CONFIDENCEOT_DEG_ROOT:-$result/robust_target_malignant_deg_all_pairs_GSE180661}
output=${CONFIDENCEOT_PYDESEQ2_ROOT:-$result/robust_target_malignant_pydeseq2_GSE180661}

cd "$repo"
export PYTHONPATH="$repo/src:$repo/cancer_metastasis${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export PYTHONUNBUFFERED=1

mkdir -p "$output" "$result/logs"
python cancer_metastasis/13_run_paired_pydeseq2.py \
  "$groups" "$output" \
  --minimum-cells-per-patient-status 20 \
  --minimum-total-count 10 \
  --maximum-fdr 0.05 \
  --absolute-log2-fold-change-thresholds 0.5 1.0 \
  --n-cpus 16
