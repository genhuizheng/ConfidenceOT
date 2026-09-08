#!/bin/bash
#SBATCH -J cot_qc_compare
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/qc_compare_%j.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/qc_compare_%j.err
#SBATCH -p gg
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
result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
pre_ot=${CONFIDENCEOT_PRE_QC_OT_ROOT:?Set CONFIDENCEOT_PRE_QC_OT_ROOT}
post_ot=${CONFIDENCEOT_POST_QC_OT_ROOT:?Set CONFIDENCEOT_POST_QC_OT_ROOT}
pre_deg=${CONFIDENCEOT_PRE_QC_DEG_ROOT:?Set CONFIDENCEOT_PRE_QC_DEG_ROOT}
post_deg=${CONFIDENCEOT_POST_QC_DEG_ROOT:?Set CONFIDENCEOT_POST_QC_DEG_ROOT}
pre_gsea=${CONFIDENCEOT_PRE_QC_GSEA_ROOT:?Set CONFIDENCEOT_PRE_QC_GSEA_ROOT}
post_gsea=${CONFIDENCEOT_POST_QC_GSEA_ROOT:?Set CONFIDENCEOT_POST_QC_GSEA_ROOT}
output=${CONFIDENCEOT_QC_COMPARISON_ROOT:?Set CONFIDENCEOT_QC_COMPARISON_ROOT}

cd "$repo"
export PYTHONPATH="$repo/src:$repo/cancer_metastasis${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTHONUNBUFFERED=1
mkdir -p "$output" "$result/logs"

python -m cancer_metastasis.gse180661.compare_pre_post_qc \
  "$pre_ot" "$post_ot" "$pre_deg" "$post_deg" \
  "$pre_gsea" "$post_gsea" "$output"
