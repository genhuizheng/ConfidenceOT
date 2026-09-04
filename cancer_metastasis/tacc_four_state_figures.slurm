#!/bin/bash
#SBATCH -J cot_four_fig
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/four_state_figures_%j.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/four_state_figures_%j.err
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 01:00:00
#SBATCH -A MCB26031
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=END,FAIL

source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
four_state=${CONFIDENCEOT_FOUR_STATE_ROOT:-$result/four_state_malignant_pseudobulk_GSE180661_20260903}
pydeseq2=${CONFIDENCEOT_PYDESEQ2_ROOT:-$result/four_state_malignant_pydeseq2_GSE180661_20260903}
gsea=${CONFIDENCEOT_GSEA_ROOT:-$result/four_state_malignant_gsea_GSE180661_20260903}
output=${CONFIDENCEOT_FIGURE_ROOT:-$result/four_state_malignant_figures_GSE180661_20260903}

cd "$repo"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

test ! -e "$output" || {
  echo "Output already exists: $output" >&2
  exit 1
}

python cancer_metastasis/22_visualize_four_state_malignant.py \
  "$four_state" "$pydeseq2" "$gsea" "$output" \
  --maximum-fdr 0.05 --minimum-absolute-log2fc 0.5
