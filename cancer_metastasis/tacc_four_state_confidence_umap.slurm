#!/bin/bash
#SBATCH -J cot_conf_umap
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/confidence_umap_%j.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/confidence_umap_%j.err
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
manifest=${CONFIDENCEOT_MANIFEST:-$result/manifest/GSE180661_malignant/pair_manifest_malignant_eligible.csv}
four_state=${CONFIDENCEOT_FOUR_STATE_ROOT:-$result/four_state_malignant_pseudobulk_GSE180661_20260903}
baseline=${CONFIDENCEOT_BASELINE_ROOT:-$result/malignant_full_GSE180661}
sensitivity=${CONFIDENCEOT_SENSITIVITY_ROOT:-$result/malignant_sourcecap_0p90_targetcap_0p95_GSE180661}
output=${CONFIDENCEOT_UMAP_ROOT:-$result/four_state_malignant_confidence_umap_separate_sides_pairwise_GSE180661_20260904}

cd "$repo"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

test ! -e "$output" || {
  echo "Output already exists: $output" >&2
  exit 1
}

python cancer_metastasis/24_visualize_four_state_confidence_umap.py \
  "$manifest" "$four_state" "$output" \
  --baseline-root "$baseline" \
  --sensitivity-root "$sensitivity" \
  --malignant-annotation Ovarian.cancer.cell \
  --n-hvg 2000 \
  --n-pcs 40 \
  --n-neighbors 20 \
  --maximum-cells 250000 \
  --seed 20260904
