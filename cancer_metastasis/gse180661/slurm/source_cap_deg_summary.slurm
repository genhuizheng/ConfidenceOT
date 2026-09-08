#!/bin/bash
#SBATCH -J cot_src_cap_sum
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/src_cap_sum_%j.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/src_cap_sum_%j.err
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

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
ot_base=${CONFIDENCEOT_SOURCE_CAP_OT_BASE:-$result/gse180661_source_cap_sensitivity_ot}
pseudobulk_base=${CONFIDENCEOT_SOURCE_CAP_PSEUDOBULK_BASE:-$result/gse180661_source_cap_sensitivity_pseudobulk}
deg_base=${CONFIDENCEOT_SOURCE_CAP_DEG_BASE:-$result/gse180661_source_cap_sensitivity_pydeseq2}
output=${CONFIDENCEOT_SOURCE_CAP_SUMMARY_ROOT:-$result/gse180661_source_cap_sensitivity_summary}
source_caps=${CONFIDENCEOT_SOURCE_CAPS:-"0.50:0.70:0.75:0.85:0.90:0.95"}
target_cap=${CONFIDENCEOT_TARGET_REJECTION_BUDGET:-0.95}

cd "$repo"
export PYTHONPATH="$repo/src:$repo/cancer_metastasis${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export PYTHONUNBUFFERED=1
read -r -a caps <<< "${source_caps//[:,]/ }"
python -m cancer_metastasis.gse180661.source_cap_deg_sensitivity \
  "$ot_base" "$pseudobulk_base" "$deg_base" "$output" \
  --source-caps "${caps[@]}" --target-cap "$target_cap"
