#!/bin/bash
#SBATCH -J cot_pan_sum
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/pan_sum_%j.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/pan_sum_%j.err
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
manifest=${CONFIDENCEOT_MANIFEST:-$result/manifest/pancancer_current/pair_manifest_eligible.csv}
input=${CONFIDENCEOT_OUTPUT_ROOT:-$result/pancancer_all_cell_sourcecap_0p85_targetcap_0p95_20260902}
output=${CONFIDENCEOT_REJECTION_SUMMARY_ROOT:-$result/pancancer_all_cell_rejection_annotation_summary_20260903}

cd "$repo"
export PYTHONPATH="$repo/src:$repo/cancer_metastasis${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTHONUNBUFFERED=1
mkdir -p "$result/logs"

if [[ -e "$output" ]]; then
  echo "STOP: output already exists: $output" >&2
  exit 1
fi

python cancer_metastasis/19_summarize_pancancer_all_cell_rejection.py \
  "$manifest" "$input" "$output"
