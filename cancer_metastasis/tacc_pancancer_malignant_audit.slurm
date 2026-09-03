#!/bin/bash
#SBATCH -J cot_pan_audit
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/pan_audit_%j.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/pan_audit_%j.err
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
converted=${CANCER_CONVERTED_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/converted}
manifest_root=${CONFIDENCEOT_PANCANCER_MANIFEST_ROOT:-$result/manifest/pancancer_current}
manifest=${CONFIDENCEOT_MANIFEST:-$manifest_root/pair_manifest_eligible.csv}
output=${CONFIDENCEOT_PANCANCER_AUDIT_ROOT:-$result/pancancer_malignant_audit}

cd "$repo"
export PYTHONPATH="$repo/src:$repo/cancer_metastasis${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTHONUNBUFFERED=1
mkdir -p "$result/logs"

if [[ -e "$output" || -e "$manifest_root" ]]; then
  echo "STOP: audit output or current manifest already exists" >&2
  exit 1
fi

python cancer_metastasis/01_build_pair_manifest.py \
  "$converted" "$manifest_root" --minimum-cells-per-side 20

python cancer_metastasis/18_audit_pancancer_malignant_labels.py \
  "$manifest" "$output" --minimum-cells-per-side 20
