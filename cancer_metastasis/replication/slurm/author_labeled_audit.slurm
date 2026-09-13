#!/bin/bash
#SBATCH -J cot_rep_audit
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/rep_audit_%j.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/rep_audit_%j.err
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 01:00:00
#SBATCH -A MCB26031
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
manifest=${CONFIDENCEOT_PANCANCER_MANIFEST:-$result/manifest/pancancer_current/pair_manifest_eligible.csv}
root=${CONFIDENCEOT_REPLICATION_ROOT:?CONFIDENCEOT_REPLICATION_ROOT is required}

cd "$repo"
export PYTHONPATH="$repo/src:$repo/cancer_metastasis${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
mkdir -p "$root/GSE181919/manifest" "$root/GSE225857/manifest" "$result/logs"

python cancer_metastasis/07_build_malignant_scope_manifest.py \
  "$manifest" "$root/GSE181919/manifest" \
  --dataset-id GSE181919 \
  --include-annotation Malignant.cells \
  --minimum-cells-per-side 20

args=()
for annotation in \
  Tu01_AREG Tu02_DEFA5 Tu03_SRRM2 Tu04_RGMB Tu05_PCNA Tu06_NKD1 \
  Tu07_MKI67 Tu08_GNG13 Tu09_MUC2 Tu10_COL3A1 Tu11_PLA2G2A; do
  args+=(--include-annotation "$annotation")
done
python cancer_metastasis/07_build_malignant_scope_manifest.py \
  "$manifest" "$root/GSE225857/manifest" \
  --dataset-id GSE225857 \
  "${args[@]}" \
  --minimum-cells-per-side 20

touch "$root/AUDIT_COMPLETE"
