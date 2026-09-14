#!/bin/bash
#SBATCH -J cot_rep_pb
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/rep_pb_%j.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/rep_pb_%j.err
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 02:00:00
#SBATCH -A MCB26031
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=END,FAIL

set -eo pipefail
source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"
set -u

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
root=${CONFIDENCEOT_REPLICATION_ROOT:?CONFIDENCEOT_REPLICATION_ROOT is required}
budget_tag=${CONFIDENCEOT_BUDGET_TAG:-budget_source_0.85_target_0.95}
cd "$repo"
export PYTHONPATH="$repo/src:$repo/cancer_metastasis${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export PYTHONUNBUFFERED=1

run_dataset() {
  local dataset=$1
  shift
  local manifest="$root/$dataset/manifest/pair_manifest_malignant_eligible.csv"
  local ot="$root/$dataset/ot"
  local output="$root/$dataset/primary_pseudobulk"
  local pair_count
  pair_count=$(python - "$manifest" <<'PY'
import csv
import sys
with open(sys.argv[1], newline="", encoding="utf-8-sig") as handle:
    print(sum(1 for _ in csv.DictReader(handle)))
PY
)
  for ((index=0; index<pair_count; index++)); do
    python -m cancer_metastasis.gse180661.primary_pseudobulk \
      "$manifest" "$ot" "$output" \
      --index "$index" \
      --budget-tag "$budget_tag" \
      --method M4-E \
      --minimum-cells-per-state 20 \
      --gate-state-labels \
      "$@"
  done
}

run_dataset GSE181919 --malignant-annotation Malignant.cells

labels=()
for annotation in \
  Tu01_AREG Tu02_DEFA5 Tu03_SRRM2 Tu04_RGMB Tu05_PCNA Tu06_NKD1 \
  Tu07_MKI67 Tu08_GNG13 Tu09_MUC2 Tu10_COL3A1 Tu11_PLA2G2A; do
  labels+=(--malignant-annotation "$annotation")
done
run_dataset GSE225857 "${labels[@]}"

touch "$root/PSEUDOBULK_COMPLETE"
