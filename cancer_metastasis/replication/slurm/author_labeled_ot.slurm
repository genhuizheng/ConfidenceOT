#!/bin/bash
#SBATCH -J cot_rep_ot
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/rep_ot_%j.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/rep_ot_%j.err
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
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
dataset=${CONFIDENCEOT_REPLICATION_DATASET:-all}
source_budget=${CONFIDENCEOT_SOURCE_REJECTION_BUDGET:-0.85}
target_budget=${CONFIDENCEOT_TARGET_REJECTION_BUDGET:-0.95}
cd "$repo"
export PYTHONPATH="$repo/src:$repo/cancer_metastasis${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
export PYTHONUNBUFFERED=1

run_dataset() {
  local dataset=$1
  shift
  local manifest="$root/$dataset/manifest/pair_manifest_malignant_eligible.csv"
  local output="$root/$dataset/ot"
  local pair_count
  pair_count=$(python - "$manifest" <<'PY'
import csv
import sys
with open(sys.argv[1], newline="", encoding="utf-8-sig") as handle:
    print(sum(1 for _ in csv.DictReader(handle)))
PY
)
  echo "===== $dataset: $pair_count eligible malignant pairs ====="
  for ((index=0; index<pair_count; index++)); do
    python cancer_metastasis/02_run_pair.py "$manifest" "$output" \
      --index "$index" \
      --analysis-scope malignant \
      "$@" \
      --cell-qc \
      --source-rejection-budget "$source_budget" \
      --target-rejection-budget "$target_budget" \
      --max-observed-cells-per-side 10000 \
      --null-calibration-replicates 5 \
      --null-validation-replicates 5 \
      --calibration-grid-size 5 \
      --tolerance 1e-4 \
      --workers 1 \
      --device cuda \
      --skip-completed
  done
}

run_gse181919() {
  run_dataset GSE181919 --include-annotation Malignant.cells
  touch "$root/GSE181919/OT_COMPLETE"
}

run_gse225857() {
  local labels=()
  local annotation
  for annotation in \
    Tu01_AREG Tu02_DEFA5 Tu03_SRRM2 Tu04_RGMB Tu05_PCNA Tu06_NKD1 \
    Tu07_MKI67 Tu08_GNG13 Tu09_MUC2 Tu10_COL3A1 Tu11_PLA2G2A; do
    labels+=(--include-annotation "$annotation")
  done
  run_dataset GSE225857 "${labels[@]}"
  touch "$root/GSE225857/OT_COMPLETE"
}

case "$dataset" in
  GSE181919)
    run_gse181919
    ;;
  GSE225857)
    run_gse225857
    ;;
  all)
    run_gse181919
    run_gse225857
    touch "$root/OT_COMPLETE"
    ;;
  *)
    echo "Unsupported CONFIDENCEOT_REPLICATION_DATASET=$dataset" >&2
    exit 2
    ;;
esac
