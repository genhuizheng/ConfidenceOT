#!/bin/bash
#SBATCH -J cot_src_cap_ot
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/src_cap_ot_%A_%a.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/src_cap_ot_%A_%a.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 24:00:00
#SBATCH -A MCB26031
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=END,FAIL

source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
manifest=${CONFIDENCEOT_MANIFEST:-$result/manifest/GSE180661_malignant/pair_manifest_malignant_eligible.csv}
output_base=${CONFIDENCEOT_SOURCE_CAP_OT_BASE:-$result/gse180661_source_cap_sensitivity_ot}
source_caps=${CONFIDENCEOT_SOURCE_CAPS:-"0.50,0.70,0.75,0.85,0.90,0.95"}
target_cap=${CONFIDENCEOT_TARGET_REJECTION_BUDGET:-0.95}
pair_count=${CONFIDENCEOT_PAIR_COUNT:-}
maximum_observed_cells=${CONFIDENCEOT_MAX_OBSERVED_CELLS_PER_SIDE:-10000}

cd "$repo"
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
export PYTHONUNBUFFERED=1
mkdir -p "$output_base" "$result/logs"

manifest_count=$(python - "$manifest" <<'PY'
import csv
import sys
with open(sys.argv[1], newline="", encoding="utf-8-sig") as handle:
    print(sum(1 for _ in csv.DictReader(handle)))
PY
)
pair_count=${pair_count:-$manifest_count}
if [[ "$pair_count" -ne "$manifest_count" ]]; then
  echo "Manifest row count ($manifest_count) != pair count ($pair_count)" >&2
  exit 2
fi

read -r -a caps <<< "${source_caps//,/ }"
cap_count=${#caps[@]}
total=$((pair_count * cap_count))
task_id=${SLURM_ARRAY_TASK_ID:-0}
task_count=${SLURM_ARRAY_TASK_COUNT:-1}
failures=0

for ((linear=task_id; linear<total; linear+=task_count)); do
  cap_index=$((linear / pair_count))
  pair_index=$((linear % pair_count))
  source_cap=${caps[$cap_index]}
  cap_slug=$(printf "%.2f" "$source_cap")
  cap_slug=${cap_slug/./p}
  output="$output_base/source_cap_$cap_slug"
  mkdir -p "$output"
  echo "===== START cap=$source_cap pair_index=$pair_index $(date --iso-8601=seconds) ====="
  if python cancer_metastasis/02_run_pair.py "$manifest" "$output" \
      --index "$pair_index" \
      --analysis-scope malignant \
      --include-annotation Ovarian.cancer.cell \
      --source-rejection-budget "$source_cap" \
      --target-rejection-budget "$target_cap" \
      --max-observed-cells-per-side "$maximum_observed_cells" \
      --null-calibration-replicates 5 \
      --null-validation-replicates 5 \
      --calibration-grid-size 5 \
      --tolerance 1e-4 \
      --workers 1 \
      --device cuda \
      --skip-completed; then
    echo "===== COMPLETE cap=$source_cap pair_index=$pair_index $(date --iso-8601=seconds) ====="
  else
    status=$?
    failures=$((failures + 1))
    echo "cap=$source_cap,pair_index=$pair_index,exit=$status" >&2
  fi
done

if (( failures > 0 )); then
  exit 1
fi
