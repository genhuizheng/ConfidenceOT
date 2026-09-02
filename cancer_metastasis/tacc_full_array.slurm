#!/bin/bash
#SBATCH -J cot_cancer_full
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/full_%A_%a.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/full_%A_%a.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 24:00:00
#SBATCH -A MCB26031
# Submit a small worker array and set CONFIDENCEOT_PAIR_COUNT. Each worker
# processes indices task_id, task_id + task_count, ... to avoid QOS array limits.
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=END,FAIL

source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
eligible=${CONFIDENCEOT_MANIFEST:-$result/manifest/pair_manifest_eligible.csv}
output_root=${CONFIDENCEOT_OUTPUT_ROOT:-$result/full}
analysis_scope=${CONFIDENCEOT_ANALYSIS_SCOPE:-all}
annotation_spec=${CONFIDENCEOT_INCLUDE_ANNOTATIONS:-}
budget=${CONFIDENCEOT_REJECTION_BUDGET:?Set CONFIDENCEOT_REJECTION_BUDGET after reviewing the pilot}
mkdir -p "$result/logs" "$output_root"
cd "$repo"
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export PYTHONUNBUFFERED=1

pair_count=${CONFIDENCEOT_PAIR_COUNT:-}
if [[ -z "$pair_count" ]]; then
  echo "CONFIDENCEOT_PAIR_COUNT must be set to the eligible manifest row count" >&2
  exit 2
fi
if ! [[ "$pair_count" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid CONFIDENCEOT_PAIR_COUNT=$pair_count" >&2
  exit 2
fi
if [[ ! -f "$eligible" ]]; then
  echo "Manifest does not exist: $eligible" >&2
  exit 2
fi
manifest_count=$(python - "$eligible" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8-sig") as handle:
    print(sum(1 for _ in csv.DictReader(handle)))
PY
)
if [[ "$manifest_count" -ne "$pair_count" ]]; then
  echo "Manifest row count ($manifest_count) != CONFIDENCEOT_PAIR_COUNT ($pair_count)" >&2
  exit 2
fi

task_id=${SLURM_ARRAY_TASK_ID:-0}
task_count=${SLURM_ARRAY_TASK_COUNT:-1}
restart_count=${SLURM_RESTART_COUNT:-0}
failure_file="$result/logs/failed_${SLURM_ARRAY_JOB_ID:-manual}_${task_id}_restart${restart_count}.txt"
failures=0

echo "Worker $task_id/$task_count processing $pair_count manifest rows"
scope_args=(--analysis-scope "$analysis_scope")
if [[ "$analysis_scope" == "malignant" ]]; then
  if [[ -z "$annotation_spec" ]]; then
    echo "CONFIDENCEOT_INCLUDE_ANNOTATIONS is required for malignant scope" >&2
    exit 2
  fi
  IFS='|' read -r -a annotations <<< "$annotation_spec"
  for annotation in "${annotations[@]}"; do
    scope_args+=(--include-annotation "$annotation")
  done
fi
for ((index=task_id; index<pair_count; index+=task_count)); do
  echo "===== START index=$index $(date --iso-8601=seconds) ====="
  if python cancer_metastasis/02_run_pair.py "$eligible" "$output_root" \
    --index "$index" \
    "${scope_args[@]}" \
    --rejection-budget "$budget" \
    --null-calibration-replicates 5 \
    --null-validation-replicates 5 \
    --calibration-grid-size 5 \
    --tolerance 1e-4 \
    --workers 1 \
    --device cuda \
    --skip-completed; then
    echo "===== COMPLETE index=$index $(date --iso-8601=seconds) ====="
  else
    status=$?
    echo "$index,$status" >> "$failure_file"
    failures=$((failures + 1))
    echo "===== FAILED index=$index exit=$status $(date --iso-8601=seconds) =====" >&2
  fi
done

echo "Worker $task_id finished with $failures failed pair(s)"
if (( failures > 0 )); then
  exit 1
fi
