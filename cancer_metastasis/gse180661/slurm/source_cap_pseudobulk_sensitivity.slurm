#!/bin/bash
#SBATCH -J cot_src_cap_pb
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/src_cap_pb_%A_%a.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/src_cap_pb_%A_%a.err
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 06:00:00
#SBATCH -A MCB26031
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=END,FAIL

source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
manifest=${CONFIDENCEOT_MANIFEST:-$result/manifest/GSE180661_malignant/pair_manifest_malignant_eligible.csv}
ot_base=${CONFIDENCEOT_SOURCE_CAP_OT_BASE:-$result/gse180661_source_cap_sensitivity_ot}
output_base=${CONFIDENCEOT_SOURCE_CAP_PSEUDOBULK_BASE:-$result/gse180661_source_cap_sensitivity_pseudobulk}
source_caps=${CONFIDENCEOT_SOURCE_CAPS:-"0.50,0.70,0.75,0.85,0.90,0.95"}
target_cap=${CONFIDENCEOT_TARGET_REJECTION_BUDGET:-0.95}
pair_count=${CONFIDENCEOT_PAIR_COUNT:-}

cd "$repo"
export PYTHONPATH="$repo/src:$repo/cancer_metastasis${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export PYTHONUNBUFFERED=1
mkdir -p "$output_base" "$result/logs"

if [[ -z "$pair_count" ]]; then
  pair_count=$(python - "$manifest" <<'PY'
import csv
import sys
with open(sys.argv[1], newline="", encoding="utf-8-sig") as handle:
    print(sum(1 for _ in csv.DictReader(handle)))
PY
  )
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
  ot_root="$ot_base/source_cap_$cap_slug"
  output="$output_base/source_cap_$cap_slug"
  budget_tag=$(printf "budget_source_%.2f_target_%.2f" "$source_cap" "$target_cap")
  mkdir -p "$output"
  if python -m cancer_metastasis.gse180661.primary_pseudobulk \
      "$manifest" "$ot_root" "$output" --index "$pair_index" \
      --budget-tag "$budget_tag" --minimum-cells-per-state 20; then
    echo "COMPLETE cap=$source_cap pair_index=$pair_index"
  else
    status=$?
    failures=$((failures + 1))
    echo "FAILED cap=$source_cap pair_index=$pair_index exit=$status" >&2
  fi
done

if (( failures > 0 )); then
  exit 1
fi
