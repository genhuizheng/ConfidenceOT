#!/bin/bash
#SBATCH -J cot_pair_umap
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/pair_umap_%A_%a.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/pair_umap_%A_%a.err
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
ot_root=${CONFIDENCEOT_MALIGNANT_OT_ROOT:-$result/malignant_pairing_GSE180661}
output=${CONFIDENCEOT_PAIRING_FIGURE_ROOT:-$result/malignant_pairing_umap_GSE180661}
cd "$repo"
export PYTHONPATH="$repo/src:$repo/cancer_metastasis${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export PYTHONUNBUFFERED=1
mkdir -p "$output" "$result/logs"

pair_count=${CONFIDENCEOT_PAIR_COUNT:-$(python - "$manifest" <<'PY'
import csv
import sys
with open(sys.argv[1], newline="", encoding="utf-8-sig") as handle:
    print(sum(1 for _ in csv.DictReader(handle)))
PY
)}
task_id=${SLURM_ARRAY_TASK_ID:-0}
task_count=${SLURM_ARRAY_TASK_COUNT:-1}
failures=0
for ((index=task_id; index<pair_count; index+=task_count)); do
  pair_id=$(python - "$manifest" "$index" <<'PY'
import csv
import sys
with open(sys.argv[1], newline="", encoding="utf-8-sig") as handle:
    rows = list(csv.DictReader(handle))
print(rows[int(sys.argv[2])]["pair_id"])
PY
)
  pair_dir=$(find "$ot_root/$pair_id/scope_malignant" -mindepth 1 -maxdepth 1 \
    -type d -exec test -f '{}/SUCCESS' \; -print | head -n 1)
  if [[ -z "$pair_dir" ]]; then
    echo "No completed malignant pairing result for $pair_id" >&2
    failures=$((failures + 1))
    continue
  fi
  destination="$output/$(printf '%03d' "$index")_${pair_id//[^A-Za-z0-9._-]/_}"
  echo "===== PAIR UMAP START index=$index pair=$pair_id $(date --iso-8601=seconds) ====="
  if python -m cancer_metastasis.gse180661.malignant_pairing \
      "$pair_dir" "$destination" --method M4-E --maximum-displayed-edges 3000; then
    echo "===== PAIR UMAP COMPLETE index=$index $(date --iso-8601=seconds) ====="
  else
    failures=$((failures + 1))
  fi
done
if (( failures > 0 )); then
  exit 1
fi
