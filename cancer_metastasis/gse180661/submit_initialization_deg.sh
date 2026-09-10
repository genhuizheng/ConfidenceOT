#!/bin/bash
set -euo pipefail

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
manifest=${CONFIDENCEOT_MANIFEST:-$result/manifest/GSE180661_malignant/pair_manifest_malignant_eligible.csv}
baseline=${CONFIDENCEOT_BASELINE_OT_ROOT:?Set CONFIDENCEOT_BASELINE_OT_ROOT}
initialization=${CONFIDENCEOT_INIT_ROOT:-$result/gse180661_m4e_initialization_sensitivity_with_gates_20260910}
summary=${CONFIDENCEOT_INIT_SUMMARY_ROOT:-$result/gse180661_m4e_initialization_sensitivity_with_gates_summary_20260910}
pseudobulk=${CONFIDENCEOT_INIT_PSEUDOBULK_ROOT:-$result/gse180661_m4e_initialization_pseudobulk_20260910}
deg=${CONFIDENCEOT_INIT_DEG_ROOT:-$result/gse180661_m4e_initialization_deg_20260910}
random_starts=${CONFIDENCEOT_RANDOM_STARTS:-5}

for path in "$initialization" "$summary" "$pseudobulk" "$deg"; do
  if [[ -e "$path" ]]; then
    echo "STOP: output already exists: $path" >&2
    exit 1
  fi
done

pair_count=$(python - "$manifest" <<'PY'
import csv
import sys
with open(sys.argv[1], newline="", encoding="utf-8-sig") as handle:
    print(sum(1 for _ in csv.DictReader(handle)))
PY
)

submit_id() {
  local response job_id
  response=$(sbatch --parsable "$@")
  printf "%s\n" "$response" >&2
  job_id=$(printf "%s\n" "$response" | sed -nE \
    's/^[[:space:]]*([0-9]+)(;.*)?[[:space:]]*$/\1/p' | tail -n 1)
  [[ -n "$job_id" ]] || { echo "Unable to parse Slurm job ID" >&2; return 1; }
  printf "%s\n" "$job_id"
}

common="ALL,CONFIDENCEOT_REPO=$repo,CONFIDENCEOT_MANIFEST=$manifest,CONFIDENCEOT_BASELINE_OT_ROOT=$baseline,CONFIDENCEOT_INIT_ROOT=$initialization,CONFIDENCEOT_INIT_SUMMARY_ROOT=$summary,CONFIDENCEOT_INIT_PSEUDOBULK_ROOT=$pseudobulk,CONFIDENCEOT_INIT_DEG_ROOT=$deg,CONFIDENCEOT_PAIR_COUNT=$pair_count,CONFIDENCEOT_RANDOM_STARTS=$random_starts"

init_job=$(submit_id --array=0-0 --export="$common" \
  cancer_metastasis/gse180661/slurm/initialization_sensitivity.slurm)
summary_job=$(submit_id --dependency="afterok:$init_job" --export="$common" \
  cancer_metastasis/gse180661/slurm/summarize_initialization_sensitivity.slurm)
pseudobulk_job=$(submit_id --array=0-3%4 --dependency="afterok:$init_job" --export="$common" \
  cancer_metastasis/gse180661/slurm/initialization_pseudobulk.slurm)
deg_job=$(submit_id --dependency="afterok:$pseudobulk_job" --export="$common" \
  cancer_metastasis/gse180661/slurm/initialization_deg.slurm)

printf "Initialization job: %s\n" "$init_job"
printf "Summary job:        %s\n" "$summary_job"
printf "Pseudobulk job:     %s\n" "$pseudobulk_job"
printf "PyDESeq2 job:       %s\n" "$deg_job"
squeue -j "$init_job,$summary_job,$pseudobulk_job,$deg_job"
