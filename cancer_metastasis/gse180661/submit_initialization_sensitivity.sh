#!/bin/bash
set -euo pipefail

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
manifest=${CONFIDENCEOT_MANIFEST:-$result/manifest/GSE180661_malignant/pair_manifest_malignant_eligible.csv}
baseline=${CONFIDENCEOT_BASELINE_OT_ROOT:?Set CONFIDENCEOT_BASELINE_OT_ROOT to the completed post-QC source-cap 0.85 OT directory}
output=${CONFIDENCEOT_INIT_ROOT:-$result/gse180661_m4e_initialization_sensitivity_sourcecap_0p85_postqc_20260908}
summary=${CONFIDENCEOT_INIT_SUMMARY_ROOT:-$result/gse180661_m4e_initialization_sensitivity_sourcecap_0p85_postqc_summary_20260908}
random_starts=${CONFIDENCEOT_RANDOM_STARTS:-5}

if [[ ! -f "$manifest" ]]; then
  echo "STOP: manifest is missing: $manifest" >&2
  exit 1
fi
if [[ ! -d "$baseline" ]]; then
  echo "STOP: baseline OT root is missing: $baseline" >&2
  exit 1
fi
for path in "$output" "$summary"; do
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
  if [[ -z "$job_id" ]]; then
    echo "Unable to parse Slurm job ID" >&2
    return 1
  fi
  printf "%s\n" "$job_id"
}

common="ALL,CONFIDENCEOT_REPO=$repo,CONFIDENCEOT_MANIFEST=$manifest,CONFIDENCEOT_BASELINE_OT_ROOT=$baseline,CONFIDENCEOT_INIT_ROOT=$output,CONFIDENCEOT_INIT_SUMMARY_ROOT=$summary,CONFIDENCEOT_PAIR_COUNT=$pair_count,CONFIDENCEOT_RANDOM_STARTS=$random_starts"

array_job=$(submit_id \
  --array=0-3%4 \
  --export="$common" \
  cancer_metastasis/gse180661/slurm/initialization_sensitivity.slurm)

summary_job=$(submit_id \
  --dependency="afterok:$array_job" \
  --export="$common" \
  cancer_metastasis/gse180661/slurm/summarize_initialization_sensitivity.slurm)

printf "Initialization array job: %s\n" "$array_job"
printf "Summary job:              %s\n" "$summary_job"
printf "Pairs:                    %s\n" "$pair_count"
printf "Random starts/strategy:   %s\n" "$random_starts"
squeue -j "$array_job,$summary_job"
