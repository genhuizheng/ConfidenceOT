#!/bin/bash
set -euo pipefail

result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
manifest=${CONFIDENCEOT_MANIFEST:-$result/manifest/GSE180661_malignant/pair_manifest_malignant_eligible.csv}
source_caps=${CONFIDENCEOT_SOURCE_CAPS:-"0.50:0.70:0.75:0.85:0.90:0.95"}
target_cap=${CONFIDENCEOT_TARGET_REJECTION_BUDGET:-0.95}
resume=${CONFIDENCEOT_RESUME:-0}
ot_base=${CONFIDENCEOT_SOURCE_CAP_OT_BASE:-$result/gse180661_source_cap_sensitivity_ot_20260908}
pseudobulk_base=${CONFIDENCEOT_SOURCE_CAP_PSEUDOBULK_BASE:-$result/gse180661_source_cap_sensitivity_pseudobulk_20260908}
deg_base=${CONFIDENCEOT_SOURCE_CAP_DEG_BASE:-$result/gse180661_source_cap_sensitivity_pydeseq2_20260908}
summary_root=${CONFIDENCEOT_SOURCE_CAP_SUMMARY_ROOT:-$result/gse180661_source_cap_sensitivity_summary_20260908}

for path in "$ot_base" "$pseudobulk_base" "$deg_base" "$summary_root"; do
  if [[ -e "$path" && "$resume" != "1" ]]; then
    echo "STOP: output already exists: $path" >&2
    exit 1
  fi
done
if [[ ! -f "$manifest" ]]; then
  echo "STOP: manifest does not exist: $manifest" >&2
  exit 1
fi

pair_count=$(python - "$manifest" <<'PY'
import csv
import sys
with open(sys.argv[1], newline="", encoding="utf-8-sig") as handle:
    print(sum(1 for _ in csv.DictReader(handle)))
PY
)

submit_id() {
  local output job_id
  output=$(sbatch --parsable "$@")
  printf "%s\n" "$output" >&2
  job_id=$(printf "%s\n" "$output" | sed -nE \
    's/^[[:space:]]*([0-9]+)(;.*)?[[:space:]]*$/\1/p' | tail -n 1)
  if [[ -z "$job_id" ]]; then
    echo "Unable to parse Slurm job ID" >&2
    return 1
  fi
  printf "%s\n" "$job_id"
}

common_export="ALL,CONFIDENCEOT_MANIFEST=$manifest,CONFIDENCEOT_PAIR_COUNT=$pair_count,CONFIDENCEOT_SOURCE_CAPS=$source_caps,CONFIDENCEOT_TARGET_REJECTION_BUDGET=$target_cap,CONFIDENCEOT_SOURCE_CAP_OT_BASE=$ot_base,CONFIDENCEOT_SOURCE_CAP_PSEUDOBULK_BASE=$pseudobulk_base,CONFIDENCEOT_SOURCE_CAP_DEG_BASE=$deg_base,CONFIDENCEOT_SOURCE_CAP_SUMMARY_ROOT=$summary_root"

ot_job=$(submit_id \
  --array=0-3%4 \
  --export="$common_export" \
  cancer_metastasis/gse180661/slurm/source_cap_ot_sensitivity.slurm)

pseudobulk_job=$(submit_id \
  --array=0-3%4 \
  --dependency="afterok:$ot_job" \
  --export="$common_export" \
  cancer_metastasis/gse180661/slurm/source_cap_pseudobulk_sensitivity.slurm)

deg_job=$(submit_id \
  --array=0-3%4 \
  --dependency="afterok:$pseudobulk_job" \
  --export="$common_export" \
  cancer_metastasis/gse180661/slurm/source_cap_deg_sensitivity.slurm)

summary_job=$(submit_id \
  --dependency="afterok:$deg_job" \
  --export="$common_export" \
  cancer_metastasis/gse180661/slurm/source_cap_deg_summary.slurm)

printf "OT job:          %s\n" "$ot_job"
printf "Pseudobulk job:  %s\n" "$pseudobulk_job"
printf "PyDESeq2 job:    %s\n" "$deg_job"
printf "Summary job:     %s\n" "$summary_job"
printf "Pair count:      %s\n" "$pair_count"
printf "Source caps:     %s\n" "$source_caps"
printf "Target cap:      %s\n" "$target_cap"
squeue -j "$ot_job,$pseudobulk_job,$deg_job,$summary_job"
