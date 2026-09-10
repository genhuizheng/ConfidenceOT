#!/bin/bash
set -euo pipefail
repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
source_manifest=${CONFIDENCEOT_SOURCE_MANIFEST:-$result/manifest/GSE180661_malignant/pair_manifest_malignant_eligible.csv}
prior=${CONFIDENCEOT_PRIOR_OT_ROOT:-$result/gse180661_sourcecap_0p85_postqc_ot_20260908/source_cap_0p85}
resume=${CONFIDENCEOT_RESUME:-0}
manifest_root=${CONFIDENCEOT_PRIOR_REJECTED_MANIFEST_ROOT:-$result/manifest/GSE180661_prior_rejected_postqc_20260910}
manifest=$manifest_root/prior_rejected_eligible.csv
ot=${CONFIDENCEOT_MALIGNANT_OT_ROOT:-$result/gse180661_prior_rejected_second_ot_postqc_20260910}
pb=${CONFIDENCEOT_PRIMARY_PSEUDOBULK_ROOT:-$result/gse180661_prior_rejected_second_pseudobulk_postqc_20260910}
deg=${CONFIDENCEOT_PRIMARY_DEG_ROOT:-$result/gse180661_prior_rejected_second_pydeseq2_postqc_20260910}
gsea=${CONFIDENCEOT_PRIMARY_GSEA_ROOT:-$result/gse180661_prior_rejected_second_gsea_postqc_20260910}
for path in "$manifest_root" "$ot" "$pb" "$deg" "$gsea"; do
  [[ ! -e "$path" || "$resume" == "1" ]] || {
    echo "STOP: output already exists: $path" >&2
    echo "Set CONFIDENCEOT_RESUME=1 to resume this exact analysis." >&2
    exit 1
  }
done
cd "$repo"
export PYTHONPATH="$repo/src:$repo/cancer_metastasis${PYTHONPATH:+:$PYTHONPATH}"
if [[ ! -f "$manifest" ]]; then
  python -m cancer_metastasis.gse180661.independent_manifests "$source_manifest" "$manifest_root" \
    --mode prior-rejected --prior-root "$prior" \
    --budget-tag budget_source_0.85_target_0.95 --method M4-E --minimum-cells 20
fi
pair_count=$(python - "$manifest" <<'PY'
import csv, sys
with open(sys.argv[1], newline="", encoding="utf-8-sig") as handle:
    print(sum(1 for _ in csv.DictReader(handle)))
PY
)
submit_id() {
  local output id
  output=$(sbatch --parsable "$@"); printf "%s\n" "$output" >&2
  id=$(printf "%s\n" "$output" | sed -nE 's/^[[:space:]]*([0-9]+)(;.*)?[[:space:]]*$/\1/p' | tail -n 1)
  [[ -n "$id" ]] || { echo "Unable to parse Slurm job ID" >&2; return 1; }
  printf "%s\n" "$id"
}
common="ALL,CONFIDENCEOT_MANIFEST=$manifest,CONFIDENCEOT_PAIR_COUNT=$pair_count,CONFIDENCEOT_MALIGNANT_OT_ROOT=$ot,CONFIDENCEOT_PRIMARY_PSEUDOBULK_ROOT=$pb,CONFIDENCEOT_PRIMARY_DEG_ROOT=$deg,CONFIDENCEOT_PRIMARY_GSEA_ROOT=$gsea,CONFIDENCEOT_BUDGET_TAG=budget_source_0.85_target_0.95"
ot_job=$(submit_id --array=0-3%4 --export="$common,CONFIDENCEOT_INPUT_GATE_ROOT=$prior,CONFIDENCEOT_INPUT_GATE_BUDGET_TAG=budget_source_0.85_target_0.95,CONFIDENCEOT_INPUT_GATE_METHOD=M4-E,CONFIDENCEOT_INPUT_GATE_STATE=rejected" cancer_metastasis/gse180661/slurm/standard_malignant_ot.slurm)
pb_job=$(submit_id --array=0-3%4 --dependency="afterok:$ot_job" --export="$common" cancer_metastasis/gse180661/slurm/primary_pseudobulk.slurm)
deg_job=$(submit_id --dependency="afterok:$pb_job" --export="$common" cancer_metastasis/gse180661/slurm/primary_deg.slurm)
gsea_job=$(submit_id --dependency="afterok:$deg_job" --export="$common" cancer_metastasis/gse180661/slurm/primary_gsea.slurm)
printf "Prior-rejected manifest pairs: %s\nSecond OT: %s\nPseudobulk: %s\nPyDESeq2: %s\nGSEA: %s\n" \
  "$pair_count" "$ot_job" "$pb_job" "$deg_job" "$gsea_job"
