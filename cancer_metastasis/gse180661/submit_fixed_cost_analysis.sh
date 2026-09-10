#!/bin/bash
set -euo pipefail
result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
manifest=${CONFIDENCEOT_MANIFEST:-$result/manifest/GSE180661_malignant/pair_manifest_malignant_eligible.csv}
costs=${CONFIDENCEOT_FIXED_COSTS:-"0.4:0.5:0.6:0.7"}
resume=${CONFIDENCEOT_RESUME:-0}
ot=${CONFIDENCEOT_FIXED_COST_OT_BASE:-$result/gse180661_fixed_cost_ot_c0p4_0p5_0p6_0p7_postqc_20260910}
pb=${CONFIDENCEOT_FIXED_COST_PB_BASE:-$result/gse180661_fixed_cost_pseudobulk_c0p4_0p5_0p6_0p7_postqc_20260910}
deg=${CONFIDENCEOT_FIXED_COST_DEG_BASE:-$result/gse180661_fixed_cost_pydeseq2_c0p4_0p5_0p6_0p7_postqc_20260910}
gsea=${CONFIDENCEOT_FIXED_COST_GSEA_BASE:-$result/gse180661_fixed_cost_gsea_c0p4_0p5_0p6_0p7_postqc_20260910}
for path in "$ot" "$pb" "$deg" "$gsea"; do
  [[ ! -e "$path" || "$resume" == "1" ]] || {
    echo "STOP: output already exists: $path" >&2
    echo "Set CONFIDENCEOT_RESUME=1 to resume this exact analysis." >&2
    exit 1
  }
done
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
common="ALL,CONFIDENCEOT_MANIFEST=$manifest,CONFIDENCEOT_PAIR_COUNT=$pair_count,CONFIDENCEOT_FIXED_COSTS=$costs,CONFIDENCEOT_FIXED_COST_OT_BASE=$ot,CONFIDENCEOT_FIXED_COST_PB_BASE=$pb,CONFIDENCEOT_FIXED_COST_DEG_BASE=$deg,CONFIDENCEOT_FIXED_COST_GSEA_BASE=$gsea"
ot_job=$(submit_id --array=0-3%4 --export="$common" cancer_metastasis/gse180661/slurm/fixed_cost_ot.slurm)
pb_job=$(submit_id --array=0-3%4 --dependency="afterok:$ot_job" --export="$common" cancer_metastasis/gse180661/slurm/fixed_cost_pseudobulk.slurm)
deg_job=$(submit_id --array=0-3%4 --dependency="afterok:$pb_job" --export="$common" cancer_metastasis/gse180661/slurm/fixed_cost_deg.slurm)
gsea_job=$(submit_id --array=0-3%4 --dependency="afterok:$deg_job" --export="$common" cancer_metastasis/gse180661/slurm/fixed_cost_gsea.slurm)
printf "Fixed-cost OT: %s\nPseudobulk: %s\nPyDESeq2: %s\nGSEA: %s\nPairs: %s\nCosts: %s\n" \
  "$ot_job" "$pb_job" "$deg_job" "$gsea_job" "$pair_count" "$costs"
