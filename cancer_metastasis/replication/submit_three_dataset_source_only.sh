#!/bin/bash
set -euo pipefail

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
root=${CONFIDENCEOT_THREE_DATASET_ROOT:-$result/source_only_GSE180661_GSE181919_GSE225857_20260913}
ov_manifest=${CONFIDENCEOT_GSE180661_MANIFEST:-$result/manifest/GSE180661_malignant/pair_manifest_malignant_eligible.csv}
logs="$result/logs"
source_budget=${CONFIDENCEOT_SOURCE_REJECTION_BUDGET:-0.85}
target_budget=${CONFIDENCEOT_TARGET_REJECTION_BUDGET:-0.00}
budget_tag=${CONFIDENCEOT_BUDGET_TAG:-budget_source_0.85_target_0.00}

cd "$repo"
if [[ -e "$root" ]]; then
  echo "STOP: output already exists: $root" >&2
  exit 1
fi
if [[ ! -f "$ov_manifest" ]]; then
  echo "STOP: GSE180661 manifest is missing: $ov_manifest" >&2
  exit 1
fi
mkdir -p "$logs"

job_id() {
  grep -E '^[0-9]+' | tail -n 1
}

ov_pair_count=$(python - "$ov_manifest" <<'PY'
import csv
import sys
with open(sys.argv[1], newline="", encoding="utf-8-sig") as handle:
    print(sum(1 for _ in csv.DictReader(handle)))
PY
)

rep_common="ALL,CONFIDENCEOT_REPO=$repo,CANCER_COT_ROOT=$result,CONFIDENCEOT_REPLICATION_ROOT=$root,CONFIDENCEOT_SOURCE_REJECTION_BUDGET=$source_budget,CONFIDENCEOT_TARGET_REJECTION_BUDGET=$target_budget,CONFIDENCEOT_BUDGET_TAG=$budget_tag"
ov_common="ALL,CONFIDENCEOT_REPO=$repo,CANCER_COT_ROOT=$result,CONFIDENCEOT_MANIFEST=$ov_manifest,CONFIDENCEOT_OUTPUT_ROOT=$root/GSE180661/ot,CONFIDENCEOT_ANALYSIS_SCOPE=malignant,CONFIDENCEOT_INCLUDE_ANNOTATIONS=Ovarian.cancer.cell,CONFIDENCEOT_SOURCE_REJECTION_BUDGET=$source_budget,CONFIDENCEOT_TARGET_REJECTION_BUDGET=$target_budget,CONFIDENCEOT_PAIR_COUNT=$ov_pair_count,CONFIDENCEOT_MAX_OBSERVED_CELLS_PER_SIDE=10000,CONFIDENCEOT_SAVE_PAIRING_EDGES=1,CONFIDENCEOT_CELL_QC=1"

audit_job=$(sbatch --parsable \
  --export="$rep_common" \
  cancer_metastasis/replication/slurm/author_labeled_audit.slurm | job_id)

ov_ot_job=$(sbatch --parsable \
  -p gh-dev -t 02:00:00 --array=0-0 \
  --export="$ov_common" \
  cancer_metastasis/tacc_full_array.slurm | job_id)

gse181919_ot_job=$(sbatch --parsable \
  --dependency="afterok:${audit_job}" \
  --export="$rep_common,CONFIDENCEOT_REPLICATION_DATASET=GSE181919" \
  cancer_metastasis/replication/slurm/author_labeled_ot.slurm | job_id)

gse225857_ot_job=$(sbatch --parsable \
  --dependency="afterok:${audit_job}" \
  --export="$rep_common,CONFIDENCEOT_REPLICATION_DATASET=GSE225857" \
  cancer_metastasis/replication/slurm/author_labeled_ot.slurm | job_id)

ov_pb_job=$(sbatch --parsable \
  --dependency="afterok:${ov_ot_job}" --array=0-3%4 \
  --export="ALL,CONFIDENCEOT_REPO=$repo,CANCER_COT_ROOT=$result,CONFIDENCEOT_MANIFEST=$ov_manifest,CONFIDENCEOT_MALIGNANT_OT_ROOT=$root/GSE180661/ot,CONFIDENCEOT_PRIMARY_PSEUDOBULK_ROOT=$root/GSE180661/primary_pseudobulk,CONFIDENCEOT_PAIR_COUNT=$ov_pair_count,CONFIDENCEOT_BUDGET_TAG=$budget_tag,CONFIDENCEOT_GATE_STATE_LABELS=1" \
  cancer_metastasis/gse180661/slurm/primary_pseudobulk.slurm | job_id)

ov_umap_job=$(sbatch --parsable \
  --dependency="afterok:${ov_ot_job}" --array=0-3%4 \
  --export="ALL,CONFIDENCEOT_REPO=$repo,CANCER_COT_ROOT=$result,CONFIDENCEOT_MANIFEST=$ov_manifest,CONFIDENCEOT_MALIGNANT_OT_ROOT=$root/GSE180661/ot,CONFIDENCEOT_PAIRING_FIGURE_ROOT=$root/GSE180661/pair_umap,CONFIDENCEOT_PAIR_COUNT=$ov_pair_count" \
  cancer_metastasis/gse180661/slurm/malignant_pairing.slurm | job_id)

ov_deg_job=$(sbatch --parsable \
  --dependency="afterok:${ov_pb_job}" \
  --export="ALL,CONFIDENCEOT_REPO=$repo,CANCER_COT_ROOT=$result,CONFIDENCEOT_PRIMARY_PSEUDOBULK_ROOT=$root/GSE180661/primary_pseudobulk,CONFIDENCEOT_PRIMARY_DEG_ROOT=$root/GSE180661/primary_deg,CONFIDENCEOT_GATE_STATE_LABELS=1" \
  cancer_metastasis/gse180661/slurm/primary_deg.slurm | job_id)

ov_gsea_job=$(sbatch --parsable \
  --dependency="afterok:${ov_deg_job}" \
  --export="ALL,CONFIDENCEOT_REPO=$repo,CANCER_COT_ROOT=$result,CONFIDENCEOT_PRIMARY_DEG_ROOT=$root/GSE180661/primary_deg,CONFIDENCEOT_PRIMARY_GSEA_ROOT=$root/GSE180661/primary_gsea" \
  cancer_metastasis/gse180661/slurm/primary_gsea.slurm | job_id)

rep_pb_job=$(sbatch --parsable \
  --dependency="afterok:${gse181919_ot_job}:${gse225857_ot_job}" \
  --export="$rep_common" \
  cancer_metastasis/replication/slurm/author_labeled_pseudobulk.slurm | job_id)

rep_umap_job=$(sbatch --parsable \
  -p gg --dependency="afterok:${gse181919_ot_job}:${gse225857_ot_job}" \
  --export="$rep_common,CONFIDENCEOT_REPLICATION_UMAP_ROOT=$root/replication_umap" \
  cancer_metastasis/replication/slurm/primary_metastasis_umap.slurm | job_id)

rep_deg_job=$(sbatch --parsable \
  --dependency="afterok:${rep_pb_job}" \
  --export="$rep_common" \
  cancer_metastasis/replication/slurm/author_labeled_deg.slurm | job_id)

rep_gsea_job=$(sbatch --parsable \
  --dependency="afterok:${rep_deg_job}" \
  --export="$rep_common" \
  cancer_metastasis/replication/slurm/author_labeled_gsea.slurm | job_id)

rep_summary_job=$(sbatch --parsable \
  --dependency="afterok:${rep_gsea_job}" \
  --export="$rep_common" \
  cancer_metastasis/replication/slurm/author_labeled_summary.slurm | job_id)

printf 'Three-dataset root: %s\n' "$root"
printf 'Source cap: %s; target cap: %s\n' "$source_budget" "$target_budget"
printf 'GSE180661 OT:       %s\n' "$ov_ot_job"
printf 'GSE180661 PB:       %s\n' "$ov_pb_job"
printf 'GSE180661 UMAP:     %s\n' "$ov_umap_job"
printf 'GSE180661 DEG:      %s\n' "$ov_deg_job"
printf 'GSE180661 GSEA:     %s\n' "$ov_gsea_job"
printf 'Replication audit: %s\n' "$audit_job"
printf 'GSE181919 OT:       %s\n' "$gse181919_ot_job"
printf 'GSE225857 OT:       %s\n' "$gse225857_ot_job"
printf 'Replication PB:    %s\n' "$rep_pb_job"
printf 'Replication UMAP:  %s\n' "$rep_umap_job"
printf 'Replication DEG:   %s\n' "$rep_deg_job"
printf 'Replication GSEA:  %s\n' "$rep_gsea_job"
printf 'Replication sum:   %s\n' "$rep_summary_job"
squeue -j "${ov_ot_job},${ov_pb_job},${ov_umap_job},${ov_deg_job},${ov_gsea_job},${audit_job},${gse181919_ot_job},${gse225857_ot_job},${rep_pb_job},${rep_umap_job},${rep_deg_job},${rep_gsea_job},${rep_summary_job}"
