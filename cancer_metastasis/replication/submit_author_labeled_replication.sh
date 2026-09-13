#!/bin/bash
set -euo pipefail

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
root=${CONFIDENCEOT_REPLICATION_ROOT:-$result/author_labeled_replication_GSE181919_GSE225857_20260912}
logs="$result/logs"

cd "$repo"
if [[ -e "$root" ]]; then
  echo "STOP: output already exists: $root" >&2
  exit 1
fi
mkdir -p "$logs"

common="ALL,CONFIDENCEOT_REPO=$repo,CANCER_COT_ROOT=$result,CONFIDENCEOT_REPLICATION_ROOT=$root"

job_id() {
  grep -E '^[0-9]+' | tail -n 1
}

audit_job=$(sbatch --parsable \
  --export="$common" \
  cancer_metastasis/replication/slurm/author_labeled_audit.slurm | job_id)

ot_job=$(sbatch --parsable \
  --dependency="afterok:${audit_job}" \
  --export="$common" \
  cancer_metastasis/replication/slurm/author_labeled_ot.slurm | job_id)

pseudobulk_job=$(sbatch --parsable \
  --dependency="afterok:${ot_job}" \
  --export="$common" \
  cancer_metastasis/replication/slurm/author_labeled_pseudobulk.slurm | job_id)

deg_job=$(sbatch --parsable \
  --dependency="afterok:${pseudobulk_job}" \
  --export="$common" \
  cancer_metastasis/replication/slurm/author_labeled_deg.slurm | job_id)

gsea_job=$(sbatch --parsable \
  --dependency="afterok:${deg_job}" \
  --export="$common" \
  cancer_metastasis/replication/slurm/author_labeled_gsea.slurm | job_id)

summary_job=$(sbatch --parsable \
  --dependency="afterok:${gsea_job}" \
  --export="$common" \
  cancer_metastasis/replication/slurm/author_labeled_summary.slurm | job_id)

printf 'Replication root: %s\n' "$root"
printf 'Audit job:       %s\n' "$audit_job"
printf 'Malignant OT:    %s\n' "$ot_job"
printf 'Pseudobulk:      %s\n' "$pseudobulk_job"
printf 'PyDESeq2:        %s\n' "$deg_job"
printf 'GSEA:            %s\n' "$gsea_job"
printf 'Cross-dataset:   %s\n' "$summary_job"
squeue -j "${audit_job},${ot_job},${pseudobulk_job},${deg_job},${gsea_job},${summary_job}"
