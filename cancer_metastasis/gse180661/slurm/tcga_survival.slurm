#!/bin/bash
#SBATCH -J cot_tcga_ov
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/tcga_ov_survival_%j.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/tcga_ov_survival_%j.err
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 02:00:00
#SBATCH -A MCB26031
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=END,FAIL

source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"
set -euo pipefail

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
data=${CONFIDENCEOT_TCGA_OV_DATA:-$result/tcga_ov_public_data}
deg=${CONFIDENCEOT_PRIMARY_DEG_ROOT:-$result/pairwise_primary_pydeseq2_GSE180661}
gsea=${CONFIDENCEOT_PRIMARY_GSEA_ROOT:-$result/pairwise_primary_gsea_GSE180661}
gmt_root=${CONFIDENCEOT_HUMAN_MSIGDB_ROOT:-$result/gene_sets/msigdb_2025.1_Hs}
output=${CONFIDENCEOT_TCGA_OV_SURVIVAL_ROOT:-$result/tcga_ov_survival_GSE180661}
signature_lfc=${CONFIDENCEOT_SIGNATURE_LOG2FC:-0.5}

expression=$data/TCGA-OV.star_tpm.tsv.gz
survival=$data/Survival_SupplementalTable_S1_20171025_xena_sp
gene_map=$data/gencode.v36.annotation.gtf.gene.probemap
deg_table=$deg/primary_compatible_vs_restricted_all_genes.csv.gz
gsea_table=$gsea/primary_compatible_vs_restricted_gsea_all.csv.gz
hallmark_gmt=$gmt_root/mh.all.v2025.1.Hs.symbols.gmt

mkdir -p "$data" "$output" "$result/logs"

download_if_missing() {
  local url=$1
  local destination=$2
  if [[ ! -s "$destination" ]]; then
    curl --fail --location --retry 5 --retry-delay 5 \
      --output "${destination}.part" "$url"
    mv "${destination}.part" "$destination"
  fi
}

download_if_missing \
  https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-OV.star_tpm.tsv.gz \
  "$expression"
download_if_missing \
  https://gdc-hub.s3.us-east-1.amazonaws.com/download/gencode.v36.annotation.gtf.gene.probemap \
  "$gene_map"
download_if_missing \
  https://pancanatlas.xenahubs.net/download/Survival_SupplementalTable_S1_20171025_xena_sp \
  "$survival"

cd "$repo"
python cancer_metastasis/14_prepare_human_msigdb_gmt.py "$gmt_root"

for required in "$deg_table" "$gsea_table" "$hallmark_gmt"; do
  if [[ ! -s "$required" ]]; then
    echo "Missing required upstream result: $required" >&2
    exit 1
  fi
done

export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export PYTHONUNBUFFERED=1

python -m cancer_metastasis.gse180661.tcga_survival \
  "$expression" \
  "$survival" \
  "$deg_table" \
  "$gsea_table" \
  "$hallmark_gmt" \
  "$output" \
  --gene-column Ensembl_ID \
  --gene-map-table "$gene_map" \
  --expression-scale log2_tpm_plus_one \
  --patient-column _PATIENT \
  --os-time-column OS.time \
  --os-event-column OS \
  --pfi-time-column PFI.time \
  --pfi-event-column PFI \
  --covariates age_at_initial_pathologic_diagnosis \
  --signature-log2fc "$signature_lfc" \
  --signature-fdr 0.05 \
  --minimum-detected-patient-fraction 0.25 \
  --minimum-patient-direction-consistency 0.70 \
  --threads 16

touch "$output/SUCCESS"
