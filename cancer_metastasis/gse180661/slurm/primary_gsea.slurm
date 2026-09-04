#!/bin/bash
#SBATCH -J cot_primary_gsea
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/primary_gsea_%j.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/primary_gsea_%j.err
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 04:00:00
#SBATCH -A MCB26031
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=END,FAIL

source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"
repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
deg=${CONFIDENCEOT_PRIMARY_DEG_ROOT:-$result/pairwise_primary_pydeseq2_GSE180661}
gmt_root=${CONFIDENCEOT_HUMAN_MSIGDB_ROOT:-$result/gene_sets/msigdb_2025.1_Hs}
gmt=${CONFIDENCEOT_HUMAN_GMT:-$gmt_root/human_hallmark_go_bp_reactome_wikipathways_v2025.1.Hs.symbols.gmt}
output=${CONFIDENCEOT_PRIMARY_GSEA_ROOT:-$result/pairwise_primary_gsea_GSE180661}
cd "$repo"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
export PYTHONUNBUFFERED=1
mkdir -p "$output" "$result/logs"
python cancer_metastasis/14_prepare_human_msigdb_gmt.py "$gmt_root"
python -m cancer_metastasis.gse180661.primary_gsea \
  "$deg" "$gmt" "$output" --permutations 2000 --threads 16
