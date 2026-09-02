#!/bin/bash
#SBATCH -J cot_gsea
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/gsea_%j.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/gsea_%j.err
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
pydeseq2=${CONFIDENCEOT_PYDESEQ2_ROOT:-$result/robust_target_malignant_pydeseq2_GSE180661}
gmt_root=${CONFIDENCEOT_HUMAN_MSIGDB_ROOT:-$result/gene_sets/msigdb_2025.1_Hs}
gmt=${CONFIDENCEOT_HUMAN_GMT:-$gmt_root/human_hallmark_go_bp_reactome_wikipathways_v2025.1.Hs.symbols.gmt}
output=${CONFIDENCEOT_GSEA_ROOT:-$result/robust_target_malignant_pydeseq2_gsea_GSE180661}

cd "$repo"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
export PYTHONUNBUFFERED=1
mkdir -p "$result/logs" "$output"

python cancer_metastasis/14_prepare_human_msigdb_gmt.py "$gmt_root"
python cancer_metastasis/15_run_pydeseq2_gseapy.py \
  "$pydeseq2" "$gmt" "$output" \
  --permutations 2000 --threads 16 --minimum-size 10 --maximum-size 500
