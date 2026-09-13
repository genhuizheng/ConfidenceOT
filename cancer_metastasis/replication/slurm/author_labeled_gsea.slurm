#!/bin/bash
#SBATCH -J cot_rep_gsea
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/rep_gsea_%j.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/rep_gsea_%j.err
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 04:00:00
#SBATCH -A MCB26031
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=END,FAIL

set -eo pipefail
source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"
set -u

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
root=${CONFIDENCEOT_REPLICATION_ROOT:?CONFIDENCEOT_REPLICATION_ROOT is required}
gmt_root=${CONFIDENCEOT_HUMAN_MSIGDB_ROOT:-$result/gene_sets/msigdb_2025.1_Hs}
gmt=${CONFIDENCEOT_HUMAN_GMT:-$gmt_root/human_hallmark_go_bp_reactome_wikipathways_v2025.1.Hs.symbols.gmt}
cd "$repo"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
export PYTHONUNBUFFERED=1

python cancer_metastasis/14_prepare_human_msigdb_gmt.py "$gmt_root"
for dataset in GSE181919 GSE225857; do
  deg="$root/$dataset/primary_deg"
  output="$root/$dataset/primary_gsea"
  if [[ ! -f "$deg/pydeseq2_report.json" ]]; then
    mkdir -p "$output"
    printf '{"status":"skipped","reason":"DEG_not_available"}\n' > "$output/GSEA_SKIPPED.json"
    continue
  fi
  python -m cancer_metastasis.gse180661.primary_gsea \
    "$deg" "$gmt" "$output" \
    --permutations 2000 \
    --threads 16
done

touch "$root/GSEA_COMPLETE"
