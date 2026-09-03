#!/bin/bash
#SBATCH -J cot_qcgsea
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/qcgsea_%j.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/qcgsea_%j.err
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

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
input=${CONFIDENCEOT_QC_MATCHED_ROOT:-$result/robust_target_malignant_qc_matched_validation_GSE180661}
gmt=${CONFIDENCEOT_HUMAN_GMT:-$result/gene_sets/msigdb_2025.1_Hs/human_hallmark_go_bp_reactome_wikipathways_v2025.1.Hs.symbols.gmt}
output=${CONFIDENCEOT_QC_MATCHED_GSEA_ROOT:-$result/robust_target_malignant_qc_matched_gsea_GSE180661}

cd "$repo"
export PYTHONPATH="$repo/src:$repo/cancer_metastasis${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
mkdir -p "$result/logs"

if [[ -e "$output" ]]; then
  echo "STOP: output already exists: $output" >&2
  exit 1
fi

python cancer_metastasis/17_run_qc_matched_gsea.py \
  "$input" "$gmt" "$output" \
  --permutations 2000 --threads 16 \
  --minimum-size 10 --maximum-size 500 \
  --minimum-stable-fraction 0.8
