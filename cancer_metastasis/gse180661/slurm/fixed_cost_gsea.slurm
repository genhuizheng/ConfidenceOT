#!/bin/bash
#SBATCH -J cot_fixed_c_gs
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/fixed_c_gs_%A_%a.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/fixed_c_gs_%A_%a.err
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 04:00:00
#SBATCH -A MCB26031

source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"
repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
deg_base=${CONFIDENCEOT_FIXED_COST_DEG_BASE:?CONFIDENCEOT_FIXED_COST_DEG_BASE is required}
gsea_base=${CONFIDENCEOT_FIXED_COST_GSEA_BASE:?CONFIDENCEOT_FIXED_COST_GSEA_BASE is required}
costs=${CONFIDENCEOT_FIXED_COSTS:-"0.4:0.5:0.6:0.7"}
gmt_root=${CONFIDENCEOT_HUMAN_MSIGDB_ROOT:-$result/gene_sets/msigdb_2025.1_Hs}
gmt=${CONFIDENCEOT_HUMAN_GMT:-$gmt_root/human_hallmark_go_bp_reactome_wikipathways_v2025.1.Hs.symbols.gmt}
cd "$repo"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
python cancer_metastasis/14_prepare_human_msigdb_gmt.py "$gmt_root"
read -r -a values <<< "${costs//[:,]/ }"
for ((i=SLURM_ARRAY_TASK_ID; i<${#values[@]}; i+=SLURM_ARRAY_TASK_COUNT)); do
  slug=$(printf "%.2f" "${values[$i]}"); slug=${slug/./p}
  python -m cancer_metastasis.gse180661.primary_gsea \
    "$deg_base/rejection_cost_$slug" "$gmt" "$gsea_base/rejection_cost_$slug" \
    --permutations 2000 --threads 16
done
