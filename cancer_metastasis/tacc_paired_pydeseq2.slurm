#!/bin/bash
#SBATCH -J cot_pydeseq2
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/pydeseq2_%j.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/pydeseq2_%j.err
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
groups=${CONFIDENCEOT_DEG_ROOT:-$result/robust_target_malignant_deg_all_pairs_GSE180661}
output=${CONFIDENCEOT_PYDESEQ2_ROOT:-$result/robust_target_malignant_pydeseq2_GSE180661}

cd "$repo"
export PYTHONPATH="$repo/src:$repo/cancer_metastasis${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export PYTHONUNBUFFERED=1

mkdir -p "$output" "$result/logs"

# Amended 2026-09-24 from the prespecification's 20 to 10. Retention runs from
# 0.37 to 0.95 across pairs, so a high-retention patient has very few rejected
# cells -- 200 primary cells at 95% retention leaves 10 -- and a 20-cell floor
# drops exactly the patients the patient-dominance check needs to count. The
# floor is the only threshold that matters here: 21_ emits a contrast whenever
# both states are non-empty, so 13_ is where patients are actually excluded.
# See the 2026-09-24 amendment in DEG_PRESPECIFICATION_2026-09-15.md.
minimum_cells=${CONFIDENCEOT_MINIMUM_CELLS_PER_STATUS:-10}

python cancer_metastasis/13_run_paired_pydeseq2.py \
  "$groups" "$output" \
  --minimum-cells-per-patient-status "$minimum_cells" \
  --minimum-total-count 10 \
  --maximum-fdr 0.05 \
  --absolute-log2-fold-change-thresholds 0.5 1.0 \
  --n-cpus 16
