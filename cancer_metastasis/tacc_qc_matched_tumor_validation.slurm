#!/bin/bash
#SBATCH -J cot_qcmatch
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/qcmatch_%j.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/qcmatch_%j.err
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 06:00:00
#SBATCH -A MCB26031
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=END,FAIL

source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
manifest=${CONFIDENCEOT_MANIFEST:-$result/manifest/GSE180661_malignant/pair_manifest_malignant_eligible.csv}
groups=${CONFIDENCEOT_DEG_ROOT:-$result/robust_target_malignant_deg_three_contrasts_GSE180661}
original=${CONFIDENCEOT_PYDESEQ2_ROOT:-$result/robust_target_malignant_pydeseq2_three_contrasts_lfc_0p5_1p0_GSE180661}
output=${CONFIDENCEOT_QC_MATCHED_ROOT:-$result/robust_target_malignant_qc_matched_validation_GSE180661}
replicates=${CONFIDENCEOT_QC_MATCHED_REPLICATES:-5}
maximum_groups=${CONFIDENCEOT_QC_MATCHED_MAX_GROUPS:-}

cd "$repo"
export PYTHONPATH="$repo/src:$repo/cancer_metastasis${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
mkdir -p "$result/logs" "$output"

extra_args=()
if [[ -n "$maximum_groups" ]]; then
  extra_args+=(--maximum-groups "$maximum_groups")
fi

python cancer_metastasis/16_run_qc_matched_tumor_validation.py \
  "$manifest" "$groups" "$original" "$output" \
  --replicates "$replicates" --caliper-scale 0.2 \
  --minimum-matched-pairs-per-group 10 \
  --minimum-cells-per-patient-status 20 --n-cpus 16 \
  "${extra_args[@]}"
