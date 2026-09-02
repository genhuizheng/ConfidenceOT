#!/bin/bash
#SBATCH -J cot_source_caps
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/source_caps_%A_%a.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/source_caps_%A_%a.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 04:00:00
#SBATCH -A MCB26031
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=END,FAIL

# Four array workers each process their strided subset at source caps 0.90 and
# 0.85. The target cap remains fixed at 0.95 for both runs.

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
manifest=${CONFIDENCEOT_MANIFEST:-$result/manifest/GSE180661_malignant/pair_manifest_malignant_eligible.csv}
pair_count=${CONFIDENCEOT_PAIR_COUNT:?Set CONFIDENCEOT_PAIR_COUNT to the malignant manifest row count}
target_cap=${CONFIDENCEOT_TARGET_REJECTION_BUDGET:-0.95}
source_caps=${CONFIDENCEOT_SOURCE_CAPS:-"0.90 0.85"}

mkdir -p "$result/logs"
cd "$repo"
failures=0

for source_cap in $source_caps; do
  source_tag=${source_cap/./p}
  target_tag=${target_cap/./p}
  output_root="$result/malignant_sourcecap_${source_tag}_targetcap_${target_tag}_GSE180661"
  echo "===== SOURCE CAP $source_cap; TARGET CAP $target_cap ====="
  if CONFIDENCEOT_MANIFEST="$manifest" \
     CONFIDENCEOT_OUTPUT_ROOT="$output_root" \
     CONFIDENCEOT_ANALYSIS_SCOPE=malignant \
     CONFIDENCEOT_INCLUDE_ANNOTATIONS=Ovarian.cancer.cell \
     CONFIDENCEOT_PAIR_COUNT="$pair_count" \
     CONFIDENCEOT_SOURCE_REJECTION_BUDGET="$source_cap" \
     CONFIDENCEOT_TARGET_REJECTION_BUDGET="$target_cap" \
     bash cancer_metastasis/tacc_full_array.slurm; then
    echo "===== COMPLETE SOURCE CAP $source_cap ====="
  else
    failures=$((failures + 1))
    echo "===== FAILED SOURCE CAP $source_cap =====" >&2
  fi
done

if (( failures > 0 )); then
  echo "$failures source-cap setting(s) failed" >&2
  exit 1
fi
