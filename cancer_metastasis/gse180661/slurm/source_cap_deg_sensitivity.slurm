#!/bin/bash
#SBATCH -J cot_src_cap_de
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/src_cap_de_%A_%a.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/src_cap_de_%A_%a.err
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
pseudobulk_base=${CONFIDENCEOT_SOURCE_CAP_PSEUDOBULK_BASE:-$result/gse180661_source_cap_sensitivity_pseudobulk}
output_base=${CONFIDENCEOT_SOURCE_CAP_DEG_BASE:-$result/gse180661_source_cap_sensitivity_pydeseq2}
source_caps=${CONFIDENCEOT_SOURCE_CAPS:-"0.50,0.70,0.75,0.85,0.90,0.95"}

cd "$repo"
export PYTHONPATH="$repo/src:$repo/cancer_metastasis${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
export PYTHONUNBUFFERED=1
mkdir -p "$output_base" "$result/logs"

read -r -a caps <<< "${source_caps//,/ }"
task_id=${SLURM_ARRAY_TASK_ID:-0}
task_count=${SLURM_ARRAY_TASK_COUNT:-1}
failures=0
for ((cap_index=task_id; cap_index<${#caps[@]}; cap_index+=task_count)); do
  source_cap=${caps[$cap_index]}
  cap_slug=$(printf "%.2f" "$source_cap")
  cap_slug=${cap_slug/./p}
  groups="$pseudobulk_base/source_cap_$cap_slug"
  output="$output_base/source_cap_$cap_slug"
  if python -m cancer_metastasis.gse180661.primary_deg \
      "$groups" "$output" --minimum-total-count 10 --n-cpus 16 \
      --absolute-log2-fold-change-thresholds 0.5 1.0; then
    echo "COMPLETE source_cap=$source_cap"
  else
    status=$?
    failures=$((failures + 1))
    echo "FAILED source_cap=$source_cap exit=$status" >&2
  fi
done

if (( failures > 0 )); then
  exit 1
fi
