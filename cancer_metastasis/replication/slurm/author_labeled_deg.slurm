#!/bin/bash
#SBATCH -J cot_rep_deg
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/rep_deg_%j.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/rep_deg_%j.err
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 02:00:00
#SBATCH -A MCB26031
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=END,FAIL

set -eo pipefail
source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"
set -u

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
root=${CONFIDENCEOT_REPLICATION_ROOT:?CONFIDENCEOT_REPLICATION_ROOT is required}
cd "$repo"
export PYTHONPATH="$repo/src:$repo/cancer_metastasis${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
export PYTHONUNBUFFERED=1

for dataset in GSE181919 GSE225857; do
  input="$root/$dataset/primary_pseudobulk"
  output="$root/$dataset/primary_deg"
  ready=$(find "$input/groups" -name PSEUDOBULK_READY -type f | wc -l)
  if (( ready < 2 )); then
    mkdir -p "$output"
    printf '{"status":"skipped","reason":"fewer_than_two_ready_pairs","ready_pair_n":%s}\n' "$ready" > "$output/DEG_SKIPPED.json"
    continue
  fi
  python -m cancer_metastasis.gse180661.primary_deg \
    "$input" "$output" \
    --minimum-total-count 10 \
    --n-cpus 16 \
    --absolute-log2-fold-change-thresholds 0.5 1.0 \
    --gate-labels
done

touch "$root/DEG_COMPLETE"
