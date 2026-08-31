#!/bin/bash
#SBATCH -J cot_cancer_pilot
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/pilot_%A_%a.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/pilot_%A_%a.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 12:00:00
#SBATCH -A MCB26031
# Supply --array=0-(pilot_rows-1)%4 at submission; see the launch commands.
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=END,FAIL

source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
pilot="$result/manifest/budget_pilot_manifest.csv"
eligible="$result/manifest/pair_manifest_eligible.csv"
mkdir -p "$result/logs" "$result/pilot"
cd "$repo"
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export PYTHONUNBUFFERED=1

read -r eligible_index budget < <(python - "$pilot" "$SLURM_ARRAY_TASK_ID" <<'PY'
import pandas as pd, sys
row = pd.read_csv(sys.argv[1]).iloc[int(sys.argv[2])]
print(int(row.eligible_index), float(row.rejection_budget))
PY
)

local_base=${TMPDIR:-/tmp}
local_work="$local_base/confidenceot_cancer_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$local_work"
cleanup_local() {
  case "$local_work" in
    "$local_base"/confidenceot_cancer_*) rm -rf -- "$local_work" ;;
    *) echo "Refusing unsafe cleanup target: $local_work" >&2 ;;
  esac
}
trap cleanup_local EXIT
export TMPDIR="$local_work"

python cancer_metastasis/02_run_pair.py "$eligible" "$result/pilot" \
  --index "$eligible_index" \
  --rejection-budget "$budget" \
  --null-calibration-replicates 5 \
  --null-validation-replicates 5 \
  --calibration-grid-size 5 \
  --tolerance 1e-4 \
  --workers 1 \
  --device cuda
