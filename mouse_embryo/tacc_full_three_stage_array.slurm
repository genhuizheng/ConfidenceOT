#!/bin/bash
#SBATCH -J cot_mosta_full
#SBATCH -o /scratch/10119/ghzheng/OT_project/mouse_embryo_results/full_three_stage/logs/%A_%a.out
#SBATCH -e /scratch/10119/ghzheng/OT_project/mouse_embryo_results/full_three_stage/logs/%A_%a.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 72
#SBATCH -a 0-35%10
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -t 48:00:00
#SBATCH -A MCB26031

set -euo pipefail
task_started=$SECONDS
source /home1/10119/ghzheng/.bashrc
conda activate worldmodeltraining

repo=/scratch/10119/ghzheng/OT_project/code/ConfidenceOT
data_root=/scratch/10119/ghzheng/OT_project/data
run_root=/scratch/10119/ghzheng/OT_project/mouse_embryo_results/full_three_stage
manifest="$run_root/pair_manifest.csv"
task_root="$run_root/pairs"
local_root="${TMPDIR:-/tmp}/confidenceot_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

cleanup_local() {
  case "$local_root" in
    "${TMPDIR:-/tmp}"/confidenceot_*) rm -rf -- "$local_root" ;;
    *) echo "Refusing unsafe cleanup target: $local_root" >&2 ;;
  esac
}
trap cleanup_local EXIT

export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export PYTHONUNBUFFERED=1
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$repo"
mkdir -p "$run_root/logs" "$task_root" "$local_root"

row=$(python - "$manifest" "$SLURM_ARRAY_TASK_ID" <<'PY'
import csv
import sys
with open(sys.argv[1], newline="") as handle:
    rows = list(csv.DictReader(handle))
record = rows[int(sys.argv[2])]
print("\t".join(record[key] for key in ("pair_id", "source_h5ad", "target_h5ad")))
PY
)
IFS=$'\t' read -r pair_id source_h5ad target_h5ad <<< "$row"
output="$task_root/$pair_id"

echo "pair_id=$pair_id"
echo "source=$source_h5ad"
echo "target=$target_h5ad"
echo "output=$output"
echo "local_work=$local_root"
nvidia-smi
python -m confidenceot
python -c 'from confidenceot.cuda import cuda_available; assert cuda_available(), "ConfidenceOT cannot see the H200 through PyTorch"'
df -h "${TMPDIR:-/tmp}"
free -h

if [[ ! -f "$output/preparation/prepared_pair.npz" ]]; then
  python mouse_embryo/02_prepare_pair.py \
    "$source_h5ad" "$target_h5ad" "$output/preparation" \
    --all-bins --n-hvg 2000 --heldout-hvg 500 --n-pcs 30 --seed 20260824
fi

if [[ ! -f "$output/analysis/SUCCESS" ]]; then
  python mouse_embryo/07_run_full_pair.py \
    "$output/preparation/prepared_pair.npz" \
    "$output/analysis" \
    "$local_root" \
    --epsilon 0.1 --lambda-a 1.0 --lambda-b 1.0 \
    --rejection-budget 0.15 --tolerance 1e-3 \
    --null-calibration-replicates 5 \
    --null-validation-replicates 5 \
    --calibration-grid-size 5 \
    --calibration-max-bins 2000 \
    --device cuda
fi

echo "COMPLETE $pair_id"
echo "TASK_WALL_SECONDS=$((SECONDS-task_started))"
