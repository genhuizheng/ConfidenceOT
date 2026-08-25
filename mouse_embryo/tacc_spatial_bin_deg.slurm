#!/bin/bash
#SBATCH -J cot_bin_deg
#SBATCH -o /scratch/10119/ghzheng/OT_project/mouse_embryo_results/spatial_bin_deg/logs/%j.out
#SBATCH -e /scratch/10119/ghzheng/OT_project/mouse_embryo_results/spatial_bin_deg/logs/%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 04:00:00
#SBATCH -A MCB26031
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=FAIL,END

set -eo pipefail
source /home1/10119/ghzheng/.bashrc
conda activate worldmodeltraining

repo=/scratch/10119/ghzheng/OT_project/code/ConfidenceOT
run_root=/scratch/10119/ghzheng/OT_project/mouse_embryo_results/full_three_stage_tol1e4
data_root=/scratch/10119/ghzheng/OT_project/data
output_root=/scratch/10119/ghzheng/OT_project/mouse_embryo_results/spatial_bin_deg

export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
export PYTHONUNBUFFERED=1

cd "$repo"
mkdir -p "$output_root/logs"

python mouse_embryo/09_spatial_bin_deg_meta.py \
  "$run_root" \
  "$data_root" \
  "$output_root" \
  --consensus-threshold 0.5 \
  --block-width-bins 8 \
  --bootstrap-replicates 400 \
  --minimum-bins 20 \
  --minimum-blocks 4 \
  --minimum-gene-total 10

echo "COMPLETE spatial-bin DEG"
