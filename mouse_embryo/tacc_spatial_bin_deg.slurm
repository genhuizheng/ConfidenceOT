#!/bin/bash
#SBATCH -J cot_bin_deg
#SBATCH -o /scratch/10119/ghzheng/OT_project/mouse_embryo_results/spatial_bin_deg_scanpy/logs/%j.out
#SBATCH -e /scratch/10119/ghzheng/OT_project/mouse_embryo_results/spatial_bin_deg_scanpy/logs/%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 02:00:00
#SBATCH -A MCB26031
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=FAIL,END

set -eo pipefail
source /home1/10119/ghzheng/.bashrc
conda activate /scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot

repo=/scratch/10119/ghzheng/OT_project/code/ConfidenceOT
run_root=/scratch/10119/ghzheng/OT_project/mouse_embryo_results/full_three_stage_tol1e4
data_root=/scratch/10119/ghzheng/OT_project/data
output_root=/scratch/10119/ghzheng/OT_project/mouse_embryo_results/spatial_bin_deg_scanpy

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1

cd "$repo"
mkdir -p "$output_root/logs"

python mouse_embryo/09_spatial_bin_deg_meta.py \
  "$run_root" \
  "$data_root" \
  "$output_root" \
  --consensus-threshold 0.5 \
  --minimum-rejected-bins 20 \
  --minimum-retained-bins 20 \
  --minimum-gene-total 10

echo "COMPLETE Scanpy rejected-bin DEG"
