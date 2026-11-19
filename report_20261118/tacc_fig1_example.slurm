#!/bin/bash
#SBATCH -J fig1_example
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:15:00
#SBATCH -A MCB26031
#SBATCH -o /scratch/10119/ghzheng/OT_project/report_20261118/logs/fig1_example_%j.out
# Figure 1d source tables: read-only on the v3 grid outputs, writes a NEW folder fig1_example/. One thread.
# No set -e / -u / pipefail (TACC rules).
source /home1/10119/ghzheng/.bashrc
conda activate /scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
R=/scratch/10119/ghzheng/OT_project
echo "job ${SLURM_JOB_ID} on $(hostname) $(date -Iseconds)"
/usr/bin/time -v python -I $R/report_20261118/code/tacc_fig1_example.py \
  --row $R/v3_grid_20261104/splatter/rows/L2_depth_dropout__rep2__N5000_rep2_L2_depth_dropout_population_lost/ranknm256-midrank__none/ranknm256-midrank__none__scale__cos__pca50 \
  --cond $R/splatter_n5000_default_paired_20261103/conditions/N5000/rep2/L2_depth_dropout/N5000_rep2_L2_depth_dropout_population_lost \
  --out $R/report_20261118/fig1_example
echo "exit $? $(date -Iseconds)"
