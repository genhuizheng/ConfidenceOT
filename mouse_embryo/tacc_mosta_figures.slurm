#!/bin/bash
#SBATCH -J mosta_figs
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 144
#SBATCH -t 08:00:00
#SBATCH -A MCB26031
#
# MOSTA figures (30_mosta_figures.py) on one gg node, CPU only.  Reads the
# finished ds run, writes only under MOSTA_FIG_OUT.  MOSTA_FIG_STAGES picks the
# stages (default all).  The log directory must exist before sbatch, so submit
# with -o/-e pointing into an existing directory, e.g.
#
#   OUT=/scratch/10119/ghzheng/OT_project/mouse_embryo_results/mosta_visualization_20261029
#   mkdir -p $OUT/logs
#   sbatch -o $OUT/logs/figures_%j.out -e $OUT/logs/figures_%j.err \
#       --export=ALL,MOSTA_FIG_OUT=$OUT,MOSTA_FIG_STAGES=all mouse_embryo/tacc_mosta_figures.slurm
#
# No set -e, set -u or pipefail: the python step's status is the job's status.
source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}" || exit 1
repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
cd "$repo" || exit 1
export PYTHONPATH="$repo/src:$repo${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg OMP_NUM_THREADS=8
echo "job=$SLURM_JOB_ID host=$(hostname) commit=$(git -C "$repo" rev-parse HEAD) stages=${MOSTA_FIG_STAGES:-all} start=$(date --iso-8601=seconds)"
python mouse_embryo/30_mosta_figures.py --out "${MOSTA_FIG_OUT:?MOSTA_FIG_OUT is not set}" \
    --stages "${MOSTA_FIG_STAGES:-all}" ${MOSTA_FIG_RUN:+--run "$MOSTA_FIG_RUN"} || exit $?
echo "end=$(date --iso-8601=seconds)"
