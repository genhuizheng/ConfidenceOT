#!/bin/bash
#SBATCH -J cot_splat_depth
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 08:00:00
#SBATCH -A MCB26031
#SBATCH -o /scratch/10119/ghzheng/OT_project/splatter_depth_generate_%j.out
#SBATCH -e /scratch/10119/ghzheng/OT_project/splatter_depth_generate_%j.err
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=END,FAIL

# Generate the depth screen's arms with splatter, so the depth ladder is set by
# a published simulator's own lib.scale rather than by our construction.
#
# The R lives in the same conda environment as the rest of the R work,
# $SCRATCH/conda_envs/infercnv_r, which is where splatter was installed for the
# scaling benchmark; its library is the repo's .r-library, exactly as
# generate_splatter_scaling.slurm and install_splatter_dependencies.slurm set
# it up.
#
# Logs go straight into OT_project rather than into a subdirectory of this
# job's own output root. Slurm opens the log files before the script runs, so a
# directory the script creates does not exist yet, and the job dies having
# written nothing -- the one failure that leaves no evidence behind.
#
# Sized from a local run of the same script against splatter 1.30.0, so this is
# not a first attempt on the cluster. Memory is the constraint, not time:
# splatter holds several dense cells-by-genes matrices at once, and the pool is
# 2 x n_cells per batch across two batches.
#
# No `set -u`: the explicit checks below catch the defined-but-empty variable
# that -u permits.
#
# Usage, production shape:
#   sbatch scripts/tacc/generate_splatter_depth_arms.slurm
#
# A different size, which writes to its own directory:
#   sbatch --export=ALL,CONFIDENCEOT_SPLATTER_N=5000 \
#       scripts/tacc/generate_splatter_depth_arms.slurm
set -eo pipefail

source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_R_ENV:-/scratch/10119/ghzheng/conda_envs/infercnv_r}"

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
root=${CONFIDENCEOT_SPLATTER_DEPTH_ROOT:-/scratch/10119/ghzheng/OT_project/depth_screen_splatter}
library=${CONFIDENCEOT_R_LIBRARY:-$repo/.r-library}

n_cells=${CONFIDENCEOT_SPLATTER_N:-1500}
n_genes=${CONFIDENCEOT_SPLATTER_GENES:-4000}
replicates=${CONFIDENCEOT_REPLICATES:-3}
seed=${CONFIDENCEOT_SPLATTER_SEED:-20260914}
median_depth=${CONFIDENCEOT_SPLATTER_MEDIAN_DEPTH:-10000}
sigmas=${CONFIDENCEOT_SPLATTER_SIGMAS:-0,0.3,0.6,0.9}
perturbed=${CONFIDENCEOT_SPLATTER_PERTURBED:-0.2}
tolerance=${CONFIDENCEOT_SPLATTER_TOLERANCE:-0.05}
pool=${CONFIDENCEOT_SPLATTER_POOL:-2}

for pair in "repo:$repo" "root:$root" "library:$library"; do
  name=${pair%%:*}
  value=${pair#*:}
  if [[ -z "$value" || "$value" != /* ]]; then
    echo "$name must be a non-empty absolute path, found: '$value'" >&2
    exit 2
  fi
done
if [[ ! -f "$repo/scripts/generate_splatter_depth_arms.R" ]]; then
  echo "no generator under $repo; did the pull land?" >&2
  exit 2
fi

# One directory per size, so a 5,000-cell run cannot land on top of a 1,500-cell
# one -- the arm names are the same in both.
output="$root/n_${n_cells}"
if [[ -e "$output/splatter_depth_manifest.csv" ]]; then
  echo "Already generated: $output" >&2
  echo "Remove it, or set CONFIDENCEOT_SPLATTER_DEPTH_ROOT elsewhere." >&2
  exit 3
fi

mkdir -p "$output"
export R_LIBS_USER="$library"
cd "$repo"

echo "=================================================="
echo "cells per side  $n_cells"
echo "genes           $n_genes"
echo "replicates      $replicates"
echo "lib.scale       $sigmas   (lib.loc = log($median_depth))"
echo "pool multiplier $pool"
echo "output          $output"
echo "R_LIBS_USER     $R_LIBS_USER"
echo "started         $(date --iso-8601=seconds)"
echo

Rscript --vanilla scripts/generate_splatter_depth_arms.R \
  "$output" "$n_cells" "$n_genes" "$replicates" "$seed" "$median_depth" \
  "$sigmas" "$perturbed" "$tolerance" "$pool"

echo
echo "finished $(date --iso-8601=seconds)"
du -sh "$output"
echo
echo "=================================================="
echo "run the screen against these counts with, for example:"
echo
echo "  export CONFIDENCEOT_SPLATTER_ARMS=$output"
echo "  sbatch --array=0-12 --export=ALL,CONFIDENCEOT_SPLATTER_ARMS=$output,CONFIDENCEOT_N_CELLS_OVERRIDE=$n_cells \\"
echo "      scripts/tacc/screen_depth_configurations.slurm"
