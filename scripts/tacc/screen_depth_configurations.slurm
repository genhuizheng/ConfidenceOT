#!/bin/bash
#SBATCH -J cot_depth_screen
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 144
#SBATCH -t 12:00:00
#SBATCH -A MCB26031
#SBATCH -o /scratch/10119/ghzheng/OT_project/depth_screen/logs/screen_%A_%a.out
#SBATCH -e /scratch/10119/ghzheng/OT_project/depth_screen/logs/screen_%A_%a.err
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=END,FAIL

# Every candidate depth treatment against the same simulated ground truth, at
# production parameters. Submit as an array, one configuration per task, so a
# configuration that fails or runs long does not hold up the rest:
#
#   sbatch --array=0-10 scripts/tacc/screen_depth_configurations.slurm
#
# The configuration this exists for is rank256_ds. The simulation had no read
# equalisation step at all, so its gene-ranking arm measured ranking alone --
# which is why that arm still showed a 0.35 depth effect at high depth spread
# while the production cancer pipeline reaches AUC 0.493 on real prostate data.
# Ranking alone on raw depth gives 0.060 there. The arm in the figure was never
# the pipeline we run.
#
# No `set -u`: the explicit checks below catch the defined-but-empty variable
# that -u permits, which is the failure that has cost job numbers here.
set -eo pipefail

source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
root=${CONFIDENCEOT_DEPTH_SCREEN_ROOT:-/scratch/10119/ghzheng/OT_project/depth_screen}

for name in repo root; do
  value=${!name}
  if [[ -z "$value" || "$value" != /* ]]; then
    echo "$name must be a non-empty absolute path, found: '$value'" >&2
    exit 2
  fi
done

cd "$repo"
export PYTHONPATH="$repo/src:$repo/cancer_metastasis:$repo${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
mkdir -p "$root/logs"

# Configurations 9 and 10 call R. The helper finds it, preferring the conda
# r-omics environment under $WORK over the container, and points R_LIBS_USER at
# the packages already installed there. The handoff files go under $root, which
# matters only in the container case: a container sees only what it is bound,
# and Python's default temp directory usually is not.
# shellcheck source=/dev/null
source "$repo/scripts/tacc/r_environment.sh"
export CONFIDENCEOT_R_WORKDIR=${CONFIDENCEOT_R_WORKDIR:-$root/r_workspace}

# label | flags. Index order is the array index order.
configurations=(
  "logcpm|--representation log_cpm"
  "logcpm_ds|--representation log_cpm --equalise-depth"
  "logcpm_cos|--representation log_cpm --cost cosine"
  "rank256|--representation rank_value --rank-top-n 256"
  "rank256_ds|--representation rank_value --rank-top-n 256 --equalise-depth"
  "rank256_ds_cos|--representation rank_value --rank-top-n 256 --equalise-depth --cost cosine"
  "pearson_ds|--representation pearson_residuals --equalise-depth"
  # External references, so the transform comparison does not rest only on our
  # own implementations. scanpy's is the analytic Pearson residual; sctransform
  # is the package Seurat's SCTransform v2 wraps, and is the field standard.
  # Both compose with equalisation, so each appears twice.
  "scanpy_pearson|--external-representation scanpy_pearson"
  "scanpy_pearson_ds|--external-representation scanpy_pearson --equalise-depth"
  "sct|--external-representation sctransform"
  "sct_ds|--external-representation sctransform --equalise-depth"
  # The screen's own result asks for these two. Residual transforms plus
  # equalisation won specificity (sct_ds 0.045, pearson_ds 0.050) while the
  # cosine cost won power (F1 0.945 against 0.85-0.88) and owned the high-depth
  # end (0.019). The flags compose, so the combination is one more run rather
  # than a new method.
  "pearson_ds_cos|--representation pearson_residuals --equalise-depth --cost cosine"
  "sct_ds_cos|--external-representation sctransform --equalise-depth --cost cosine"
  # The one empty cell in the table, and now the decisive one. Every other
  # family has an "alone" arm; ours does not, so when scanpy's analytic
  # residual and sctransform's fitted one swapped places between the two count
  # constructions there was no way to ask whether our analytic residual
  # behaves like scanpy's. If it does, the analytic-versus-fitted distinction
  # is real; if it does not, one of the implementations is doing something
  # unexpected and the swap is not a property of the transforms.
  "pearson|--representation pearson_residuals"
)

task=${SLURM_ARRAY_TASK_ID:-0}
if (( task < 0 || task >= ${#configurations[@]} )); then
  echo "Array index $task is outside [0,$(( ${#configurations[@]} - 1 ))]" >&2
  exit 2
fi

entry=${configurations[$task]}
label=${entry%%|*}
flags=${entry#*|}
# Anything that changes what a run means earns its own label. A confirmation
# round at more replicates, or a run on splatter's counts, would otherwise
# overwrite in place the very result it exists to compare against, and nobody
# would have to forget anything for that to happen -- the path is the same.
replicates=${CONFIDENCEOT_REPLICATES:-3}
splatter_arms=${CONFIDENCEOT_SPLATTER_ARMS:-}
n_cells=${CONFIDENCEOT_N_CELLS:-1500}
suffix=""
if (( replicates != 3 )); then
  suffix="${suffix}_r${replicates}"
fi
if [[ -n "$splatter_arms" ]]; then
  if [[ ! -f "$splatter_arms/splatter_depth_manifest.csv" ]]; then
    echo "no splatter arms at $splatter_arms; generate them first with" >&2
    echo "  sbatch scripts/tacc/generate_splatter_depth_arms.slurm" >&2
    exit 2
  fi
  suffix="${suffix}_splatter"
fi
if (( n_cells != 1500 )); then
  suffix="${suffix}_n${n_cells}"
fi
out="$root/${label}${suffix}"

# Passed as an array so an empty value contributes no argument at all.
extra=()
if [[ -n "$splatter_arms" ]]; then
  extra+=(--splatter-arms "$splatter_arms")
fi

echo "=================================================="
echo "configuration $task  $label"
echo "flags   $flags"
echo "cells   $n_cells"
echo "counts  ${splatter_arms:-internal construction}"
echo "output  $out"
echo "started $(date --iso-8601=seconds)"

# Production parameters, matching the defaults the earlier figure was drawn
# from, so the new rows are readable beside the stored ones.
#
# CONFIDENCEOT_REPLICATES raises the replicate count for a confirmation round.
# Three separates 0.49 from 0.05 easily; it does not separate 0.045 from
# 0.062, where the per-replicate ranges overlap and the pass threshold falls
# inside both. Point it at the few configurations actually in contention, not
# at all thirteen: runtime scales linearly with it.
"$CONDA_PREFIX/bin/python" scripts/validate_depth_null_specificity.py "$out" \
  $flags \
  "${extra[@]}" \
  --replicates "$replicates" \
  --n-cells "$n_cells" \
  --n-genes 4000 \
  --n-hvg 2000 \
  --n-pcs 30 \
  --calibration-max-cells 2000

echo "finished $(date --iso-8601=seconds)"
echo "summary  $out/depth_null_arm_summary.csv"
