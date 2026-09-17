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

# Configurations 9 and 10 call R. Seurat is not on PATH here; it lives in a
# container, so the whole invocation is passed through rather than assumed. The
# handoff files go under $root so the bind in CONFIDENCEOT_RSCRIPT covers them
# -- a container sees only what it is given, and the default temp directory is
# the usual reason this fails.
container=${CONFIDENCEOT_R_SIF:-/scratch/10119/ghzheng/containers/r-ver-4.4.sif}
r_libs=${CONFIDENCEOT_R_LIBS:-/scratch/10119/ghzheng/r_libs}
if [[ -z "${CONFIDENCEOT_RSCRIPT:-}" && -f "$container" ]]; then
  # apptainer is a module here, and it is refused on the login nodes, so the
  # load has to happen inside the job. Without it the container tasks fail with
  # "apptainer: command not found" on the compute node too.
  module load tacc-apptainer 2>/dev/null || true
  export CONFIDENCEOT_RSCRIPT="apptainer exec -B /scratch --env R_LIBS_USER=$r_libs $container Rscript"
fi
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
  # own implementations. scanpy's is the analytic Pearson residual; Seurat's
  # SCTransform v2 is the field standard and needs Rscript with Seurat and
  # Matrix. Both compose with equalisation, so each appears twice.
  "scanpy_pearson|--external-representation scanpy_pearson"
  "scanpy_pearson_ds|--external-representation scanpy_pearson --equalise-depth"
  "sct|--external-representation sctransform"
  "sct_ds|--external-representation sctransform --equalise-depth"
)

task=${SLURM_ARRAY_TASK_ID:-0}
if (( task < 0 || task >= ${#configurations[@]} )); then
  echo "Array index $task is outside [0,$(( ${#configurations[@]} - 1 ))]" >&2
  exit 2
fi

entry=${configurations[$task]}
label=${entry%%|*}
flags=${entry#*|}
out="$root/$label"

echo "=================================================="
echo "configuration $task  $label"
echo "flags   $flags"
echo "output  $out"
echo "started $(date --iso-8601=seconds)"

# Production parameters, matching the defaults the earlier figure was drawn
# from, so the new rows are readable beside the stored ones.
"$CONDA_PREFIX/bin/python" scripts/validate_depth_null_specificity.py "$out" \
  $flags \
  --replicates 3 \
  --n-cells 1500 \
  --n-genes 4000 \
  --n-hvg 2000 \
  --n-pcs 30 \
  --calibration-max-cells 2000

echo "finished $(date --iso-8601=seconds)"
echo "summary  $out/depth_null_arm_summary.csv"
