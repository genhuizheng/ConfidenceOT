#!/bin/bash
#SBATCH -J cot_bench_gen
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 04:00:00
#SBATCH -A MCB26031
#SBATCH -o /scratch/10119/ghzheng/OT_project/benchmark_technical/logs/gen_%A_%a.out
#SBATCH -e /scratch/10119/ghzheng/OT_project/benchmark_technical/logs/gen_%A_%a.err
#
# Splatter generation and condition construction for one (N, replicate).
#
# Both stages live in one task because the second is minutes and the first is
# what costs: splitting them would double the queue slots for no gain, and the
# 40-job submission limit is the binding constraint here.
#
# The array index is decoded against the size list and the replicate count, so
# resubmitting one index reruns exactly one replicate. Anything already
# carrying its completion flag is skipped unless CONFIDENCEOT_BENCH_FORCE=1.

set -eo pipefail
source /home1/10119/ghzheng/.bashrc
set -u

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
root=${CONFIDENCEOT_BENCH_ROOT:-/scratch/10119/ghzheng/OT_project/benchmark_technical}
sizes=(${CONFIDENCEOT_BENCH_SIZES:-1000 5000 10000})
replicates=${CONFIDENCEOT_BENCH_REPLICATES:-5}
theta=${CONFIDENCEOT_BENCH_THETA:-0.50}
dropout=${CONFIDENCEOT_BENCH_DROPOUT:-0.55}
rank_top_n=${CONFIDENCEOT_BENCH_RANK_TOP_N:-256}
force=${CONFIDENCEOT_BENCH_FORCE:-0}
task=${SLURM_ARRAY_TASK_ID:?Submit as an array over sizes x replicates}

size_index=$(( task / replicates ))
replicate=$(( task % replicates + 1 ))
if (( size_index >= ${#sizes[@]} )); then
    echo "task $task is past the end of the size list" >&2
    exit 2
fi
n_cells=${sizes[$size_index]}

generated=$root/generated/N${n_cells}/rep${replicate}
conditions=$root/conditions/N${n_cells}/rep${replicate}
mkdir -p "$root/logs" "$generated" "$conditions"

if [[ "$force" != "1" && -f "$conditions/CONSTRUCTION_DONE" ]]; then
    echo "N=$n_cells rep=$replicate already built; skipping"
    exit 0
fi

# ---- Splatter ---------------------------------------------------------------
if [[ "$force" == "1" || ! -f "$generated/GENERATION_DONE" ]]; then
    # The R is whatever CONFIDENCEOT_RSCRIPT names, so a conda
    # environment, a bind-mounted library and an apptainer image are all
    # the same to this job. Only the conda route needs the environment
    # activated, and activating it is harmless for the others.
    conda activate "${CONFIDENCEOT_R_ENV:-/scratch/10119/ghzheng/conda_envs/infercnv_r}" || true
    # Where install_splatter_dependencies.slurm puts Splatter. Inside a
    # container this is the bind-mounted path, which is writable where
    # the image is not.
    export R_LIBS_USER=${CONFIDENCEOT_R_LIBRARY:-$repo/.r-library}
    rscript=${CONFIDENCEOT_RSCRIPT:-Rscript}
    if ! $rscript -e 'quit(status = !requireNamespace("splatter", quietly = TRUE))'; then
        echo "splatter will not load for: $rscript" >&2
        echo "  R_LIBS_USER=$R_LIBS_USER" >&2
        echo "  install it with scripts/tacc/install_splatter_dependencies.slurm," >&2
        echo "  or point CONFIDENCEOT_RSCRIPT at an R that has it" >&2
        exit 4
    fi
    $rscript "$repo/benchmark/tacc/10_generate_splatter.R" \
        --out "$generated" \
        --n "$n_cells" \
        --replicate "$replicate" \
        --groups "${CONFIDENCEOT_BENCH_GROUPS:-5}" \
        --seed "${CONFIDENCEOT_BENCH_SEED:-20260925}" \
        ${CONFIDENCEOT_BENCH_ESTIMATE_FROM:+--estimate-from "$CONFIDENCEOT_BENCH_ESTIMATE_FROM"}
    conda deactivate
else
    echo "generation already present for N=$n_cells rep=$replicate"
fi

# ---- levels and cases -------------------------------------------------------
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"
python "$repo/benchmark/20_build_conditions.py" \
    --generated "$generated" \
    --out "$conditions" \
    --n-cells "$n_cells" \
    --replicate "$replicate" \
    --theta "$theta" \
    --dropout "$dropout" \
    --rank-top-n "$rank_top_n" \
    --seed "${CONFIDENCEOT_BENCH_SEED:-20260925}"

echo "done N=$n_cells rep=$replicate"
