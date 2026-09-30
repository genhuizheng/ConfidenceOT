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
# Splatter generation and condition construction for one
# (size, replicate, technical level).
#
# One task per level, not per replicate: the three levels are independent
# realizations of the same biological parameters, each with its own seed, so
# none of them reuses another's counts. Deriving them from a single draw made
# part of the difference between levels a difference in the same cells, and
# left L2 defined as L1 with more dropout rather than as a condition of its
# own.
#
# The array index decodes to (size, replicate, level), so resubmitting one
# index reruns exactly one condition. Anything already carrying its completion
# flag is skipped unless CONFIDENCEOT_BENCH_FORCE=1.

# No set -u. TACC's module files and conda's shell functions read variables
# they do not define -- module load tacc-apptainer sources a completion script
# that reads BASH_COMPLETION_DEBUG -- and under set -u that kills the job in
# seconds with nothing in the log. Every expansion below carries its default.
set -eo pipefail
source /home1/10119/ghzheng/.bashrc

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
root=${CONFIDENCEOT_BENCH_ROOT:-/scratch/10119/ghzheng/OT_project/benchmark_technical}
sizes=(${CONFIDENCEOT_BENCH_SIZES:-1000 5000 10000})
replicates=${CONFIDENCEOT_BENCH_REPLICATES:-5}
theta=${CONFIDENCEOT_BENCH_THETA:-0.50}
dropout_mid=${CONFIDENCEOT_BENCH_DROPOUT_MID:--1.0}
# Where the depth mismatch comes from. 'thinning' is the original: Splatter
# draws both sides at one library size and 20_build_conditions.py thins the
# target's counts. 'splatter' asks the simulator for it, through lib.loc per
# side, and then nothing downstream touches the counts -- which is the point,
# because the read-equalisation arm inverts binomial thinning exactly and an
# equalised L1 result was therefore never independent evidence.
#
# The default is the original, so an existing root re-run reproduces itself.
depth_mechanism=${CONFIDENCEOT_BENCH_DEPTH_MECHANISM:-thinning}
lib_loc=${CONFIDENCEOT_BENCH_LIB_LOC:-11.0}
lib_scale=${CONFIDENCEOT_BENCH_LIB_SCALE:-0.2}
depth_ratio=${CONFIDENCEOT_BENCH_DEPTH_RATIO:-0.50}
# Splatter's dropout.mid is a logistic midpoint on log expression, so the two
# mechanisms want different values: the old generator put the source at -10
# and the target at this, the native one puts dropout on the target call
# alone. Kept separate rather than reused so neither default drifts with the
# other.
native_dropout_mid=${CONFIDENCEOT_BENCH_NATIVE_DROPOUT_MID:-1.0}
rank_top_n=${CONFIDENCEOT_BENCH_RANK_TOP_N:-256}
base_seed=${CONFIDENCEOT_BENCH_SEED:-20260925}
force=${CONFIDENCEOT_BENCH_FORCE:-0}
task=${SLURM_ARRAY_TASK_ID:?Submit as an array over sizes x replicates x levels}

levels=(L0_matched L1_depth L2_depth_dropout)
per_size=$(( replicates * ${#levels[@]} ))
size_index=$(( task / per_size ))
within=$(( task % per_size ))
replicate=$(( within / ${#levels[@]} + 1 ))
level_index=$(( within % ${#levels[@]} ))
if (( size_index >= ${#sizes[@]} )); then
    echo "task $task is past the end of the size list" >&2
    exit 2
fi
n_cells=${sizes[$size_index]}
level=${levels[$level_index]}

# Distinct per (replicate, level), so the levels are independent draws and the
# number in generation.json is the number that was used.
seed=$(( base_seed + 1000 * replicate + 137 * level_index ))

generated=$root/generated/N${n_cells}/rep${replicate}/${level}
conditions=$root/conditions/N${n_cells}/rep${replicate}/${level}
mkdir -p "$root/logs" "$generated" "$conditions"

if [[ "$force" != "1" && -f "$conditions/CONSTRUCTION_DONE" ]]; then
    echo "N=$n_cells rep=$replicate $level already built; skipping"
    exit 0
fi

# ---- Splatter ---------------------------------------------------------------
if [[ "$force" == "1" || ! -f "$generated/GENERATION_DONE" ]]; then
    # The R is whatever CONFIDENCEOT_RSCRIPT names, so a conda environment, a
    # bind-mounted library and an apptainer image are all the same to this
    # job. Only the conda route needs the environment activated, and
    # activating it is harmless for the others.
    conda activate "${CONFIDENCEOT_R_ENV:-/scratch/10119/ghzheng/conda_envs/infercnv_r}" || true
    export R_LIBS_USER=${CONFIDENCEOT_R_LIBRARY:-$repo/.r-library}
    # The container by default, because it is the route that was built and
    # verified here: splatter.def ends by running the same splatSimulate call
    # this job makes. Plain Rscript resolves to the conda R, where splatter
    # does not load at all, and the bind-mounted .r-library holds a splatter
    # compiled against a different R. Either of those is still selectable by
    # setting CONFIDENCEOT_RSCRIPT; neither should be what happens by
    # default.
    rscript=${CONFIDENCEOT_RSCRIPT:-$repo/benchmark/tacc/rscript.sh}
    if ! $rscript -e 'quit(status = !requireNamespace("splatter", quietly = TRUE))'; then
        echo "splatter will not load for: $rscript" >&2
        echo "  R_LIBS_USER=$R_LIBS_USER" >&2
        echo "  build the image with benchmark/tacc/build_container.slurm," >&2
        echo "  or point CONFIDENCEOT_RSCRIPT at an R that has it" >&2
        exit 4
    fi

    if [[ "$depth_mechanism" == "splatter" ]]; then
        # The native generator decides its own per-level settings from
        # --level, so the dropout flag is not conditional here: it names the
        # value L2 would use and the script ignores it at L0 and L1. One less
        # place for the level and its treatment to disagree.
        $rscript "$repo/benchmark/tacc/10_generate_splatter_native.R" \
            --out "$generated" \
            --n "$n_cells" \
            --replicate "$replicate" \
            --level "$level" \
            --groups "${CONFIDENCEOT_BENCH_GROUPS:-5}" \
            --seed "$seed" \
            --lib-loc "$lib_loc" \
            --lib-scale "$lib_scale" \
            --depth-ratio "$depth_ratio" \
            --dropout-mid-target "$native_dropout_mid"
    else
        # Only the dropout level asks Splatter for dropout, and it asks for it
        # on the target batch alone.
        dropout_flag=()
        [[ "$level" == "L2_depth_dropout" ]] && \
            dropout_flag=(--dropout-mid-target "$dropout_mid")

        $rscript "$repo/benchmark/tacc/10_generate_splatter.R" \
            --out "$generated" \
            --n "$n_cells" \
            --replicate "$replicate" \
            --level "$level" \
            --groups "${CONFIDENCEOT_BENCH_GROUPS:-5}" \
            --seed "$seed" \
            "${dropout_flag[@]}" \
            ${CONFIDENCEOT_BENCH_ESTIMATE_FROM:+--estimate-from "$CONFIDENCEOT_BENCH_ESTIMATE_FROM"}
    fi
    conda deactivate || true
else
    echo "generation already present for N=$n_cells rep=$replicate $level"
fi

# ---- the three biological cases ---------------------------------------------
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"
# With the native mechanism the depth difference is already in the counts
# Splatter drew, so the thinning is switched off by asking for a read survival
# probability of one: thin() returns its input unchanged at theta >= 1.0. The
# mechanism is passed through as well, so the manifest states where the depth
# came from instead of leaving a reader to infer it from a theta of one.
build_theta=$theta
[[ "$depth_mechanism" == "splatter" ]] && build_theta=1.0
python "$repo/benchmark/20_build_conditions.py" \
    --generated "$generated" \
    --out "$conditions" \
    --n-cells "$n_cells" \
    --replicate "$replicate" \
    --level "$level" \
    --theta "$build_theta" \
    --depth-mechanism "$depth_mechanism" \
    --rank-top-n "$rank_top_n" \
    --seed "$base_seed"

echo "done N=$n_cells rep=$replicate $level"
