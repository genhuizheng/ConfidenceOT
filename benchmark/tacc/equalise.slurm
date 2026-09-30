#!/bin/bash
#SBATCH -J cot_bench_ds
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 08:00:00
#SBATCH -A MCB26031
#SBATCH -o /scratch/10119/ghzheng/OT_project/benchmark_technical/logs/ds_%A_%a.out
#SBATCH -e /scratch/10119/ghzheng/OT_project/benchmark_technical/logs/ds_%A_%a.err
#
# Read equalisation for the synthetic conditions, one target depth per
# condition.
#
# **Per condition, not per grid.** The real-data run gave all 118 pairs one
# shared target, which is right there: the depth variation it removes is a
# nuisance across cells and samples. Here the depth varies for two reasons,
# and only one of them is a nuisance. Within a condition it is the cell-level
# spread, which is what equalisation exists to remove. Between L0, L1 and L2
# it is the construction -- the experimental axis itself. One target across
# the whole grid would pull all three levels to the same depth and delete the
# distinction the benchmark is built on, so each condition is equalised
# within itself.
#
# What that does not avoid, and cannot: inside a condition the source is
# undegraded and the target was thinned, so equalising the two to one depth
# removes the mismatch L1 was constructed to create. DESIGN.md states this --
# L1 downsampling results are not independent evidence about low-depth
# robustness. L2 is not reversed the same way, because dropout removes genes
# rather than thinning reads proportionally.
#
# Every cell is kept. A simulation has no non-malignant compartment to select
# out of, which is what --all-cells is for.
#
# No set -u. TACC's module files and conda's shell functions read variables
# they do not define, and under set -u that kills the job in seconds with
# nothing in the log. Every expansion below carries its own default.
set -eo pipefail
source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
root=${CONFIDENCEOT_BENCH_ROOT:-/scratch/10119/ghzheng/OT_project/benchmark_technical}
quantile=${CONFIDENCEOT_BENCH_DS_QUANTILE:-0.10}
chunk_size=${CONFIDENCEOT_BENCH_DS_CHUNK:-8}
force=${CONFIDENCEOT_BENCH_FORCE:-0}

cd "$repo"
export PYTHONPATH="$repo/src:$repo/cancer_metastasis:$repo${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

shopt -s nullglob
manifests=("$root"/conditions/*/*/*/manifest.csv)
total=${#manifests[@]}
if (( total == 0 )); then
    echo "no condition manifests under $root/conditions" >&2
    exit 2
fi

task=${SLURM_ARRAY_TASK_ID:?Submit as a chunk array}
start=$(( task * chunk_size ))
if (( start >= total )); then
    echo "chunk $task starts at $start, past the $total conditions; nothing to do"
    exit 0
fi
stop=$(( start + chunk_size ))
(( stop > total )) && stop=$total
echo "chunk $task covers conditions [$start,$stop) of $total"

failures=0
for (( position = start; position < stop; position++ )); do
    manifest=${manifests[$position]}
    # conditions/N1000/rep1/L0_matched -> conditions_ds/N1000/rep1/L0_matched,
    # so the two trees mirror each other and a reader can see which condition
    # an equalised one came from without consulting anything.
    relative=${manifest#"$root"/conditions/}
    relative=$(dirname "$relative")
    destination=$root/conditions_ds/$relative
    if [[ "$force" != "1" && -f "$destination/pair_manifest_downsampled.csv" ]]; then
        echo "  $relative already equalised"
        continue
    fi
    mkdir -p "$destination"
    echo "===== START $relative $(date --iso-8601=seconds) ====="
    if python "$repo/cancer_metastasis/27_downsample_counts.py" \
        "$manifest" "$destination" \
        --all-cells \
        --target-quantile "$quantile" \
        --minimum-total-counts 0 \
        --minimum-detected-genes 0 \
        --maximum-mitochondrial-percent 100 > "$destination.log" 2>&1; then
        echo "===== COMPLETE $relative $(date --iso-8601=seconds) ====="
    else
        status=$?
        failures=$(( failures + 1 ))
        echo "===== FAILED $relative exit=$status ====="
        echo "===== FAILED $relative exit=$status =====" >&2
        tail -20 "$destination.log" >&2 || true
    fi
done

echo "chunk $task finished with $failures failed condition(s)"
(( failures > 0 )) && exit 1
exit 0
