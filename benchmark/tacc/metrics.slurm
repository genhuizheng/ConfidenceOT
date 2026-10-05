#!/bin/bash
#SBATCH -J cot_bench_score
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 08:00:00
#SBATCH -A MCB26031
#SBATCH -o /scratch/10119/ghzheng/OT_project/benchmark_technical/logs/score_%A.out
#SBATCH -e /scratch/10119/ghzheng/OT_project/benchmark_technical/logs/score_%A.err
#
# Score every finished run and pool the result into one table, on TACC. The
# point of doing it here rather than after a download is that the couplings
# are the large objects and they never need to leave the filesystem: only the
# scored table does.
#
# Missing runs are reported, not fatal. A benchmark that refuses to summarise
# until every last unit has landed cannot tell you which ones are missing.

# No set -u. TACC's module files and conda's shell functions read
# variables they do not define -- module load tacc-apptainer sources a
# completion script that reads BASH_COMPLETION_DEBUG -- and under set -u
# that kills the job in seconds with nothing in the log. Every expansion
# below carries its own default.
set -eo pipefail
source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
root=${CONFIDENCEOT_BENCH_ROOT:-/scratch/10119/ghzheng/OT_project/benchmark_technical}
# Which run tree to score: runs/ by default, or another one ot_array.slurm
# wrote, such as runs_acceptance_0.99/. Another tree gets its own metrics
# directory and its own pooled table, so scoring it never clears or merges
# into the default one.
runs_dir=${CONFIDENCEOT_BENCH_RUNS_DIR:-runs}
metrics_dir=$root/metrics
pooled_table=$root/benchmark_metrics.csv
if [[ "$runs_dir" != "runs" ]]; then
    metrics_dir=$root/metrics_$runs_dir
    pooled_table=$root/benchmark_metrics_$runs_dir.csv
fi
# Cleared, not merged into. The file name carries the condition and one
# table now holds every configuration of it, where the previous layout was
# one table per (condition, configuration) -- and the old names still match
# the new glob. Pooling over both put 515 of the 540 old cells into the
# table a second time: 6,330 rows where 3,240 were real, and every median
# over them double counted most of the grid.
#
# The directory is derived output and is rebuilt in full by this job, so
# emptying it is the correct operation rather than a convenience.
rm -rf "$metrics_dir"
mkdir -p "$metrics_dir"

shopt -s nullglob
# Every configuration of a condition in one pass. Twelve separate invocations
# re-read the same h5ads twelve times for readouts that do not depend on the
# preprocessing, and paid twelve interpreter startups to do it. That is what
# ran the scoring past its two-hour wall with every one of the 1,620 runs
# already finished. The run roots do not depend on the condition, so they are
# listed once.
roots=()
for run_root in "$root/$runs_dir"/*; do
    [[ -d "$run_root" ]] || continue
    roots+=(--run-root "$run_root")
done
if (( ${#roots[@]} == 0 )); then
    echo "no run roots under $root/$runs_dir" >&2
    exit 2
fi

# The conditions are independent -- each writes its own table -- and they
# used to run one after another on a job that had asked for sixteen cores.
# The fifteen N=10000 conditions dominate, so sixteen at once puts all of them
# in the first wave and the wall becomes roughly one of them rather than the
# sum. One BLAS thread per scorer, or sixteen scorers each starting a thread
# per core put 256 threads on sixteen cores.
parallel=${CONFIDENCEOT_BENCH_SCORE_PARALLEL:-${SLURM_CPUS_PER_TASK:-16}}
echo "scoring $(ls "$root"/conditions/*/*/*/manifest.csv | wc -l) conditions, $parallel at a time"

score_one() {
    # A failure is written down rather than returned: under set -e the exit
    # status of a background job would either be lost or, through wait,
    # kill the loop with the other scorers still running.
    local manifest=$1 tag=$2
    if ! OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        python "$repo/benchmark/40_score.py" \
            "${roots[@]}" \
            --manifest "$manifest" \
            --scope all \
            --out "$metrics_dir/${tag}.csv" > "$metrics_dir/${tag}.log" 2>&1; then
        echo "$tag" >> "$metrics_dir/FAILED"
    fi
}

for manifest in "$root"/conditions/*/*/*/manifest.csv; do
    level_dir=$(dirname "$manifest")
    replicate_dir=$(dirname "$level_dir")
    tag=$(basename "$(dirname "$replicate_dir")")_$(basename "$replicate_dir")_$(basename "$level_dir")
    score_one "$manifest" "$tag" &
    while (( $(jobs -rp | wc -l) >= parallel )); do
        wait -n || true
    done
done
wait || true

if [[ -s "$metrics_dir/FAILED" ]]; then
    echo "$(wc -l < "$metrics_dir/FAILED") condition(s) failed to score:" >&2
    cat "$metrics_dir/FAILED" >&2
    echo "each has its own log in $metrics_dir/<condition>.log" >&2
fi

python - "$metrics_dir" "$pooled_table" <<'PY'
import sys
from pathlib import Path

import pandas as pd

metrics_dir, pooled_table = Path(sys.argv[1]), Path(sys.argv[2])
# One file per condition now, holding every configuration.
parts = sorted(metrics_dir.glob("*.csv"))
frames = [pd.read_csv(path) for path in parts if path.stat().st_size > 0]
if not frames:
    raise SystemExit("no scored tables to pool")
pooled = pd.concat(frames, ignore_index=True)
pooled.to_csv(pooled_table, index=False)
print(f"pooled {len(pooled)} rows from {len(frames)} tables into "
      f"{pooled_table}")
outside = pooled.loc[~pooled["rank_cut_valid"].astype(bool), "pair_id"].unique()
if len(outside):
    print(f"{len(outside)} conditions are outside the valid rank-encoding "
          f"regime and must not be read as performance points: "
          f"{sorted(outside)[:8]}")
PY

echo "scoring finished"
