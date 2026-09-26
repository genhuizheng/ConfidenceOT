#!/bin/bash
#SBATCH -J cot_bench_score
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 02:00:00
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
mkdir -p "$root/metrics"

shopt -s nullglob
for manifest in "$root"/conditions/*/*/*/manifest.csv; do
    level_dir=$(dirname "$manifest")
    replicate_dir=$(dirname "$level_dir")
    tag=$(basename "$(dirname "$replicate_dir")")_$(basename "$replicate_dir")_$(basename "$level_dir")
    for run_root in "$root"/runs/*; do
        [[ -d "$run_root" ]] || continue
        preprocessing=$(basename "$run_root")
        python "$repo/benchmark/40_score.py" \
            --run-root "$run_root" \
            --manifest "$manifest" \
            --preprocessing "$preprocessing" \
            --scope all \
            --out "$root/metrics/${tag}__${preprocessing}.csv"
    done
done

python - "$root" <<'PY'
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
parts = sorted((root / "metrics").glob("*__*.csv"))
frames = [pd.read_csv(path) for path in parts if path.stat().st_size > 0]
if not frames:
    raise SystemExit("no scored tables to pool")
pooled = pd.concat(frames, ignore_index=True)
pooled.to_csv(root / "benchmark_metrics.csv", index=False)
print(f"pooled {len(pooled)} rows from {len(frames)} tables into "
      f"{root / 'benchmark_metrics.csv'}")
outside = pooled.loc[~pooled["rank_cut_valid"].astype(bool), "pair_id"].unique()
if len(outside):
    print(f"{len(outside)} conditions are outside the valid rank-encoding "
          f"regime and must not be read as performance points: "
          f"{sorted(outside)[:8]}")
PY

echo "scoring finished"
