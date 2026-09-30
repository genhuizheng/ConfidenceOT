#!/bin/bash
#SBATCH -J cot_bench_list
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 00:20:00
#SBATCH -A MCB26031
#SBATCH -o /scratch/10119/ghzheng/OT_project/benchmark_technical/logs/list_%A.out
#SBATCH -e /scratch/10119/ghzheng/OT_project/benchmark_technical/logs/list_%A.err
#
# Numbers every unit of work once, after the manifests exist and before any
# solver task reads them.
#
# A job of its own rather than a step inside the array: every task has to see
# the same numbering, and the alternative -- each task building the list if it
# is missing -- is a race whose losers read a half-written file. Twenty
# minutes of one slot buys a numbering that a resubmitted chunk can be trusted
# to reproduce.
#
# It covers the sizes it was given, not everything on disk. submit.sh sizes
# the chunk array from --sizes, so a list that quietly included a size the
# invocation was not about would leave the tail of it uncovered -- the array
# would run to the end of its chunks and stop, and the remainder would look
# like work that had simply not been reached yet. The sizes go into the file
# name for the same reason: two runs on two partitions must not share one
# list, or the second renumbers the units the first is still solving.

# No set -u. TACC's module files and conda's shell functions read
# variables they do not define -- module load tacc-apptainer sources a
# completion script that reads BASH_COMPLETION_DEBUG -- and under set -u
# that kills the job in seconds with nothing in the log. Every expansion
# below carries its own default.
set -eo pipefail
source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"

root=${CONFIDENCEOT_BENCH_ROOT:-/scratch/10119/ghzheng/OT_project/benchmark_technical}
worklist=${CONFIDENCEOT_BENCH_WORKLIST:-$root/worklist.csv}
labels=${CONFIDENCEOT_BENCH_PREPROCESSING:?Set CONFIDENCEOT_BENCH_PREPROCESSING}
sizes=${CONFIDENCEOT_BENCH_SIZES:-1000 5000 10000}

python - "$root" "$worklist" "$sizes" $labels <<'PY'
import sys
from pathlib import Path

import pandas as pd
from pathlib import Path

root, out, sizes, *labels = sys.argv[1:]
wanted = {int(size) for size in sizes.split()}

# Which labels carry read equalisation, asked of the class that owns them.
# An equalised arm has to read the equalised counts or it misreports itself:
# the label records the correction and 27_downsample_counts.py applies it, so
# a _ds arm pointed at the original matrix would carry a name for something
# that never happened.
sys.path.insert(0, str(Path(root).parent))
from confidenceot.preprocessing import Preprocessing  # noqa: E402

equalised_labels = {label for label in labels
                    if Preprocessing.from_label(label).equalise_depth}
rows = []
skipped = set()
missing_equalised = set()
for manifest in sorted(Path(root).glob("conditions/*/*/*/manifest.csv")):
    table = pd.read_csv(manifest)
    # Filtered on the column rather than on the directory name: the manifest
    # is what the solver reads, so it is what decides.
    present = int(table["n_cells"].iloc[0])
    if present not in wanted:
        skipped.add(present)
        continue
    # conditions/N1000/rep1/L0_matched -> conditions_ds/N1000/rep1/L0_matched
    relative = manifest.parent.relative_to(Path(root) / "conditions")
    equalised_path = (Path(root) / "conditions_ds" / relative
                      / "pair_manifest_downsampled.csv")
    equalised = None
    if equalised_path.exists():
        # Looked up by pair_id rather than trusted to be in the same order:
        # the rewritten manifest drops a pair whose sample had no analysed
        # cells, and a positional index would then silently name a different
        # pair.
        table_ds = pd.read_csv(equalised_path)
        equalised = {str(name): position for position, name
                     in enumerate(table_ds["pair_id"].astype(str))}
    for index, row in table.iterrows():
        for label in labels:
            if label in equalised_labels:
                if equalised is None:
                    missing_equalised.add(str(relative))
                    continue
                position = equalised.get(str(row["pair_id"]))
                if position is None:
                    continue
                rows.append({"manifest": str(equalised_path),
                             "pair_index": int(position),
                             "pair_id": row["pair_id"], "preprocessing": label})
            else:
                rows.append({"manifest": str(manifest),
                             "pair_index": int(index),
                             "pair_id": row["pair_id"], "preprocessing": label})
if not rows:
    raise SystemExit("no manifests found; the generation stage has not run")
frame = pd.DataFrame(rows)
# Written beside the target and moved into place, so a reader either sees the
# previous list or the complete new one, never a partial file.
staging = Path(out).with_suffix(".partial")
frame.to_csv(staging, index=False)
staging.replace(out)
print(f"work list: {len(frame)} units, {frame['pair_id'].nunique()} pairs, "
      f"{len(labels)} configurations -> {out}")
if skipped:
    print(f"left out {sorted(skipped)}, which this invocation is not about")
if missing_equalised:
    raise SystemExit(
        f"{len(missing_equalised)} conditions have no equalised counts, so "
        f"the _ds arms have nothing to read: run --stage equalise first. "
        f"First few: {sorted(missing_equalised)[:4]}")
PY
