#!/bin/bash
#SBATCH -J cot_solvers
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 72
#SBATCH -t 24:00:00
#SBATCH -A MCB26031
#SBATCH -p gg
#SBATCH -o /scratch/10119/ghzheng/OT_project/benchmark_technical/logs/solv_%A_%a.out
#SBATCH -e /scratch/10119/ghzheng/OT_project/benchmark_technical/logs/solv_%A_%a.err
#
# Four solvers on the benchmark's conditions, at one locked preprocessing.
#
# One array task walks a contiguous run of the condition manifests, three
# pairs each, so the task count stays inside the submission cap. The label is
# a parameter: this stage runs after the preprocessing grid has settled on
# one, and it must be the same one for every task or the comparison is
# between configurations again.
#
# Partial OT's transported mass is swept inside the runner, so one pair
# produces several Partial OT rows. That is deliberate -- m decides that
# method's whole result, and a single value chosen by us is a value chosen
# with the answer in view.
#
# No set -u. TACC's module files and conda's shell functions read variables
# they do not define, and under set -u that kills the job in seconds with
# nothing in the log. Every expansion below carries its own default.
set -eo pipefail
source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
root=${CONFIDENCEOT_BENCH_ROOT:-/scratch/10119/ghzheng/OT_project/benchmark_technical}
label=${CONFIDENCEOT_SOLVER_LABEL:?Set CONFIDENCEOT_SOLVER_LABEL to the locked configuration}
out=${CONFIDENCEOT_SOLVER_ROOT:-$root/solvers/$label}
scope=${CONFIDENCEOT_BENCH_SCOPE:-all}
chunk_size=${CONFIDENCEOT_SOLVER_CHUNK:-3}
epsilon=${CONFIDENCEOT_BENCH_EPSILON:-0.1}
reg_m=${CONFIDENCEOT_SOLVER_REG_M:-1.0}
uot_method=${CONFIDENCEOT_SOLVER_UOT_METHOD:-sinkhorn_stabilized}
force=${CONFIDENCEOT_SOLVER_FORCE:-0}

cd "$repo"
export PYTHONPATH="$repo/src:$repo/cancer_metastasis:$repo${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${CONFIDENCEOT_THREADS:-16}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONUNBUFFERED=1

# The manifests, in one fixed order so a resubmitted chunk covers what it
# covered before.
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
    echo "chunk $task starts at $start, past the $total manifests; nothing to do"
    exit 0
fi
stop=$(( start + chunk_size ))
(( stop > total )) && stop=$total
echo "chunk $task covers manifests [$start,$stop) of $total"
echo "label       $label"
echo "uot method  $uot_method"

failures=0
for (( position = start; position < stop; position++ )); do
    manifest=${manifests[$position]}
    level_dir=$(dirname "$manifest")
    tag=$(basename "$(dirname "$(dirname "$level_dir")")")_$(basename "$(dirname "$level_dir")")_$(basename "$level_dir")
    rows=$(python - "$manifest" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8-sig") as handle:
    print(sum(1 for _ in csv.DictReader(handle)))
PY
)
    for (( index = 0; index < rows; index++ )); do
        destination=$out/$tag/index_$index
        if [[ "$force" != "1" && -f "$destination/solver_comparison.csv" ]]; then
            echo "  $tag index=$index already scored"
            continue
        fi
        echo "===== START $tag index=$index $(date --iso-8601=seconds) ====="
        if python "$repo/benchmark/50_solver_comparison.py" \
            "$manifest" "$destination" \
            --index "$index" \
            --preprocessing "$label" \
            --confidenceot-root "$root/runs/$label" \
            --scope "$scope" \
            --epsilon "$epsilon" \
            --reg-m "$reg_m" \
            --uot-method "$uot_method"; then
            echo "===== COMPLETE $tag index=$index $(date --iso-8601=seconds) ====="
        else
            status=$?
            failures=$(( failures + 1 ))
            # To stdout as well as stderr: a failure written only to stderr
            # sits in a different file from the unit lines, and an array whose
            # tasks all "finished" has hidden one before.
            echo "===== FAILED $tag index=$index exit=$status ====="
            echo "===== FAILED $tag index=$index exit=$status =====" >&2
        fi
    done
done

echo "chunk $task finished with $failures failed pair(s)"
(( failures > 0 )) && exit 1
exit 0
