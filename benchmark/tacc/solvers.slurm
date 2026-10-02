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
# The units are the OT stage's own: the rows of its work list that carry the
# locked label, each with the manifest and the pair index ConfidenceOT was run
# on. Walking the condition manifests instead would hand a _ds label the
# original counts rather than the equalised ones under conditions_ds/, and
# the positions there need not match: the rewritten manifest drops a pair
# whose sample had no analysed cells. Either way the solver comparison would
# rebuild a different cost matrix from the one ConfidenceOT saw, and its
# geometry check would refuse every pair.
#
# One array task walks a contiguous run of those units, so the task count
# stays inside the submission cap. The label is a parameter: this stage runs
# after the preprocessing grid has settled on one, and it must be the same one
# for every task or the comparison is between configurations again.
#
# Partial OT's transported mass is swept inside the runner, so one pair
# produces several Partial OT rows. That is deliberate -- m decides that
# method's whole result, and a single value chosen by us is a value chosen
# with the answer in view.
#
# The #SBATCH log lines point at the technical root; for another root pass -o
# and -e on the sbatch line, which override them.
#
# No set -u. TACC's module files and conda's shell functions read variables
# they do not define, and under set -u that kills the job in seconds with
# nothing in the log. Every expansion below carries its own default.
set -eo pipefail
source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
root=${CONFIDENCEOT_BENCH_ROOT:-/scratch/10119/ghzheng/OT_project/benchmark_technical}
worklist=${CONFIDENCEOT_BENCH_WORKLIST:-$root/worklist_1000_5000_10000.csv}
label=${CONFIDENCEOT_SOLVER_LABEL:?Set CONFIDENCEOT_SOLVER_LABEL to the locked configuration}
# 1 runs only Oracle-budget Partial OT (50_solver_comparison.py
# --oracle-budget), into its own tree so it can never be read as one of the
# usable methods.
oracle_budget=${CONFIDENCEOT_SOLVER_ORACLE_BUDGET:-0}
mode_args=()
if [[ "$oracle_budget" == "1" ]]; then
    mode_args=(--oracle-budget)
    out=${CONFIDENCEOT_SOLVER_ROOT:-$root/solvers_oracle_budget/$label}
else
    out=${CONFIDENCEOT_SOLVER_ROOT:-$root/solvers/$label}
fi
# Empty means every size in the work list; "1000 5000" runs those alone, so
# the small sizes can come back before the N = 10,000 pairs are queued.
sizes=${CONFIDENCEOT_SOLVER_SIZES:-}
scope=${CONFIDENCEOT_BENCH_SCOPE:-all}
chunk_size=${CONFIDENCEOT_SOLVER_CHUNK:-6}
# The seed the OT stage ran with, under the same name and default as in
# ot_array.slurm. The PCA and the cost scale are drawn from it.
seed=${CONFIDENCEOT_BENCH_SEED:-20260925}
epsilon=${CONFIDENCEOT_BENCH_EPSILON:-0.1}
reg_m=${CONFIDENCEOT_SOLVER_REG_M:-1.0}
uot_method=${CONFIDENCEOT_SOLVER_UOT_METHOD:-sinkhorn_stabilized}
# Partial OT's transported mass, one or several values. Empty runs the
# runner's own sweep; "0.85" is the single value of the earlier Splatter
# comparison, 1 minus its 15% rejection budget.
partial_mass=${CONFIDENCEOT_SOLVER_PARTIAL_MASS:-}
mass_args=()
for value in $partial_mass; do mass_args+=(--partial-mass "$value"); done
force=${CONFIDENCEOT_SOLVER_FORCE:-0}

cd "$repo"
export PYTHONPATH="$repo/src:$repo/cancer_metastasis:$repo${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${CONFIDENCEOT_THREADS:-16}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONUNBUFFERED=1

if [[ ! -f "$worklist" ]]; then
    echo "no work list at $worklist; the OT stage writes it" >&2
    exit 2
fi

# The label's units, in the work list's own order, so a resubmitted chunk
# covers what it covered before. One line each: manifest, pair index, pair id.
units_file=$(mktemp)
trap 'rm -f "$units_file"' EXIT
python - "$worklist" "$label" "$sizes" > "$units_file" <<'PY'
import sys

import pandas as pd

worklist, label, sizes = sys.argv[1], sys.argv[2], sys.argv[3]
table = pd.read_csv(worklist)
table = table[table["preprocessing"].astype(str) == label]
if sizes.split():
    # pair_id is N{N}_rep{r}_{level}_{case}, as 20_build_conditions.py names it.
    wanted = {f"N{size}" for size in sizes.split()}
    table = table[table["pair_id"].astype(str).str.split("_").str[0].isin(wanted)]
for row in table.itertuples(index=False):
    print(f"{row.manifest},{int(row.pair_index)},{row.pair_id}")
PY
total=$(wc -l < "$units_file")
if (( total == 0 )); then
    echo "the work list has no units for $label${sizes:+ at sizes $sizes}" >&2
    exit 2
fi

task=${SLURM_ARRAY_TASK_ID:?Submit as a chunk array}
start=$(( task * chunk_size ))
if (( start >= total )); then
    echo "chunk $task starts at $start, past the $total units; nothing to do"
    exit 0
fi
stop=$(( start + chunk_size ))
(( stop > total )) && stop=$total
echo "chunk $task covers units [$start,$stop) of $total"
echo "label       $label"
echo "work list   $worklist"
echo "seed        $seed"
echo "uot method  $uot_method"
echo "partial m   ${partial_mass:-the default sweep}"

failures=0
for (( unit = start; unit < stop; unit++ )); do
    IFS=',' read -r manifest pair_index pair_id <<< "$(sed -n "$(( unit + 1 ))p" "$units_file")"
    destination=$out/$pair_id
    if [[ "$force" != "1" && -f "$destination/solver_comparison.csv" ]]; then
        echo "  $pair_id already scored"
        continue
    fi
    echo "===== START $pair_id $(date --iso-8601=seconds) ====="
    if python "$repo/benchmark/50_solver_comparison.py" \
        "$manifest" "$destination" \
        --index "$pair_index" \
        --preprocessing "$label" \
        --confidenceot-root "$root/runs/$label" \
        --scope "$scope" \
        --seed "$seed" \
        --epsilon "$epsilon" \
        --reg-m "$reg_m" \
        --uot-method "$uot_method" \
        "${mass_args[@]}" "${mode_args[@]}"; then
        echo "===== COMPLETE $pair_id $(date --iso-8601=seconds) ====="
    else
        status=$?
        failures=$(( failures + 1 ))
        # To stdout as well as stderr: a failure written only to stderr
        # sits in a different file from the unit lines, and an array whose
        # tasks all "finished" has hidden one before.
        echo "===== FAILED $pair_id exit=$status ====="
        echo "===== FAILED $pair_id exit=$status =====" >&2
    fi
done

echo "chunk $task finished with $failures failed pair(s)"
(( failures > 0 )) && exit 1
exit 0
