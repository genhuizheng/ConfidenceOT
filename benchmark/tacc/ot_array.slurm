#!/bin/bash
#SBATCH -J cot_bench_ot
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 72
#SBATCH -t 24:00:00
#SBATCH -A MCB26031
#SBATCH -o /scratch/10119/ghzheng/OT_project/benchmark_technical/logs/ot_%A_%a.out
#SBATCH -e /scratch/10119/ghzheng/OT_project/benchmark_technical/logs/ot_%A_%a.err
#
# One array task runs a contiguous chunk of the work list through the
# production solver. Chunking rather than one task per pair: the list is
# pairs x preprocessing configurations, which reaches the thousands, and the
# submission limit is 40.
#
# The work list is the joined manifest, written once by the submission helper,
# so a task never has to reconstruct which pair is which.
#
# Resume is by SUCCESS flag, which 02_run_pair.py writes itself. A task that
# finds one skips that unit, so a failed chunk can be resubmitted alone and
# will redo only what is missing.
#
# No rejection-budget flags. Passing target_rejection_budget 0.00 makes that
# side structurally unable to reject -- _cpu_uot.py:298 keeps a zero budget
# binding whether or not the floor is enforced -- and the emerged case needs
# the target to reject. The defaults are 0.95 on both sides, and any non-zero
# budget with the floor off leaves the gate unconstrained.

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
worklist=${CONFIDENCEOT_BENCH_WORKLIST:-$root/worklist.csv}
chunk=${SLURM_ARRAY_TASK_ID:?Submit as a chunk array}
chunk_size=${CONFIDENCEOT_BENCH_CHUNK:-24}
# CPU by default: the GPU queue here is a long wait for a solve that
# runs on cores anyway. Null calibration fits ten matrices per cost on
# the grid, which is the expensive part and the part --workers
# parallelises, so the core count and the workers go together.
device=${CONFIDENCEOT_BENCH_DEVICE:-cpu}
workers=${CONFIDENCEOT_BENCH_WORKERS:-8}
force=${CONFIDENCEOT_BENCH_FORCE:-0}

total=$(( $(wc -l < "$worklist") - 1 ))
start=$(( chunk * chunk_size ))
if (( start >= total )); then
    echo "chunk $chunk starts at $start, past the $total units; nothing to do"
    exit 0
fi
stop=$(( start + chunk_size ))
(( stop > total )) && stop=$total
echo "chunk $chunk covers units [$start,$stop) of $total"

for (( unit = start; unit < stop; unit++ )); do
    line=$(( unit + 2 ))
    IFS=',' read -r manifest pair_index pair_id preprocessing <<< \
        "$(sed -n "${line}p" "$worklist" | cut -d',' -f1-4)"
    output=$root/runs/$preprocessing
    if [[ "$force" != "1" && -f "$output/$pair_id/scope_all/SUCCESS" ]]; then
        echo "  unit $unit  $pair_id  $preprocessing  already complete"
        continue
    fi
    echo "  unit $unit  $pair_id  $preprocessing"
    # The raw levels of the factorial resolve to the 'precomputed' transform,
    # which 02_run_pair.py refuses unless asked, because an unrecognised stem
    # resolves to it too and a typo must not silently run the counts as the
    # representation. Asking here is deliberate; a typo still fails.
    allow=()
    [[ "$preprocessing" == raw* ]] && allow=(--allow-precomputed)
    # The manifest and the output root are positional, and the row is
    # --index. There are no --manifest, --output-root or --pair-index flags.
    python "$repo/cancer_metastasis/02_run_pair.py" \
        "$manifest" "$output" \
        "${allow[@]}" \
        --index "$pair_index" \
        --analysis-scope all \
        --preprocessing "$preprocessing" \
        --calibration-null within_side_split \
        --epsilon "${CONFIDENCEOT_BENCH_EPSILON:-0.1}" \
        --lambda-a 1.0 --lambda-b 1.0 \
        --tolerance 1e-4 \
        --save-coupling --save-pairing-edges \
        --device "$device" \
        --workers "$workers" \
        --seed "${CONFIDENCEOT_BENCH_SEED:-20260925}" \
        || echo "  unit $unit FAILED; leaving it for a resubmission"
done

echo "chunk $chunk finished"
