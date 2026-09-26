#!/bin/bash
# Submit the whole benchmark: generation, then OT, then scoring.
#
#   bash benchmark/tacc/submit.sh --dry-run
#   bash benchmark/tacc/submit.sh --stage container   # build the R image once
#   bash benchmark/tacc/submit.sh --sizes "1000 5000" --partition gh-dev --device cuda
#
# gg has no GPU and is usually open; gh and gh-dev have one and gh is often
# deep in backlog. Which size goes where is a judgement about the queue on the
# day, so it is an argument rather than an SBATCH line, and sbatch --partition
# overrides the directive in the script.
#   bash benchmark/tacc/submit.sh --sizes "1000" --replicates 1
#   bash benchmark/tacc/submit.sh --stage ot          # resubmit one stage
#
# Every stage goes through here, including the one-off container build, for a
# reason that cost a job already: SLURM opens the log file named in #SBATCH -o
# before the script runs, so a script cannot create its own log directory. The
# directory is made here, before anything is submitted.
#
# The chain is held together by afterok dependencies, so a failed generation
# stops the OT rather than running it against files that are not there.
#
# Two limits shape the shape of it. Array tasks count individually against the
# 40-job submission cap, so the OT stage is chunked: each task walks a
# contiguous run of the work list instead of one unit. And the work list is
# written here rather than inside a job, so every task reads the same numbering
# and a resubmitted chunk covers exactly what it covered before.
#
# Resume is the default everywhere. Generation skips a replicate that carries
# its construction flag, the OT stage skips a unit whose SUCCESS file exists,
# and scoring reports what is missing instead of refusing to run.

set -eo pipefail

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
root=${CONFIDENCEOT_BENCH_ROOT:-/scratch/10119/ghzheng/OT_project/benchmark_technical}
sizes=${CONFIDENCEOT_BENCH_SIZES:-1000 5000 10000}
replicates=${CONFIDENCEOT_BENCH_REPLICATES:-5}
chunk=${CONFIDENCEOT_BENCH_CHUNK:-24}
stage=all
dry_run=0
partition=""
device=""

# The twelve configurations the production Preprocessing can express. The
# remaining four cells of the factorial put library-size normalisation and
# rank encoding on together, which is one slot in the production code and
# cannot be named; they are a bit-for-bit identity with rank alone and are
# asserted in tests/test_benchmark_identity.py rather than run here.
preprocessing=${CONFIDENCEOT_BENCH_PREPROCESSING:-"\
raw raw_noscale raw_cos raw_noscale_cos \
logcpm logcpm_noscale logcpm_cos logcpm_noscale_cos \
ranknm256 ranknm256_noscale ranknm256_cos ranknm256_noscale_cos"}

while (( $# )); do
    case "$1" in
        --dry-run) dry_run=1; shift ;;
        --stage) stage=$2; shift 2 ;;
        --sizes) sizes=$2; shift 2 ;;
        --replicates) replicates=$2; shift 2 ;;
        --chunk) chunk=$2; shift 2 ;;
        --preprocessing) preprocessing=$2; shift 2 ;;
        --partition) partition=$2; shift 2 ;;
        --device) device=$2; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

size_array=($sizes)
preprocessing_array=($preprocessing)
# A dry run must not touch the filesystem, so that it can be read on a laptop
# where $root does not exist.
(( dry_run )) || mkdir -p "$root/logs"

queued_now() { squeue -u "$USER" -h -r 2> /dev/null | wc -l; }

# Empty unless asked for, so an unset option changes nothing about the job.
where=()
[[ -n "$partition" ]] && where=(--partition="$partition")
device_export=""
[[ -n "$device" ]] && device_export=",CONFIDENCEOT_BENCH_DEVICE=$device"

submit() {
    # $1 = description, rest = sbatch arguments. Prints the job id.
    local description=$1; shift
    if (( dry_run )); then
        echo "DRY RUN  $description" >&2
        echo "         sbatch $*" >&2
        echo "DRYRUN"
        return 0
    fi
    local id
    id=$(sbatch --parsable "$@" | tail -n 1)
    echo "submitted $description as $id" >&2
    echo "$id"
}

# ---- stage 0: the container, when that is the R route -----------------------
if [[ "$stage" == "container" ]]; then
    submit "container build" "${where[@]}" \
        --export=ALL,CONFIDENCEOT_BENCH_ROOT="$root",CONFIDENCEOT_REPO="$repo" \
        "$repo/benchmark/tacc/build_container.slurm" > /dev/null
    echo
    echo "root        $root"
    (( dry_run )) && echo "nothing was submitted (--dry-run)"
    exit 0
fi

# ---- stage 1: generation and construction -----------------------------------
# One task per (size, replicate, technical level): the levels are
# independent realizations, so each is generated on its own.
levels=3
generation_tasks=$(( ${#size_array[@]} * replicates * levels ))
generation_id=""
if [[ "$stage" == "all" || "$stage" == "generate" ]]; then
    required=$(( generation_tasks + 2 ))
    if (( dry_run == 0 )) && (( $(queued_now) + required > 40 )); then
        echo "would exceed the 40-job submission limit: $(queued_now) queued, "\
             "$required more wanted" >&2
        exit 3
    fi
    generation_id=$(CONFIDENCEOT_BENCH_SIZES="$sizes" \
        CONFIDENCEOT_BENCH_REPLICATES="$replicates" \
        submit "generation (${generation_tasks} tasks)" \
            --array=0-$(( generation_tasks - 1 )) \
            "${where[@]}" \
            --export=ALL,CONFIDENCEOT_BENCH_SIZES="$sizes",CONFIDENCEOT_BENCH_REPLICATES="$replicates",CONFIDENCEOT_BENCH_ROOT="$root",CONFIDENCEOT_REPO="$repo" \
            "$repo/benchmark/tacc/generate.slurm")
fi

# ---- stage 2: number the work, then solve -----------------------------------
# The unit count is deterministic: three biological cases per generated
# technical condition, times the configurations.
worklist=$root/worklist.csv
pairs=$(( ${#size_array[@]} * replicates * levels * 3 ))
units=$(( pairs * ${#preprocessing_array[@]} ))
chunks=$(( (units + chunk - 1) / chunk ))
(( chunks < 1 )) && chunks=1

list_id=""
ot_id=""
if [[ "$stage" == "all" || "$stage" == "ot" ]]; then
    required=$(( chunks + 3 ))
    if (( dry_run == 0 )) && (( $(queued_now) + required > 40 )); then
        echo "the OT stage wants $chunks chunks plus the work list, which "\
             "would exceed the 40-job limit; raise --chunk" >&2
        exit 3
    fi

    depend=()
    [[ -n "$generation_id" && "$generation_id" != "DRYRUN" ]] && \
        depend=(--dependency=afterok:"$generation_id")
    list_id=$(submit "work list ($units units)" \
        "${depend[@]}" "${where[@]}" \
        --export=ALL,CONFIDENCEOT_BENCH_ROOT="$root",CONFIDENCEOT_REPO="$repo",CONFIDENCEOT_BENCH_WORKLIST="$worklist",CONFIDENCEOT_BENCH_PREPROCESSING="$preprocessing" \
        "$repo/benchmark/tacc/worklist.slurm")

    depend=()
    [[ -n "$list_id" && "$list_id" != "DRYRUN" ]] && \
        depend=(--dependency=afterok:"$list_id")
    ot_id=$(submit "OT ($chunks chunks of $chunk units)" \
        --array=0-$(( chunks - 1 )) \
        "${depend[@]}" "${where[@]}" \
        --export=ALL,CONFIDENCEOT_BENCH_ROOT="$root",CONFIDENCEOT_REPO="$repo",CONFIDENCEOT_BENCH_CHUNK="$chunk",CONFIDENCEOT_BENCH_WORKLIST="$worklist"$device_export \
        "$repo/benchmark/tacc/ot_array.slurm")
fi

# ---- stage 3: scoring -------------------------------------------------------
if [[ "$stage" == "all" || "$stage" == "metrics" ]]; then
    depend=()
    [[ -n "$ot_id" && "$ot_id" != "DRYRUN" ]] && \
        depend=(--dependency=afterany:"$ot_id")
    submit "scoring" "${depend[@]}" "${where[@]}" \
        --export=ALL,CONFIDENCEOT_BENCH_ROOT="$root",CONFIDENCEOT_REPO="$repo" \
        "$repo/benchmark/tacc/metrics.slurm" > /dev/null
fi

echo
echo "root        $root"
echo "sizes       $sizes"
echo "replicates  $replicates"
echo "preprocess  ${#preprocessing_array[@]} configurations"
echo "chunk       $chunk units per array task"
echo "partition   ${partition:-the default in each script (gg)}"
echo "device      ${device:-the default in ot_array.slurm (cpu)}"
(( dry_run )) && echo "nothing was submitted (--dry-run)"
