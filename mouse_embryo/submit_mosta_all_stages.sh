#!/bin/bash
# Submit the all-stage MOSTA run (mouse_embryo/20-25) in two phases.
#
#   bash mouse_embryo/submit_mosta_all_stages.sh prepare --dry-run
#   bash mouse_embryo/submit_mosta_all_stages.sh prepare
#       The manifest and the common-depth equalisation (gg), then the
#       representations of every pair (gg, MOSTA_REPRESENTATION_JOBS jobs),
#       then the dense-against-blockwise check on the early pairs (gh) and,
#       only if it passes, the largest pair as a scale probe in the same job.
#       Equalisation that already finished is not redone.
#
#   bash mouse_embryo/submit_mosta_all_stages.sh run
#       Refuses unless equivalence/EQUIVALENCE_PASS exists.  Then the OT
#       workers (gh, MOSTA_OT_JOBS jobs walking one pair list, largest pairs
#       first) and, after them, the collection (gg).
#
#   bash mouse_embryo/submit_mosta_all_stages.sh collect
#       The audit, tables, pseudobulks and download list again, for instance
#       after a resubmission of run.
#
# Resubmitting a phase continues where the last one stopped: finished pairs
# are skipped, and the claims of unfinished pairs are cleared first.  That is
# also why a phase refuses while any MOSTA job is still queued or running: two
# workers would otherwise share a pair.
#
# No set -u, as everywhere in this repo.
set -eo pipefail

phase=${1:-}
[[ $# -gt 0 ]] && shift
dry_run=0
while (( $# )); do
    case "$1" in
        --dry-run) dry_run=1; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
root=${MOSTA_ROOT:-/scratch/10119/ghzheng/OT_project/mouse_embryo_results/mosta_all_stages}
data=${MOSTA_DATA:-/scratch/10119/ghzheng/OT_project/data}
representation_jobs=${MOSTA_REPRESENTATION_JOBS:-4}
ot_jobs=${MOSTA_OT_JOBS:-16}
cap=${CONFIDENCEOT_JOB_CAP:-40}
job=$repo/mouse_embryo/tacc_mosta_stage.slurm
account=MCB26031
names=mosta_prep,mosta_rep,mosta_equiv,mosta_ot,mosta_collect

case "$phase" in
    prepare|run|collect) ;;
    *) echo "usage: $0 prepare|run|collect [--dry-run]" >&2; exit 2 ;;
esac

problems=0
for path in "$job" "$repo/mouse_embryo/23_run_mosta_pair.py" "$repo/src/confidenceot/blockwise.py"; do
    if [[ ! -f "$path" ]]; then
        echo "missing: $path  (git pull in $repo first)" >&2
        problems=1
    fi
done
if [[ "$phase" == prepare && ! -d "$data" ]]; then
    echo "missing data directory: $data" >&2
    problems=1
fi
# Settings the job reads from the environment that this script sets itself.
for name in MOSTA_STAGE MOSTA_WORKERS MOSTA_THREADS MOSTA_STOP_HOURS; do
    if [[ -n "${!name}" ]]; then
        echo "set in this shell, would reach every job: $name=${!name}  (unset $name)" >&2
        problems=1
    fi
done
if (( problems && ! dry_run )); then
    exit 2
fi

active=$(squeue -u "$USER" -h -n "$names" -o %i 2> /dev/null | wc -l) || active=0
if (( active > 0 && ! dry_run )); then
    echo "$active MOSTA jobs are still queued or running (squeue -u $USER -n $names); wait for them" >&2
    exit 3
fi

submit() {
    # $1 = description, rest = sbatch arguments. Prints the job id only.
    local what=$1; shift
    if (( dry_run )); then
        echo "DRY RUN  $what" >&2
        echo "         sbatch $*" >&2
        echo "DRYRUN"
        return 0
    fi
    local id
    id=$(sbatch --parsable "$@" | tail -n 1)
    # Inside $(...) set -e does not stop the script, so a refused submission
    # has to end it here, before anything is made to wait on it.
    if [[ ! "$id" =~ ^[0-9]+$ ]]; then
        echo "sbatch returned no job id for $what; stopping. Already submitted above." >&2
        exit 1
    fi
    echo "submitted $what as $id" >&2
    echo "$id"
}

clear_claims() {
    # $1 = claim kind, $2 = output directory, $3 = marker of a finished pair.
    # Clears the claims of pairs without that marker: a job that ended mid-pair
    # leaves its claim behind, and the pair would otherwise never be retried.
    local kind=$1 outputs=$2 marker=$3 directory pair
    for directory in "$root/claims/$kind"/*; do
        [[ -d "$directory" ]] || continue
        pair=$(basename "$directory")
        [[ -f "$outputs/$pair/$marker" ]] && continue
        if (( dry_run )); then
            echo "would clear stale claim $directory" >&2
        else
            rm -rf -- "$directory"
        fi
    done
}

stage_args() {
    # $1 = stage. The environment every job of that stage gets.
    echo "--export=ALL,MOSTA_STAGE=$1,MOSTA_ROOT=$root,MOSTA_DATA=$data,CONFIDENCEOT_REPO=$repo"
}

# SLURM opens the log files before the job starts, so the directory has to
# exist before the first sbatch.
(( dry_run )) || mkdir -p "$root/logs"
queued=$(squeue -u "$USER" -h -r 2> /dev/null | wc -l) || queued=0

if [[ "$phase" == prepare ]]; then
    needed=$(( representation_jobs + 2 ))
    if (( ! dry_run && queued + needed > cap )); then
        echo "$queued jobs already queued and this adds $needed, over the cap of $cap" >&2
        exit 3
    fi
    clear_claims representation "$root/representations" REPRESENTATION_SUCCESS
    dependency=()
    if [[ -f "$root/equalisation/report.json" ]]; then
        echo "equalisation already done ($root/equalisation/report.json); not redone" >&2
    else
        prep=$(submit "prepare (manifest + common-depth equalisation)" \
            -p gg -c 72 -t 12:00:00 -A "$account" -J mosta_prep \
            -o "$root/logs/prepare_%j.out" -e "$root/logs/prepare_%j.err" \
            "$(stage_args prepare)",MOSTA_WORKERS=3,MOSTA_THREADS=16 "$job")
        dependency=(--dependency="afterok:$prep")
    fi
    representation_ids=()
    for (( i = 1; i <= representation_jobs; i++ )); do
        representation_ids+=("$(submit "representation worker $i" \
            -p gg -c 72 -t 24:00:00 -A "$account" -J mosta_rep "${dependency[@]}" \
            -o "$root/logs/representation_%j.out" -e "$root/logs/representation_%j.err" \
            "$(stage_args representation)",MOSTA_WORKERS=3,MOSTA_THREADS=16 "$job")")
    done
    after=$(IFS=:; echo "${representation_ids[*]}")
    submit "equivalence check (dense against blockwise)" \
        -p gh -c 72 -t 24:00:00 -A "$account" -J mosta_equiv --dependency="afterany:$after" \
        -o "$root/logs/equivalence_%j.out" -e "$root/logs/equivalence_%j.err" \
        "$(stage_args equivalence)",MOSTA_THREADS=16 "$job" > /dev/null
    echo "When the equivalence job ends, read $root/equivalence/summary.json; run phase 'run' only after it says PASS." >&2
    echo "The same job then runs the largest pair; its pairs/<id>/pair_metrics.csv and run.json give the time and GPU memory." >&2
fi

if [[ "$phase" == run ]]; then
    if [[ ! -f "$root/equivalence/EQUIVALENCE_PASS" ]]; then
        echo "no $root/equivalence/EQUIVALENCE_PASS: run phase prepare and read equivalence/summary.json first" >&2
        (( dry_run )) || exit 3
    fi
    needed=$(( ot_jobs + 1 ))
    if (( ! dry_run && queued + needed > cap )); then
        echo "$queued jobs already queued and this adds $needed, over the cap of $cap" >&2
        exit 3
    fi
    clear_claims ot "$root/pairs" SUCCESS
    ot_ids=()
    for (( i = 1; i <= ot_jobs; i++ )); do
        ot_ids+=("$(submit "OT worker $i" \
            -p gh -c 72 -t 48:00:00 -A "$account" -J mosta_ot \
            -o "$root/logs/ot_%j.out" -e "$root/logs/ot_%j.err" \
            "$(stage_args ot)",MOSTA_THREADS=16,MOSTA_STOP_HOURS=40 "$job")")
    done
    after=$(IFS=:; echo "${ot_ids[*]}")
    submit "collection (audit, tables, pseudobulks, download list)" \
        -p gg -c 72 -t 24:00:00 -A "$account" -J mosta_collect --dependency="afterany:$after" \
        -o "$root/logs/collect_%j.out" -e "$root/logs/collect_%j.err" \
        "$(stage_args collect)",MOSTA_THREADS=16 "$job" > /dev/null
fi

if [[ "$phase" == collect ]]; then
    submit "collection (audit, tables, pseudobulks, download list)" \
        -p gg -c 72 -t 24:00:00 -A "$account" -J mosta_collect \
        -o "$root/logs/collect_%j.out" -e "$root/logs/collect_%j.err" \
        "$(stage_args collect)",MOSTA_THREADS=16 "$job" > /dev/null
fi
