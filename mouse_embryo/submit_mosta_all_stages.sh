#!/bin/bash
# Submit the all-stage MOSTA run (mouse_embryo/20-25).
#
#   bash mouse_embryo/submit_mosta_all_stages.sh prepare [--dry-run]
#       The manifest and the common-depth equalisation, then the
#       representations of every pair, then the equivalence check.  A step
#       whose output is already complete is not redone.
#
#   bash mouse_embryo/submit_mosta_all_stages.sh equivalence [--dry-run]
#       The dense-against-blockwise check on the early pairs, on its own.
#       Earlier equivalence output is cleared first.
#
#   bash mouse_embryo/submit_mosta_all_stages.sh run [--dry-run]
#       Refuses unless equivalence/EQUIVALENCE_PASS exists for the device the
#       run uses.  With MOSTA_GATE=float64-identity (cuda only) it instead
#       accepts a recorded check in which every criterion-A (float64)
#       comparison passed on all 52 test pairs, the configuration the run uses.  Then the OT workers (MOSTA_OT_JOBS jobs walking one pair
#       list, largest pairs first) and, after them, the collection.
#
#   bash mouse_embryo/submit_mosta_all_stages.sh collect [--dry-run]
#       The audit, tables, pseudobulks and download list again, for instance
#       after a resubmission of run.
#
# MOSTA_DEVICE picks the device for the equivalence check and the run; one run
# uses one device for every pair.
#   cuda (default, the production run): gh nodes, the GPU production path
#                  (torch CUDA, float64).
#   cpu:           gg nodes, for validation and reference comparisons only:
#                  the CPU production path (NumPy reference calibration, torch
#                  solvers in float64), MOSTA_THREADS 144.
# Everything else runs on gg.
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
device=${MOSTA_DEVICE:-cuda}
representation_jobs=${MOSTA_REPRESENTATION_JOBS:-4}
equivalence_shards=${MOSTA_EQUIVALENCE_SHARDS:-8}
cap=${CONFIDENCEOT_JOB_CAP:-40}
job=$repo/mouse_embryo/tacc_mosta_stage.slurm
account=MCB26031
names=mosta_prep,mosta_rep,mosta_equiv,mosta_equivm,mosta_ot,mosta_collect
expected_pairs=464

case "$phase" in
    prepare|equivalence|run|collect) ;;
    *) echo "usage: $0 prepare|equivalence|run|collect [--dry-run]" >&2; exit 2 ;;
esac
case "$device" in
    cpu) compute=(-p gg -c 144); threads=144; ot_jobs=${MOSTA_OT_JOBS:-30} ;;
    cuda) compute=(-p gh -c 72); threads=16; ot_jobs=${MOSTA_OT_JOBS:-16} ;;
    *) echo "MOSTA_DEVICE must be cpu or cuda; got '$device'" >&2; exit 2 ;;
esac
cpu_node=(-p gg -c 144)

problems=0
for path in "$job" "$repo/mouse_embryo/23_run_mosta_pair.py" "$repo/src/confidenceot/blockwise.py"; do
    if [[ ! -f "$path" ]]; then
        echo "missing: $path  (git pull in $repo first)" >&2
        problems=1
    fi
done
if ! grep -q "equivalence_merge" "$job" 2> /dev/null; then
    echo "$job is older than this script: git pull in $repo first" >&2
    problems=1
fi
if [[ "$phase" == prepare && ! -d "$data" ]]; then
    echo "missing data directory: $data" >&2
    problems=1
fi
# Settings the job reads from the environment that this script sets itself.
for name in MOSTA_STAGE MOSTA_WORKERS MOSTA_THREADS MOSTA_STOP_HOURS MOSTA_SHARD MOSTA_N_SHARDS; do
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
    echo "$active MOSTA jobs are still queued or running (squeue -u $USER -n $names); wait for them or scancel them" >&2
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
    echo "--export=ALL,MOSTA_STAGE=$1,MOSTA_ROOT=$root,MOSTA_DATA=$data,CONFIDENCEOT_REPO=$repo,MOSTA_DEVICE=$device"
}

check_cap() {
    # $1 = jobs this phase adds.
    local queued
    queued=$(squeue -u "$USER" -h -r 2> /dev/null | wc -l) || queued=0
    if (( ! dry_run && queued + $1 > cap )); then
        echo "$queued jobs already queued and this adds $1, over the cap of $cap" >&2
        exit 3
    fi
}

submit_equivalence() {
    # $@ = optional dependency argument. Clears the previous check's output,
    # then submits the check for $device.
    local dependency=("$@") ids=() shard after
    if (( dry_run )); then
        echo "would clear $root/equivalence/{report.csv,summary.json,EQUIVALENCE_PASS,EQUIVALENCE_FAIL,shards}" >&2
    else
        rm -rf -- "$root/equivalence/shards"
        rm -f -- "$root/equivalence/report.csv" "$root/equivalence/summary.json" \
            "$root/equivalence/EQUIVALENCE_PASS" "$root/equivalence/EQUIVALENCE_FAIL"
    fi
    if [[ "$device" == cuda ]]; then
        submit "equivalence check (gh, then the largest pair as a scale probe)" \
            "${compute[@]}" -t 24:00:00 -A "$account" -J mosta_equiv "${dependency[@]}" \
            -o "$root/logs/equivalence_%j.out" -e "$root/logs/equivalence_%j.err" \
            "$(stage_args equivalence)",MOSTA_THREADS=$threads "$job" > /dev/null
        return 0
    fi
    for (( shard = 0; shard < equivalence_shards; shard++ )); do
        ids+=("$(submit "equivalence shard $shard of $equivalence_shards (gg)" \
            "${compute[@]}" -t 24:00:00 -A "$account" -J mosta_equiv "${dependency[@]}" \
            -o "$root/logs/equivalence_%j.out" -e "$root/logs/equivalence_%j.err" \
            "$(stage_args equivalence)",MOSTA_THREADS=$threads,MOSTA_SHARD=$shard,MOSTA_N_SHARDS=$equivalence_shards "$job")")
    done
    after=$(IFS=:; echo "${ids[*]}")
    submit "equivalence merge" \
        "${cpu_node[@]}" -t 01:00:00 -A "$account" -J mosta_equivm --dependency="afterany:$after" \
        -o "$root/logs/equivalence_merge_%j.out" -e "$root/logs/equivalence_merge_%j.err" \
        "$(stage_args equivalence_merge)",MOSTA_THREADS=16 "$job" > /dev/null
}

equivalence_jobs() {
    if [[ "$device" == cuda ]]; then echo 1; else echo $(( equivalence_shards + 1 )); fi
}

# SLURM opens the log files before the job starts, so the directory has to
# exist before the first sbatch.
(( dry_run )) || mkdir -p "$root/logs"

if [[ "$phase" == prepare ]]; then
    check_cap $(( representation_jobs + 1 + $(equivalence_jobs) ))
    dependency=()
    if [[ -f "$root/equalisation/report.json" ]]; then
        echo "equalisation already done ($root/equalisation/report.json); not redone" >&2
    else
        prep=$(submit "prepare (manifest + common-depth equalisation)" \
            "${cpu_node[@]}" -t 12:00:00 -A "$account" -J mosta_prep \
            -o "$root/logs/prepare_%j.out" -e "$root/logs/prepare_%j.err" \
            "$(stage_args prepare)",MOSTA_WORKERS=3,MOSTA_THREADS=16 "$job")
        dependency=(--dependency="afterok:$prep")
    fi
    done_representations=$(ls "$root"/representations/*/REPRESENTATION_SUCCESS 2> /dev/null | wc -l) || done_representations=0
    if (( ${#dependency[@]} == 0 && done_representations >= expected_pairs )); then
        echo "all $done_representations representations already done; not redone" >&2
        submit_equivalence
    else
        clear_claims representation "$root/representations" REPRESENTATION_SUCCESS
        representation_ids=()
        for (( i = 1; i <= representation_jobs; i++ )); do
            representation_ids+=("$(submit "representation worker $i" \
                "${cpu_node[@]}" -t 24:00:00 -A "$account" -J mosta_rep "${dependency[@]}" \
                -o "$root/logs/representation_%j.out" -e "$root/logs/representation_%j.err" \
                "$(stage_args representation)",MOSTA_WORKERS=3,MOSTA_THREADS=16 "$job")")
        done
        after=$(IFS=:; echo "${representation_ids[*]}")
        submit_equivalence --dependency="afterany:$after"
    fi
    echo "When the equivalence check ends, read $root/equivalence/summary.json; submit run only after it says PASS." >&2
fi

if [[ "$phase" == equivalence ]]; then
    check_cap "$(equivalence_jobs)"
    submit_equivalence
    echo "When it ends, read $root/equivalence/summary.json; submit run only after it says PASS." >&2
fi

float64_identity_passed() {
    # True when the recorded check has every criterion-A comparison (blockwise
    # against dense, both float64, on a float64 cost) passing for every test
    # pair: the comparison of the configuration the run uses since the CUDA
    # solver went to float64. The record and its criteria are read, not changed.
    local report=$root/equivalence/report.csv
    [[ -f "$report" ]] || return 1
    awk -F, 'NR==1{for(i=1;i<=NF;i++)c[$i]=i; next}
        $c["outcome"]=="missing representation"{bad++}
        $c["criterion"]=="A"{a++; if($c["outcome"]!="pass") bad++; pairs[$c["pair_id"]]=1}
        END{n=0; for(p in pairs) n++; print "criterion A comparisons " a+0 ", failing or missing " bad+0 ", pairs " n > "/dev/stderr"; exit !(a>0 && bad==0 && n>=52)}' "$report"
}

if [[ "$phase" == run ]]; then
    if [[ ! -f "$root/equivalence/EQUIVALENCE_PASS" && "$device" == cuda && "${MOSTA_GATE:-}" == float64-identity ]] && float64_identity_passed; then
        echo "gate: every float64 comparison in the recorded equivalence check passed (MOSTA_GATE=float64-identity)" >&2
        (( dry_run )) || printf 'run gated on criterion A of %s, %s\n' "$root/equivalence/report.csv" "$(date --iso-8601=seconds)" > "$root/equivalence/RUN_GATE_FLOAT64_IDENTITY"
    elif [[ ! -f "$root/equivalence/EQUIVALENCE_PASS" ]]; then
        echo "no $root/equivalence/EQUIVALENCE_PASS: run the equivalence check and read equivalence/summary.json first" >&2
        (( dry_run )) || exit 3
    elif grep -q '"devices"' "$root/equivalence/summary.json" 2> /dev/null; then
        if ! grep -q "\"$device\"" "$root/equivalence/summary.json"; then
            echo "the equivalence check that passed did not run on $device; check $device first (MOSTA_DEVICE=$device ... equivalence)" >&2
            (( dry_run )) || exit 3
        fi
    elif [[ "$device" != cuda ]]; then
        # A summary without "devices" comes from 2b5f3bb, whose check ran on the GPU only.
        echo "the equivalence check that passed ran on cuda; check $device first" >&2
        (( dry_run )) || exit 3
    fi
    check_cap $(( ot_jobs + 1 ))
    clear_claims ot "$root/pairs" SUCCESS
    ot_ids=()
    for (( i = 1; i <= ot_jobs; i++ )); do
        ot_ids+=("$(submit "OT worker $i ($device)" \
            "${compute[@]}" -t 48:00:00 -A "$account" -J mosta_ot \
            -o "$root/logs/ot_%j.out" -e "$root/logs/ot_%j.err" \
            "$(stage_args ot)",MOSTA_THREADS=$threads,MOSTA_STOP_HOURS=40 "$job")")
    done
    after=$(IFS=:; echo "${ot_ids[*]}")
    submit "collection (audit, tables, pseudobulks, download list)" \
        "${cpu_node[@]}" -t 24:00:00 -A "$account" -J mosta_collect --dependency="afterany:$after" \
        -o "$root/logs/collect_%j.out" -e "$root/logs/collect_%j.err" \
        "$(stage_args collect)",MOSTA_THREADS=16 "$job" > /dev/null
fi

if [[ "$phase" == collect ]]; then
    check_cap 1
    submit "collection (audit, tables, pseudobulks, download list)" \
        "${cpu_node[@]}" -t 24:00:00 -A "$account" -J mosta_collect \
        -o "$root/logs/collect_%j.out" -e "$root/logs/collect_%j.err" \
        "$(stage_args collect)",MOSTA_THREADS=16 "$job" > /dev/null
fi
