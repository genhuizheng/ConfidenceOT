#!/bin/bash
# Splatter N = 5,000 benchmark, levels L0-L5 + ablations, on TACC. The user submits; nothing runs by itself.
#
#   bash realdata_benchmark/splatter_n5000/tacc/submit_n5000.sh [--dry-run] [--pilot] [--param-set S] [--procs P] chain
#   bash realdata_benchmark/splatter_n5000/tacc/submit_n5000.sh [--dry-run] cleanup-audit
#   bash realdata_benchmark/splatter_n5000/tacc/submit_n5000.sh [--dry-run] [--pilot] generate
#   bash realdata_benchmark/splatter_n5000/tacc/submit_n5000.sh [--dry-run] [--pilot] [--after IDS] worklist
#   bash realdata_benchmark/splatter_n5000/tacc/submit_n5000.sh [--dry-run] [--pilot] [--after IDS] [--nodes N] [--hours H] run
#   bash realdata_benchmark/splatter_n5000/tacc/submit_n5000.sh [--dry-run] [--pilot] [--after IDS] collect
#   bash realdata_benchmark/splatter_n5000/tacc/submit_n5000.sh [--dry-run] [--pilot] [--after IDS] f1
#   bash realdata_benchmark/splatter_n5000/tacc/submit_n5000.sh [--pilot] status
#   bash realdata_benchmark/splatter_n5000/tacc/submit_n5000.sh [--dry-run] --param-set splatter_default --pairing paired generate-audit
#   bash realdata_benchmark/splatter_n5000/tacc/submit_n5000.sh [--dry-run] --param-set splatter_default --pairing paired [--procs P] fit
#   bash realdata_benchmark/splatter_n5000/tacc/submit_n5000.sh [--dry-run] [--after IDS] pairing-audit | case-report
#
# chain          everything in one submission: generate -> [--pairing paired: pairing audit 29_, afterok] -> worklist -> run (afterok
#                the worklist job) -> collect (afterany every run job, so failed rows are still collected and listed)
#                -> f1 (afterok collect); beside it the batch-effect audit (batch_effect_audit.slurm, afterok every
#                generation job, read only). A failed generation or worklist leaves the later jobs in
#                DependencyNeverSatisfied: cancel them. --pilot: replicate 1, the 18-row pilot worklist, 1 node, the
#                _PILOT tables (no audit).
# cleanup-audit  one short job, read-only: TACC_CLEANUP_MANIFEST.tsv of the old Splatter directories; deletes nothing
# generate       one gg job per replicate (generate_n5000.slurm): 8 levels, L0-L2 checked against the native
#                generation, conditions + truth. A generation job exits non-zero if any level fails.
# worklist       one short job: registry (from the code) + 31_build_n5000_worklist.py (--pilot: the pilot worklist)
# run            N nodes (default 4 x 6 h; pilot 1 x 3 h) of realdata_benchmark/tacc/benchmark_workers.slurm
# collect        one short job: 32_collect_n5000.py (--pilot: the _PILOT tables)
# f1             one short job: 33_f1_summary.py on the collected tables (SPLATTER_N5000_F1_ROWS / F1_SUMMARY, --pilot:
#                _PILOT); also the last job of chain
# --after IDS    colon-separated job ids this stage waits for (afterok; collect: afterany)
# generate-audit generate -> pairing audit (29_, afterok every generation job) and batch audit (27_): read them before fitting
# fit            worklist -> run -> collect -> f1 -> case report (34_) on an already generated root
# pairing-audit  one job: 29_pairing_audit.py -> ROOT/PAIRING_AUDIT_<date>/ (any root, also one generated without pairing)
# case-report    one job: 34_case_report.py -> ROOT/CASE_REPORT*.{tsv,md}
# --calibration-nulls "N ..."  calibration arms of the worklist (default within_side_split; e.g. "within_side_split
#                cross_side_rotation" fits every row under both, rotation rows' config id + |cross_side_rotation)
# --pairing P    none (default) or paired (20_generate_n5000.R --pairing: batch factors drawn at every level and set to 1
#                where the level has no batch; counts by the paired sampler); paired needs --param-set splatter_default
#                and has its own root splatter_n5000_default_paired_20261103 unless RB5_ROOT is set
# --param-set S  benchmark (default; the frozen run, root splatter_n5000_benchmark_20261101) or splatter_default (every
#                Splatter parameter at its installed default, 20_generate_n5000.R; own root splatter_n5000_default_20261103
#                unless RB5_ROOT is set; the frozen root is refused for it)
# --procs P      workers per run node (default 18 on gg, 9 on gh). Each worker runs one row; only the calibration is
#                threaded (up to 5 null fits at once), the final M4-E / M4-R fits, comparators and extras are single
#                threaded, so 18 workers keep about a quarter of a gg node busy; more workers fill it (memory about
#                2-3 GB per N = 5,000 row)
# --partition P  gg (default, every stage) or gh-dev / gh (72 cores, so 9 workers per node; gh-dev's 2 h limit clips
#                the job times and limits pilot generation to the two levels it reads, L2_depth_dropout and L4_batch_perturb)
# No set -e / -u / pipefail (TACC rules). Root is versioned; frozen roots are read only.

dry_run=0; pilot=0; nodes=""; hours=""; stage=""; after=""; partition=""; param_set=benchmark; procs_opt=""; pairing=none; nulls=within_side_split
while (( $# )); do
    case "$1" in
        --dry-run) dry_run=1; shift ;;
        --pilot) pilot=1; shift ;;
        --nodes) nodes=$2; shift 2 ;;
        --hours) hours=$2; shift 2 ;;
        --after) after=$2; shift 2 ;;
        --partition) partition=$2; shift 2 ;;
        --param-set) param_set=$2; shift 2 ;;
        --procs) procs_opt=$2; shift 2 ;;
        --pairing) pairing=$2; shift 2 ;;
        --calibration-nulls) nulls=$2; shift 2 ;;
        chain|cleanup-audit|generate|worklist|run|status|collect|f1|generate-audit|fit|pairing-audit|case-report) stage=$1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$stage" ]] || { echo "usage: bash realdata_benchmark/splatter_n5000/tacc/submit_n5000.sh [--dry-run] [--pilot] [--after IDS] [--nodes N] [--hours H] chain|cleanup-audit|generate|worklist|run|status|collect|f1" >&2; exit 2; }
repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
case $param_set in
    benchmark) root=${RB5_ROOT:-/scratch/10119/ghzheng/OT_project/splatter_n5000_benchmark_20261101} ;;
    splatter_default) if [[ $pairing == paired ]]; then root=${RB5_ROOT:-/scratch/10119/ghzheng/OT_project/splatter_n5000_default_paired_20261103}
        else root=${RB5_ROOT:-/scratch/10119/ghzheng/OT_project/splatter_n5000_default_20261103}; fi
        [[ $(basename "$root") == splatter_n5000_benchmark_20261101 ]] && { echo "the frozen root is not used for splatter_default" >&2; exit 2; } ;;
    *) echo "unknown --param-set $param_set" >&2; exit 2 ;;
esac
native=${RB5_NATIVE_ROOT:-/scratch/10119/ghzheng/OT_project/benchmark_native}
frozen_fits=${RB5_FROZEN_FITS:-/scratch/10119/ghzheng/OT_project/realdata_benchmark_20261031/splatter/acc0.99}
env_path=${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}
cap=${CONFIDENCEOT_JOB_CAP:-40}
activate="source /home1/10119/ghzheng/.bashrc; conda activate $env_path; cd $repo; export PYTHONPATH=$repo/src:$repo/cancer_metastasis:$repo/realdata_benchmark:$repo"
wl=$root/TACC_N5000_WORKLIST.tsv; pilot_flag=""
(( pilot )) && { wl=$root/TACC_N5000_PILOT_WORKLIST.tsv; pilot_flag="--pilot"; }
partition=${partition:-gg}
case $partition in
    gg) cores=144; procs=18; max_hours=48 ;;
    gh-dev) cores=72; procs=9; max_hours=2 ;;
    gh) cores=72; procs=9; max_hours=48 ;;
    *) echo "unknown partition $partition" >&2; exit 2 ;;
esac
[[ -n "$procs_opt" ]] && procs=$procs_opt
part=(-p "$partition" -c "$cores")
clip() { if (( $1 > max_hours )); then echo "$max_hours"; else echo "$1"; fi; }

submit() {  # prints the job id
    if (( dry_run )); then echo "DRY RUN  sbatch $*" | cut -c1-400 >&2; echo "DRY$RANDOM"; return 0; fi
    local id; id=$(sbatch --parsable "$@" | tail -n 1)
    [[ "$id" =~ ^[0-9]+$ ]] || { echo "sbatch returned no job id" >&2; return 1; }
    echo "$id"
}
dep_args() {  # $1 kind (afterok / afterany), $2 ids
    [[ -n "$2" ]] && echo "--dependency=$1:$2"
}
(( dry_run )) || mkdir -p "$root/logs/run" || exit 1

do_generate() {  # prints colon-joined ids
    [[ -d "$native/generated" ]] || { echo "missing $native/generated" >&2; return 1; }
    local reps=(1 2 3) ids=() rep id levels=""
    (( pilot )) && reps=(1)
    # gh-dev's 2 h limit: generate only the two levels the pilot reads; on gg the pilot generates every level of rep 1
    [[ $pilot == 1 && $partition == gh-dev ]] && levels="L2_depth_dropout L4_batch_perturb"
    for rep in "${reps[@]}"; do
        id=$(submit "${part[@]}" -t "$(printf '%02d:00:00' "$(clip 6)")" -J rb5_gen -o "$root/logs/generate_rep${rep}_%j.out" \
            --export=ALL,CONFIDENCEOT_REPO="$repo",CONFIDENCEOT_ENV="$env_path",RB5_ROOT="$root",RB5_REP="$rep",RB5_NATIVE_ROOT="$native",RB5_LEVELS="$levels",RB5_PARAM_SET="$param_set",RB5_PAIRING="$pairing" \
            "$repo/realdata_benchmark/splatter_n5000/tacc/generate_n5000.slurm") || return 1
        echo "generate rep $rep: job $id" >&2; ids+=("$id")
    done
    (IFS=:; echo "${ids[*]}")
}
do_worklist() {  # $1 ids to wait for
    [[ -f "$wl" ]] && { echo "$wl exists; not rebuilt" >&2; return 2; }
    local cmd="$activate && { [[ -f $root/PREPROCESSING_CONFIG_REGISTRY.tsv ]] || python realdata_benchmark/01_config_registry.py --out $root; } && python realdata_benchmark/splatter_n5000/31_build_n5000_worklist.py --root $root --registry $root/PREPROCESSING_CONFIG_REGISTRY.tsv --frozen-fits $frozen_fits --calibration-nulls $nulls $pilot_flag"
    local id; id=$(submit $(dep_args afterok "$1") -p "$partition" -N 1 -n 1 -t 00:30:00 -A MCB26031 -J rb5_work -o "$root/logs/worklist_%j.out" --wrap "$cmd") || return 1
    echo "worklist job $id${1:+ (after $1)}" >&2; echo "$id"
}
do_run() {  # $1 ids to wait for; the worklist may not exist yet when it is a dependency
    [[ -n "$1" || -f "$wl" ]] || { echo "missing $wl" >&2; return 1; }
    if (( pilot )); then nodes=${nodes:-1}; hours=${hours:-3}; else nodes=${nodes:-4}; hours=${hours:-6}; fi
    hours=$(clip "$hours")
    local queued; queued=$(squeue -u "$USER" -h -r 2> /dev/null | wc -l)
    (( dry_run || queued + nodes <= cap )) || { echo "$queued queued; $nodes more would pass the cap of $cap" >&2; return 3; }
    local ids=() n id
    for (( n = 0; n < nodes; n++ )); do
        id=$(submit $(dep_args afterok "$1") "${part[@]}" -t "$(printf '%02d:00:00' "$hours")" -J rb5_run -o "$root/logs/run/node_%j.out" \
            --export=ALL,CONFIDENCEOT_REPO="$repo",CONFIDENCEOT_ENV="$env_path",RB_WORKLIST="$wl",RB_REGISTRY="$root/PREPROCESSING_CONFIG_REGISTRY.tsv",RB_LOGDIR="$root/logs/run",RB_NODE="$n",RB_NODES="$nodes",RB_PROCS="$procs" \
            "$repo/realdata_benchmark/tacc/benchmark_workers.slurm") || return 1
        echo "run node $n of $nodes: job $id${1:+ (after $1)}" >&2; ids+=("$id")
    done
    (IFS=:; echo "${ids[*]}")
}
do_collect() {  # $1 ids to wait for (afterany)
    local cmd="$activate && python realdata_benchmark/splatter_n5000/32_collect_n5000.py --root $root $pilot_flag"
    local id; id=$(submit $(dep_args afterany "$1") -p "$partition" -N 1 -n 1 -t 01:00:00 -A MCB26031 -J rb5_coll -o "$root/logs/collect_%j.out" --wrap "$cmd") || return 1
    echo "collect job $id${1:+ (after $1)}" >&2; echo "$id"
}

do_f1() {  # $1 ids to wait for (afterok)
    local id; id=$(submit $(dep_args afterok "$1") -p "$partition" -N 1 -n 1 -t 00:30:00 -A MCB26031 -J rb5_f1 -o "$root/logs/f1_%j.out" \
        --wrap "$activate && python realdata_benchmark/splatter_n5000/33_f1_summary.py --root $root $pilot_flag") || return 1
    echo "F1 summary job $id${1:+ (after $1)} -> $root/SPLATTER_N5000_F1_SUMMARY${pilot_flag:+_PILOT}.tsv" >&2; echo "$id"
}
do_pairing_audit() {  # $1 ids to wait for (afterok)
    local id; id=$(submit $(dep_args afterok "$1") -p "$partition" -N 1 -n 1 -t 02:00:00 -A MCB26031 -J rb5_pair_audit -o "$root/logs/pairing_audit_%j.out" \
        --wrap "$activate && python realdata_benchmark/splatter_n5000/29_pairing_audit.py --root $root --out ${RB5_PAIR_AUDIT_OUT:-$root/PAIRING_AUDIT_$(date +%Y%m%d)}") || return 1
    echo "pairing audit job $id${1:+ (after $1)} -> $root/PAIRING_AUDIT_<date>/PAIRING_AUDIT.md" >&2; echo "$id"
}
do_case_report() {  # $1 ids to wait for (afterok)
    local id; id=$(submit $(dep_args afterok "$1") -p "$partition" -N 1 -n 1 -t 00:30:00 -A MCB26031 -J rb5_cases -o "$root/logs/case_report_%j.out" \
        --wrap "$activate && python realdata_benchmark/splatter_n5000/34_case_report.py --root $root") || return 1
    echo "case report job $id${1:+ (after $1)} -> $root/CASE_REPORT.md" >&2; echo "$id"
}
do_audit() {  # $1 ids to wait for (afterok)
    local id; id=$(submit $(dep_args afterok "$1") -o "$root/logs/batch_audit_%j.out" \
        --export=ALL,CONFIDENCEOT_REPO="$repo",CONFIDENCEOT_ENV="$env_path",RB5_ROOT="$root" \
        "$repo/realdata_benchmark/splatter_n5000/tacc/batch_effect_audit.slurm") || return 1
    echo "batch audit job $id${1:+ (after $1)} -> $root/BATCH_EFFECT_AUDIT_<date>" >&2; echo "$id"
}

case $stage in
chain)
    [[ -f "$wl" ]] && { echo "$wl exists: the chain would not rebuild it; submit run / collect on their own" >&2; exit 2; }
    g=$(do_generate) || exit 1
    if [[ $pairing == paired ]]; then  # the fit waits for a passing pairing audit (29_ exits 1 when a check fails)
        pa=$(do_pairing_audit "$g") || exit 1
        w=$(do_worklist "$pa") || exit 1
    else
        w=$(do_worklist "$g") || exit 1
    fi
    (( pilot )) || do_audit "$g" > /dev/null || exit 1
    r=$(do_run "$w") || exit 1
    c=$(do_collect "$r") || exit 1
    f=$(do_f1 "$c") || exit 1
    do_case_report "$f" > /dev/null || exit 1
    ;;
generate-audit)
    [[ $pairing == paired ]] || echo "note: generating without --pairing paired" >&2
    g=$(do_generate) || exit 1
    do_pairing_audit "$g" > /dev/null || exit 1
    do_audit "$g" > /dev/null || exit 1
    ;;
fit)
    [[ -f "$wl" ]] && { echo "$wl exists: submit run / collect on their own" >&2; exit 2; }
    w=$(do_worklist "$after") || exit 1
    r=$(do_run "$w") || exit 1
    c=$(do_collect "$r") || exit 1
    f=$(do_f1 "$c") || exit 1
    do_case_report "$f" > /dev/null || exit 1
    ;;
pairing-audit) do_pairing_audit "$after" > /dev/null || exit 1 ;;
case-report) do_case_report "$after" > /dev/null || exit 1 ;;
cleanup-audit)
    id=$(submit -p gg -N 1 -n 1 -t 01:00:00 -A MCB26031 -J rb5_audit -o "$root/logs/cleanup_audit_%j.out" \
        --wrap "bash $repo/realdata_benchmark/splatter_n5000/tacc/cleanup_audit.sh $root/TACC_CLEANUP_MANIFEST.tsv") || exit 1
    echo "cleanup audit job $id -> $root/TACC_CLEANUP_MANIFEST.tsv (read-only)"
    ;;
generate) do_generate > /dev/null || exit 1 ;;
worklist) do_worklist "$after" > /dev/null || exit 1 ;;
run) do_run "$after" > /dev/null || exit 1 ;;
collect) do_collect "$after" > /dev/null || exit 1 ;;
f1) do_f1 "$after" > /dev/null || exit 1 ;;
status)
    [[ -f "$wl" ]] || { echo "no worklist at $wl"; exit 0; }
    rows=0; done_n=0; fail_n=0
    # the out column by header name: read -a with a tab IFS merges empty fields and shifts the index
    while IFS= read -r out; do
        rows=$(( rows + 1 ))
        [[ -f "$out/SUCCESS" ]] && done_n=$(( done_n + 1 )); [[ -f "$out/FAILED" ]] && fail_n=$(( fail_n + 1 ))
    done < <(awk -F'\t' 'NR == 1 { for (i = 1; i <= NF; i++) if ($i == "out") c = i; next } { print $c }' "$wl")
    echo "$(basename "$wl"): $rows rows, $done_n done, $fail_n failed, $(( rows - done_n - fail_n )) pending"
    ;;
esac
(( dry_run )) || [[ $stage == status ]] || echo "$stage${pilot_flag:+ $pilot_flag} submitted $(date -Iseconds), commit $(git -C "$repo" rev-parse --short HEAD 2> /dev/null)" >> "$root/SUBMITTED.txt"
