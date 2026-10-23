#!/bin/bash
# The method comparison with log1p in place of raw (user, 2026-10-22: raw is discarded from every
# heatmap; log1p takes its place). log1p is the code label `logcpm`: raw counts -> library size
# 1e4 -> log1p -> 2,000 HVGs -> scaled -> PCA 30, squared Euclidean cost; the 32-arm grid ran it
# at the runner default, so runs/logcpm (acceptance 0.90) already exists and is not rerun.
#
#   bash benchmark/tacc/submit_log1p_comparison.sh --dry-run
#   bash benchmark/tacc/submit_log1p_comparison.sh
#
# What runs, all on the 180 logcpm pairs of the native work list, dealt round robin by size:
#   1. ConfidenceOT at within-side acceptance 0.95 and 0.99 (ot_array.slurm, 9 tasks x 20 each)
#      -> runs_acceptance_<a>/logcpm, beside the three arms already there
#   2. the comparators (solvers.slurm, 4 tasks x 45 each, three passes):
#      default (Vanilla UOT, Partial OT sweep, Balanced OT) -> solvers/logcpm
#      --oracle-budget (Partial OT, truth-derived m)          -> solvers_oracle_budget/logcpm
#      IC-POT at c = 0.5                                       -> solvers_icpot/logcpm
#      with solvers.slurm's defaults (epsilon 0.1, reg_m 1.0, sinkhorn_stabilized); the dry run
#      prints the provenance of an existing raw comparator run so those can be checked first
#   3. after all of them: 56_comparator_metrics.py on ranknm256_noscale_ds_cos,
#      ranknm256_noscale_cos and logcpm -> comparator_metrics_log1p/, the 57_ audit, a tarball.
# Nothing existing is written: any of those logcpm directories, or the output, already there
# stops the submission. 31 array tasks + 1 job; refused over the 40-task cap.
#
# No set -u, as everywhere in this repo.
set -eo pipefail

dry_run=0
[[ "${1:-}" == --dry-run ]] && dry_run=1

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
root=${CONFIDENCEOT_BENCH_ROOT:-/scratch/10119/ghzheng/OT_project/benchmark_native}
env_path=${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}
arm=logcpm
source_list=$root/worklist_1000_5000_10000.csv
worklist=$root/worklist_${arm}_balanced.csv
out=$root/comparator_metrics_log1p
logs=$root/logs
blocks=9
chunk=20
solver_chunk=45
acceptances="0.95 0.99"

problems=0
for path in "$source_list" "$root/runs/$arm" "$root/solvers/raw"; do
    [[ -e "$path" ]] || { echo "missing: $path" >&2; problems=1; }
done
for d in runs_acceptance_0.95/$arm runs_acceptance_0.99/$arm solvers/$arm solvers_oracle_budget/$arm solvers_icpot/$arm; do
    [[ -e "$root/$d" ]] && { echo "$root/$d exists; not rewritten" >&2; problems=1; }
done
[[ -e "$out" ]] && { echo "$out exists; not rewritten" >&2; problems=1; }

# The logcpm rows, sorted by size and dealt round robin, as submit_log1p_pcc.sh does; awk on the
# login node, not Python.
rows=$(awk -F',' -v arm="$arm" -v blocks="$blocks" '
    NR == 1 { next }
    $4 == arm {
        size = $3; sub(/^N/, "", size); sub(/_.*/, "", size)
        n[size]++; row[size, n[size]] = $0
    }
    END {
        split("1000 5000 10000", order, " ")
        total = 0
        for (k = 1; k <= 3; k++) for (i = 1; i <= n[order[k]]; i++) all[++total] = row[order[k], i]
        for (b = 0; b < blocks; b++) for (i = b + 1; i <= total; i += blocks) print all[i]
    }' "$source_list")
count=$(printf '%s\n' "$rows" | grep -c . || true)
[[ "$count" -eq 180 ]] || { echo "expected 180 $arm rows in $source_list, found $count" >&2; problems=1; }

queued=$( { squeue -u "$USER" -h -r 2> /dev/null || true; } | wc -l )
wanted=$(( 2 * blocks + 3 * 4 + 1 ))
(( ! dry_run && queued + wanted > 40 )) && { echo "$queued tasks queued; $wanted more would pass the 40-task cap" >&2; problems=1; }
(( problems && ! dry_run )) && exit 2

if (( dry_run )); then
    echo "DRY RUN  $count units -> $worklist"
    printf '%s\n' "$rows" | head -n 2 | cut -c1-200
    example=$(ls -d "$root"/solvers/raw/*/ 2> /dev/null | head -n 1)
    echo "== an existing raw comparator run, to check the solver settings against solvers.slurm's defaults: $example"
    [[ -n "$example" ]] && head -c 2500 "$example/provenance.json" && echo
else
    mkdir -p "$logs"
    { head -n 1 "$source_list"; printf '%s\n' "$rows"; } > "$worklist.tmp"
    if [[ -f "$worklist" ]] && ! cmp -s "$worklist" "$worklist.tmp"; then
        echo "$worklist exists and differs; not overwritten" >&2; rm -f "$worklist.tmp"; exit 2
    fi
    mv "$worklist.tmp" "$worklist"
fi

submit() {
    if (( dry_run )); then echo "DRY RUN  sbatch $*" | cut -c1-300 >&2; echo 0; return 0; fi
    local id
    id=$(sbatch --parsable "$@" | tail -n 1 | tr -dc '0-9')
    [[ -n "$id" ]] || { echo "sbatch returned no job id; stopping (anything above was submitted)" >&2; exit 1; }
    echo "$id"
}

ids=()
for acceptance in $acceptances; do
    id=$(submit --array=0-$(( blocks - 1 )) -J l1p_ot_$acceptance \
        --export=ALL,CONFIDENCEOT_REPO=$repo,CONFIDENCEOT_BENCH_ROOT=$root,CONFIDENCEOT_BENCH_WORKLIST=$worklist,CONFIDENCEOT_BENCH_CHUNK=$chunk,CONFIDENCEOT_BENCH_WITHIN_SIDE_ACCEPTANCE=$acceptance \
        -o "$logs/l1p_ot_${acceptance}_%A_%a.out" -e "$logs/l1p_ot_${acceptance}_%A_%a.err" \
        "$repo/benchmark/tacc/ot_array.slurm")
    ids+=("$id"); echo "ConfidenceOT logcpm at $acceptance: $id" >&2
done
common="ALL,CONFIDENCEOT_REPO=$repo,CONFIDENCEOT_BENCH_ROOT=$root,CONFIDENCEOT_BENCH_WORKLIST=$worklist,CONFIDENCEOT_SOLVER_LABEL=$arm,CONFIDENCEOT_SOLVER_CHUNK=$solver_chunk"
for pass in default oracle icpot; do
    extra=""
    [[ "$pass" == oracle ]] && extra=",CONFIDENCEOT_SOLVER_ORACLE_BUDGET=1"
    [[ "$pass" == icpot ]] && extra=",CONFIDENCEOT_SOLVER_ICPOT_COST=0.5"
    id=$(submit --array=0-3 --time=12:00:00 -J l1p_solv_$pass --export="$common$extra" \
        -o "$logs/l1p_solv_${pass}_%A_%a.out" -e "$logs/l1p_solv_${pass}_%A_%a.err" \
        "$repo/benchmark/tacc/solvers.slurm")
    ids+=("$id"); echo "comparators ($pass) on logcpm: $id" >&2
done
after=$(IFS=:; echo "${ids[*]}")
score="source /home1/10119/ghzheng/.bashrc; conda activate $env_path; cd $repo; python benchmark/56_comparator_metrics.py $root $out --arm ranknm256_noscale_ds_cos --arm ranknm256_noscale_cos --arm $arm && python benchmark/57_audit_main_decisions.py $out/main_decisions_per_pair.csv $root > $out/audit_57.txt && cd $root && tar czf comparator_metrics_log1p.tgz comparator_metrics_log1p && md5sum comparator_metrics_log1p.tgz > comparator_metrics_log1p.tgz.md5 && cat comparator_metrics_log1p.tgz.md5"
id=$(submit --dependency=afterany:$after -p gg -N 1 -n 1 -c 16 -t 02:00:00 -A MCB26031 -J l1p_metrics \
    -o "$logs/l1p_metrics_%j.out" -e "$logs/l1p_metrics_%j.err" --wrap "$score")
echo "metrics: $id -> $out (and $root/comparator_metrics_log1p.tgz)" >&2
echo "commit $(git -C "$repo" rev-parse --short HEAD 2> /dev/null || echo unknown)"
