#!/bin/bash
# The native simulation benchmark with one more representation, asked for on
# 2026-10-19: log1p of the counts, then Pearson correlation over ALL genes as the
# OT geometry (no HVG, no PCA, no gene scaling). Label log1p_noscale_pcc; the cost
# is 2 - 2r, divided by the median sampled pair like every other arm, and the
# gate, null calibration and solver settings are those of ot_array.slurm.
#
#   bash benchmark/tacc/submit_log1p_pcc.sh --dry-run
#   bash benchmark/tacc/submit_log1p_pcc.sh
#
# Inputs: the same 180 pairs (3 sizes x 5 replicates x 3 levels x 4 cases) and the
# same un-equalised counts as the raw arm: the work list is the raw rows of the
# 32-arm work list, relabelled, dealt round robin by size into 9 blocks of 20.
#
# Two acceptances, each its own runs directory, so nothing existing is written:
#   0.99 -> runs_acceptance_0.99/log1p_noscale_pcc   (beside the three arms rerun there)
#   0.90 -> runs_acceptance_0.90/log1p_noscale_pcc   (0.90 is the runner default; the
#           32-arm grid at 0.90 is runs/, already scored in benchmark_metrics.csv)
# Then one scoring job per acceptance (score_arms.slurm, afterany), into
#   metrics_log1p_pcc/acceptance_<a>/ ; at 0.99 the three arms already there are
#   scored beside it with the same scorer.
#
# No joint coordinates are written (CONFIDENCEOT_BENCH_SAVE_EDGES=0): here they would
# be cells x 10,000 genes as text, and the scorer reads only the gate and coupling.
#
# No set -u, as everywhere in this repo.
set -eo pipefail

dry_run=0
[[ "${1:-}" == --dry-run ]] && dry_run=1

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
root=${CONFIDENCEOT_BENCH_ROOT:-/scratch/10119/ghzheng/OT_project/benchmark_native}
arm=log1p_noscale_pcc
source_list=$root/worklist_1000_5000_10000.csv
worklist=$root/worklist_${arm}.csv
blocks=9
chunk=20
logs=$root/logs

test -f "$source_list" || { echo "no $source_list" >&2; exit 2; }

# The raw rows, relabelled, sorted by size and dealt round robin into the blocks,
# so every task carries about the same number of N=10000 pairs. awk, not Python:
# this runs on the login node.
rows=$(awk -F',' -v OFS=',' -v arm="$arm" '
    NR == 1 { next }
    $4 == "raw" {
        size = $3; sub(/^N/, "", size); sub(/_.*/, "", size)
        $4 = arm
        n[size]++; row[size, n[size]] = $0
    }
    END {
        split("1000 5000 10000", order, " ")
        total = 0
        for (k = 1; k <= 3; k++) for (i = 1; i <= n[order[k]]; i++) all[++total] = row[order[k], i]
        for (b = 0; b < blocks; b++) for (i = b + 1; i <= total; i += blocks) print all[i]
    }' blocks="$blocks" "$source_list")
count=$(printf '%s\n' "$rows" | grep -c . || true)
[[ "$count" -eq 180 ]] || { echo "expected 180 raw rows in $source_list, found $count" >&2; exit 2; }
bad=$(printf '%s\n' "$rows" | awk -F',' -v arm="$arm" '$4 != arm' | wc -l)
[[ "$bad" -eq 0 ]] || { echo "relabelling failed on $bad rows" >&2; exit 2; }

queued=$( { squeue -u "$USER" -h -r 2> /dev/null || true; } | wc -l )
wanted=$(( 2 * blocks + 2 ))
if (( queued + wanted > 40 )); then
    echo "$queued jobs queued; $wanted more would pass the 40-job cap" >&2
    exit 3
fi

if (( dry_run )); then
    echo "DRY RUN  $count units -> $worklist ($blocks tasks x $chunk per acceptance)"
    printf '%s\n' "$rows" | head -n 3 | cut -c1-200
else
    mkdir -p "$logs"
    { head -n 1 "$source_list"; printf '%s\n' "$rows"; } > "$worklist.tmp"
    if [[ -f "$worklist" ]] && ! cmp -s "$worklist" "$worklist.tmp"; then
        echo "$worklist exists and differs; not overwritten" >&2; rm -f "$worklist.tmp"; exit 2
    fi
    mv "$worklist.tmp" "$worklist"
fi

submit() {
    if (( dry_run )); then echo "DRY RUN  sbatch $*" | cut -c1-400 >&2; echo 0; return 0; fi
    sbatch --parsable "$@" | tail -n 1 | tr -dc '0-9'
}

for acceptance in 0.99 0.90; do
    runs_dir=runs_acceptance_$acceptance
    ot=$(submit --array=0-$(( blocks - 1 )) -J cot_pcc_$acceptance \
            --export=ALL,CONFIDENCEOT_REPO=$repo,CONFIDENCEOT_BENCH_ROOT=$root,CONFIDENCEOT_BENCH_WORKLIST=$worklist,CONFIDENCEOT_BENCH_CHUNK=$chunk,CONFIDENCEOT_BENCH_WITHIN_SIDE_ACCEPTANCE=$acceptance,CONFIDENCEOT_BENCH_SAVE_EDGES=0 \
            -o "$logs/pcc_ot_${acceptance}_%A_%a.out" -e "$logs/pcc_ot_${acceptance}_%A_%a.err" \
            "$repo/benchmark/tacc/ot_array.slurm")
    arms=$arm
    [[ "$acceptance" == 0.99 ]] && arms="$arm+ranknm256_noscale_ds_cos+ranknm256_noscale_cos+raw"
    submit --dependency=afterany:$ot -J cot_pcc_score \
            --export=ALL,CONFIDENCEOT_REPO=$repo,CONFIDENCEOT_BENCH_ROOT=$root,CONFIDENCEOT_SCORE_RUNS=$runs_dir,CONFIDENCEOT_SCORE_ARMS=$arms,CONFIDENCEOT_SCORE_OUT=$root/metrics_log1p_pcc/acceptance_$acceptance \
            -o "$logs/pcc_score_${acceptance}_%j.out" -e "$logs/pcc_score_${acceptance}_%j.err" \
            "$repo/benchmark/tacc/score_arms.slurm" > /dev/null
    echo "acceptance $acceptance: OT array ${ot}, then scoring of [$arms]"
done
echo "commit $(git -C "$repo" rev-parse --short HEAD 2> /dev/null || echo unknown)"
