#!/bin/bash
# Read-only audit of the old Splatter directories on TACC -> TACC_CLEANUP_MANIFEST.tsv. Deletes nothing.
#
#   bash realdata_benchmark/splatter_n5000/tacc/cleanup_audit.sh OUT_TSV
#
# One row per candidate: absolute path, files, size, purpose, rebuildable, related running / queued jobs (by name),
# recommendation. Only directories owned by the old Splatter experiments are listed; LTL331, KP-Tracer / Quinn, other
# OT experiments, frozen baselines, code, shared data and environments are not candidates and are not listed as such.
# The recommendation is advice for the user, who decides; no rm is run here. No set -e / -u / pipefail.

out=${1:?usage: cleanup_audit.sh OUT_TSV}
base=/scratch/10119/ghzheng/OT_project
jobs=$(squeue -u "$USER" -h -o "%i %j" 2> /dev/null | tr '\n' ';')
printf "path\tfiles\tsize\tpurpose\trebuildable\trelated_jobs\trecommendation\n" > "$out"
row() {  # path purpose rebuildable recommendation
    local p=$1
    if [[ ! -e "$p" ]]; then printf "%s\t0\t-\t%s\t%s\t-\tabsent\n" "$p" "$2" "$3" >> "$out"; return; fi
    local n s
    n=$(find "$p" -type f 2> /dev/null | wc -l)
    s=$(du -sh "$p" 2> /dev/null | cut -f1)
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$p" "$n" "$s" "$2" "$3" "${jobs:-none}" "$4" >> "$out"
}
row "$base/benchmark_technical" "superseded thinning Splatter benchmark (3 cases, no disjoint; not archived, ARCHIVE_MANIFEST.md)" \
    "yes (benchmark/tacc/generate.slurm with --depth-mechanism thinning)" "candidate: delete only after the user confirms it is no longer needed (it is the only copy)"
row "$base/benchmark_native" "original Splatter benchmark (frozen baseline; archived to corral)" "yes, and archived" "keep (frozen baseline, reused by L0-L2)"
row "$base/realdata_benchmark_20261031/splatter" "joint benchmark Arm A fits (reused by the N=5,000 L0-L2 rows)" "yes (worklist + worker)" "keep (reused)"
inputs=$(find "$base/realdata_benchmark_20261031" -type d -name input 2> /dev/null | wc -l)
isz=$(find "$base/realdata_benchmark_20261031" -type d -name input -prune -exec du -sc {} + 2> /dev/null | tail -n 1 | cut -f1)
printf "%s\t%s\t%s KB\t%s\t%s\t%s\t%s\n" "$base/realdata_benchmark_20261031/**/input" "$inputs dirs" "${isz:-0}" \
    "per-row prepared input h5ad (equalised / encoded copies of the conditions)" "yes (rebuilt by the worker from the manifests)" \
    "${jobs:-none}" "candidate after the N=5,000 collect: the fits read them only while fitting; delete with: find $base/realdata_benchmark_20261031 -type d -name input -prune -exec rm -r {} +" >> "$out"
column -t -s $'\t' "$out" | cut -c1-240
echo "-> $out (nothing deleted)"
