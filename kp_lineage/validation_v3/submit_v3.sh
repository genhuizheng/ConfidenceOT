#!/bin/bash
# KP-Tracer validation_v3 (Level 2/3), phase by phase. Each phase is one small gg job.
#
#   bash kp_lineage/validation_v3/submit_v3.sh audit [--dry-run]
#   bash kp_lineage/validation_v3/submit_v3.sh pilot [--dry-run]
#   bash kp_lineage/validation_v3/submit_v3.sh compare [--dry-run]
#   bash kp_lineage/validation_v3/submit_v3.sh couplings [--dry-run]
#
# audit  01_audit_v3.py: data inventory, shared-tree pairs, lineage informativeness of
#        L, Cassiopeia vs neighbor-joining, character-matrix statistics. Read only;
#        computes nothing from any gate or coupling. -> <root>/validation_v3/audit/
# pilot  Coupling reproduction on 3 pairs x 2 labels: 02_rerun_with_coupling.py with
#        the original calibrated cost, then 03_compare_reproduction.py against the
#        original outputs. -> <root>/validation_v3/coupling_pilot/
#        Nothing larger is submitted unless the pilot passes.
#
# Every phase refuses when its output already holds SUCCESS, at submission and again
# when the job starts. The original experiment is only read.
#
# No set -u, as everywhere in this repo.
set -eo pipefail

phase=${1:-}
[[ $# -gt 0 ]] && shift
dry_run=0
[[ "${1:-}" == --dry-run ]] && dry_run=1

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
root=${KPTRACER_ROOT:-/scratch/10119/ghzheng/OT_project/kp_tracer_confidenceot}
tracer=${KPTRACER_DATA:-/scratch/10119/ghzheng/OT_project/data/kp_tracer/KPTracer-Data}
env_path=${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}
v3=$root/validation_v3
logs=$v3/logs
activate="source /home1/10119/ghzheng/.bashrc; conda activate $env_path; cd $repo; export PYTHONPATH=$repo/src:$repo/cancer_metastasis:$repo"
(( dry_run )) || mkdir -p "$logs"

# The pilot pairs: small, medium and the largest source tumour, all same-family and
# lineage-covered, chosen by size only.
pilot_pairs="3457__3457_Apc_T4__3457_Apc_L1 3515__3515_Lkb1_T1__3515_Lkb1_K2 3724__3724_NT_T1__3724_NT_L1"

submit() {
    local what=$1; shift
    if (( dry_run )); then
        echo "DRY RUN  $what" >&2; echo "         sbatch $*" >&2; echo 0; return 0
    fi
    local id
    id=$(sbatch --parsable "$@" | tail -n 1 | tr -dc '0-9')
    if [[ -z "$id" ]]; then echo "sbatch returned no job id for $what" >&2; exit 1; fi
    echo "submitted $what as $id" >&2
    echo "$id"
}

case "$phase" in
    audit)
        out=$v3/audit
        [[ -f "$out/SUCCESS" ]] && { echo "$out/SUCCESS exists; the audit is not rerun" >&2; exit 2; }
        submit "lineage audit v3" -p gg -N 1 -n 1 -t 02:00:00 -A MCB26031 -J kp_audit3 \
            -o "$logs/audit_%j.out" -e "$logs/audit_%j.err" \
            --wrap "test ! -f $out/SUCCESS || exit 2; $activate && python kp_lineage/validation_v3/01_audit_v3.py --tracer $tracer --prepared $root/prepared --ot-block $root/ot/kptracer --out $out" > /dev/null
        ;;
    pilot)
        out=$v3/coupling_pilot
        [[ -f "$out/report/SUCCESS" ]] && { echo "$out/report/SUCCESS exists; the pilot is not rerun" >&2; exit 2; }
        rerun="python kp_lineage/validation_v3/02_rerun_with_coupling.py --root $root --out $out --pairs $pilot_pairs"
        compare="python kp_lineage/validation_v3/03_compare_reproduction.py --root $root --rerun $out --out $out/report"
        submit "coupling reproduction pilot" -p gg -N 1 -n 1 -t 02:00:00 -A MCB26031 -J kp_pilot \
            -o "$logs/pilot_%j.out" -e "$logs/pilot_%j.err" \
            --wrap "test ! -f $out/report/SUCCESS || exit 2; $activate && $rerun --label ranknm256_noscale_ds_cos && $rerun --label raw && $compare" > /dev/null
        ;;
    compare)
        # The comparison step of the pilot alone, on the reruns already in coupling_pilot/.
        # Added after pilot job 1075006: all six reruns exited 0 and reproduction.tsv was
        # written, but 03_ crashed while writing reproduction.json (fixed). Nothing is refit.
        out=$v3/coupling_pilot
        [[ -f "$out/report/SUCCESS" ]] && { echo "$out/report/SUCCESS exists; the comparison is not rerun" >&2; exit 2; }
        compare="python kp_lineage/validation_v3/03_compare_reproduction.py --root $root --rerun $out --out $out/report"
        submit "coupling reproduction comparison" -p gg -N 1 -n 1 -t 00:30:00 -A MCB26031 -J kp_compare \
            -o "$logs/compare_%j.out" -e "$logs/compare_%j.err" \
            --wrap "test ! -f $out/report/SUCCESS || exit 2; $activate && $compare" > /dev/null
        ;;
    couplings)
        # Stage D: coupling reruns for every shared-tree pair with more than one distinct L
        # (design rules 1-3; 53 - 3 = 50), both labels, at the original calibrated cost.
        # Then 03_compare_reproduction.py over all of them (rule 4). No lineage quantity is
        # computed. Refuses unless the design is frozen and the audit and pilot passed.
        out=$v3/coupling
        design=$repo/kp_lineage/validation_v3/LEVEL3_VALIDATION_DESIGN.md
        [[ -f "$out/report/SUCCESS" ]] && { echo "$out/report/SUCCESS exists; the couplings are not rerun" >&2; exit 2; }
        grep -q '^Status: \*\*FROZEN' "$design" || { echo "the Level 3 design is not frozen" >&2; exit 2; }
        [[ -f "$v3/audit/SUCCESS" ]] || { echo "the audit has not completed" >&2; exit 2; }
        [[ -f "$v3/coupling_pilot/report/SUCCESS" ]] || { echo "the pilot has not passed" >&2; exit 2; }
        pairs=$(awk -F'\t' 'NR==1{for(i=1;i<=NF;i++)c[$i]=i; next} $c["status"]=="shared tree" && ($c["L_distinct"]+0)>1 {print $c["pair_id"]}' "$v3/audit/LINEAGE_PAIR_ELIGIBILITY.tsv" | sort)
        count=$(printf '%s\n' "$pairs" | grep -c . || true)
        [[ "$count" -eq 50 ]] || { echo "expected 50 eligible pairs, found $count" >&2; exit 2; }
        list=$out/pairs.txt
        tasks=8
        if (( dry_run )); then
            echo "DRY RUN  $count pairs, $tasks tasks per label (round robin over the sorted list)" >&2
        else
            mkdir -p "$out"
            printf '%s\n' "$pairs" > "$list"
        fi
        ids=""
        for label in ranknm256_noscale_ds_cos raw; do
            rerun="python kp_lineage/validation_v3/02_rerun_with_coupling.py --root $root --label $label --pair-list $list --task-index \$SLURM_ARRAY_TASK_ID --task-count $tasks --out $out"
            id=$(submit "coupling reruns $label" --array=0-$((tasks - 1)) -p gg -N 1 -n 1 -t 04:00:00 -A MCB26031 -J kp_cpl \
                -o "$logs/couplings_${label}_%A_%a.out" -e "$logs/couplings_${label}_%A_%a.err" \
                --wrap "test ! -f $out/report/SUCCESS || exit 2; $activate && $rerun")
            ids="$ids:$id"
        done
        compare="python kp_lineage/validation_v3/03_compare_reproduction.py --root $root --rerun $out --out $out/report"
        submit "coupling reproduction check" --dependency=afterok$ids -p gg -N 1 -n 1 -t 01:00:00 -A MCB26031 -J kp_cplcmp \
            -o "$logs/couplings_compare_%j.out" -e "$logs/couplings_compare_%j.err" \
            --wrap "test ! -f $out/report/SUCCESS || exit 2; $activate && $compare" > /dev/null
        ;;
    *)
        echo "usage: $0 audit|pilot|compare|couplings [--dry-run]" >&2; exit 2 ;;
esac
echo "commit $(git -C "$repo" rev-parse --short HEAD 2> /dev/null || echo unknown)"
