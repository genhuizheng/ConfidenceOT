#!/bin/bash
# KP-Tracer validation_v3 (Level 2/3), phase by phase. Each phase is one small gg job.
#
#   bash kp_lineage/validation_v3/submit_v3.sh audit [--dry-run]
#   bash kp_lineage/validation_v3/submit_v3.sh pilot [--dry-run]
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
    *)
        echo "usage: $0 audit|pilot [--dry-run]" >&2; exit 2 ;;
esac
echo "commit $(git -C "$repo" rev-parse --short HEAD 2> /dev/null || echo unknown)"
