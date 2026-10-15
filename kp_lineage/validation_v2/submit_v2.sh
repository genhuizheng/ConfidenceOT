#!/bin/bash
# KP-Tracer validation_v2, phase by phase. Each phase is one small gg job.
#
#   bash kp_lineage/validation_v2/submit_v2.sh audit   [--dry-run]
#   bash kp_lineage/validation_v2/submit_v2.sh readout [--dry-run]
#   bash kp_lineage/validation_v2/submit_v2.sh lineage [--dry-run]
#
# audit    03_audit_lineage.py: trees, character matrices, identifier mapping,
#          coverage of the same-family pairs, observed gate semantics. Read only;
#          computes no association between lineage and any gate outcome.
#          Writes <root>/validation_v2/audit/.
# readout  The existing, preregistered primary readout (02_kptracer_readout.py)
#          with its established arguments, into <root>/readout/. Refuses if that
#          directory already holds a readout_summary.json, so it never overwrites.
#
# lineage  D1: 04_lineage_concordance.py, E1/E1b and their sensitivity analyses
#          under the preregistration frozen 2026-10-15 (text of b0817f1), both
#          labels, into <root>/validation_v2/lineage/. Reads the existing gates and
#          the authors' trees only; no OT is rerun. Refuses if that directory
#          already holds SUCCESS.
#
# Later phases (cross-mouse control, baselines) are added after D1.
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
v2=$root/validation_v2
logs=$v2/logs
activate="source /home1/10119/ghzheng/.bashrc; conda activate $env_path; cd $repo; export PYTHONPATH=$repo/src:$repo/cancer_metastasis:$repo"
mkdir -p "$logs"

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
        out=$v2/audit
        mkdir -p "$out"
        submit "lineage data audit" -p gg -N 1 -n 1 -t 01:00:00 -A MCB26031 -J kp_audit \
            -o "$logs/audit_%j.out" -e "$logs/audit_%j.err" \
            --wrap "$activate && python kp_lineage/validation_v2/03_audit_lineage.py --tracer $tracer --prepared $root/prepared --ot-block $root/ot/kptracer --out $out" > /dev/null
        ;;
    readout)
        out=$root/readout
        if [[ -f "$out/readout_summary.json" ]]; then
            echo "$out/readout_summary.json exists; the primary readout is not rerun" >&2
            exit 2
        fi
        mkdir -p "$out"
        submit "preregistered primary readout" -p gg -N 1 -n 1 -t 00:30:00 -A MCB26031 -J kp_readout \
            -o "$out/readout_%j.log" -e "$out/readout_%j.err" \
            --wrap "$activate && python kp_lineage/02_kptracer_readout.py $root/prepared $root/ot/kptracer $out" > /dev/null
        ;;
    lineage)
        out=$v2/lineage
        if [[ -f "$out/SUCCESS" ]]; then
            echo "$out/SUCCESS exists; D1 is not rerun into the same directory" >&2
            exit 2
        fi
        if ! grep -q "Status: \*\*FROZEN 2026-10-15 b0817f1" "$repo/kp_lineage/validation_v2/VALIDATION_V2_PREREGISTRATION.md"; then
            echo "the preregistration in $repo is not the frozen one; pull the repository first" >&2
            exit 2
        fi
        mkdir -p "$out"
        submit "lineage concordance D1" -p gg -N 1 -n 1 -t 02:00:00 -A MCB26031 -J kp_d1             -o "$logs/lineage_%j.out" -e "$logs/lineage_%j.err"             --wrap "$activate && python kp_lineage/validation_v2/04_lineage_concordance.py --tracer $tracer --prepared $root/prepared --ot-block $root/ot/kptracer --depth $root/downsampled/predownsample_depth.csv.gz --out $out" > /dev/null
        ;;
    *)
        echo "usage: $0 audit|readout|lineage [--dry-run]" >&2; exit 2 ;;
esac
echo "commit $(git -C "$repo" rev-parse --short HEAD 2> /dev/null || echo unknown)"
