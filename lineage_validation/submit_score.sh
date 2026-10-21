#!/bin/bash
# Scoring of the lineage validation against lineage (VALIDATION_PROTOCOL.md, FROZEN 508609f).
#
#   bash lineage_validation/submit_score.sh --dry-run
#   bash lineage_validation/submit_score.sh [--after JOBID[:JOBID...]] [--tasks 12] [--baseline-tasks 6]
#                                           [--phases labels,score,baselines,umap,aggregate,figures]
#
# labels     02_labels.py: lineage per cell, tree leaves (once; skipped if labels/SUCCESS)
# score      03_score_runs.py as an array of --tasks tasks: §7-§10 per run, M4-E and M4-R
# baselines  04_baselines.py as an array of --baseline-tasks tasks: §10 balanced OT and kNN
# umap       48_reference_umap.py pairs, one job per block: the fixed joint UMAP -> <root>/umap/<block>
# aggregate  05_aggregate.py --replace: tests and tables -> <root>/summary
# figures    06_figures.py -> <root>/figures (a new figures_<time> if it exists)
# --after makes labels/score/baselines/umap wait for those jobs (afterany), e.g. the OT jobs of
# submit_runs.sh. Runs that are not complete are marked MISSING and scored on a later submission;
# completed scores are skipped. Array tasks count one by one against the 40-job cap.
# No set -u, as everywhere in this repo.
set -eo pipefail

dry_run=0; after=""; tasks=12; btasks=6; phases="labels,score,baselines,umap,aggregate,figures"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) dry_run=1 ;;
        --after) after=$2; shift ;;
        --tasks) tasks=$2; shift ;;
        --baseline-tasks) btasks=$2; shift ;;
        --phases) phases=$2; shift ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
    shift
done
repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
root=${LINEAGE_VALIDATION_ROOT:-/scratch/10119/ghzheng/OT_project/lineage_validation}
env_path=${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}
logs=$root/logs
cap=${CONFIDENCEOT_JOB_CAP:-40}
activate="source /home1/10119/ghzheng/.bashrc; conda activate $env_path; cd $repo; export PYTHONPATH=$repo/src:$repo/cancer_metastasis:$repo"
has() { [[ ",$phases," == *",$1,"* ]]; }

[[ -f "$root/inputs/SUCCESS" ]] || { echo "missing: $root/inputs/SUCCESS (run submit_runs.sh first)" >&2; exit 1; }
grep -q "FROZEN 2026-10-21 508609f" "$repo/lineage_validation/VALIDATION_PROTOCOL.md" || { echo "protocol on TACC is not the frozen one; git pull first" >&2; exit 1; }
need=0
has labels && need=$((need + 1)); has score && need=$((need + tasks)); has baselines && need=$((need + btasks))
has umap && need=$((need + 4)); has aggregate && need=$((need + 1)); has figures && need=$((need + 1))
queued=$(squeue -u "$USER" -h -r 2> /dev/null | wc -l) || queued=0
if (( ! dry_run && queued + need > cap )); then echo "$queued queued; $need more would pass the cap of $cap" >&2; exit 3; fi
(( dry_run )) || mkdir -p "$logs"

submit() {
    if (( dry_run )); then echo "DRY RUN  sbatch $*" | cut -c1-500 >&2; echo DRYRUN; return 0; fi
    local id
    id=$(sbatch --parsable "$@" | tail -n 1)
    [[ "$id" =~ ^[0-9]+$ ]] || { echo "sbatch returned no job id; stopping (anything above was submitted)" >&2; exit 1; }
    echo "$id"
}
common=(-p gg -N 1 -n 1 -A MCB26031)
wait_after=(); [[ -n "$after" ]] && wait_after=(--dependency=afterany:$after)
labels_dep=("${wait_after[@]}")
ids=""

if has labels && [[ ! -f "$root/labels/SUCCESS" ]]; then
    id=$(submit "${common[@]}" "${wait_after[@]}" -t 02:00:00 -J lv_labels -o "$logs/labels_%j.out" \
        --wrap "test ! -f $root/labels/SUCCESS || exit 0; $activate && python lineage_validation/02_labels.py --root $root")
    ids="$ids labels=$id"; labels_dep=(--dependency=afterok:$id${after:+,afterany:$after})
fi
score_ids=""
if has score; then
    id=$(submit "${common[@]}" "${labels_dep[@]}" -t 24:00:00 -J lv_score --array=0-$((tasks - 1)) -o "$logs/score_%A_%a.out" \
        --wrap "$activate && python lineage_validation/03_score_runs.py --root $root --task-index \$SLURM_ARRAY_TASK_ID --task-count $tasks")
    ids="$ids score=$id"; score_ids=$id
fi
if has baselines; then
    id=$(submit "${common[@]}" "${labels_dep[@]}" -t 24:00:00 -J lv_base --array=0-$((btasks - 1)) -o "$logs/baselines_%A_%a.out" \
        --wrap "$activate && python lineage_validation/04_baselines.py --root $root --task-index \$SLURM_ARRAY_TASK_ID --task-count $btasks")
    ids="$ids baselines=$id"; score_ids="$score_ids${score_ids:+:}$id"
fi
umap_ids=""
if has umap; then
    for block in kp_D_A kp_D_B kp_D_C3 quinn_D_B; do
        dataset=KPTracer; [[ $block == quinn_* ]] && dataset=Quinn2021
        id=$(submit "${common[@]}" "${wait_after[@]}" -t 12:00:00 -J "lv_umap_$block" -o "$logs/umap_${block}_%j.out" \
            --wrap "$activate && python cancer_metastasis/48_reference_umap.py pairs --manifest $root/inputs/manifests/${block}_raw.csv --datasets $dataset --block $dataset=$block --gates-root $root/runs/acc0.99 --out $root/umap/$block")
        ids="$ids umap_$block=$id"; umap_ids="$umap_ids${umap_ids:+:}$id"
    done
fi
agg_id=""
if has aggregate; then
    dep=(); [[ -n "$score_ids" ]] && dep=(--dependency=afterany:$score_ids)
    id=$(submit "${common[@]}" "${dep[@]}" -t 02:00:00 -J lv_aggregate -o "$logs/aggregate_%j.out" \
        --wrap "$activate && python lineage_validation/05_aggregate.py --root $root --replace")
    ids="$ids aggregate=$id"; agg_id=$id
fi
if has figures; then
    deps="$agg_id${umap_ids:+${agg_id:+:}$umap_ids}"
    dep=(); [[ -n "$deps" ]] && dep=(--dependency=afterany:$deps)
    id=$(submit "${common[@]}" "${dep[@]}" -t 06:00:00 -J lv_figures -o "$logs/figures_%j.out" \
        --wrap "$activate && python lineage_validation/06_figures.py --root $root")
    ids="$ids figures=$id"
fi
summary="commit $(git -C "$repo" rev-parse --short HEAD 2> /dev/null || echo unknown); protocol FROZEN 508609f
phases $phases; score tasks $tasks; baseline tasks $btasks; after ${after:-none}
jobs$ids"
echo "$summary"
(( dry_run )) || echo "$summary" > "$root/SUBMITTED_score_$(date +%Y%m%d_%H%M%S).txt"
