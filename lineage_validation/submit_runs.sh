#!/bin/bash
# OT runs of the lineage validation (VALIDATION_DESIGN.md §2), through the existing pipeline.
#
#   bash lineage_validation/submit_runs.sh --dry-run
#   bash lineage_validation/submit_runs.sh
#
# 1. inputs: lineage_validation/01_build_inputs.py -> <root>/inputs (pooled sides, manifests,
#    cell_sites.tsv.gz). Skipped if <root>/inputs/SUCCESS exists.
# 2. quinn_ds: 27_downsample_counts.py on inputs/manifests/quinn_D_B_raw.csv (defaults, the
#    project's preprocessing for a new dataset) -> inputs/h5ad/Quinn2021/ds. Skipped if done.
# 3. One job per block (kp_D_A, kp_D_B, kp_D_C3, quinn_D_B): tacc_cnv_rna_strategies.slurm with
#    STRAT_STAGE=all (rna, cnv, rna_cnv, then cnv_then_rna, rna_then_cnv), acceptance 0.99,
#    ranknm256_noscale_ds_cos, --save-coupling --save-pairing-edges, no cell cap
#    (CONFIDENCEOT_MAX_OBSERVED_CELLS_PER_SIDE=0). Output <root>/runs/acc0.99/<strategy>/arm_A/<block>/.
#    CNV: KP-Tracer infercnvpy_mouse (reference = the authors' normal cells), Quinn
#    infercnvpy_experiment (no reference).
# Pairs that finished are skipped by the pipeline (--skip-completed), so a resubmission only
# redoes failures. Nothing outside <root> is written.
# No set -u, as everywhere in this repo.
set -eo pipefail

dry_run=0
[[ "${1:-}" == --dry-run ]] && dry_run=1
repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
root=${LINEAGE_VALIDATION_ROOT:-/scratch/10119/ghzheng/OT_project/lineage_validation}
kp=/scratch/10119/ghzheng/OT_project/kp_tracer_confidenceot
quinn=/scratch/10119/ghzheng/OT_project/quinn2021_confidenceot
env_path=${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}
inputs=$root/inputs
qds=$inputs/h5ad/Quinn2021/ds
logs=$root/logs
acceptance=0.99
cap=${CONFIDENCEOT_JOB_CAP:-40}
activate="source /home1/10119/ghzheng/.bashrc; conda activate $env_path; cd $repo; export PYTHONPATH=$repo/src:$repo/cancer_metastasis:$repo"

for path in "$kp/downsampled/pair_manifest_downsampled.csv" "$kp/downsampled/predownsample_depth.csv.gz" \
            "$kp/cnv_profiles/infercnvpy_mouse/KPTracer" "$quinn/cnv_profiles/infercnvpy_experiment/Quinn2021"; do
    [[ -e "$path" ]] || { echo "missing: $path" >&2; exit 1; }
done
queued=$(squeue -u "$USER" -h -r 2> /dev/null | wc -l) || queued=0
if (( ! dry_run && queued + 6 > cap )); then echo "$queued queued; 6 more would pass the cap of $cap" >&2; exit 3; fi
(( dry_run )) || mkdir -p "$logs"

submit() {
    if (( dry_run )); then echo "DRY RUN  sbatch $*" | cut -c1-400 >&2; echo DRYRUN; return 0; fi
    local id
    id=$(sbatch --parsable "$@" | tail -n 1)
    [[ "$id" =~ ^[0-9]+$ ]] || { echo "sbatch returned no job id; stopping (anything above was submitted)" >&2; exit 1; }
    echo "$id"
}

wait_inputs=()
if [[ ! -f "$inputs/SUCCESS" ]]; then
    id=$(submit -p gg -N 1 -n 1 -t 02:00:00 -A MCB26031 -J lv_inputs -o "$logs/inputs_%j.out" \
        --wrap "test ! -f $inputs/SUCCESS || exit 2; $activate && python lineage_validation/01_build_inputs.py --out $inputs")
    echo "inputs job $id" >&2
    wait_inputs=(--dependency=afterok:$id)
fi
wait_qds=("${wait_inputs[@]}")
if [[ ! -f "$qds/pair_manifest_downsampled.csv" ]]; then
    id=$(submit "${wait_inputs[@]}" -p gg -N 1 -n 1 -t 02:00:00 -A MCB26031 -J lv_quinn_ds -o "$logs/quinn_ds_%j.out" \
        --wrap "test ! -f $qds/pair_manifest_downsampled.csv || exit 2; $activate && python cancer_metastasis/27_downsample_counts.py $inputs/manifests/quinn_D_B_raw.csv $qds --malignant-column malignant")
    echo "quinn equalisation job $id" >&2
    wait_qds=(--dependency=afterok:$id)
fi

ids=""
for block in kp_D_A kp_D_B kp_D_C3 quinn_D_B; do
    if [[ $block == quinn_* ]]; then
        m=$inputs/manifests/${block}_raw.csv; mds=$qds/pair_manifest_downsampled.csv; d=$qds/predownsample_depth.csv.gz
        cnv=$quinn/cnv_profiles/infercnvpy_experiment; wait=("${wait_qds[@]}")
    else
        m=$inputs/manifests/${block}_raw.csv; mds=$inputs/manifests/${block}_ds.csv; d=$kp/downsampled/predownsample_depth.csv.gz
        cnv=$kp/cnv_profiles/infercnvpy_mouse; wait=("${wait_inputs[@]}")
    fi
    id=$(submit "${wait[@]}" -J "lv_${block}" -o "$logs/runs_${block}_%j.out" -e "$logs/runs_${block}_%j.err" \
        --export=ALL,CONFIDENCEOT_REPO="$repo",CANCER_COT_ROOT="$root",STRAT_STAGE=all,STRAT_ACCEPTANCE="$acceptance",STRAT_BLOCK="$block",STRAT_OUT="$root/runs/acc$acceptance",STRAT_CNV_ROOT="$cnv",STRAT_MANIFEST="$m",STRAT_MANIFEST_DS="$mds",STRAT_DEPTH="$d",STRAT_ANNOTATIONS=,STRAT_EXTRA_ARGS=--save-pairing-edges,CONFIDENCEOT_MAX_OBSERVED_CELLS_PER_SIDE=0 \
        "$repo/cancer_metastasis/tacc_cnv_rna_strategies.slurm")
    echo "$block job $id" >&2
    ids="$ids $block=$id"
done
summary="commit $(git -C "$repo" rev-parse --short HEAD 2> /dev/null || echo unknown)
acceptance $acceptance; strategies rna cnv rna_cnv cnv_then_rna rna_then_cnv; no cell cap; couplings + coordinates saved
runs $root/runs/acc$acceptance/<strategy>/arm_A/<block>/ranknm256_noscale_ds_cos
jobs$ids"
echo "$summary"
(( dry_run )) || echo "$summary" > "$root/SUBMITTED_runs_$(date +%Y%m%d_%H%M%S).txt"
