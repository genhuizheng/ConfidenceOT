#!/bin/bash
# The UMAP settings grid of 56_umap_grid.py on TACC (visualisation only, no OT), then two packages:
#   <out>_summary.tar.gz  summary.tsv (every setting's primary-metastasis mixing ratio) + thumbnails
#   <out>_coords.tar.gz   every setting's coordinates, for the settings chosen from the summary
#
#   bash cancer_metastasis/submit_umap_grid.sh --dry-run
#   bash cancer_metastasis/submit_umap_grid.sh [--export export_20261021b] [--datasets "GSE180661 GSE225857 GSE271675"]
#
# One task per dataset x representation x PCA size (3 x 3 x 2 = 18 by default); a setting with DONE
# is skipped, so a resubmission redoes only what is missing. The summary is written after the array
# whatever its state; FAILED_*.txt files are listed and no md5 is written while any exists.
#
# No set -u, as everywhere in this repo.
set -eo pipefail

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
env_path=${CONFIDENCEOT_ENV_EXPORT:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}
raw_uniform=$result/manifest/pancancer_20260924/pair_manifest_eligible.csv
raw_prostate=$result/prepared_GSE271675_20260916/pair_manifest_eligible.csv
ds_uniform=$result/manifest/factorial_20260929/uniform_ds.csv
ds_prostate=$result/downsampled_GSE271675_20260916/pair_manifest_downsampled.csv
export_name=export_20261021b
datasets="GSE180661 GSE225857 GSE271675"
dry_run=0
while (( $# )); do
    case "$1" in
        --export) export_name=$2; shift 2 ;;
        --datasets) datasets=$2; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
cells=$result/cnv_rna_strategies_export/$export_name/cells
out=$result/umap_grid_$export_name
logs=$out/logs
tasks=$(( $(wc -w <<< "$datasets") * 3 * 2 ))

problems=0
for path in "$raw_uniform" "$raw_prostate" "$ds_uniform" "$ds_prostate" "$cells" "$repo/cancer_metastasis/56_umap_grid.py"; do
    [[ -e "$path" ]] || { echo "missing: $path" >&2; problems=1; }
done
[[ -f "$out"_summary.tar.gz.md5 ]] && { echo "${out}_summary.tar.gz.md5 exists; not rewritten" >&2; problems=1; }
queued=$( { squeue -u "$USER" -h -r 2> /dev/null || true; } | wc -l )
(( ! dry_run && queued + tasks + 1 > 40 )) && { echo "$queued tasks queued, this adds $((tasks + 1)); over the cap of 40" >&2; problems=1; }
(( problems && ! dry_run )) && exit 2

activate="source /home1/10119/ghzheng/.bashrc; conda activate $env_path; cd $repo; export PYTHONPATH=$repo/src:$repo/cancer_metastasis:$repo; export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMBA_NUM_THREADS=16 MPLBACKEND=Agg PYTHONUNBUFFERED=1"
submit() {
    if (( dry_run )); then echo "DRY RUN  sbatch $*" | cut -c1-400 >&2; echo 0; return 0; fi
    local id
    id=$(sbatch --parsable "$@" | tail -n 1 | tr -dc '0-9')
    [[ -n "$id" ]] || { echo "sbatch returned no job id; stopping" >&2; exit 1; }
    echo "$id"
}
(( dry_run )) || mkdir -p "$logs"

array=$(submit -p gg -N 1 -n 1 -c 16 -t 08:00:00 -A MCB26031 -J ug_grid --array=0-$((tasks - 1)) \
    -o "$logs/grid_%A_%a.out" -e "$logs/grid_%A_%a.err" \
    --wrap "$activate && python cancer_metastasis/56_umap_grid.py --cells $cells --manifest $raw_uniform --manifest $raw_prostate --ds-manifest $ds_uniform --ds-manifest $ds_prostate --datasets $datasets --out $out --task-index \$SLURM_ARRAY_TASK_ID --task-count $tasks")
echo "UMAP grid: $array ($tasks tasks) -> $out" >&2
summary="import glob, json, pandas as pd; rows = [json.load(open(f)) for f in sorted(glob.glob('$out/*/*/metrics.json'))]; pd.DataFrame(rows).sort_values(['dataset', 'mixing_ratio']).to_csv('$out/summary.tsv', sep='\\t', index=False); print(len(rows), 'settings')"
pack="$activate && python -c \"$summary\" && cd $result && (ls $out/*/FAILED_*.txt 2> /dev/null || true) && tar -czf ${out}_summary.tar.gz.partial \$(basename $out)/summary.tsv \$(cd $result && ls -d \$(basename $out)/*/*/thumb.png \$(basename $out)/*/*/metrics.json) && mv ${out}_summary.tar.gz.partial ${out}_summary.tar.gz && tar -czf ${out}_coords.tar.gz.partial \$(cd $result && ls -d \$(basename $out)/*/*/coords.csv.gz) && mv ${out}_coords.tar.gz.partial ${out}_coords.tar.gz && { if ls $out/*/FAILED_*.txt > /dev/null 2>&1; then echo 'some units FAILED: no md5 written'; exit 1; fi; } && md5sum ${out}_summary.tar.gz ${out}_coords.tar.gz | tee ${out}_summary.tar.gz.md5 && du -h ${out}_summary.tar.gz ${out}_coords.tar.gz"
id=$(submit --dependency=afterany:$array -p gg -N 1 -n 1 -c 4 -t 01:00:00 -A MCB26031 -J ug_pack \
    -o "$logs/pack_%j.out" -e "$logs/pack_%j.err" --wrap "$pack")
echo "pack: $id -> ${out}_summary.tar.gz, ${out}_coords.tar.gz" >&2
echo "commit $(git -C "$repo" rev-parse --short HEAD 2> /dev/null || echo unknown); datasets $datasets; cells $cells"
