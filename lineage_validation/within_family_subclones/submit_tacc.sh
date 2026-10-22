#!/bin/bash
# Within-family subclone analysis on TACC for every eligible family / clone (SUBCLONE_RULES.md, a545f1c).
#
#   bash lineage_validation/within_family_subclones/submit_tacc.sh --dry-run
#   bash lineage_validation/within_family_subclones/submit_tacc.sh
#
# One gg job, read only on existing data:
#   1. define_subclones.py on the authors' trees and matrices (KPTracer-Data/trees, the Quinn lineage folders of
#      local_lineage_visualization_export) -> <root>/within_family_subclones/defs
#   2. classify_subclones.py on the saved couplings of the runs (runs/acc0.99) -> .../classification
#   3. <root>/within_family_subclones.tar.gz (definitions, classification tables, UMAP coordinates; no coupling)
# Refuses if the output folder or the archive exists. No set -u, as everywhere in this repo.
set -eo pipefail

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
root=${LINEAGE_VALIDATION_ROOT:-/scratch/10119/ghzheng/OT_project/lineage_validation}
tracer=/scratch/10119/ghzheng/OT_project/data/kp_tracer/KPTracer-Data
env_path=${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}
out=$root/within_family_subclones
for path in "$out" "$root/within_family_subclones.tar.gz"; do
    [[ -e "$path" ]] && { echo "exists: $path; nothing is overwritten" >&2; exit 2; }
done
[[ -f "$root/local_lineage_visualization_export/SUCCESS" ]] || { echo "missing: the lineage export (submit_export.sh)" >&2; exit 1; }
w=lineage_validation/within_family_subclones
cmd="source /home1/10119/ghzheng/.bashrc; conda activate $env_path; cd $repo; export PYTHONPATH=$repo/src:$repo/cancer_metastasis:$repo"
cmd="$cmd; python $w/define_subclones.py --out $out/defs --kp-trees $tracer/trees --kp-meta $tracer/KPTracer_meta.csv --lineage-export $root/local_lineage_visualization_export"
cmd="$cmd && python $w/classify_subclones.py --defs $out/defs --out $out/classification --runs $root/runs/acc0.99 --labels $root/labels --kp-meta $tracer/KPTracer_meta.csv --umap-root $root/umap"
cmd="$cmd && cd $root && tar -czf within_family_subclones.tar.gz within_family_subclones && ls -l within_family_subclones.tar.gz"
if [[ "${1:-}" == --dry-run ]]; then echo "DRY RUN  $cmd"; exit 0; fi
mkdir -p "$root/logs"
sbatch -p gg -N 1 -n 1 -t 06:00:00 -A MCB26031 -J lv_subclones -o "$root/logs/subclones_%j.out" --wrap "test ! -e $out || exit 2; $cmd"
