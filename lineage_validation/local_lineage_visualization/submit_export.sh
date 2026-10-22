#!/bin/bash
# Export the real lineage data and existing OT outputs of the chosen mice, then pack them for download.
#
#   bash lineage_validation/local_lineage_visualization/submit_export.sh --dry-run
#   bash lineage_validation/local_lineage_visualization/submit_export.sh
#
# One gg job: tacc_export.py (read only) -> <root>/local_lineage_visualization_export/, then two
# archives in <root>: local_lineage_visualization_core.tar.gz (everything but the couplings) and
# local_lineage_visualization_couplings.tar (the byte-identical coupling .npz files, optional).
# Refuses if the export folder or an archive already exists. No set -u, as everywhere in this repo.
set -eo pipefail

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
root=${LINEAGE_VALIDATION_ROOT:-/scratch/10119/ghzheng/OT_project/lineage_validation}
env_path=${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}
name=local_lineage_visualization_export
for path in "$root/$name" "$root/local_lineage_visualization_core.tar.gz" "$root/local_lineage_visualization_couplings.tar"; do
    [[ -e "$path" ]] && { echo "exists: $path; nothing is overwritten" >&2; exit 2; }
done
cmd="source /home1/10119/ghzheng/.bashrc; conda activate $env_path; cd $repo; export PYTHONPATH=$repo/src:$repo/cancer_metastasis:$repo"
cmd="$cmd; python lineage_validation/local_lineage_visualization/tacc_export.py --out $root/$name"
cmd="$cmd && cd $root && tar -czf local_lineage_visualization_core.tar.gz --exclude=$name/couplings $name"
cmd="$cmd && tar -cf local_lineage_visualization_couplings.tar $name/couplings && ls -l local_lineage_visualization_core.tar.gz local_lineage_visualization_couplings.tar"
if [[ "${1:-}" == --dry-run ]]; then echo "DRY RUN  $cmd"; exit 0; fi
mkdir -p "$root/logs"
sbatch -p gg -N 1 -n 1 -t 03:00:00 -A MCB26031 -J lv_export -o "$root/logs/export_%j.out" --wrap "test ! -e $root/$name || exit 2; $cmd"
