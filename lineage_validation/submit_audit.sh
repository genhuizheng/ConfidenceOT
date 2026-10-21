#!/bin/bash
# Read-only data audit for the lineage ground-truth validation (KP-Tracer, Quinn 2021).
#
#   bash lineage_validation/submit_audit.sh [--dry-run]
#
# One gg job running lineage_validation/00_audit.py; writes only <root>/audit/. Refuses if
# the audit already has SUCCESS. No set -u, as everywhere in this repo.
set -eo pipefail

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
root=${LINEAGE_VALIDATION_ROOT:-/scratch/10119/ghzheng/OT_project/lineage_validation}
env_path=${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}
out=$root/audit
[[ -f "$out/SUCCESS" ]] && { echo "$out/SUCCESS exists; the audit is not rerun" >&2; exit 2; }
cmd="source /home1/10119/ghzheng/.bashrc; conda activate $env_path; cd $repo; export PYTHONPATH=$repo/src:$repo/cancer_metastasis:$repo; python lineage_validation/00_audit.py --out $out"
if [[ "${1:-}" == --dry-run ]]; then echo "DRY RUN  $cmd"; exit 0; fi
mkdir -p "$root/logs"
sbatch -p gg -N 1 -n 1 -t 01:00:00 -A MCB26031 -J lv_audit -o "$root/logs/audit_%j.out" --wrap "test ! -f $out/SUCCESS || exit 2; $cmd"
