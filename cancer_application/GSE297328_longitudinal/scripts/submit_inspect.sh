#!/bin/bash
# Read-only inspection of the GSE297328 download on a gh-dev node (00_inspect_data.py).
#
#   bash cancer_application/GSE297328_longitudinal/scripts/submit_inspect.sh [--dry-run]
#
# Reads $data (nothing written there); writes $root/audit/inspect_<time>/ (inspect_report.json, INSPECT_SUMMARY.md).
# Download first, on the login node (commands at the end). No set -e / -u / pipefail (TACC rules).

dry_run=0
[[ "${1:-}" == --dry-run ]] && dry_run=1
repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
data=${GSE297328_DATA:-/scratch/10119/ghzheng/OT_project/data/ltl331_GSE297328}
root=${GSE297328_ROOT:-/scratch/10119/ghzheng/OT_project/gse297328_longitudinal}
env_path=${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}
out=$root/audit/inspect_$(date +%Y%m%d_%H%M%S)

[[ -f "$data/GSE297328_ltl331_annotated.h5ad" ]] || { echo "missing $data/GSE297328_ltl331_annotated.h5ad (download first)" >&2; exit 1; }
[[ -d "$data/raw" ]] || { echo "missing $data/raw (extract GSE297328_RAW.tar first)" >&2; exit 1; }
cmd="source /home1/10119/ghzheng/.bashrc; conda activate $env_path; cd $repo; python cancer_application/GSE297328_longitudinal/scripts/00_inspect_data.py --data $data --out $out"
if (( dry_run )); then echo "DRY RUN  sbatch -p gh-dev -N 1 -n 1 -t 00:40:00 -A MCB26031 -J g297_insp --wrap \"$cmd\""; exit 0; fi
mkdir -p "$root/logs" || exit 1
id=$(sbatch --parsable -p gh-dev -N 1 -n 1 -t 00:40:00 -A MCB26031 -J g297_insp -o "$root/logs/inspect_%j.out" --wrap "$cmd" | tail -n 1)
[[ "$id" =~ ^[0-9]+$ ]] || { echo "sbatch returned no job id" >&2; exit 1; }
echo "inspection job $id -> $out (log $root/logs/inspect_$id.out)"

# Download, once, on the login node (about 3.3 GB; velocity 8.8 GB is optional and not needed for the inspection):
#   D=/scratch/10119/ghzheng/OT_project/data/ltl331_GSE297328 && mkdir -p $D/raw && cd $D
#   U=https://ftp.ncbi.nlm.nih.gov/geo/series/GSE297nnn/GSE297328/suppl
#   wget -q $U/GSE297328_README.txt $U/filelist.txt $U/GSE297328_RAW.tar $U/GSE297328_ltl331_annotated.h5ad
#   test $(stat -c %s GSE297328_RAW.tar) -eq 694087680 && tar -xf GSE297328_RAW.tar -C raw
