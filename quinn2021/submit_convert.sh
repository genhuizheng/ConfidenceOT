#!/bin/bash
# Convert GSE161363 (Quinn et al. 2021) on a compute node.
#
#   bash quinn2021/submit_convert.sh [--dry-run]
#
# Expects GSE161363_RAW.tar already downloaded and extracted in <data>/raw (see the
# commands at the end). Writes <data>/converted/{h5ad,lineage,conversion_report.json} with
# quinn2021/01_convert_geo.py; refuses if conversion_report.json already exists.
# No set -u, as everywhere in this repo.
set -eo pipefail

dry_run=0
replace=0
for a in "$@"; do
    case "$a" in
        --dry-run) dry_run=1 ;;
        --replace) replace=1 ;;   # keep the earlier conversion as converted_<time>, then convert again
        *) echo "unknown option: $a" >&2; exit 2 ;;
    esac
done
repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
data=${QUINN_DATA:-/scratch/10119/ghzheng/OT_project/data/quinn2021_GSE161363}
env_path=${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}
raw=$data/raw
out=$data/converted

n=$(ls $raw/GSM*_matrix.*.mtx.gz 2> /dev/null | wc -l)
(( n > 0 )) || { echo "no GSM*_matrix.*.mtx.gz in $raw" >&2; exit 1; }
if [[ -f "$out/conversion_report.json" ]]; then
    if (( ! replace )); then echo "$out/conversion_report.json exists; not rewritten (--replace keeps it aside and converts again)" >&2; exit 2; fi
    kept=${out}_$(date +%Y%m%d_%H%M%S)
    if (( dry_run )); then echo "DRY RUN  would move $out to $kept" >&2; else mv "$out" "$kept" && echo "earlier conversion kept as $kept" >&2; fi
fi
echo "$n experiments in $raw" >&2
cmd="source /home1/10119/ghzheng/.bashrc; conda activate $env_path; cd $repo; python quinn2021/01_convert_geo.py --raw $raw --out $out"
if (( dry_run )); then echo "DRY RUN  $cmd"; exit 0; fi
mkdir -p "$data/logs"
sbatch -p gg -N 1 -n 1 -t 01:00:00 -A MCB26031 -J quinn_conv -o "$data/logs/convert_%j.out" --wrap "$cmd"

# Download and extract, once, on the login node (782 MB):
#   D=/scratch/10119/ghzheng/OT_project/data/quinn2021_GSE161363 && mkdir -p $D/raw && cd $D
#   wget -q https://ftp.ncbi.nlm.nih.gov/geo/series/GSE161nnn/GSE161363/suppl/GSE161363_RAW.tar
#   wget -q https://ftp.ncbi.nlm.nih.gov/geo/series/GSE161nnn/GSE161363/suppl/filelist.txt
#   test $(stat -c %s GSE161363_RAW.tar) -eq 820428800 && tar -xf GSE161363_RAW.tar -C raw
