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
[[ "${1:-}" == --dry-run ]] && dry_run=1
repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
data=${QUINN_DATA:-/scratch/10119/ghzheng/OT_project/data/quinn2021_GSE161363}
env_path=${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}
raw=$data/raw
out=$data/converted

n=$(ls $raw/GSM*_matrix.*.mtx.gz 2> /dev/null | wc -l)
(( n > 0 )) || { echo "no GSM*_matrix.*.mtx.gz in $raw" >&2; exit 1; }
[[ -f "$out/conversion_report.json" ]] && { echo "$out/conversion_report.json exists; not rewritten" >&2; exit 2; }
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
