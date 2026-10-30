#!/bin/bash
# Download every GSE297328 supplementary file from GEO (about 28 GB) and check each size against the server's Content-Length.
#
#   nohup bash cancer_application/GSE297328_longitudinal/scripts/download_gse297328.sh > download_gse297328.log 2>&1 &
#
# Resumable (wget -c): rerun the same command after an interruption. Extracts GSE297328_RAW.tar into raw/ once its size
# matches. Writes $data/download_check.tsv (file, expected bytes, local bytes, ok). No set -e / -u / pipefail (TACC rules).

data=${GSE297328_DATA:-/scratch/10119/ghzheng/OT_project/data/ltl331_GSE297328}
url=https://ftp.ncbi.nlm.nih.gov/geo/series/GSE297nnn/GSE297328/suppl
files="GSE297328_README.txt filelist.txt GSE297328_RAW.tar GSE297328_ltl331_annotated.h5ad GSE297328_ltl331_harmony_integrated.h5ad GSE297328_ltl331_velocity.h5ad GSE297328_LTL331.RDS"
mkdir -p "$data/raw" || exit 1
cd "$data" || exit 1
printf "file\texpected_bytes\tlocal_bytes\tok\n" > download_check.tsv
for f in $files; do
    echo "$(date '+%F %T') downloading $f"
    wget -c -q "$url/$f" -O "$f"
    expected=$(curl -sIL "$url/$f" | tr -d '\r' | awk 'tolower($1)=="content-length:"{n=$2} END{print n}')
    local_size=$(stat -c %s "$f" 2>/dev/null)
    ok=no
    [[ -n "$expected" && "$expected" == "$local_size" ]] && ok=yes
    printf "%s\t%s\t%s\t%s\n" "$f" "$expected" "$local_size" "$ok" >> download_check.tsv
    echo "$(date '+%F %T') $f expected $expected local $local_size ok $ok"
done
if grep -q "^GSE297328_RAW.tar.*yes$" download_check.tsv; then
    tar -xf GSE297328_RAW.tar -C raw && echo "RAW.tar extracted: $(ls raw | wc -l) archives in $data/raw"
else
    echo "RAW.tar size does not match; not extracted"
fi
column -t download_check.tsv
