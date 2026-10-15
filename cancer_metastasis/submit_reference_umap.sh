#!/bin/bash
# Frozen reference UMAP per pair (RNA, plus a CNV embedding from the inferCNV
# profiles where they exist), with every cycling x CNV gate's labels on it.
#
#   bash cancer_metastasis/submit_reference_umap.sh [--tasks N] [--dry-run]
#
# 1. An array of N tasks (default 10), each computing the reference UMAP of
#    every N-th pair of GSE180661, GSE225857 and GSE271675
#    (48_reference_umap.py pairs). A pair with DONE is skipped, so a resubmission
#    after a failure only redoes the failed pairs.
# 2. After the array, whatever its exit state (failed pairs are listed, not
#    hidden): the export (48_reference_umap.py export) into <out>/export with
#    MANIFEST.tsv and MD5SUMS, then one tarball and its md5 for download.
#
# Reads the existing gates and the original counts; no OT is rerun and no gate
# result is written. Refuses if <out>/export/MANIFEST.tsv already exists.
#
# No set -u, as everywhere in this repo.
set -eo pipefail

tasks=10
dry_run=0
while (( $# )); do
    case "$1" in
        --tasks) tasks=$2; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
env_path=${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}
uniform_manifest=$result/manifest/pancancer_20260924/pair_manifest_eligible.csv
prostate_manifest=$result/prepared_GSE271675_20260916/pair_manifest_eligible.csv
gates=$result/cycling_cnv_acc0.99
cnv_root=$result/cnv_profiles/infercnv_r_patient
out=$result/reference_umap_cycling_cnv_acc0.99
logs=$out/logs
tarball=$out/reference_umap_export.tar.gz

for path in "$uniform_manifest" "$prostate_manifest"; do
    if [[ ! -f "$path" ]]; then echo "missing: $path" >&2; (( dry_run )) || exit 1; fi
done
if [[ ! -d "$gates" ]]; then echo "missing: $gates" >&2; (( dry_run )) || exit 1; fi
if [[ -f "$out/export/MANIFEST.tsv" ]]; then
    echo "$out/export/MANIFEST.tsv exists; the export is not rewritten" >&2
    exit 2
fi
(( dry_run )) || mkdir -p "$logs"

common="--manifest $uniform_manifest --manifest $prostate_manifest --datasets GSE180661 GSE225857 GSE271675 --gates-root $gates --out $out --cnv-root $cnv_root"
activate="source /home1/10119/ghzheng/.bashrc; conda activate $env_path; cd $repo; export PYTHONPATH=$repo/src:$repo/cancer_metastasis:$repo; export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMBA_NUM_THREADS=16 MPLBACKEND=Agg PYTHONUNBUFFERED=1"

submit() {
    local what=$1; shift
    if (( dry_run )); then
        echo "DRY RUN  $what" >&2; echo "         sbatch $*" >&2; echo 0; return 0
    fi
    local id
    id=$(sbatch --parsable "$@" | tail -n 1 | tr -dc '0-9')
    if [[ -z "$id" ]]; then echo "sbatch returned no job id for $what" >&2; exit 1; fi
    echo "submitted $what as $id" >&2
    echo "$id"
}

array=$(submit "reference UMAP, $tasks tasks" -p gg -N 1 -n 1 -c 16 -t 02:00:00 -A MCB26031 -J ref_umap \
    --array=0-$((tasks - 1)) -o "$logs/pairs_%A_%a.out" -e "$logs/pairs_%A_%a.err" \
    --wrap "$activate && python cancer_metastasis/48_reference_umap.py pairs $common --task-index \$SLURM_ARRAY_TASK_ID --task-count $tasks")

archive="tar -C $out -czf $tarball export && cd $out && md5sum $(basename $tarball) > $(basename $tarball).md5 && cat $(basename $tarball).md5"
dependency=()
(( dry_run )) || dependency=(--dependency=afterany:$array)
submit "export and archive" -p gg -N 1 -n 1 -c 16 -t 02:00:00 -A MCB26031 -J ref_umap_export "${dependency[@]}" \
    -o "$logs/export_%j.out" -e "$logs/export_%j.err" \
    --wrap "test ! -f $out/export/MANIFEST.tsv || exit 2; $activate && python cancer_metastasis/48_reference_umap.py export $common && $archive" > /dev/null

echo "jobs: $tasks array tasks + 1 export"
echo "out $out"
echo "commit $(git -C "$repo" rev-parse --short HEAD 2> /dev/null || echo unknown)"
