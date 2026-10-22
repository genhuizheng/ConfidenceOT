#!/bin/bash
# One frozen joint UMAP per patient (53_patient_umap.py), for the patient-by-patient view of the
# five CNV / RNA strategies, then a small tarball of the coordinates for download.
#
#   bash cancer_metastasis/submit_patient_umap.sh --dry-run
#   bash cancer_metastasis/submit_patient_umap.sh [--export export_20261021b] [--tasks 8]
#
# Cells: every reference cell of every pair of the patient in the strategy export package
# (<export>/cells), the union by cell_uid. Counts: the original h5ads named by the manifests
# 48_reference_umap.py reads. 48_'s recipe and seed. A patient with DONE is skipped, so a
# resubmission redoes only failed patients. No OT is run; no gate result is read or written.
#
# Output: <result>/patient_umap_<export>/patients/<dataset>/<patient>/ and, after the array
# (whatever its state; failed patients are listed in the log), patient_umap_<export>.tar.gz + md5.
#
# No set -u, as everywhere in this repo.
set -eo pipefail

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
env_path=${CONFIDENCEOT_ENV_EXPORT:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}
uniform_manifest=$result/manifest/pancancer_20260924/pair_manifest_eligible.csv
prostate_manifest=$result/prepared_GSE271675_20260916/pair_manifest_eligible.csv
export_name=export_20261021b
tasks=8
dry_run=0
while (( $# )); do
    case "$1" in
        --export) export_name=$2; shift 2 ;;
        --tasks) tasks=$2; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
cells=$result/cnv_rna_strategies_export/$export_name/cells
out=$result/patient_umap_$export_name
logs=$out/logs

problems=0
for path in "$uniform_manifest" "$prostate_manifest" "$cells" "$repo/cancer_metastasis/53_patient_umap.py"; do
    [[ -e "$path" ]] || { echo "missing: $path" >&2; problems=1; }
done
[[ -f "$out.tar.gz" ]] && { echo "$out.tar.gz exists; not rewritten" >&2; problems=1; }
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

array=$(submit -p gg -N 1 -n 1 -c 16 -t 04:00:00 -A MCB26031 -J pu_umap --array=0-$((tasks - 1)) \
    -o "$logs/umap_%A_%a.out" -e "$logs/umap_%A_%a.err" \
    --wrap "$activate && python cancer_metastasis/53_patient_umap.py --cells $cells --manifest $uniform_manifest --manifest $prostate_manifest --out $out --task-index \$SLURM_ARRAY_TASK_ID --task-count $tasks")
echo "patient UMAPs: $array ($tasks tasks) -> $out/patients" >&2
pack="cd $out && echo \"DONE \$(ls patients/*/*/DONE | wc -l), FAILED \$(ls patients/*/*/FAILED 2> /dev/null | wc -l)\" && (ls patients/*/*/FAILED 2> /dev/null || true) && tar -czf $out.tar.gz patients && md5sum $out.tar.gz > $out.tar.gz.md5 && cat $out.tar.gz.md5 && du -h $out.tar.gz"
id=$(submit --dependency=afterany:$array -p gg -N 1 -n 1 -c 4 -t 01:00:00 -A MCB26031 -J pu_pack \
    -o "$logs/pack_%j.out" -e "$logs/pack_%j.err" --wrap "$pack")
echo "pack: $id -> $out.tar.gz" >&2
echo "commit $(git -C "$repo" rev-parse --short HEAD 2> /dev/null || echo unknown); cells $cells"
