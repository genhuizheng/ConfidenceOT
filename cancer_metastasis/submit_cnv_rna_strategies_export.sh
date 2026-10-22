#!/bin/bash
# The download package for the five CNV / RNA strategies (51_strategy_export.py), and the joint
# UMAPs it needs, on TACC.
#
#   bash cancer_metastasis/submit_cnv_rna_strategies_export.sh --dry-run
#   bash cancer_metastasis/submit_cnv_rna_strategies_export.sh [--scope three|all] [--name NAME]
#
# UMAP: one frozen joint UMAP per pair. The three cancers' pairs reuse the reference UMAP the
# cycling x CNV export already computed (reference_umap_cycling_cnv_acc0.99/pairs): the same
# malignant cells the strategies gate, so nothing is recomputed. With --scope all, the uniform
# block's other datasets have none, so 48_reference_umap.py pairs computes them once, from the
# strategies' own gates (acc0.99), into strategy_umap/pairs; a pair with a DONE file is skipped.
#
# Export: 51_strategy_export.py over every acceptance -> <result>/cnv_rna_strategies_export/<NAME>/,
# then <NAME>.tar.gz and its md5 beside it. An export with MANIFEST.tsv is never rewritten.
# Couplings are not in the package; they stay beside the gates.
#
# No set -u, as everywhere in this repo.
set -eo pipefail

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
env_path=${CONFIDENCEOT_ENV_EXPORT:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}
# the manifests the strategy OT ran on (submit_cnv_rna_strategies.sh); the pancancer eligible
# manifest is wider (for the three cancers it also lists 36 ovarian, 2 colorectal and a separate
# 70-pair GSE271675 pairing, none of which any gate ran on) and is only reported against
uniform_manifest=$result/manifest/factorial_20260929/uniform.csv
prostate_manifest=$result/prepared_GSE271675_20260916/pair_manifest_eligible.csv
eligible_manifest=$result/manifest/pancancer_20260924/pair_manifest_eligible.csv
gates=$result/cnv_rna_strategies_infercnvpy
down=$result/downstream_cnv_rna_strategies
old_umap=$result/reference_umap_cycling_cnv_acc0.99
new_umap=$result/strategy_umap
three="GSE180661 GSE225857 GSE271675"
others="HRA000036 GSE188737 GSE197177 GSE181919 GSE178318 GSE315534 GSE167036 GSE163558"
acceptances="0.99 0.95 0.90"
scope=three
name=export_$(date +%Y%m%d)
dry_run=0
tasks=10
while (( $# )); do
    case "$1" in
        --scope) scope=$2; shift 2 ;;
        --name) name=$2; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[[ "$scope" == three || "$scope" == all ]] || { echo "--scope three or all" >&2; exit 2; }
datasets=$three
[[ "$scope" == all ]] && datasets="$three $others"
out=$result/cnv_rna_strategies_export/$name
logs=$result/cnv_rna_strategies_export/logs

problems=0
for path in "$uniform_manifest" "$prostate_manifest" "$eligible_manifest" "$gates" "$old_umap/pairs" "$repo/cancer_metastasis/51_strategy_export.py"; do
    [[ -e "$path" ]] || { echo "missing: $path" >&2; problems=1; }
done
[[ -f "$out/MANIFEST.tsv" ]] && { echo "$out/MANIFEST.tsv exists; not rewritten (use --name)" >&2; problems=1; }
running=$( { squeue -u "$USER" -h -o "%j" 2> /dev/null || true; } | grep -cE '^(st_|dg_)' || true)
(( running > 0 )) && { echo "$running strategy OT or downstream job(s) still queued or running" >&2; problems=1; }
(( problems && ! dry_run )) && exit 2

activate="source /home1/10119/ghzheng/.bashrc; conda activate $env_path; cd $repo; export PYTHONPATH=$repo/src:$repo/cancer_metastasis:$repo; export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMBA_NUM_THREADS=16 MPLBACKEND=Agg PYTHONUNBUFFERED=1"
manifests="--manifest $uniform_manifest --manifest $prostate_manifest"
submit() {
    if (( dry_run )); then echo "DRY RUN  sbatch $*" | cut -c1-400 >&2; echo 0; return 0; fi
    local id
    id=$(sbatch --parsable "$@" | tail -n 1 | tr -dc '0-9')
    [[ -n "$id" ]] || { echo "sbatch returned no job id; stopping" >&2; exit 1; }
    echo "$id"
}
(( dry_run )) || mkdir -p "$logs"

wait_for=()
umap_roots="--umap-root $old_umap"
if [[ "$scope" == all ]]; then
    array=$(submit -p gg -N 1 -n 1 -c 16 -t 04:00:00 -A MCB26031 -J sx_umap --array=0-$((tasks - 1)) \
        -o "$logs/umap_%A_%a.out" -e "$logs/umap_%A_%a.err" \
        --wrap "$activate && python cancer_metastasis/48_reference_umap.py pairs $manifests --datasets $others --gates-root $gates/acc0.99 --out $new_umap --methods M4-E --task-index \$SLURM_ARRAY_TASK_ID --task-count $tasks")
    echo "UMAP for the other uniform datasets: $array ($tasks tasks) -> $new_umap/pairs" >&2
    wait_for=(--dependency=afterany:$array)
    umap_roots="$umap_roots --umap-root $new_umap"
fi
archive="tar -C $(dirname "$out") -czf $out.tar.gz $name && md5sum $out.tar.gz > $out.tar.gz.md5 && cat $out.tar.gz.md5 && du -h $out.tar.gz"
id=$(submit "${wait_for[@]}" -p gg -N 1 -n 1 -c 16 -t 06:00:00 -A MCB26031 -J sx_export \
    -o "$logs/export_%j.out" -e "$logs/export_%j.err" \
    --wrap "test ! -f $out/MANIFEST.tsv || exit 2; $activate && python cancer_metastasis/51_strategy_export.py $manifests --datasets $datasets --gates-root $gates --acceptances $acceptances $umap_roots --downstream-root $down --eligible-manifest $eligible_manifest --out $out && $archive")
echo "export: $id -> $out (and $out.tar.gz)" >&2
echo "commit $(git -C "$repo" rev-parse --short HEAD 2> /dev/null || echo unknown); scope $scope; datasets $datasets"
