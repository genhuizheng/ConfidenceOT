#!/bin/bash
# Submit DEG and hallmark GSEA for every arm of one dataset in the
# preprocessing factorial, as one array over the arms.
#
#   bash cancer_metastasis/submit_factorial_downstream.sh ovarian
#   DRY_RUN=1 bash cancer_metastasis/submit_factorial_downstream.sh prostate
#
# The arms are discovered from the block on disk, not taken from a list: the
# prostate block has 25 of its 32 arms while seven are being re-run, and an
# array over 32 would spend seven tasks failing on gates that do not exist.
# They are kept in the factorial's own order, so task N is the same arm every
# time and the heatmap rows come out grouped by axis.
#
# The lesion-size table is taken from the block's own record. Each block wrote
# PREDOWNSAMPLE_DEPTH, the depth table beside the 27_downsample_counts.py run
# that equalised it, and downsample_per_sample.csv sits in the same directory.
# Deriving it from the record rather than from a dated name means it cannot be
# a different run's table.
#
# Then, once every arm is DONE:
#
#   python cancer_metastasis/41_preprocessing_strategy_heatmaps.py OUT \
#       $(for d in $ROOT/*/; do l=$(basename $d); \
#         echo --deg $l=$d/deg/contrasts/primary_rejected_vs_primary_retained/pydeseq2_all_gene_discovery.csv \
#              --gsea $l=$d/gsea/contrasts/primary_rejected_vs_primary_retained/all_gene_discovery_gsea_results.csv; done)
#
# which this script prints with the paths filled in.
#
# No set -u, for the same reason as every script here.
set -eo pipefail

dataset=${1:-}
repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
factorial=${CONFIDENCEOT_FACTORIAL_ROOT:-$result/preprocessing_factorial}
out_root=${CONFIDENCEOT_DOWNSTREAM_ROOT:-$result/downstream_factorial}
gmt_root=${CONFIDENCEOT_HUMAN_MSIGDB_ROOT:-$result/gene_sets/msigdb_2025.1_Hs}
gmt=${CONFIDENCEOT_HUMAN_GMT:-$gmt_root/h.all.v2025.1.Hs.symbols.gmt}
partition=${PARTITION:-gg}
cap=${GG_JOB_CAP:-40}
pancancer=${PANCANCER_MANIFEST:-$result/manifest/pancancer_20260924/pair_manifest_eligible.csv}

# The accession, the block it was gated in, the manifest whose ORIGINAL counts
# the expression stages read, and the malignant compartment -- which must be
# the one the gate used.
#
# colorectal is the cancer, not a deposit: its evaluable pairs are split over
# three deposits (GSE225857 4, GSE178318 2, GSE315534 1), none of which can
# carry a paired fit alone, so they are merged into one analysis named for the
# cancer. One paired fit over all of them is valid because each patient sits
# inside one deposit; the job refuses the merge if a patient_id is shared.
# The single deposits stay reachable under their own names.
malignant_column=
annotations=
# Not "name": the file check below loops over a variable of that name, and a
# loop variable keeps its last value -- which once sent every analysis into
# one directory called gmt.
analysis=
case "$dataset" in
    ovarian)     accession=GSE180661; block=uniform; source=$pancancer; malignant_column=malignant ;;
    colorectal)  accession="GSE225857 GSE178318 GSE315534"; analysis=colorectal
                 block=uniform; source=$pancancer; malignant_column=malignant ;;
    colorectal_GSE225857) accession=GSE225857; block=uniform; source=$pancancer; malignant_column=malignant ;;
    colorectal_GSE178318) accession=GSE178318; block=uniform; source=$pancancer; malignant_column=malignant ;;
    colorectal_GSE315534) accession=GSE315534; block=uniform; source=$pancancer; malignant_column=malignant ;;
    breast)      accession=GSE167036; block=uniform; source=$pancancer; malignant_column=malignant ;;
    gastric)     accession=GSE163558; block=uniform; source=$pancancer; malignant_column=malignant ;;
    pancreatic)  accession=GSE197177; block=uniform; source=$pancancer; malignant_column=malignant ;;
    headneck)    accession=GSE181919; block=uniform; source=$pancancer; malignant_column=malignant ;;
    headneck2)   accession=GSE188737; block=uniform; source=$pancancer; malignant_column=malignant ;;
    prostate)
        accession=GSE271675; block=prostate
        source=$result/prepared_GSE271675_20260916/pair_manifest_eligible.csv
        annotations="Epithelial|Basal Epithelial|Neuroendocrine" ;;
    *)
        echo "Usage: $0 {ovarian|prostate|colorectal|colorectal_GSE225857|colorectal_GSE178318|colorectal_GSE315534|breast|gastric|pancreatic|headneck|headneck2}" >&2
        exit 2 ;;
esac
# A single deposit is named by its accession, as before.
analysis=${analysis:-$accession}

block_dir=$factorial/$block
if [[ ! -d "$block_dir" ]]; then
    echo "no factorial block at $block_dir" >&2
    exit 2
fi
if [[ ! -f "$block_dir/PREDOWNSAMPLE_DEPTH" ]]; then
    echo "$block_dir has no PREDOWNSAMPLE_DEPTH record, so its lesion-size table cannot be found" >&2
    exit 2
fi
size_csv=$(dirname "$(cat "$block_dir/PREDOWNSAMPLE_DEPTH")")/downsample_per_sample.csv
for name in source size_csv gmt; do
    if [[ ! -f "${!name}" ]]; then
        echo "$name does not exist: ${!name}" >&2
        [[ "$name" == gmt ]] && echo "  prepare it once: python cancer_metastasis/14_prepare_human_msigdb_gmt.py $gmt_root" >&2
        exit 2
    fi
done

# The factorial's order: normalisation outermost, then the binary tags.
canonical="raw raw_cos raw_ds raw_ds_cos raw_noscale raw_noscale_cos raw_noscale_ds raw_noscale_ds_cos \
logcpm logcpm_cos logcpm_ds logcpm_ds_cos logcpm_noscale logcpm_noscale_cos logcpm_noscale_ds logcpm_noscale_ds_cos \
rank256 rank256_cos rank256_ds rank256_ds_cos rank256_noscale rank256_noscale_cos rank256_noscale_ds rank256_noscale_ds_cos \
ranknm256 ranknm256_cos ranknm256_ds ranknm256_ds_cos ranknm256_noscale ranknm256_noscale_cos ranknm256_noscale_ds ranknm256_noscale_ds_cos"
labels=()
absent=()
shopt -s nullglob
for label in ${CONFIDENCEOT_FACTORIAL_LABELS:-$canonical}; do
    gates=("$block_dir/$label"/*/scope_malignant/*/cell_confidence.csv)
    if (( ${#gates[@]} > 0 )); then labels+=("$label"); else absent+=("$label"); fi
done
if (( ${#labels[@]} == 0 )); then
    echo "no arm of block $block has a gate" >&2
    exit 2
fi

# Several arms per task: the job reads its node's core count and runs the
# pseudobulk of every (arm, patient) it owns in one pool, so a task with four
# arms costs one job against the cap rather than four.
arms_per_task=${ARMS_PER_TASK:-4}
tasks=$(( (${#labels[@]} + arms_per_task - 1) / arms_per_task ))

queued=0
[[ -z "${DRY_RUN:-}" ]] && queued=$(squeue -u "$USER" -p "$partition" -h -r 2>/dev/null | wc -l) || true
if (( queued + tasks > cap )); then
    echo "refusing: $tasks tasks with $queued already queued on $partition is over $cap" >&2
    exit 3
fi

echo "dataset      $dataset ($accession), block $block"
echo "arms         ${#labels[@]} with a gate${absent:+, absent: ${absent[*]}}"
echo "tasks        $tasks, $arms_per_task arms each"
echo "manifest     $source (original counts)"
echo "lesion size  $size_csv"
echo "gene sets    $gmt"
if [[ -n "$malignant_column" ]]; then
    echo "malignant    $malignant_column == malignant (uniform inferCNV)"
else
    echo "malignant    cell_type in $annotations (deposit labels)"
fi
echo "output       $out_root/$block/$analysis/<label>/"

export_list="ALL,CONFIDENCEOT_REPO=$repo,CANCER_COT_ROOT=$result,CONFIDENCEOT_FACTORIAL_ROOT=$factorial"
export_list+=",CONFIDENCEOT_FACTORIAL_BLOCK=$block,CONFIDENCEOT_DOWNSTREAM_ACCESSION=$accession"
export_list+=",CONFIDENCEOT_DOWNSTREAM_NAME=$analysis"
export_list+=",CONFIDENCEOT_DOWNSTREAM_SOURCE_MANIFEST=$source,CONFIDENCEOT_METASTASIS_SIZE_CSV=$size_csv"
export_list+=",CONFIDENCEOT_DOWNSTREAM_ROOT=$out_root,CONFIDENCEOT_HUMAN_GMT=$gmt"
export_list+=",CONFIDENCEOT_FACTORIAL_LABELS=${labels[*]}"
export_list+=",CONFIDENCEOT_DOWNSTREAM_ARMS_PER_TASK=$arms_per_task"
if [[ -n "$malignant_column" ]]; then
    export_list+=",CONFIDENCEOT_MALIGNANT_COLUMN=$malignant_column"
else
    export_list+=",CONFIDENCEOT_MALIGNANT_ANNOTATIONS=$annotations"
fi

command=(sbatch --parsable --partition="$partition" --array=0-$(( tasks - 1 ))
         --export="$export_list" "$repo/cancer_metastasis/tacc_factorial_downstream.slurm")
if [[ -n "${DRY_RUN:-}" ]]; then
    echo
    echo "DRY_RUN ${command[*]}"
else
    job=$("${command[@]}" | tail -n 1 | tr -dc '0-9_')
    echo
    echo "submitted    $job (array 0-$(( tasks - 1 )), $arms_per_task arms per task)"
fi

echo
echo "heatmaps, once every arm has its DONE file:"
root=$out_root/$block/$analysis
contrast=primary_rejected_vs_primary_retained
printf 'python %s/cancer_metastasis/41_preprocessing_strategy_heatmaps.py %s/heatmaps' "$repo" "$root"
for label in "${labels[@]}"; do
    printf ' \\\n  --deg %s=%s/%s/deg/contrasts/%s/pydeseq2_all_gene_discovery.csv' "$label" "$root" "$label" "$contrast"
    printf ' \\\n  --gsea %s=%s/%s/gsea/contrasts/%s/all_gene_discovery_gsea_results.csv' "$label" "$root" "$label" "$contrast"
done
echo
