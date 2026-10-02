#!/bin/bash
#SBATCH -J cot_fact_deg
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 12:00:00
#SBATCH -A MCB26031
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/factdeg_%A_%a.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/factdeg_%A_%a.err
#
# Differential expression and hallmark GSEA for one arm of the preprocessing
# factorial, one array task per arm.
#
# The factorial's thirty-two arms were run to ask how much of the biological
# conclusion is a property of the preprocessing choice, and nothing ever
# carried them past the gate. The downstream chain that exists
# (submit_downstream_deg.sh) could not: it is six jobs per arm -- trim, a
# pseudobulk array over patients, DEG, GSEA, disqualifiers, leave-one-out --
# so thirty-two arms are over four hundred jobs against a cap of forty, and it
# reads gates from ot_<accession>_<label>_<stamp>, a layout the factorial does
# not use.
#
# Here one task does one arm end to end: trim the original manifest to the
# pairs this arm gated, pseudobulk every patient in sequence, the paired DEG,
# then GSEA. Thirty-two arms are thirty-two tasks.
#
# The disqualifiers and the leave-one-patient-out refit are not run. They are
# robustness checks on a result someone has chosen to report, and running them
# on every arm of a screen whose point is the comparison would be the
# expensive half of the chain spent on arms nobody will report. They belong to
# the arm the lock selects, through submit_downstream_deg.sh.
#
# The gates are read where the factorial wrote them. 21_ looks for
# <gate_root>/<pair>/scope_malignant/*/cell_confidence.csv and requires exactly
# one, which is the factorial's <block>/<label>/<pair>/scope_malignant/
# budget_0.95/ as it stands, so no stage below is modified.
#
# The expression stages read the ORIGINAL matrices whichever arm the gate came
# from, the same split submit_downstream_deg.sh makes and for its reason:
# equalisation was justified for the representation and the gate, never priced
# for what it costs the differential expression. So arms that differ only in
# _ds share their counts here and differ only in which cells are retained.
#
# GSEA is hallmark only. The GSEA job's default is four collections -- hallmark,
# GO BP, Reactome, WikiPathways -- and that default has to be overridden every
# time, so here it is the default and the four-collection file is reachable
# only by naming it.
#
# No set -u. TACC's module files and conda's shell functions read variables
# they do not define, and under set -u that kills the job with nothing in the
# log. Every expansion below carries its own default or an explicit check.
set -eo pipefail
source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
factorial=${CONFIDENCEOT_FACTORIAL_ROOT:-$result/preprocessing_factorial}
block=${CONFIDENCEOT_FACTORIAL_BLOCK:-}
accession=${CONFIDENCEOT_DOWNSTREAM_ACCESSION:-}
source_manifest=${CONFIDENCEOT_DOWNSTREAM_SOURCE_MANIFEST:-}
size_csv=${CONFIDENCEOT_METASTASIS_SIZE_CSV:-}
labels=(${CONFIDENCEOT_FACTORIAL_LABELS:-})
out_root=${CONFIDENCEOT_DOWNSTREAM_ROOT:-$result/downstream_factorial}
gmt_root=${CONFIDENCEOT_HUMAN_MSIGDB_ROOT:-$result/gene_sets/msigdb_2025.1_Hs}
gmt=${CONFIDENCEOT_HUMAN_GMT:-$gmt_root/h.all.v2025.1.Hs.symbols.gmt}
minimum_cells=${CONFIDENCEOT_MINIMUM_CELLS_PER_STATUS:-10}
force=${CONFIDENCEOT_DOWNSTREAM_FORCE:-0}

for name in block accession; do
    if [[ -z "${!name}" ]]; then
        echo "$name is required (CONFIDENCEOT_FACTORIAL_BLOCK, CONFIDENCEOT_DOWNSTREAM_ACCESSION)" >&2
        exit 2
    fi
done
for name in source_manifest size_csv gmt; do
    if [[ ! -f "${!name}" ]]; then
        echo "$name does not exist: '${!name}'" >&2
        exit 2
    fi
done
if (( ${#labels[@]} == 0 )); then
    echo "no labels: set CONFIDENCEOT_FACTORIAL_LABELS" >&2
    exit 2
fi

task=${SLURM_ARRAY_TASK_ID:?submit as an array over the labels}
if (( task >= ${#labels[@]} )); then
    echo "task $task is past the ${#labels[@]} labels; nothing to do"
    exit 0
fi
label=${labels[$task]}
gate=$factorial/$block/$label
out=$out_root/$block/$accession/$label
trimmed=$out/manifest_trimmed.csv
four_state=$out/four_state
deg=$out/deg
gsea=$out/gsea

echo "block      $block"
echo "accession  $accession"
echo "label      $label"
echo "gate       $gate"
echo "manifest   $source_manifest (original counts)"
echo "gene sets  $gmt"
echo "out        $out"

if [[ ! -d "$gate" ]]; then
    echo "no gate for this arm: $gate" >&2
    exit 2
fi
if [[ "$force" != "1" && -f "$out/DONE" ]]; then
    echo "already complete; set CONFIDENCEOT_DOWNSTREAM_FORCE=1 to redo"
    exit 0
fi

# The compartment the pseudobulk uses must be the one the gate used, or a
# patient's pseudobulk mixes two definitions of malignant.
extra=(--metastasis-size-csv "$size_csv")
if [[ -n "${CONFIDENCEOT_MALIGNANT_COLUMN:-}" && -n "${CONFIDENCEOT_MALIGNANT_ANNOTATIONS:-}" ]]; then
    echo "set CONFIDENCEOT_MALIGNANT_COLUMN or CONFIDENCEOT_MALIGNANT_ANNOTATIONS, not both" >&2
    exit 2
fi
if [[ -n "${CONFIDENCEOT_MALIGNANT_COLUMN:-}" ]]; then
    extra+=(--malignant-column "$CONFIDENCEOT_MALIGNANT_COLUMN")
elif [[ -n "${CONFIDENCEOT_MALIGNANT_ANNOTATIONS:-}" ]]; then
    IFS='|' read -r -a annotation_list <<< "$CONFIDENCEOT_MALIGNANT_ANNOTATIONS"
    for annotation in "${annotation_list[@]}"; do
        [[ -n "$annotation" ]] && extra+=(--malignant-annotation "$annotation")
    done
else
    echo "the malignant compartment is unset: CONFIDENCEOT_MALIGNANT_COLUMN or CONFIDENCEOT_MALIGNANT_ANNOTATIONS" >&2
    exit 2
fi

cd "$repo"
export PYTHONPATH="$repo/src:$repo/cancer_metastasis${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
export PYTHONUNBUFFERED=1
rm -rf "$four_state" "$deg" "$gsea"
mkdir -p "$out" "$four_state" "$deg" "$gsea"

stamp() { echo "===== $1 $(date --iso-8601=seconds) ====="; }

stamp "trim"
python cancer_metastasis/35_trim_manifest_to_gate.py \
    "$source_manifest" "$trimmed" --gate-root "$gate" --dataset-id "$accession"

stamp "pseudobulk"
patients=$(python cancer_metastasis/21_prepare_four_state_malignant_pseudobulk.py \
    "$trimmed" "$gate" "$four_state" --print-patient-count "${extra[@]}")
echo "patients   $patients"
# Anything but a positive integer here -- a warning printed to stdout, an
# empty trim -- would make the loop below run zero times in silence, and the
# DEG would then fit an empty cohort and the arm would be marked DONE.
if ! [[ "$patients" =~ ^[0-9]+$ ]] || (( patients < 1 )); then
    echo "patient count is not a positive integer: '$patients'" >&2
    echo "  the trimmed manifest may hold no pairs of $accession gated by $label" >&2
    exit 2
fi
failures=0
for (( index = 0; index < patients; index++ )); do
    if ! python cancer_metastasis/21_prepare_four_state_malignant_pseudobulk.py \
        "$trimmed" "$gate" "$four_state" --index "$index" "${extra[@]}"; then
        failures=$(( failures + 1 ))
        echo "  patient $index failed" >&2
    fi
done
# A missing patient does not make the DEG fail: it is silently absent from the
# fit. So a failure here stops the arm rather than producing a result that
# looks complete and is not.
if (( failures > 0 )); then
    echo "$failures of $patients patients failed to pseudobulk; not fitting a partial cohort" >&2
    exit 1
fi

stamp "paired DEG"
python cancer_metastasis/13_run_paired_pydeseq2.py \
    "$four_state" "$deg" \
    --minimum-cells-per-patient-status "$minimum_cells" \
    --minimum-total-count 10 \
    --maximum-fdr 0.05 \
    --absolute-log2-fold-change-thresholds 0.5 1.0 \
    --n-cpus 16

stamp "GSEA, hallmark"
python cancer_metastasis/15_run_pydeseq2_gseapy.py \
    "$deg" "$gmt" "$gsea" \
    --permutations 2000 --threads 16 --minimum-size 10 --maximum-size 500

printf 'block=%s\naccession=%s\nlabel=%s\ngate=%s\ngene_sets=%s\npatients=%s\n' \
    "$block" "$accession" "$label" "$gate" "$gmt" "$patients" > "$out/DONE"
stamp "done"
