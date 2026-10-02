#!/bin/bash
#SBATCH -J cot_fact_deg
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
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

# ---- which arms this task owns ---------------------------------------------
# Several arms per task, because one arm per task left a node doing one
# patient at a time. The arms of one task share the node and, more usefully,
# its page cache: every arm reads the same ORIGINAL counts for a patient and
# differs only in which gate it joins, so the pseudobulk below runs patient by
# patient across the arms and the reads after the first are served from memory
# rather than from Lustre.
#
# The core count is read off the node. No -c is requested: the node is
# allocated whole either way, and a number written here would only be a guess
# at what the node already says.
arms_per_task=${CONFIDENCEOT_DOWNSTREAM_ARMS_PER_TASK:-4}
cores=${CONFIDENCEOT_DOWNSTREAM_CORES:-$(nproc)}
threads_per_worker=${CONFIDENCEOT_DOWNSTREAM_THREADS_PER_WORKER:-4}
workers=${CONFIDENCEOT_DOWNSTREAM_WORKERS:-$(( cores / threads_per_worker ))}
(( workers < 1 )) && workers=1

task=${SLURM_ARRAY_TASK_ID:?submit as an array over the labels}
first=$(( task * arms_per_task ))
if (( first >= ${#labels[@]} )); then
    echo "task $task starts at arm $first, past the ${#labels[@]} labels; nothing to do"
    exit 0
fi
mine=("${labels[@]:first:arms_per_task}")

echo "block      $block"
echo "accession  $accession"
echo "arms       ${mine[*]}"
echo "manifest   $source_manifest (original counts)"
echo "gene sets  $gmt"
echo "parallel   $workers pseudobulk workers x $threads_per_worker threads on $cores cores"

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
export PYTHONUNBUFFERED=1
stamp() { echo "===== $1 $(date --iso-8601=seconds) ====="; }
# One arm's directory, recomputed from the label wherever it is needed, so a
# background job never reads a path left in a variable by the loop that
# launched it.
arm_dir() { echo "$out_root/$block/$accession/$1"; }
fail_arm() {
    # $1 = label, rest = reason. The arm stops; the task carries on with the
    # others, which share nothing with it but the node.
    local label=$1
    shift
    echo "$*" >> "$(arm_dir "$label")/FAILED"
    echo "===== ARM FAILED $label: $* =====" >&2
}

# ---- 1. per arm: trim and count -------------------------------------------
declare -A patients_of=()
active=()
for label in "${mine[@]}"; do
    out=$(arm_dir "$label")
    gate=$factorial/$block/$label
    if [[ "$force" != "1" && -f "$out/DONE" ]]; then
        echo "$label already complete; set CONFIDENCEOT_DOWNSTREAM_FORCE=1 to redo"
        continue
    fi
    rm -rf "$out"
    mkdir -p "$out/four_state" "$out/deg" "$out/gsea" "$out/logs"
    if [[ ! -d "$gate" ]]; then
        fail_arm "$label" "no gate at $gate"
        continue
    fi
    stamp "trim $label"
    if ! python cancer_metastasis/35_trim_manifest_to_gate.py \
        "$source_manifest" "$out/manifest_trimmed.csv" \
        --gate-root "$gate" --dataset-id "$accession" > "$out/logs/trim.log" 2>&1; then
        fail_arm "$label" "trim failed, see $out/logs/trim.log"
        continue
    fi
    count=$(OMP_NUM_THREADS=1 python cancer_metastasis/21_prepare_four_state_malignant_pseudobulk.py \
        "$out/manifest_trimmed.csv" "$gate" "$out/four_state" --print-patient-count "${extra[@]}" \
        2> "$out/logs/count.log") || count=""
    # Anything but a positive integer -- a warning printed to stdout, an empty
    # trim -- would make the pseudobulk run zero patients in silence, the DEG
    # would fit an empty cohort, and the arm would be marked DONE.
    if ! [[ "$count" =~ ^[0-9]+$ ]] || (( count < 1 )); then
        fail_arm "$label" "patient count is not a positive integer: '$count'; the trim may hold no pairs of $accession gated by $label"
        continue
    fi
    patients_of[$label]=$count
    active+=("$label")
    echo "$label: $count patients"
done
if (( ${#active[@]} == 0 )); then
    echo "no arm of this task has work to do"
    for label in "${mine[@]}"; do [[ -f "$(arm_dir "$label")/FAILED" ]] && exit 1; done
    exit 0
fi

# ---- 2. pseudobulk, every (arm, patient) in one pool, patient by patient ---
pseudobulk_one() {
    local label=$1 index=$2
    local out
    out=$(arm_dir "$label")
    if ! OMP_NUM_THREADS=$threads_per_worker OPENBLAS_NUM_THREADS=$threads_per_worker \
         MKL_NUM_THREADS=$threads_per_worker NUMEXPR_NUM_THREADS=$threads_per_worker \
         python cancer_metastasis/21_prepare_four_state_malignant_pseudobulk.py \
            "$out/manifest_trimmed.csv" "$factorial/$block/$label" "$out/four_state" \
            --index "$index" "${extra[@]}" > "$out/logs/patient_${index}.log" 2>&1; then
        # A missing patient does not make the DEG fail: it is silently absent
        # from the fit. So any patient failing stops its arm before the DEG.
        echo "$index" >> "$out/PSEUDOBULK_FAILED"
    fi
}
stamp "pseudobulk ${#active[@]} arms"
most=0
for label in "${active[@]}"; do (( patients_of[$label] > most )) && most=${patients_of[$label]}; done
for (( index = 0; index < most; index++ )); do
    for label in "${active[@]}"; do
        (( index < patients_of[$label] )) || continue
        pseudobulk_one "$label" "$index" &
        while (( $(jobs -rp | wc -l) >= workers )); do
            wait -n || true
        done
    done
done
wait || true

ready=()
for label in "${active[@]}"; do
    failed_file=$(arm_dir "$label")/PSEUDOBULK_FAILED
    if [[ -s "$failed_file" ]]; then
        fail_arm "$label" "$(wc -l < "$failed_file") of ${patients_of[$label]} patients failed to pseudobulk; not fitting a partial cohort (patients: $(tr '\n' ' ' < "$failed_file"))"
    else
        ready+=("$label")
    fi
done

# ---- 3 and 4. DEG, then GSEA, the arms side by side ------------------------
# The cores are shared between the arms still standing, so two arms left get
# half the node each rather than a quarter with the rest idle.
share=$cores
(( ${#ready[@]} > 0 )) && share=$(( cores / ${#ready[@]} ))
(( share < 1 )) && share=1

deg_one() {
    local label=$1 out
    out=$(arm_dir "$label")
    if ! OMP_NUM_THREADS=$share OPENBLAS_NUM_THREADS=$share MKL_NUM_THREADS=$share NUMEXPR_NUM_THREADS=$share \
        python cancer_metastasis/13_run_paired_pydeseq2.py \
            "$out/four_state" "$out/deg" \
            --minimum-cells-per-patient-status "$minimum_cells" \
            --minimum-total-count 10 \
            --maximum-fdr 0.05 \
            --absolute-log2-fold-change-thresholds 0.5 1.0 \
            --n-cpus "$share" > "$out/logs/deg.log" 2>&1; then
        fail_arm "$label" "paired DEG failed, see $out/logs/deg.log"
    fi
}
gsea_one() {
    local label=$1 out
    out=$(arm_dir "$label")
    [[ -f "$out/FAILED" ]] && return 0
    if ! OMP_NUM_THREADS=$share OPENBLAS_NUM_THREADS=$share MKL_NUM_THREADS=$share NUMEXPR_NUM_THREADS=$share \
        python cancer_metastasis/15_run_pydeseq2_gseapy.py \
            "$out/deg" "$gmt" "$out/gsea" \
            --permutations 2000 --threads "$share" --minimum-size 10 --maximum-size 500 \
            > "$out/logs/gsea.log" 2>&1; then
        fail_arm "$label" "GSEA failed, see $out/logs/gsea.log"
    fi
}

if (( ${#ready[@]} > 0 )); then
    stamp "paired DEG ${#ready[@]} arms, $share cores each"
    for label in "${ready[@]}"; do deg_one "$label" & done
    wait || true
    stamp "GSEA, hallmark, ${#ready[@]} arms"
    for label in "${ready[@]}"; do gsea_one "$label" & done
    wait || true
fi

# ---- per arm: DONE, or the reason it is not --------------------------------
failed_any=0
for label in "${mine[@]}"; do
    out=$(arm_dir "$label")
    if [[ -f "$out/FAILED" ]]; then
        failed_any=1
        echo "$label: FAILED -- $(tail -1 "$out/FAILED")"
    elif [[ -n "${patients_of[$label]:-}" ]]; then
        printf 'block=%s\naccession=%s\nlabel=%s\ngate=%s\ngene_sets=%s\npatients=%s\n' \
            "$block" "$accession" "$label" "$factorial/$block/$label" "$gmt" \
            "${patients_of[$label]}" > "$out/DONE"
        echo "$label: done, ${patients_of[$label]} patients"
    fi
done
stamp "task finished"
exit $failed_any
