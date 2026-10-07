#!/bin/bash
# DEG and hallmark GSEA for the cycling x representation factorial, the way
# tacc_factorial_downstream.slurm ran them for the preprocessing factorial.
#
#   bash cancer_metastasis/submit_cycling_cnv_downstream.sh --dry-run
#   bash cancer_metastasis/submit_cycling_cnv_downstream.sh
#
# The three analyses of the 2026-10-02 figure -- ovarian GSE180661 and
# colorectal GSE225857 from the uniform block, prostate GSE271675 -- over every
# OT root submit_cycling_cnv_factorial.sh wrote: one array task per (analysis,
# root), each running that root's labels end to end with the 2026-10-02
# settings. The original counts, the malignant compartment the gate used,
# paired PyDESeq2, hallmark GSEA. Written to
# <down>/<rna|rna_cnv|cnv>/arm_<X>/<block>/<accession>/<label>/.
#
# One thing differs, the manifest the arms read. On 2026-10-02 each arm was cut
# to its own gated pairs, which were the same pairs in every arm. Here the cell
# filter leaves three uniform pairs unevaluable -- their metastasis keeps two or
# three malignant cells -- and 21_ names one lesion per patient, the largest
# gated one, so a patient could change lesion between arm A and arm B. Every arm
# of a block therefore reads one manifest: the original-count manifest cut by
# 35_trim_manifest_to_gate.py to the pairs every root of the block gated with an
# M4-E calibration 21_ accepts. The same patients and lesions in every arm.
#
# The DEG reads every gene, in arms C and D too: the cell-cycle genes were taken
# out of the representation the gate saw, not out of the transcriptome the
# readout compares. In arms B and D the cells outside G1 are not in the gate, so
# they are in no pseudobulk.
#
# No set -u, as everywhere in this repo.
set -eo pipefail

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
acceptance=0.99
dry_run=0
cap=${CONFIDENCEOT_JOB_CAP:-40}
while (( $# )); do
    case "$1" in
        --dry-run) dry_run=1; shift ;;
        --acceptance) acceptance=$2; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

ot=$result/cycling_cnv_acc$acceptance
down=$result/downstream_cycling_cnv_acc$acceptance
job=$repo/cancer_metastasis/tacc_factorial_downstream.slurm
gmt=$result/gene_sets/msigdb_2025.1_Hs/h.all.v2025.1.Hs.symbols.gmt
env_python=${CONFIDENCEOT_PYTHON:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot/bin/python}
labels="ranknm256_noscale_ds_cos raw"
prostate_annotations="Epithelial|Basal Epithelial|Neuroendocrine"
# The original-count manifests and lesion-size tables the 2026-10-02 run read.
declare -A source=(
    [uniform]=$result/manifest/pancancer_20260924/pair_manifest_eligible.csv
    [prostate]=$result/prepared_GSE271675_20260916/pair_manifest_eligible.csv)
declare -A size=(
    [uniform]=$result/downsampled_factorial_20260929/downsample_per_sample.csv
    [prostate]=$result/downsampled_GSE271675_20260916/downsample_per_sample.csv)
# The roots of each block, as rep/arm:labels.
declare -A roots=(
    [uniform]="rna/A rna/B rna/C rna/D rna_cnv/A rna_cnv/B rna_cnv/C rna_cnv/D cnv/A cnv/B"
    [prostate]="rna/A rna/B rna/C rna/D")
labels_of() { [[ "$1" == cnv/* ]] && echo raw || echo "$labels"; }
# analysis: accession, block, short name for the job.
analyses="GSE180661:uniform:ov GSE225857:uniform:crc GSE271675:prostate:pr"

problems=0
for path in "${source[@]}" "${size[@]}" "$gmt" "$job" "$env_python"; do
    [[ -f "$path" ]] || { echo "missing: $path" >&2; problems=1; }
done
for block in uniform prostate; do
    for root in ${roots[$block]}; do
        for label in $(labels_of "$root"); do
            gate=$ot/${root%/*}/arm_${root#*/}/$block/$label
            [[ -d "$gate" ]] || { echo "no gate root: $gate" >&2; problems=1; }
        done
    done
done
if ! grep -q -- '--require-m4e-calibration' "$repo/cancer_metastasis/35_trim_manifest_to_gate.py" 2> /dev/null; then
    echo "$repo is older than this script: git pull there first" >&2
    problems=1
fi
# Settings the array reads from the environment and this does not set.
for name in CONFIDENCEOT_ENV CONFIDENCEOT_MINIMUM_CELLS_PER_STATUS CONFIDENCEOT_DOWNSTREAM_CORES \
            CONFIDENCEOT_DOWNSTREAM_THREADS_PER_WORKER CONFIDENCEOT_DOWNSTREAM_WORKERS; do
    if [[ -n "${!name}" ]]; then
        echo "set in this shell, would reach every task: $name=${!name}  (unset $name)" >&2
        problems=1
    fi
done
if (( problems )); then
    exit 2
fi

tasks=0
for analysis in $analyses; do
    block=$(cut -d: -f2 <<< "$analysis")
    tasks=$(( tasks + $(wc -w <<< "${roots[$block]}") ))
done
queued=$(squeue -u "$USER" -h -r 2> /dev/null | wc -l) || queued=0
if (( ! dry_run && queued + tasks > cap )); then
    echo "$queued tasks already queued and this adds $tasks, over the cap of $cap" >&2
    exit 3
fi

# ---- one manifest per block: the pairs every arm gated and calibrated -------
# Light: a few hundred manifest rows and one small run.json per pair and arm. A
# dry run writes it to a scratch directory, so the counts can be read first.
common_dir=$down
(( dry_run )) && common_dir=$(mktemp -d)
mkdir -p "$common_dir"
declare -A common=()
for block in uniform prostate; do
    gate_args=()
    for root in ${roots[$block]}; do
        for label in $(labels_of "$root"); do
            gate_args+=(--gate-root "$ot/${root%/*}/arm_${root#*/}/$block/$label")
        done
    done
    common[$block]=$common_dir/common_pairs_$block.csv
    if ! PYTHONPATH="$repo/src:$repo/cancer_metastasis" "$env_python" \
        "$repo/cancer_metastasis/35_trim_manifest_to_gate.py" "${source[$block]}" "${common[$block]}" \
        "${gate_args[@]}" --require-m4e-calibration > "$common_dir/common_pairs_$block.log" 2>&1; then
        tail -n 5 "$common_dir/common_pairs_$block.log" >&2
        echo "could not build the $block manifest; nothing was submitted" >&2
        exit 2
    fi
    "$env_python" - "${common[$block]%.csv}.trim.json" "$block" "${#gate_args[@]}" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
refused = report["m4e_calibration_refused"] or {}
print(f"{sys.argv[2]:<9} {report['rows_kept']} pairs of {report['patient_n_after']} patients "
      f"gated in all {int(sys.argv[3]) // 2} roots, {len(refused)} more refused by calibration"
      + (f": {sorted(refused)}" if refused else ""))
PY
done

submit() {
    # $1 = description, rest = sbatch arguments. Called inside $(...), where
    # set -e does not reach, so a refused submission ends the script here.
    local what=$1; shift
    if (( dry_run )); then
        echo "DRY RUN  $what" >&2
        echo "         sbatch $*" >&2
        echo "DRYRUN"
        return 0
    fi
    local id
    id=$(sbatch --parsable "$@" | tail -n 1)
    if [[ ! "$id" =~ ^[0-9]+$ ]]; then
        echo "sbatch returned no job id for $what; stopping. Already submitted above." >&2
        exit 1
    fi
    echo "submitted $what as $id" >&2
    echo "$id"
}

ids=()
for analysis in $analyses; do
    IFS=: read -r accession block short <<< "$analysis"
    if [[ "$block" == prostate ]]; then
        compartment="CONFIDENCEOT_MALIGNANT_COLUMN=,CONFIDENCEOT_MALIGNANT_ANNOTATIONS=$prostate_annotations"
    else
        compartment="CONFIDENCEOT_MALIGNANT_COLUMN=malignant,CONFIDENCEOT_MALIGNANT_ANNOTATIONS="
    fi
    for root in ${roots[$block]}; do
        rep=${root%/*}; arm=${root#*/}
        id=$(submit "$accession $rep arm $arm" --array=0-0 -J "dg_${short}_${rep}_$arm" \
            --export="ALL,CONFIDENCEOT_REPO=$repo,CANCER_COT_ROOT=$result,CONFIDENCEOT_FACTORIAL_ROOT=$ot/$rep/arm_$arm,CONFIDENCEOT_FACTORIAL_BLOCK=$block,CONFIDENCEOT_DOWNSTREAM_ACCESSION=$accession,CONFIDENCEOT_DOWNSTREAM_NAME=$accession,CONFIDENCEOT_DOWNSTREAM_SOURCE_MANIFEST=${common[$block]},CONFIDENCEOT_METASTASIS_SIZE_CSV=${size[$block]},CONFIDENCEOT_DOWNSTREAM_ROOT=$down/$rep/arm_$arm,CONFIDENCEOT_HUMAN_GMT=$gmt,CONFIDENCEOT_FACTORIAL_LABELS=$(labels_of "$root"),CONFIDENCEOT_DOWNSTREAM_ARMS_PER_TASK=4,CONFIDENCEOT_DOWNSTREAM_FORCE=0,$compartment" \
            "$job")
        ids+=("${short}_${rep}_$arm=$id")
    done
done

summary=$(cat <<EOF
commit       $(git -C "$repo" rev-parse --short HEAD 2> /dev/null || echo unknown)
gates        $ot
output       $down/<rep>/arm_<X>/<block>/<accession>/<label>/
manifests    ${common[uniform]}
             ${common[prostate]}
sizes        ${size[uniform]}
             ${size[prostate]}
gene sets    $gmt
tasks        ${ids[*]}
EOF
)
echo
echo "$summary"
if (( dry_run )); then
    echo "nothing was submitted (--dry-run); the manifests above are in a scratch directory"
else
    echo "$summary" > "$down/SUBMITTED_$(date +%Y%m%d_%H%M%S).txt"
fi
