#!/bin/bash
# DEG and hallmark GSEA for the cycling x representation factorial, through
# tacc_factorial_downstream.slurm exactly as the preprocessing factorial ran it
# on 2026-10-02.
#
#   bash cancer_metastasis/submit_cycling_cnv_downstream.sh --dry-run
#   bash cancer_metastasis/submit_cycling_cnv_downstream.sh
#
# The three analyses of the 2026-10-02 figure -- ovarian GSE180661 and
# colorectal GSE225857 from the uniform block, prostate GSE271675 -- over every
# OT root submit_cycling_cnv_factorial.sh wrote: one array task per (analysis,
# root), each running that root's labels end to end. Written to
# <down>/<rna|rna_cnv|cnv>/arm_<X>/<block>/<accession>/<label>/.
#
# Nothing here selects pairs. Every task reads the original-count manifest the
# 2026-10-02 run read and the job trims it to the pairs its own arm gated. 21_
# runs with its own --allow-invalid-calibration, so it consumes every completed
# gate as it stands and does not judge the calibration again. A pair an arm is
# missing is reported afterwards by 46_report_downstream_eligibility.py, not
# replaced.
#
# The lesion-size table is the one each block recorded, as 2026-10-02 took it:
# PREDOWNSAMPLE_DEPTH beside the block, whose directory holds
# downsample_per_sample.csv. A cnv root ran only the raw label, which writes no
# such record, so it takes the record of its block's rna/arm_A -- the same depth
# table every root of the block was submitted with; all records present must
# agree, or nothing is submitted.
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
labels="ranknm256_noscale_ds_cos raw"
prostate_annotations="Epithelial|Basal Epithelial|Neuroendocrine"
# The original-count manifests submit_factorial_downstream.sh reads.
declare -A source=(
    [uniform]=$result/manifest/pancancer_20260924/pair_manifest_eligible.csv
    [prostate]=$result/prepared_GSE271675_20260916/pair_manifest_eligible.csv)
declare -A roots=(
    [uniform]="rna/A rna/B rna/C rna/D rna_cnv/A rna_cnv/B rna_cnv/C rna_cnv/D cnv/A cnv/B"
    [prostate]="rna/A rna/B rna/C rna/D")
labels_of() { [[ "$1" == cnv/* ]] && echo raw || echo "$labels"; }
block_dir() { echo "$ot/${2%/*}/arm_${2#*/}/$1"; }   # $1 block, $2 rep/arm
# analysis: accession, block, short name for the job.
analyses="GSE180661:uniform:ov GSE225857:uniform:crc GSE271675:prostate:pr"

problems=0
for path in "${source[@]}" "$gmt" "$job"; do
    [[ -f "$path" ]] || { echo "missing: $path" >&2; problems=1; }
done
declare -A size=()
for block in uniform prostate; do
    record=$(block_dir "$block" rna/A)/PREDOWNSAMPLE_DEPTH
    if [[ ! -f "$record" ]]; then
        echo "no PREDOWNSAMPLE_DEPTH record at $record" >&2
        problems=1
        continue
    fi
    for root in ${roots[$block]}; do
        other=$(block_dir "$block" "$root")/PREDOWNSAMPLE_DEPTH
        if [[ -f "$other" ]] && ! cmp -s "$record" "$other"; then
            echo "$other names a different depth table from $record" >&2
            problems=1
        fi
        for label in $(labels_of "$root"); do
            [[ -d "$(block_dir "$block" "$root")/$label" ]] || {
                echo "no gate root: $(block_dir "$block" "$root")/$label" >&2; problems=1; }
        done
    done
    size[$block]=$(dirname "$(cat "$record")")/downsample_per_sample.csv
    [[ -f "${size[$block]}" ]] || { echo "missing: ${size[$block]}" >&2; problems=1; }
done
if [[ ! -f "$repo/cancer_metastasis/46_report_downstream_eligibility.py" ]] ||
   ! grep -q 'CONFIDENCEOT_DOWNSTREAM_PSEUDOBULK_ARGS' "$job" 2> /dev/null; then
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
(( dry_run )) || mkdir -p "$down"

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
            --export="ALL,CONFIDENCEOT_REPO=$repo,CANCER_COT_ROOT=$result,CONFIDENCEOT_FACTORIAL_ROOT=$ot/$rep/arm_$arm,CONFIDENCEOT_FACTORIAL_BLOCK=$block,CONFIDENCEOT_DOWNSTREAM_ACCESSION=$accession,CONFIDENCEOT_DOWNSTREAM_NAME=$accession,CONFIDENCEOT_DOWNSTREAM_SOURCE_MANIFEST=${source[$block]},CONFIDENCEOT_METASTASIS_SIZE_CSV=${size[$block]},CONFIDENCEOT_DOWNSTREAM_ROOT=$down/$rep/arm_$arm,CONFIDENCEOT_HUMAN_GMT=$gmt,CONFIDENCEOT_FACTORIAL_LABELS=$(labels_of "$root"),CONFIDENCEOT_DOWNSTREAM_ARMS_PER_TASK=4,CONFIDENCEOT_DOWNSTREAM_FORCE=0,CONFIDENCEOT_DOWNSTREAM_PSEUDOBULK_ARGS=--allow-invalid-calibration,$compartment" \
            "$job")
        ids+=("${short}_${rep}_$arm=$id")
    done
done

summary=$(cat <<EOF
commit       $(git -C "$repo" rev-parse --short HEAD 2> /dev/null || echo unknown)
gates        $ot
output       $down/<rep>/arm_<X>/<block>/<accession>/<label>/
manifests    ${source[uniform]}
             ${source[prostate]}
sizes        ${size[uniform]}
             ${size[prostate]}
gene sets    $gmt
tasks        ${ids[*]}
report       python cancer_metastasis/46_report_downstream_eligibility.py $down $ot
EOF
)
echo
echo "$summary"
if (( dry_run )); then
    echo "nothing was submitted (--dry-run)"
else
    echo "$summary" > "$down/SUBMITTED_$(date +%Y%m%d_%H%M%S).txt"
fi
