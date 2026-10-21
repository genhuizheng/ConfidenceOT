#!/bin/bash
# DEG and hallmark GSEA for the five CNV / RNA gate strategies, primary and metastasis apart.
#
#   bash cancer_metastasis/submit_cnv_rna_strategies_downstream.sh --dry-run
#   bash cancer_metastasis/submit_cnv_rna_strategies_downstream.sh [--acceptance "0.99 0.95 0.90"] [--scope three|all] [--force]
#
# The pipeline is the preprocessing factorial's own, unchanged: tacc_factorial_downstream.slurm
# (35_ trim to the gated pairs -> 21_ patient pseudobulk from the original raw counts -> 13_
# PyDESeq2, ~patient + status, rejected vs retained -> 15_ GSEApy prerank on the Wald
# statistic, hallmark only, 2,000 permutations), run as units one after another in one job
# (tacc_downstream_units.slurm), as submit_cycling_cnv_downstream.sh does.
#
# Units: every strategy x side x accession. A side is one 21_ contrast, run as its own unit
# so a contrast that cannot be fitted (too few patients with both states) fails alone:
#   primary     primary_rejected_vs_primary_retained
#   metastasis  metastasis_rejected_vs_metastasis_retained
# For a two-step strategy the gate root holds only the cells step 1 retained, so its contrast
# is step 2 rejected vs finally retained. Step 1 excluded vs step 1 retained is the single
# strategy's own contrast (cnv for cnv_then_rna, rna for rna_then_cnv) and is not recomputed.
#
# Scope:
#   three (default): ovarian GSE180661 and colorectal GSE225857 (uniform block), prostate
#     GSE271675; one job per acceptance x accession
#   all: also the uniform block's other datasets, each accession its own analysis, all of
#     them in one job per acceptance
#
# Output: <result>/downstream_cnv_rna_strategies/acc<a>/<side>/<strategy>/arm_A/<block>/<accession>/<label>/
# (the cycling layout under <side>, so 43_/47_ read it). Nothing selects pairs: the trim keeps
# what each strategy gated, and 46_report_downstream_eligibility.py reports what is missing.
#
# No set -u, as everywhere in this repo.
set -eo pipefail

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
gates=$result/cnv_rna_strategies_infercnvpy
down=$result/downstream_cnv_rna_strategies
job=$repo/cancer_metastasis/tacc_factorial_downstream.slurm
units_job=$repo/cancer_metastasis/tacc_downstream_units.slurm
gmt=$result/gene_sets/msigdb_2025.1_Hs/h.all.v2025.1.Hs.symbols.gmt
label=ranknm256_noscale_ds_cos
strategies="rna cnv rna_cnv cnv_then_rna rna_then_cnv"
prostate_annotations="Epithelial|Basal Epithelial|Neuroendocrine"
declare -A source=(
    [uniform]=$result/manifest/pancancer_20260924/pair_manifest_eligible.csv
    [prostate]=$result/prepared_GSE271675_20260916/pair_manifest_eligible.csv)
declare -A contrast=(
    [primary]=primary_rejected_vs_primary_retained
    [metastasis]=metastasis_rejected_vs_metastasis_retained)
others="HRA000036 GSE188737 GSE197177 GSE181919 GSE178318 GSE315534 GSE167036 GSE163558"
acceptances="0.99 0.95 0.90"
scope=three
force=0
dry_run=0
cap=${CONFIDENCEOT_JOB_CAP:-40}
while (( $# )); do
    case "$1" in
        --acceptance) acceptances=$2; shift 2 ;;
        --scope) scope=$2; shift 2 ;;
        --force) force=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[[ "$scope" == three || "$scope" == all ]] || { echo "--scope three or all" >&2; exit 2; }
# analysis groups: "short:block:accession[,accession...]" -- one job each per acceptance
groups="ov:uniform:GSE180661 crc:uniform:GSE225857 pr:prostate:GSE271675"
[[ "$scope" == all ]] && groups="$groups other:uniform:${others// /,}"

problems=0
for path in "${source[@]}" "$gmt" "$job" "$units_job"; do [[ -f "$path" ]] || { echo "missing: $path" >&2; problems=1; }; done
grep -q 'CONFIDENCEOT_DOWNSTREAM_CONTRASTS' "$job" 2> /dev/null || { echo "$repo is older than this script: git pull" >&2; problems=1; }
for name in CONFIDENCEOT_ENV CONFIDENCEOT_DOWNSTREAM_CORES CONFIDENCEOT_DOWNSTREAM_THREADS_PER_WORKER \
            CONFIDENCEOT_DOWNSTREAM_WORKERS CONFIDENCEOT_DOWNSTREAM_CONTRASTS; do
    [[ -z "${!name}" ]] || { echo "set in this shell, would reach every unit: $name=${!name}  (unset $name)" >&2; problems=1; }
done
declare -A size=()
for a in $acceptances; do
    for block in uniform prostate; do
        record=$gates/acc$a/rna/arm_A/$block/PREDOWNSAMPLE_DEPTH
        if [[ ! -f "$record" ]]; then echo "no depth record: $record" >&2; problems=1; continue; fi
        for s in $strategies; do
            other=$gates/acc$a/$s/arm_A/$block/PREDOWNSAMPLE_DEPTH
            [[ -d "$gates/acc$a/$s/arm_A/$block/$label" ]] || { echo "no gate root: $gates/acc$a/$s/arm_A/$block/$label" >&2; problems=1; }
            [[ ! -f "$other" ]] || cmp -s "$record" "$other" || { echo "$other names a different depth table" >&2; problems=1; }
        done
        size[$a:$block]=$(dirname "$(cat "$record")")/downsample_per_sample.csv
        [[ -f "${size[$a:$block]}" ]] || { echo "missing: ${size[$a:$block]}" >&2; problems=1; }
    done
done
# The trim takes the pairs a gate has completed when the unit runs, and a unit marked DONE is
# not redone; so no strategy run may still be queued or running.
running=$( { squeue -u "$USER" -h -o "%j" 2> /dev/null || true; } | grep -c '^st_' || true)
if (( running > 0 )); then
    echo "$running strategy OT job(s) (st_*) are still queued or running; submit after they finish" >&2
    problems=1
fi
(( problems && ! dry_run )) && exit 2

wanted=$(( $(wc -w <<< "$acceptances") * $(wc -w <<< "$groups") ))
queued=$( { squeue -u "$USER" -h -r 2> /dev/null || true; } | wc -l )
if (( ! dry_run && queued + wanted > cap )); then
    echo "$queued tasks already queued and this adds $wanted, over the cap of $cap" >&2
    exit 3
fi
units_dir=$down
(( dry_run )) && units_dir=$(mktemp -d)
mkdir -p "$units_dir"

submit() {
    if (( dry_run )); then echo "DRY RUN  sbatch $*" | cut -c1-300 >&2; echo DRYRUN; return 0; fi
    local id
    id=$(sbatch --parsable "$@" | tail -n 1)
    [[ "$id" =~ ^[0-9]+$ ]] || { echo "sbatch returned no job id; stopping (anything above was submitted)" >&2; exit 1; }
    echo "$id"
}

ids=()
for a in $acceptances; do
    for group in $groups; do
        IFS=: read -r short block accessions <<< "$group"
        if [[ "$block" == prostate ]]; then
            compartment=("CONFIDENCEOT_MALIGNANT_COLUMN=" "CONFIDENCEOT_MALIGNANT_ANNOTATIONS=$prostate_annotations")
        else
            compartment=("CONFIDENCEOT_MALIGNANT_COLUMN=malignant" "CONFIDENCEOT_MALIGNANT_ANNOTATIONS=")
        fi
        units=$units_dir/units_acc${a}_${short}.tsv
        : > "$units"
        for accession in ${accessions//,/ }; do
            for side in primary metastasis; do
                for s in $strategies; do
                    settings=("CONFIDENCEOT_REPO=$repo" "CANCER_COT_ROOT=$result"
                              "CONFIDENCEOT_FACTORIAL_ROOT=$gates/acc$a/$s/arm_A" "CONFIDENCEOT_FACTORIAL_BLOCK=$block"
                              "CONFIDENCEOT_DOWNSTREAM_ACCESSION=$accession" "CONFIDENCEOT_DOWNSTREAM_NAME=$accession"
                              "CONFIDENCEOT_DOWNSTREAM_SOURCE_MANIFEST=${source[$block]}"
                              "CONFIDENCEOT_METASTASIS_SIZE_CSV=${size[$a:$block]}"
                              "CONFIDENCEOT_DOWNSTREAM_ROOT=$down/acc$a/$side/$s/arm_A" "CONFIDENCEOT_HUMAN_GMT=$gmt"
                              "CONFIDENCEOT_FACTORIAL_LABELS=$label" "CONFIDENCEOT_DOWNSTREAM_CONTRASTS=${contrast[$side]}"
                              "CONFIDENCEOT_DOWNSTREAM_ARMS_PER_TASK=4" "CONFIDENCEOT_DOWNSTREAM_FORCE=$force"
                              "${compartment[@]}")
                    (IFS=$'\t'; echo "${settings[*]}") >> "$units"
                done
            done
        done
        id=$(submit -p gg -N 1 -n 1 -c 72 -t 48:00:00 -A MCB26031 -J "dg_${short}_$a" \
            -o "$result/logs/strategy_downstream_${short}_acc${a}_%j.out" -e "$result/logs/strategy_downstream_${short}_acc${a}_%j.err" \
            --export="ALL,CONFIDENCEOT_REPO=$repo,CONFIDENCEOT_UNITS=$units" "$units_job")
        ids+=("${short}/acc$a=$id")
        echo "$short at $a: $(wc -l < "$units") units -> $id" >&2
    done
done

summary="commit       $(git -C "$repo" rev-parse --short HEAD 2> /dev/null || echo unknown)
scope        $scope
acceptances  $acceptances
gates        $gates/acc<a>/<strategy>/arm_A/<block>/$label
output       $down/acc<a>/<primary|metastasis>/<strategy>/arm_A/<block>/<accession>/$label/
gene sets    $gmt
units        $units_dir/units_acc<a>_<group>.tsv
jobs         ${ids[*]}"
echo; echo "$summary"
if (( dry_run )); then echo "nothing was submitted (--dry-run); unit files in $units_dir"; else echo "$summary" > "$down/SUBMITTED_${scope}_$(date +%Y%m%d_%H%M%S).txt"; fi
