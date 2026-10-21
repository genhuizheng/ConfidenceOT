#!/bin/bash
# Five CNV / RNA gate strategies on the real pairs, with the infercnvpy CNV profiles.
#
#   bash cancer_metastasis/submit_cnv_rna_strategies.sh --stage single   --acceptance 0.99 [--block uniform|prostate|both] [--dry-run]
#   bash cancer_metastasis/submit_cnv_rna_strategies.sh --stage two_step --acceptance 0.99 [--block ...] [--dry-run]
#
# Strategies (directory names), all at one RNA preprocessing, ranknm256_noscale_ds_cos,
# with ConfidenceOT M4-E (M4-R saved beside it) at the given within-side acceptance:
#   rna            RNA only
#   cnv            CNV only: each cell's own infercnvpy profile (PCA) as the representation
#   rna_cnv        CNV + RNA: each block divided by its own median cross-side cost, then
#                  concatenated, so the cost is the equal-weight sum of the two normalised
#                  costs (02_run_pair.py --representation-source rna_cnv)
#   cnv_then_rna   step 2 of CNV -> RNA: RNA gate on the cells the cnv strategy retained
#   rna_then_cnv   step 2 of RNA -> CNV: CNV gate on the cells the rna strategy retained
# Step 1 of a two-step strategy is the single strategy itself (cnv, rna), read only.
# The null calibration of step 2 is redone on the restricted cells (--input-gate-*).
#
# Stages: "single" submits rna, cnv and rna_cnv; "two_step" submits the two step 2
# runs and refuses until its step 1 exists, taking the budget tag step 1 was written
# under. A pair whose step 1 did not complete, or left under 4 cells on a side, fails
# with that reason in its log and is reported, never dropped silently.
#
# Output: <result>/cnv_rna_strategies_infercnvpy/acc<a>/<strategy>/arm_A/<block>/<label>/
# The arm_A level keeps the tree the cycling x CNV factorial used, so 43_/47_ layouts and
# 48_'s configuration glob (<root>/*/arm_*/*/*) find these runs.
#
# Everything else is the factorial's own (tacc_preprocessing_factorial.slurm): manifests,
# pairs, QC (recorded, filters nothing), seeds, solver, null. --save-coupling is added so
# every run keeps its coupling matrices.
#
# Blocks: uniform (ovarian GSE180661 and colorectal GSE225857, 110 pairs, the collection's
# malignant call) and prostate (GSE271675, 24 pairs, Faming Zhao's cells and labels). The
# prostate CNV strategies need the prostate profiles in Faming's cell names
# (cnv_profiles/infercnvpy_patient_faming, EXPORT_COMPLETE); without them the prostate
# block is refused for every strategy that reads CNV.
#
# No set -u, as everywhere in this repo.
set -eo pipefail

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
base=${CNV_BASE:-/scratch/10119/ghzheng/primary_metastatic_cancer}
result=${CANCER_COT_ROOT:-$base/confidenceot_results}
uniform_manifest=$result/manifest/factorial_20260929/uniform.csv
uniform_manifest_ds=$result/manifest/factorial_20260929/uniform_ds.csv
uniform_depth=$result/downsampled_factorial_20260929/predownsample_depth.csv.gz
prostate_manifest=$result/prepared_GSE271675_20260916/pair_manifest_eligible.csv
prostate_manifest_ds=$result/downsampled_GSE271675_20260916/pair_manifest_downsampled.csv
prostate_depth=$result/downsampled_GSE271675_20260916/predownsample_depth.csv.gz
prostate_annotations="Epithelial|Basal Epithelial|Neuroendocrine"
cnv_uniform=$result/cnv_profiles/infercnvpy_patient
cnv_prostate=$result/cnv_profiles/infercnvpy_patient_faming
label=ranknm256_noscale_ds_cos
job=$repo/cancer_metastasis/tacc_preprocessing_factorial.slurm
cap=${CONFIDENCEOT_JOB_CAP:-40}
stage=""
acceptance=""
blocks="uniform prostate"
dry_run=0
while (( $# )); do
    case "$1" in
        --stage) stage=$2; shift 2 ;;
        --acceptance) acceptance=$2; shift 2 ;;
        --block) case "$2" in both) blocks="uniform prostate" ;; uniform|prostate) blocks=$2 ;;
                 *) echo "--block uniform, prostate or both" >&2; exit 2 ;; esac; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[[ "$stage" == single || "$stage" == two_step ]] || { echo "--stage single or two_step" >&2; exit 2; }
[[ "$acceptance" =~ ^0\.[0-9]+$ ]] || { echo "--acceptance, e.g. 0.99" >&2; exit 2; }
out=$result/cnv_rna_strategies_infercnvpy/acc$acceptance

problems=0
need() { [[ "$1" == /* && -e "$1" ]] || { echo "missing: $1" >&2; problems=1; }; }
for path in "$job" "$uniform_manifest" "$uniform_manifest_ds" "$uniform_depth"; do need "$path"; done
[[ " $blocks " == *" prostate "* ]] && for path in "$prostate_manifest" "$prostate_manifest_ds" "$prostate_depth"; do need "$path"; done
need "$cnv_uniform"
if [[ " $blocks " == *" prostate "* && ! -f "$cnv_prostate/EXPORT_COMPLETE" ]]; then
    echo "no prostate CNV in Faming's names ($cnv_prostate/EXPORT_COMPLETE): use --block uniform," \
         "or build it first" >&2
    problems=1
fi
grep -q -- '--input-gate-budget-tag' "$repo/cancer_metastasis/02_run_pair.py" 2> /dev/null &&
grep -q 'CONFIDENCEOT_FACTORIAL_EXTRA_ARGS' "$job" 2> /dev/null || { echo "$repo is older than this script: git pull" >&2; problems=1; }
# Settings the array reads from the environment and this does not set; --export=ALL
# would carry a leftover into every task without any output path recording it.
for name in CONFIDENCEOT_ENV CONFIDENCEOT_MALIGNANT_VALUE CONFIDENCEOT_CALIBRATION_NULL \
            CONFIDENCEOT_MAX_OBSERVED_CELLS_PER_SIDE CONFIDENCEOT_MINIMUM_TOTAL_COUNTS \
            CONFIDENCEOT_MINIMUM_DETECTED_GENES CONFIDENCEOT_MAXIMUM_MITOCHONDRIAL_PERCENT \
            CONFIDENCEOT_WORKERS CONFIDENCEOT_THREADS CONFIDENCEOT_FACTORIAL_PAIR_PARALLEL; do
    [[ -z "${!name}" ]] || { echo "set in this shell, would reach every task: $name=${!name}  (unset $name)" >&2; problems=1; }
done
(( problems && ! dry_run )) && exit 2

# The budget tag step 1 wrote, read from its SUCCESS files: exactly one, or refuse.
step1_tag() {  # $1 = step 1 label directory
    local tags
    tags=$(find "$1" -mindepth 4 -maxdepth 4 -name SUCCESS -path "*/scope_malignant/*" -printf '%h\n' 2> /dev/null \
           | xargs -r -n1 basename | sort -u)
    [[ $(printf '%s\n' "$tags" | grep -c .) -eq 1 ]] || { echo "$1: no single budget tag [$tags]" >&2; return 1; }
    echo "$tags"
}

plan=()   # entries: strategy|block|extra args
for block in $blocks; do
    if [[ "$block" == uniform ]]; then cnv_root=$cnv_uniform; else cnv_root=$cnv_prostate; fi
    common="--within-side-acceptance-minimum $acceptance --save-coupling"
    if [[ "$stage" == single ]]; then
        plan+=("rna|$block|$common")
        plan+=("cnv|$block|$common --representation-source cnv --cnv-root $cnv_root")
        plan+=("rna_cnv|$block|$common --representation-source rna_cnv --cnv-root $cnv_root")
    else
        for pair in "cnv_then_rna:cnv:" "rna_then_cnv:rna:--representation-source cnv --cnv-root $cnv_root"; do
            strategy=${pair%%:*}; rest=${pair#*:}; first=${rest%%:*}; own=${rest#*:}
            step1=$out/$first/arm_A/$block/$label
            if [[ ! -d "$step1" ]]; then
                echo "step 1 missing for $strategy/$block: $step1 (submit --stage single first)" >&2
                (( dry_run )) || exit 2
                tag=budget_0.95
            else
                done_n=$(find "$step1" -mindepth 4 -maxdepth 4 -name SUCCESS -path "*/scope_malignant/*" | wc -l)
                tag=$(step1_tag "$step1") || exit 2
                echo "$strategy/$block: step 1 $first has $done_n completed pairs, tag $tag" >&2
            fi
            plan+=("$strategy|$block|$common $own --input-gate-root $step1 --input-gate-state retained --input-gate-method M4-E --input-gate-budget-tag $tag")
        done
    fi
done

queued=$( { squeue -u "$USER" -h -r 2> /dev/null || true; } | wc -l )
if (( ! dry_run && queued + ${#plan[@]} > cap )); then
    echo "$queued tasks already queued and this adds ${#plan[@]}, over the cap of $cap" >&2
    exit 3
fi

submit() {  # prints the job id; refuses to continue on a failed submission
    if (( dry_run )); then echo "DRY RUN  sbatch $*" | cut -c1-600 >&2; echo DRYRUN; return 0; fi
    local id
    id=$(sbatch --parsable "$@" | tail -n 1)
    [[ "$id" =~ ^[0-9]+$ ]] || { echo "sbatch returned no job id; stopping (anything above was submitted)" >&2; exit 1; }
    echo "$id"
}

(( dry_run )) || mkdir -p "$out" "$result/logs"
ids=()
for entry in "${plan[@]}"; do
    IFS='|' read -r strategy block extra <<< "$entry"
    if [[ "$block" == uniform ]]; then
        manifest=$uniform_manifest; manifest_ds=$uniform_manifest_ds; depth=$uniform_depth; annotations=""
    else
        manifest=$prostate_manifest; manifest_ds=$prostate_manifest_ds; depth=$prostate_depth; annotations=$prostate_annotations
    fi
    id=$(submit --array=0-0 -t 48:00:00 -J "st_${strategy}" \
        --export=ALL,CONFIDENCEOT_REPO="$repo",CANCER_COT_ROOT="$result",CONFIDENCEOT_FACTORIAL_MANIFEST="$manifest",CONFIDENCEOT_FACTORIAL_MANIFEST_DS="$manifest_ds",CONFIDENCEOT_PREDOWNSAMPLE_DEPTH="$depth",CONFIDENCEOT_FACTORIAL_ROOT="$out/$strategy/arm_A",CONFIDENCEOT_FACTORIAL_LABELS="$label",CONFIDENCEOT_FACTORIAL_WORKERS=1,CONFIDENCEOT_ANALYSIS_SCOPE=malignant,CONFIDENCEOT_MALIGNANT_COLUMN=malignant,CONFIDENCEOT_DEVICE=cpu,CONFIDENCEOT_FACTORIAL_FORCE=0,CONFIDENCEOT_FACTORIAL_BLOCK="$block",CONFIDENCEOT_INCLUDE_ANNOTATIONS="$annotations",CONFIDENCEOT_MINIMUM_SCOPE_CELLS=4,CONFIDENCEOT_FACTORIAL_EXTRA_ARGS="$extra" \
        "$job")
    ids+=("$strategy/$block=$id")
    echo "$strategy/$block: $id" >&2
    echo "    -> $out/$strategy/arm_A/$block/$label" >&2
    echo "    02_run_pair.py extra: $extra" >&2
done

summary="commit       $(git -C "$repo" rev-parse --short HEAD 2> /dev/null || echo unknown)
stage        $stage
acceptance   $acceptance
blocks       $blocks
label        $label
cnv profiles uniform $cnv_uniform; prostate $cnv_prostate
output       $out/<strategy>/arm_A/<block>/$label
arrays       ${ids[*]}"
echo; echo "$summary"
if (( dry_run )); then echo "nothing was submitted (--dry-run)"; else echo "$summary" > "$out/SUBMITTED_${stage}_$(date +%Y%m%d_%H%M%S).txt"; fi
