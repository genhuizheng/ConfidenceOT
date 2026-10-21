#!/bin/bash
# Five CNV / RNA gate strategies on the real pairs, with the infercnvpy CNV profiles.
#
#   bash cancer_metastasis/submit_cnv_rna_strategies.sh --stage single --dry-run
#   bash cancer_metastasis/submit_cnv_rna_strategies.sh --stage single   [--acceptance "0.99 0.95 0.90"] [--block uniform|prostate|both]
#   bash cancer_metastasis/submit_cnv_rna_strategies.sh --stage two_step [...]   (after the single stage has finished)
#
# --stage single runs rna, cnv and rna_cnv (and the prostate CNV export first, when it is
# missing); --stage two_step runs cnv_then_rna and rna_then_cnv on the single stage's output.
#
# Strategies (directory names), all at one RNA preprocessing, ranknm256_noscale_ds_cos,
# with ConfidenceOT M4-E (M4-R saved beside it) at each within-side acceptance:
#   rna            RNA only
#   cnv            CNV only: each cell's own infercnvpy profile (PCA) as the representation
#   rna_cnv        CNV + RNA: each block divided by its own median cross-side cost, then
#                  concatenated, so the cost is the equal-weight sum of the two normalised
#                  costs (02_run_pair.py --representation-source rna_cnv)
#   cnv_then_rna   step 2 of CNV -> RNA: RNA gate on the cells the cnv strategy retained
#   rna_then_cnv   step 2 of RNA -> CNV: CNV gate on the cells the rna strategy retained
# Step 1 of a two-step strategy is the single strategy itself (cnv, rna), read only; the
# null calibration of step 2 is redone on the restricted cells (--input-gate-*). One c is
# shared by primary and metastasis, as in every run.
#
# Cell cycle: arm A of the cycling x CNV factorial only. Cycling (non-G1) cells and the 97
# cell-cycle genes are kept (user, 2026-10-20): no --cell-filter, no --exclude-genes.
#
# Jobs: one per acceptance x block (tacc_cnv_rna_strategies.slurm runs the five strategies
# in sequence, step 2 after its step 1, reading step 1's own budget tag), plus, for the
# prostate block, one job that first re-exports the prostate CNV profiles in Faming Zhao's
# cell names (50_prostate_cnv_faming_names.py) unless that export is already complete. The
# prostate jobs wait on it with afterok, so they start only if every analysed prostate cell
# has a profile. Default: three acceptances x two blocks = 6 jobs, plus the export = 7.
#
# Output: <result>/cnv_rna_strategies_infercnvpy/acc<a>/<strategy>/arm_A/<block>/<label>/
# (the arm_A level keeps the cycling factorial's tree, so 43_/47_ layouts and 48_'s
# configuration glob <root>/*/arm_*/*/* find these runs). Couplings are saved.
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
env=/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot
job=$repo/cancer_metastasis/tacc_cnv_rna_strategies.slurm
cap=${CONFIDENCEOT_JOB_CAP:-40}
acceptances="0.99 0.95 0.90"
blocks="uniform prostate"
stage=""
dry_run=0
while (( $# )); do
    case "$1" in
        --stage) stage=$2; shift 2 ;;
        --acceptance) acceptances=$2; shift 2 ;;
        --block) case "$2" in both) blocks="uniform prostate" ;; uniform|prostate) blocks=$2 ;;
                 *) echo "--block uniform, prostate or both" >&2; exit 2 ;; esac; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[[ "$stage" == single || "$stage" == two_step || "$stage" == all ]] || { echo "--stage single, two_step or all" >&2; exit 2; }
for a in $acceptances; do [[ "$a" =~ ^0\.[0-9]+$ ]] || { echo "bad acceptance $a" >&2; exit 2; }; done
root=$result/cnv_rna_strategies_infercnvpy

problems=0
need() { [[ "$1" == /* && -e "$1" ]] || { echo "missing: $1" >&2; problems=1; }; }
for path in "$job" "$repo/cancer_metastasis/tacc_preprocessing_factorial.slurm" "$cnv_uniform"; do need "$path"; done
[[ " $blocks " == *" uniform "* ]] && for path in "$uniform_manifest" "$uniform_manifest_ds" "$uniform_depth"; do need "$path"; done
if [[ " $blocks " == *" prostate "* ]]; then
    for path in "$prostate_manifest" "$prostate_manifest_ds" "$prostate_depth" \
                "$repo/cancer_metastasis/50_prostate_cnv_faming_names.py"; do need "$path"; done
fi
grep -q -- '--input-gate-budget-tag' "$repo/cancer_metastasis/02_run_pair.py" 2> /dev/null &&
grep -q 'CONFIDENCEOT_FACTORIAL_EXTRA_ARGS' "$repo/cancer_metastasis/tacc_preprocessing_factorial.slurm" 2> /dev/null ||
    { echo "$repo is older than this script: git pull" >&2; problems=1; }
# Settings the factorial reads from the environment and this does not set; --export=ALL
# would carry a leftover into every run without any output path recording it.
for name in CONFIDENCEOT_ENV CONFIDENCEOT_MALIGNANT_VALUE CONFIDENCEOT_CALIBRATION_NULL \
            CONFIDENCEOT_MAX_OBSERVED_CELLS_PER_SIDE CONFIDENCEOT_MINIMUM_TOTAL_COUNTS \
            CONFIDENCEOT_MINIMUM_DETECTED_GENES CONFIDENCEOT_MAXIMUM_MITOCHONDRIAL_PERCENT \
            CONFIDENCEOT_WORKERS CONFIDENCEOT_THREADS CONFIDENCEOT_FACTORIAL_PAIR_PARALLEL; do
    [[ -z "${!name}" ]] || { echo "set in this shell, would reach every run: $name=${!name}  (unset $name)" >&2; problems=1; }
done
(( problems && ! dry_run )) && exit 2

if [[ "$stage" == two_step ]]; then
    for a in $acceptances; do for block in $blocks; do for first in cnv rna; do
        d=$root/acc$a/$first/arm_A/$block/ranknm256_noscale_ds_cos
        [[ -d "$d" ]] || { echo "step 1 missing: $d (run --stage single first)" >&2; problems=1; }
    done; done; done
    (( problems && ! dry_run )) && exit 2
fi
make_export=0
[[ "$stage" != two_step && " $blocks " == *" prostate "* && ! -f "$cnv_prostate/EXPORT_COMPLETE" ]] && make_export=1
wanted=$(( $(wc -w <<< "$acceptances") * $(wc -w <<< "$blocks") + make_export ))
queued=$( { squeue -u "$USER" -h -r 2> /dev/null || true; } | wc -l )
if (( ! dry_run && queued + wanted > cap )); then
    echo "$queued tasks already queued and this adds $wanted, over the cap of $cap" >&2
    exit 3
fi

submit() {  # prints the job id; stops on a refused submission
    if (( dry_run )); then echo "DRY RUN  sbatch $*" | cut -c1-500 >&2; echo DRYRUN; return 0; fi
    local id
    id=$(sbatch --parsable "$@" | tail -n 1)
    [[ "$id" =~ ^[0-9]+$ ]] || { echo "sbatch returned no job id; stopping (anything above was submitted)" >&2; exit 1; }
    echo "$id"
}

(( dry_run )) || mkdir -p "$root" "$result/logs"
export_wait=()
if (( make_export )); then
    export_id=$(submit -p gg -N 1 -n 1 -t 01:00:00 -A MCB26031 -J pr_cnvmap \
        -o "$result/logs/prostate_cnv_faming_%j.out" -e "$result/logs/prostate_cnv_faming_%j.err" \
        --wrap "source /home1/10119/ghzheng/.bashrc; conda activate $env && cd $repo && python cancer_metastasis/50_prostate_cnv_faming_names.py --manifest $prostate_manifest --cnv-root $cnv_uniform --out $cnv_prostate")
    echo "prostate CNV export (Faming names): $export_id" >&2
    export_wait=(--dependency=afterok:$export_id)
elif [[ " $blocks " == *" prostate "* ]]; then
    echo "prostate CNV export already complete: $cnv_prostate" >&2
fi

ids=()
for acceptance in $acceptances; do
    for block in $blocks; do
        if [[ "$block" == uniform ]]; then
            m=$uniform_manifest; mds=$uniform_manifest_ds; d=$uniform_depth; ann=""; cnv=$cnv_uniform; wait_for=()
        else
            m=$prostate_manifest; mds=$prostate_manifest_ds; d=$prostate_depth; ann=$prostate_annotations; cnv=$cnv_prostate
            wait_for=("${export_wait[@]}")
        fi
        id=$(submit "${wait_for[@]}" -J "st_${stage}_${block}_${acceptance}" \
            -o "$result/logs/strategies_${stage}_${block}_acc${acceptance}_%j.out" -e "$result/logs/strategies_${stage}_${block}_acc${acceptance}_%j.err" \
            --export=ALL,CONFIDENCEOT_REPO="$repo",CANCER_COT_ROOT="$result",STRAT_STAGE="$stage",STRAT_ACCEPTANCE="$acceptance",STRAT_BLOCK="$block",STRAT_OUT="$root/acc$acceptance",STRAT_CNV_ROOT="$cnv",STRAT_MANIFEST="$m",STRAT_MANIFEST_DS="$mds",STRAT_DEPTH="$d",STRAT_ANNOTATIONS="$ann" \
            "$job")
        ids+=("$block/acc$acceptance=$id")
        echo "$block at $acceptance: $id  -> $root/acc$acceptance/<strategy>/arm_A/$block/ranknm256_noscale_ds_cos" >&2
    done
done

summary="commit       $(git -C "$repo" rev-parse --short HEAD 2> /dev/null || echo unknown)
stage        $stage
acceptances  $acceptances
blocks       $blocks
cnv profiles uniform $cnv_uniform; prostate $cnv_prostate (export job ${export_id:-not needed})
output       $root/acc<a>/<strategy>/arm_A/<block>/ranknm256_noscale_ds_cos
jobs         ${ids[*]}"
echo; echo "$summary"
if (( dry_run )); then echo "nothing was submitted (--dry-run)"; else echo "$summary" > "$root/SUBMITTED_${stage}_$(date +%Y%m%d_%H%M%S).txt"; fi
