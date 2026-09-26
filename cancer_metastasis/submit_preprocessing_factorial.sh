#!/bin/bash
# Submit the preprocessing factorial on the real pairs: audit, then OT, then
# the gate diagnostics.
#
#   bash cancer_metastasis/submit_preprocessing_factorial.sh --dry-run
#   bash cancer_metastasis/submit_preprocessing_factorial.sh --partition gg --time 48:00:00
#   bash cancer_metastasis/submit_preprocessing_factorial.sh --stage diagnose
#
# Twelve arms over the pan-cancer manifest, one output root each, everything
# else held fixed. What comes back is not a performance number -- the real
# pairs have no ground truth -- but a covariate statistic per arm: whether the
# gate tracks sequencing depth, detected genes or mitochondrial fraction, and
# how much of the retained set the calibrated rule accounts for. The benchmark
# says which configuration recovers a known answer; this says which one the
# real gate agrees with, and the two questions need both halves.
#
# Three stages, chained with afterok so a failure leaves the rest PENDING
# rather than running against absent output:
#
#   J1 audit      tools/audit_rank_top_n.py, and only when a rank arm is in
#                 the set. The rank cut has to stay below the shallowest
#                 cell's detected-gene count or the encoding manufactures the
#                 low-content artefact it exists to remove. Submitted as a
#                 gate, not a report: it exits non-zero and the OT never runs.
#   J2 ot         tacc_preprocessing_factorial.slurm, labels x workers tasks
#   J3 diagnose   25_diagnose_gate_covariates.py across all twelve arms at
#                 once, source and target. It takes --dataset LABEL=ROOT
#                 repeatedly, so one invocation pools the arms into one table
#                 with the label in the dataset column -- which is the
#                 comparison, with no new scoring code to keep in step.
#
# Two TACC limits shape it. Array tasks count one by one against the
# submission cap, which is per queue, so twelve labels times the workers plus
# two has to stay under it: the default of two workers is 26 jobs. And a wall
# limit above the queue's own is refused rather than trimmed, so --time is
# here; gg reports no limit of its own, and 48 hours costs nothing to ask for
# when a worker carrying half the manifest may want it.
#
# No set -u. It fires on things that are fine here, such as an empty array
# expansion, while permitting the failure that has actually cost this project
# job numbers: a path variable that is defined but empty. The explicit
# non-empty absolute-path check is the protection that matters.
set -eo pipefail

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
manifest=${CONFIDENCEOT_FACTORIAL_MANIFEST:-$result/manifest/pancancer_20260924/pair_manifest_eligible.csv}
out=${CONFIDENCEOT_FACTORIAL_ROOT:-$result/preprocessing_factorial}
labels=${CONFIDENCEOT_FACTORIAL_LABELS:-"\
raw raw_noscale raw_cos raw_noscale_cos \
logcpm logcpm_noscale logcpm_cos logcpm_noscale_cos \
ranknm256 ranknm256_noscale ranknm256_cos ranknm256_noscale_cos"}
workers=2
scope=${CONFIDENCEOT_ANALYSIS_SCOPE:-malignant}
malignant_column=${CONFIDENCEOT_MALIGNANT_COLUMN:-malignant}
malignant_value=${CONFIDENCEOT_MALIGNANT_VALUE:-malignant}
budget_tag=${CONFIDENCEOT_FACTORIAL_BUDGET_TAG:-budget_0.95}
block=${CONFIDENCEOT_FACTORIAL_BLOCK:-uniform}
annotations=${CONFIDENCEOT_INCLUDE_ANNOTATIONS:-}
minimum_cells=${CONFIDENCEOT_MINIMUM_SCOPE_CELLS:-1}
device=${CONFIDENCEOT_DEVICE:-cpu}
partition=""
limit=""
stage=all
dry_run=0
force=0
cap=${CONFIDENCEOT_JOB_CAP:-40}
# Named explicitly rather than taken from the PATH. This runs on a
# login node, where the environment may not be activated, and a bare
# python there is whatever the shell happens to have -- which would
# fail the label probe below and block a submission that is fine.
env_python=${CONFIDENCEOT_PYTHON:-${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}/bin/python}

while (( $# )); do
    case "$1" in
        --dry-run) dry_run=1; shift ;;
        --stage) stage=$2; shift 2 ;;
        --labels) labels=$2; shift 2 ;;
        --workers) workers=$2; shift 2 ;;
        --partition) partition=$2; shift 2 ;;
        --time) limit=$2; shift 2 ;;
        --manifest) manifest=$2; shift 2 ;;
        --out) out=$2; shift 2 ;;
        --scope) scope=$2; shift 2 ;;
        --block) block=$2; shift 2 ;;
        --annotations) annotations=$2; shift 2 ;;
        --minimum-cells) minimum_cells=$2; shift 2 ;;
        --device) device=$2; shift 2 ;;
        --force) force=1; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

for name in repo result manifest out; do
    value=${!name}
    if [[ -z "$value" ]]; then
        echo "$name is empty; export the paths in the submitting shell" >&2
        exit 2
    fi
    if [[ "$value" != /* ]]; then
        echo "$name is not an absolute path: $value" >&2
        exit 2
    fi
done

label_array=($labels)
tasks=$(( ${#label_array[@]} * workers ))

# A dry run must not touch the filesystem, so it can be read on a laptop where
# $result does not exist. SLURM opens the file named in #SBATCH -o before the
# script runs, so the log directory is made here or the job dies with no log
# at all -- which has cost a job number on this project already.
(( dry_run )) || mkdir -p "$result/logs" "$out"

# Counted within the partition being submitted to, because the cap is per
# queue. Under pipefail a missing squeue would otherwise abort the script
# from inside the courtesy check it was meant to serve.
queued_now() {
    local n
    local scope_args=()
    [[ -n "$partition" ]] && scope_args=(-p "$partition")
    n=$(squeue -u "$USER" -h -r "${scope_args[@]}" 2> /dev/null | wc -l) || n=0
    echo "${n:-0}"
}

where=()
[[ -n "$partition" ]] && where+=(--partition="$partition")
wall=()
[[ -n "$limit" ]] && wall=(--time="$limit")

submit() {
    local description=$1; shift
    if (( dry_run )); then
        echo "DRY RUN  $description" >&2
        echo "         sbatch $*" >&2
        echo "DRYRUN"
        return 0
    fi
    local id
    id=$(sbatch --parsable "$@" | tail -n 1)
    echo "submitted $description as $id" >&2
    echo "$id"
}

# Does this set contain an arm whose representation is a rank encoding? Asked
# of the class that defines the label, never matched against the string here.
# A shell that guessed wrong would skip the cut audit for an arm that needs it,
# and the encoding would then manufacture the artefact it exists to remove
# while the label still read ranknm256.
rank_cut=$(PYTHONPATH="$repo/src" "$env_python" - $labels <<'PY'
import sys

from confidenceot.preprocessing import Preprocessing

cuts = set()
for label in sys.argv[1:]:
    configuration = Preprocessing.from_label(label)
    if str(configuration.normalisation).startswith("rank"):
        cuts.add(int(configuration.rank_top_n))
if len(cuts) > 1:
    raise SystemExit(f"several rank cuts in one submission: {sorted(cuts)}")
print(cuts.pop() if cuts else "")
PY
) || {
    echo "could not read the labels with confidenceot.Preprocessing," >&2
    echo "using $env_python." >&2
    echo "Fix that before submitting: the alternative is guessing which" >&2
    echo "arms need the rank-cut audit." >&2
    exit 2
}

total=$(( tasks + 2 ))
if [[ "$stage" == "all" ]] && (( dry_run == 0 )) && \
   (( $(queued_now) + total > cap )); then
    echo "the chain is $total jobs (${#label_array[@]} labels x $workers" \
         "workers, plus the audit and the diagnostics) against a cap of" \
         "$cap, with $(queued_now) already queued on ${partition:-this queue}." >&2
    echo "lower --workers, or run --stage ot on its own once the queue" \
         "drains." >&2
    exit 3
fi

# ---- J1: the rank cut is a gate, not a report -------------------------------
audit_id=""
if [[ "$stage" == "all" || "$stage" == "audit" ]] && [[ -n "$rank_cut" ]]; then
    audit_id=$(submit "rank-cut audit at $rank_cut ($block)" "${where[@]}" \
        --export=ALL,CONFIDENCEOT_REPO="$repo",CANCER_COT_ROOT="$result",CONFIDENCEOT_AUDIT_MANIFEST="$manifest",CONFIDENCEOT_AUDIT_RANK_TOP_N="$rank_cut",CONFIDENCEOT_AUDIT_SCOPE="$scope",CONFIDENCEOT_AUDIT_MALIGNANT_COLUMN="$malignant_column",CONFIDENCEOT_AUDIT_INCLUDE_ANNOTATIONS="$annotations",CONFIDENCEOT_AUDIT_OUT="$result/logs/rank_cut_audit_${block}_${rank_cut}.csv" \
        "$repo/cancer_metastasis/tacc_rank_cut_audit.slurm")
elif [[ -z "$rank_cut" ]]; then
    echo "no rank arm in this set; the cut audit is skipped" >&2
fi

# ---- J2: the arms -----------------------------------------------------------
# Submitted as two arrays when the set mixes them, because the cut audit is a
# gate on the rank arms alone. One array under afterok meant an unsafe cut
# cancelled the eight arms that never touch rank encoding, which is a
# statement about a parameter they do not use. Splitting costs no extra array
# tasks -- the same twelve labels, divided -- only one more job id.
submit_arm_array() {
    # $1 = space-separated labels, $2 = extra sbatch args as a name, rest passed
    local arm_labels=$1; shift
    local arm_count
    arm_count=$(wc -w <<< "$arm_labels")
    (( arm_count == 0 )) && return 0
    submit "OT ($block: $arm_count labels x $workers workers)" \
        --array=0-$(( arm_count * workers - 1 )) \
        "$@" "${where[@]}" "${wall[@]}" \
        --export=ALL,CONFIDENCEOT_REPO="$repo",CANCER_COT_ROOT="$result",CONFIDENCEOT_FACTORIAL_MANIFEST="$manifest",CONFIDENCEOT_FACTORIAL_ROOT="$out",CONFIDENCEOT_FACTORIAL_LABELS="$arm_labels",CONFIDENCEOT_FACTORIAL_WORKERS="$workers",CONFIDENCEOT_ANALYSIS_SCOPE="$scope",CONFIDENCEOT_MALIGNANT_COLUMN="$malignant_column",CONFIDENCEOT_DEVICE="$device",CONFIDENCEOT_FACTORIAL_FORCE="$force",CONFIDENCEOT_FACTORIAL_BLOCK="$block",CONFIDENCEOT_INCLUDE_ANNOTATIONS="$annotations",CONFIDENCEOT_MINIMUM_SCOPE_CELLS="$minimum_cells" \
        "$repo/cancer_metastasis/tacc_preprocessing_factorial.slurm"
}

ot_id=""
rank_ot_id=""
if [[ "$stage" == "all" || "$stage" == "ot" ]]; then
    # Asked of the class that defines each label, not matched against its
    # text, for the same reason the cut itself is.
    IFS=$'\t' read -r plain_labels rank_labels < <(
        PYTHONPATH="$repo/src" "$env_python" - $labels <<'PYSPLIT'
import sys

from confidenceot.preprocessing import Preprocessing

plain, ranked = [], []
for label in sys.argv[1:]:
    target = ranked if str(
        Preprocessing.from_label(label).normalisation).startswith("rank") \
        else plain
    target.append(label)
print(" ".join(plain) + "\t" + " ".join(ranked))
PYSPLIT
    )

    # The arms that do not rank anything run regardless of the cut.
    ot_id=$(submit_arm_array "$plain_labels")

    # The rank arms wait on the audit, and are the only thing it can cancel.
    depend=()
    [[ -n "$audit_id" && "$audit_id" != "DRYRUN" ]] && \
        depend=(--dependency=afterok:"$audit_id")
    rank_ot_id=$(submit_arm_array "$rank_labels" "${depend[@]}")
fi

# ---- J3: one diagnostic table over all twelve -------------------------------
if [[ "$stage" == "all" || "$stage" == "diagnose" ]]; then
    depend=()
    # afterany, not afterok: an arm that fails on some pairs still has a gate
    # worth reading on the rest, and the diagnostic reports what is missing.
    for finished in "$ot_id" "$rank_ot_id"; do
        [[ -n "$finished" && "$finished" != "DRYRUN" ]] && \
            depend+=(--dependency=afterany:"$finished")
    done
    # One --dependency wins over another, so several afterany ids go in one.
    if (( ${#depend[@]} > 1 )); then
        joined=$(printf '%s:' "${depend[@]}")
        joined=${joined//--dependency=afterany:/}
        depend=(--dependency=afterany:"${joined%:}")
    fi
    submit "gate diagnostics" "${depend[@]}" "${where[@]}" \
        --export=ALL,CONFIDENCEOT_REPO="$repo",CANCER_COT_ROOT="$result",CONFIDENCEOT_FACTORIAL_ROOT="$out",CONFIDENCEOT_FACTORIAL_LABELS="$labels",CONFIDENCEOT_ANALYSIS_SCOPE="$scope",CONFIDENCEOT_FACTORIAL_BUDGET_TAG="$budget_tag" \
        "$repo/cancer_metastasis/tacc_factorial_diagnostics.slurm" > /dev/null
fi

echo
echo "manifest    $manifest"
echo "output      $out"
echo "labels      ${#label_array[@]}"
echo "workers     $workers per label"
echo "jobs        $tasks OT + 2 for the whole chain"
echo "scope       $scope, budget tag $budget_tag"
echo "block       $block"
echo "minimum     $minimum_cells cells per side (must match the trim)"
echo "compartment ${annotations:-uniform call $malignant_column==$malignant_value}"
echo "rank cut    ${rank_cut:-none in this set}"
echo "partition   ${partition:-the default in each script (gg)}"
echo "device      $device"
echo "time        ${limit:-24:00:00, the default in the array}"
(( dry_run )) && echo "nothing was submitted (--dry-run)"
