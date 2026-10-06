#!/bin/bash
# Submit the cycling x representation factorial on the real pairs, in one go.
#
#   bash cancer_metastasis/submit_cycling_cnv_factorial.sh --dry-run
#   bash cancer_metastasis/submit_cycling_cnv_factorial.sh
#
# The pairs, manifests and depth tables are the 2026-10-02 run's, as its own
# submissions printed them: the uniform block (110 pairs, the collection's
# malignant call) and the prostate block (24 pairs, Faming Zhao's cells and
# labels). Every arm runs two RNA preprocessings, ranknm256_noscale_ds_cos and
# raw, at one within-side acceptance minimum (default 0.99), through the array
# the preprocessing factorial used, each under its own output root:
#
#   <out>/rna/arm_A..D        uniform and prostate
#   <out>/rna_cnv/arm_A..D    uniform only: prostate has no inferred CNV yet
#   <out>/cnv/arm_A, arm_B    uniform only, label raw; the gene filter does not
#                             reach the CNV profiles, so C would repeat A and D B
#
# A no cycling filter; B malignant cells outside G1 removed on both sides
# (keep_arm_b); C the 97 cell-cycle genes removed; D both (keep_arm_d). Two
# short jobs come first unless their output already exists: the inferred CNV
# profiles exported from the uniform patients' R inferCNV objects
# (44_export_cnv_cells.R), which the CNV arms wait on, and the prostate cycle
# filter in Faming Zhao's cell names (45_build_prostate_cycle_filter.py), which
# the prostate B and D arms wait on.
#
# Rerunning it after a failure resubmits every array; pairs that finished are
# skipped by --skip-completed, so only the failed ones run again. Do not rerun
# while the earlier arrays are still running: two tasks would share a pair.
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
acceptance=0.99
dry_run=0
cap=${CONFIDENCEOT_JOB_CAP:-40}
while (( $# )); do
    case "$1" in
        --dry-run) dry_run=1; shift ;;
        --acceptance) acceptance=$2; shift 2 ;;
        --uniform-manifest) uniform_manifest=$2; shift 2 ;;
        --uniform-manifest-ds) uniform_manifest_ds=$2; shift 2 ;;
        --uniform-depth) uniform_depth=$2; shift 2 ;;
        --prostate-manifest) prostate_manifest=$2; shift 2 ;;
        --prostate-manifest-ds) prostate_manifest_ds=$2; shift 2 ;;
        --prostate-depth) prostate_depth=$2; shift 2 ;;
        --prostate-annotations) prostate_annotations=$2; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

out=$result/cycling_cnv_acc$acceptance
cycle=$base/cycle
cnv_root=$result/cnv_profiles/infercnv_r_patient
prostate_filter=$result/cycle_filters/GSE271675_faming_ot_cell_filter.tsv.gz
labels="ranknm256_noscale_ds_cos raw"
job=$repo/cancer_metastasis/tacc_preprocessing_factorial.slurm
env_cnv=/scratch/10119/ghzheng/conda_envs/cnv
env_r=/scratch/10119/ghzheng/conda_envs/infercnv_r

# Every input, checked before anything is queued. A dry run reports instead of
# stopping, so it can be read on a machine without the data.
problems=0
for path in "$uniform_manifest" "$uniform_manifest_ds" "$uniform_depth" "$prostate_manifest" \
            "$prostate_manifest_ds" "$prostate_depth" "$cycle/ot_cell_filter.tsv.gz" \
            "$cycle/ot_gene_filter.txt" "$cycle/GSE271675.cells.tsv.gz" "$job"; do
    if [[ "$path" != /* || ! -f "$path" ]]; then
        echo "missing: $path" >&2
        problems=1
    fi
done
# The checkout must carry the options this passes. Without them 02_run_pair.py
# refuses every pair, after each has waited in the queue.
if ! grep -q -- '--representation-source' "$repo/cancer_metastasis/02_run_pair.py" 2> /dev/null ||
   ! grep -q 'CONFIDENCEOT_FACTORIAL_EXTRA_ARGS' "$job" 2> /dev/null; then
    echo "$repo is older than this script: git pull there first" >&2
    problems=1
fi
# Settings the array reads from the environment and this does not set. Left in
# the submitting shell by some earlier run, --export=ALL would carry them into
# every task, and no output path would record the difference.
for name in CONFIDENCEOT_ENV CONFIDENCEOT_MALIGNANT_VALUE CONFIDENCEOT_CALIBRATION_NULL \
            CONFIDENCEOT_MAX_OBSERVED_CELLS_PER_SIDE CONFIDENCEOT_MINIMUM_TOTAL_COUNTS \
            CONFIDENCEOT_MINIMUM_DETECTED_GENES CONFIDENCEOT_MAXIMUM_MITOCHONDRIAL_PERCENT \
            CONFIDENCEOT_WORKERS CONFIDENCEOT_THREADS CONFIDENCEOT_FACTORIAL_PAIR_PARALLEL; do
    if [[ -n "${!name}" ]]; then
        echo "set in this shell, would reach every task: $name=${!name}  (unset $name)" >&2
        problems=1
    fi
done
if (( problems && ! dry_run )); then
    exit 2
fi

submit() {
    # $1 = description, rest = sbatch arguments. Prints the job id: the last
    # line only, since sbatch on Vista prints a banner before it.
    local what=$1; shift
    if (( dry_run )); then
        echo "DRY RUN  $what" >&2
        echo "         sbatch $*" >&2
        echo "DRYRUN"
        return 0
    fi
    local id
    id=$(sbatch --parsable "$@" | tail -n 1)
    # Called inside $(...), where set -e does not reach: a refused submission
    # has to end the script here, before anything is made to wait on it.
    if [[ ! "$id" =~ ^[0-9]+$ ]]; then
        echo "sbatch returned no job id for $what; stopping. Already submitted above." >&2
        exit 1
    fi
    echo "submitted $what as $id" >&2
    echo "$id"
}

# The two inputs that have to be made first, reused when a complete one exists.
export_wait=()
make_export=1
if [[ -f "$cnv_root/EXPORT_COMPLETE" &&
      "$(cat "$cnv_root/EXPORT_COMPLETE")" == "$(readlink -f "$uniform_manifest")" ]]; then
    make_export=0
fi
filter_wait=()
make_filter=1
[[ -f "$prostate_filter" ]] && make_filter=0

tasks=$(( 8 + 8 + 8 + 2 + make_export + make_filter ))
queued=$(squeue -u "$USER" -h -r 2> /dev/null | wc -l) || queued=0
if (( ! dry_run && queued + tasks > cap )); then
    echo "$queued tasks already queued and this adds $tasks, over the cap of $cap" >&2
    exit 3
fi
(( dry_run )) || mkdir -p "$result/logs" "$out" "$(dirname "$prostate_filter")"

if (( make_export )); then
    export_id=$(submit "CNV export, the uniform patients" \
        -p gg -N 1 -t 08:00:00 -A MCB26031 -J cnv_export \
        -o "$result/logs/cnv_export_%j.out" \
        --wrap "source /home1/10119/ghzheng/.bashrc; conda activate $env_r && Rscript $repo/cancer_metastasis/44_export_cnv_cells.R $uniform_manifest $cnv_root")
    export_wait=(--dependency=afterok:$export_id)
else
    echo "reusing the CNV export in $cnv_root" >&2
fi
if (( make_filter )); then
    filter_id=$(submit "prostate cycle filter" \
        -p gg -N 1 -t 01:00:00 -A MCB26031 -J prostate_cycle \
        -o "$result/logs/prostate_cycle_filter_%j.out" \
        --wrap "source /home1/10119/ghzheng/.bashrc; conda activate $env_cnv && python $repo/cancer_metastasis/45_build_prostate_cycle_filter.py $prostate_filter")
    filter_wait=(--dependency=afterok:$filter_id)
else
    echo "reusing the prostate cycle filter $prostate_filter" >&2
fi

arm_args() {
    # $1 = arm, $2 = the cell filter table of the block
    local args="--within-side-acceptance-minimum $acceptance"
    case "$1" in
        B) args+=" --cell-filter $2 --cell-filter-column keep_arm_b" ;;
        C) args+=" --exclude-genes $cycle/ot_gene_filter.txt" ;;
        D) args+=" --cell-filter $2 --cell-filter-column keep_arm_d --exclude-genes $cycle/ot_gene_filter.txt" ;;
    esac
    echo "$args"
}

array() {
    # $1 job name, $2 output root, $3 labels, $4 block, $5 manifest, $6 equalised
    # manifest, $7 depth table, $8 annotations, $9 further 02_run_pair.py
    # arguments, ${10} wall time; the rest go to sbatch.
    local name=$1 root=$2 arm_labels=$3 block=$4 manifest=$5 manifest_ds=$6 depth=$7
    local annotations=$8 extra=$9 limit=${10}
    shift 10
    local count
    count=$(wc -w <<< "$arm_labels")
    submit "$name ($block, $root, $arm_labels)" --array=0-$(( count - 1 )) -t "$limit" -J "$name" "$@" \
        --export=ALL,CONFIDENCEOT_REPO="$repo",CANCER_COT_ROOT="$result",CONFIDENCEOT_FACTORIAL_MANIFEST="$manifest",CONFIDENCEOT_FACTORIAL_MANIFEST_DS="$manifest_ds",CONFIDENCEOT_PREDOWNSAMPLE_DEPTH="$depth",CONFIDENCEOT_FACTORIAL_ROOT="$root",CONFIDENCEOT_FACTORIAL_LABELS="$arm_labels",CONFIDENCEOT_FACTORIAL_WORKERS=1,CONFIDENCEOT_ANALYSIS_SCOPE=malignant,CONFIDENCEOT_MALIGNANT_COLUMN=malignant,CONFIDENCEOT_DEVICE=cpu,CONFIDENCEOT_FACTORIAL_FORCE=0,CONFIDENCEOT_FACTORIAL_BLOCK="$block",CONFIDENCEOT_INCLUDE_ANNOTATIONS="$annotations",CONFIDENCEOT_MINIMUM_SCOPE_CELLS=4,CONFIDENCEOT_FACTORIAL_EXTRA_ARGS="$extra" \
        "$job"
}

ids=()
for arm in A B C D; do
    uniform_extra=$(arm_args $arm "$cycle/ot_cell_filter.tsv.gz")
    id=$(array "rna_$arm" "$out/rna/arm_$arm" "$labels" uniform \
        "$uniform_manifest" "$uniform_manifest_ds" "$uniform_depth" "" "$uniform_extra" 48:00:00)
    ids+=("rna_$arm=$id")
    id=$(array "rnacnv_$arm" "$out/rna_cnv/arm_$arm" "$labels" uniform \
        "$uniform_manifest" "$uniform_manifest_ds" "$uniform_depth" "" \
        "$uniform_extra --representation-source rna_cnv --cnv-root $cnv_root" 48:00:00 \
        "${export_wait[@]}")
    ids+=("rnacnv_$arm=$id")
    wait_for=()
    [[ "$arm" == B || "$arm" == D ]] && wait_for=("${filter_wait[@]}")
    id=$(array "pr_rna_$arm" "$out/rna/arm_$arm" "$labels" prostate \
        "$prostate_manifest" "$prostate_manifest_ds" "$prostate_depth" "$prostate_annotations" \
        "$(arm_args $arm "$prostate_filter")" 24:00:00 "${wait_for[@]}")
    ids+=("pr_rna_$arm=$id")
done
for arm in A B; do
    id=$(array "cnv_$arm" "$out/cnv/arm_$arm" raw uniform \
        "$uniform_manifest" "$uniform_manifest_ds" "$uniform_depth" "" \
        "$(arm_args $arm "$cycle/ot_cell_filter.tsv.gz") --representation-source cnv --cnv-root $cnv_root" \
        48:00:00 "${export_wait[@]}")
    ids+=("cnv_$arm=$id")
done

summary=$(cat <<EOF
commit       $(git -C "$repo" rev-parse --short HEAD 2> /dev/null || echo unknown)
output       $out
acceptance   $acceptance
labels       $labels (cnv arms: raw)
uniform      $uniform_manifest
             $uniform_manifest_ds
             $uniform_depth
prostate     $prostate_manifest
             $prostate_manifest_ds
             $prostate_depth
             $prostate_annotations
cell filter  $cycle/ot_cell_filter.tsv.gz; prostate $prostate_filter
gene filter  $cycle/ot_gene_filter.txt
cnv profiles $cnv_root (export ${export_id:-reused})
prostate filter job ${filter_id:-reused}
arrays       ${ids[*]}
EOF
)
echo
echo "$summary"
if (( dry_run )); then
    echo "nothing was submitted (--dry-run)"
else
    echo "$summary" > "$out/SUBMITTED_$(date +%Y%m%d_%H%M%S).txt"
fi
