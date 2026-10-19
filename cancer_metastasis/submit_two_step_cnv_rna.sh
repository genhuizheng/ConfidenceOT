#!/bin/bash
# Two-step gate on the real pairs: CNV first, then gene expression on what CNV kept.
#
#   bash cancer_metastasis/submit_two_step_cnv_rna.sh --dry-run
#   bash cancer_metastasis/submit_two_step_cnv_rna.sh
#
# Step 1 already exists and is only read: the CNV-only arm of the cycling x CNV
# factorial (cycling_cnv_acc0.99/cnv/arm_A/uniform/raw), where the cost is built
# from each cell's own inferCNV profile, M4-E at within-side acceptance 0.99.
#
# Step 2 is new: for every uniform-block pair, 02_run_pair.py restricted, on both
# sides, to the cells step 1's M4-E gate retained (--input-gate-root,
# --input-gate-state retained), with the RNA representation, for
# ranknm256_noscale_ds_cos and raw, at the same acceptance. The null calibration
# is redone on the restricted cells. Every cell then falls in one class:
#   rejected by CNV | retained by CNV, rejected by RNA | retained by both.
#
# The array, QC (wide open), seeds, solver and manifests are the factorial's
# own (tacc_preprocessing_factorial.slurm), so step 2 differs from the RNA arm A
# run only by the input restriction. Uniform block only: prostate has no inferred
# CNV yet, so it has no step 1. A pair whose step 1 left fewer than 4 cells on a
# side cannot run step 2; it fails with that reason and is reported, not dropped.
#
# No set -u, as everywhere in this repo.
set -eo pipefail

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
base=${CNV_BASE:-/scratch/10119/ghzheng/primary_metastatic_cancer}
result=${CANCER_COT_ROOT:-$base/confidenceot_results}
uniform_manifest=$result/manifest/factorial_20260929/uniform.csv
uniform_manifest_ds=$result/manifest/factorial_20260929/uniform_ds.csv
uniform_depth=$result/downsampled_factorial_20260929/predownsample_depth.csv.gz
acceptance=0.99
step1=$result/cycling_cnv_acc$acceptance/cnv/arm_A/uniform/raw
out=$result/two_step_cnv_rna_acc$acceptance
labels="ranknm256_noscale_ds_cos raw"
job=$repo/cancer_metastasis/tacc_preprocessing_factorial.slurm
cap=${CONFIDENCEOT_JOB_CAP:-40}
dry_run=0
while (( $# )); do
    case "$1" in
        --dry-run) dry_run=1; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

for path in "$uniform_manifest" "$uniform_manifest_ds" "$uniform_depth" "$job"; do
    [[ -f "$path" ]] || { echo "missing: $path" >&2; exit 1; }
done
[[ -d "$step1" ]] || { echo "step 1 output missing: $step1" >&2; exit 1; }

# Step 1, read only: every completed pair and the one budget tag it was written under.
done_n=$(find "$step1" -mindepth 4 -maxdepth 4 -name SUCCESS -path "*/scope_malignant/*" | wc -l)
tags=$(find "$step1" -mindepth 4 -maxdepth 4 -name SUCCESS -path "*/scope_malignant/*" -printf '%h\n' \
       | xargs -r -n1 basename | sort -u)
if [[ $(printf '%s\n' "$tags" | grep -c .) -ne 1 ]]; then
    echo "step 1 has no single budget tag: [$tags]" >&2
    exit 1
fi
pairs_n=$(( $(wc -l < "$uniform_manifest") - 1 ))
echo "step 1: $done_n of $pairs_n uniform pairs completed, budget tag $tags" >&2

queued=$(squeue -u "$USER" -h -r 2> /dev/null | wc -l) || queued=0
count=$(wc -w <<< "$labels")
if (( ! dry_run && queued + count > cap )); then
    echo "$queued tasks already queued and this adds $count, over the cap of $cap" >&2
    exit 3
fi

extra="--within-side-acceptance-minimum $acceptance --input-gate-root $step1 --input-gate-state retained --input-gate-method M4-E --input-gate-budget-tag $tags"
args=(--array=0-$(( count - 1 )) -t 48:00:00 -J two_step
      --export=ALL,CONFIDENCEOT_REPO="$repo",CANCER_COT_ROOT="$result",CONFIDENCEOT_FACTORIAL_MANIFEST="$uniform_manifest",CONFIDENCEOT_FACTORIAL_MANIFEST_DS="$uniform_manifest_ds",CONFIDENCEOT_PREDOWNSAMPLE_DEPTH="$uniform_depth",CONFIDENCEOT_FACTORIAL_ROOT="$out",CONFIDENCEOT_FACTORIAL_LABELS="$labels",CONFIDENCEOT_FACTORIAL_WORKERS=1,CONFIDENCEOT_ANALYSIS_SCOPE=malignant,CONFIDENCEOT_MALIGNANT_COLUMN=malignant,CONFIDENCEOT_DEVICE=cpu,CONFIDENCEOT_FACTORIAL_FORCE=0,CONFIDENCEOT_FACTORIAL_BLOCK=uniform,CONFIDENCEOT_INCLUDE_ANNOTATIONS=,CONFIDENCEOT_MINIMUM_SCOPE_CELLS=4,CONFIDENCEOT_FACTORIAL_EXTRA_ARGS="$extra"
      "$job")
if (( dry_run )); then
    echo "DRY RUN  sbatch ${args[*]}" >&2
    echo "nothing was submitted (--dry-run)"
    exit 0
fi
mkdir -p "$out"
id=$(sbatch --parsable "${args[@]}" | tail -n 1)
[[ "$id" =~ ^[0-9]+$ ]] || { echo "sbatch returned no job id" >&2; exit 1; }
summary="commit $(git -C "$repo" rev-parse --short HEAD 2> /dev/null || echo unknown)
step 1  $step1 ($done_n pairs, $tags)
step 2  $out/uniform/{ranknm256_noscale_ds_cos,raw}
labels  $labels
array   $id"
echo "$summary"
echo "$summary" > "$out/SUBMITTED_$(date +%Y%m%d_%H%M%S).txt"
