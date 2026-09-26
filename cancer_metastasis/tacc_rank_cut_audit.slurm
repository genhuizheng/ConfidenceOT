#!/bin/bash
#SBATCH -J cot_rank_audit
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/rankaudit_%j.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/rankaudit_%j.err
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 02:00:00
#SBATCH -A MCB26031
#
# The rank cut, checked against the cells it will be applied to, before any
# transport runs.
#
# A gate, not a report. rank_value_encode keeps min(top_n, detected) genes, so
# a cell detecting fewer genes than the cut is encoded on fewer coordinates
# and the encoding becomes partly a detection-breadth readout -- the artefact
# the rank arms exist to remove. The audit exits 3 when more than one percent
# of any pair's cells fall below the cut, and the OT array depends on it with
# afterok, so the four rank arms stay unsubmitted rather than landing in a
# directory named ranknm256 for a run that is not the screened configuration.
#
# It runs on the same scoping the transport will use, because that is the
# matrix the encoder sees: a cut that is safe over all cells can be unsafe
# over the malignant compartment alone.
#
# No set -u; TACC's module files and conda's shell functions read variables
# they do not define, and under set -u that kills the job with nothing in the
# log. Every expansion carries its own default.
set -eo pipefail
source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
manifest=${CONFIDENCEOT_AUDIT_MANIFEST:?Set CONFIDENCEOT_AUDIT_MANIFEST}
rank_top_n=${CONFIDENCEOT_AUDIT_RANK_TOP_N:?Set CONFIDENCEOT_AUDIT_RANK_TOP_N}
scope=${CONFIDENCEOT_AUDIT_SCOPE:-malignant}
malignant_column=${CONFIDENCEOT_AUDIT_MALIGNANT_COLUMN:-malignant}
malignant_value=${CONFIDENCEOT_AUDIT_MALIGNANT_VALUE:-malignant}
max_fraction_short=${CONFIDENCEOT_AUDIT_MAX_FRACTION_SHORT:-0.01}
out=${CONFIDENCEOT_AUDIT_OUT:-$result/logs/rank_cut_audit_${rank_top_n}.csv}

cd "$repo"
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$result/logs"

scope_args=(--analysis-scope "$scope")
if [[ "$scope" == "malignant" ]]; then
    scope_args+=(--malignant-column "$malignant_column"
                 --malignant-value "$malignant_value")
fi

echo "manifest    $manifest"
echo "rank cut    $rank_top_n"
echo "scope       $scope"
echo "tolerance   $max_fraction_short of cells may be encoded short"

python cancer_metastasis/tools/audit_rank_top_n.py "$manifest" \
    --rank-top-n "$rank_top_n" \
    "${scope_args[@]}" \
    --max-fraction-short "$max_fraction_short" \
    --out "$out"

echo "rank cut audit passed; the rank arms may run"
