#!/bin/bash
#SBATCH -J cot_fact_diag
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/factdiag_%j.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/factdiag_%j.err
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 04:00:00
#SBATCH -A MCB26031
#
# One diagnostic table across all twelve arms.
#
# 25_diagnose_gate_covariates.py takes --dataset LABEL=ROOT repeatedly and
# writes the label into the dataset column, so a single invocation pools the
# arms into one table and one per-arm summary. That is the whole comparison,
# with no new scoring code to keep in step with the diagnostic that the rest
# of the project already reads.
#
# Run per side, because they answer different questions. The source side is
# the primary compartment the gate selects within, which is where a depth
# artefact shows up as cells rejected for being shallow. The target side is
# the metastatic compartment the transport lands on, where the same artefact
# shows up as mass concentrating on deep cells instead.
#
# What to read, per arm: auc_total_counts and auc_n_genes_by_counts near 0.5,
# and spearman_decision_cost_total_counts near 0. Those are the statistics the
# real data can answer. Nothing here is a detection rate -- the real pairs
# have no ground truth, and an arm cannot be preferred for producing a more
# interesting biological result. The benchmark supplies the other half.
#
# An arm that failed on some pairs is still read on the rest: the stage is
# chained with afterany and the diagnostic reports what is missing rather than
# refusing to run. An arm with no completed pair at all raises, which is the
# correct outcome -- a silently absent arm is a comparison with a hole in it.
#
# No set -u; TACC's module files and conda's shell functions read variables
# they do not define, and under set -u that kills the job with nothing in the
# log. Every expansion carries its own default.
set -eo pipefail
source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
out=${CONFIDENCEOT_FACTORIAL_ROOT:-$result/preprocessing_factorial}
labels=(${CONFIDENCEOT_FACTORIAL_LABELS:-\
raw raw_noscale raw_cos raw_noscale_cos \
logcpm logcpm_noscale logcpm_cos logcpm_noscale_cos \
ranknm256 ranknm256_noscale ranknm256_cos ranknm256_noscale_cos})
scope=${CONFIDENCEOT_ANALYSIS_SCOPE:-malignant}
budget_tag=${CONFIDENCEOT_FACTORIAL_BUDGET_TAG:-budget_0.95}
method=${CONFIDENCEOT_FACTORIAL_METHOD:-M4-E}

cd "$repo"
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

# Only the arms that actually produced a gate. An arm whose directory is
# absent is named in the log rather than passed to the diagnostic, which would
# raise on it and take the other eleven down with it.
datasets=()
missing=()
for label in "${labels[@]}"; do
    if compgen -G "$out/$label/*/scope_$scope/$budget_tag/cell_confidence.csv" \
        > /dev/null; then
        datasets+=(--dataset "$label=$out/$label")
    else
        missing+=("$label")
    fi
done
if (( ${#missing[@]} > 0 )); then
    echo "no completed gate for: ${missing[*]}" >&2
fi
if (( ${#datasets[@]} == 0 )); then
    echo "not one arm produced a gate under scope_$scope/$budget_tag" >&2
    exit 2
fi
echo "pooling $(( ${#datasets[@]} / 2 )) of ${#labels[@]} arms"

for side in source target; do
    echo "===== $side ====="
    python cancer_metastasis/25_diagnose_gate_covariates.py \
        "$out/diagnostics/$side" \
        "${datasets[@]}" \
        --scope "scope_$scope" \
        --budget-tag "$budget_tag" \
        --method "$method" \
        --side "$side"
done

# One table for reading, with the side carried as a column. The per-side
# directories keep their own copies; this is the join the comparison is
# actually read from.
python - "$out/diagnostics" <<'PY'
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
frames = []
for side in ("source", "target"):
    path = root / side / "gate_covariate_pair_statistics.csv"
    if not path.exists():
        continue
    table = pd.read_csv(path)
    table.insert(0, "side", side)
    frames.append(table)
if not frames:
    raise SystemExit("no per-side statistics to join")
pooled = pd.concat(frames, ignore_index=True)
# 'dataset' is the preprocessing label here, because that is what was passed
# as --dataset. Renamed so a reader coming from the other diagnostics does
# not take it for a deposit.
pooled = pooled.rename(columns={"dataset": "preprocessing"})
pooled.to_csv(root / "factorial_gate_statistics.csv", index=False)
print(f"wrote {len(pooled)} rows to {root / 'factorial_gate_statistics.csv'}")

# Named to match 25_diagnose_gate_covariates.py's own COVARIATES, which
# are total_counts, n_genes_by_counts and pct_counts_mitochondrial. A name
# that does not match is reported rather than dropped: a readout quietly
# missing from the comparison is worse than one that is absent loudly,
# because the table still looks complete.
wanted = ("retained_fraction", "sign_rule_retained_fraction",
          "auc_decision_cost",
          "auc_total_counts", "auc_n_genes_by_counts",
          "auc_pct_counts_mitochondrial",
          "spearman_decision_cost_total_counts",
          "spearman_decision_cost_n_genes_by_counts")
readouts = [column for column in wanted if column in pooled]
absent = [column for column in wanted if column not in pooled]
if absent:
    print(f"not in the pair statistics, so not summarised: {absent}")
summary = (pooled.groupby(["side", "preprocessing"])[readouts]
           .median().reset_index())
summary.to_csv(root / "factorial_gate_summary.csv", index=False)
print(f"wrote {len(summary)} rows to {root / 'factorial_gate_summary.csv'}")
with pd.option_context("display.width", 200, "display.max_columns", 50):
    print(summary.to_string(index=False))
PY

echo "factorial diagnostics finished"
