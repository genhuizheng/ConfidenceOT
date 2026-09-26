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
# One table across every arm of every block.
#
# A block is one (manifest, compartment call). There are two: the deposits
# carrying the uniform inferCNV labels, and the prostate object as Faming Zhao
# annotated it -- which is the authoritative call for that deposit, since the
# GEO conversion of the same accession has no call at all. Both are correct
# for the data they cover; the block travels as a column so the comparison can
# say which pairs came from which.
#
# 25_diagnose_gate_covariates.py takes --dataset LABEL=ROOT repeatedly and
# writes the label into its dataset column, so one invocation pools everything
# and the label is split back into block and preprocessing here. That keeps
# the comparison a pooling of the diagnostic the rest of the project reads,
# with no second scorer to fall out of step with it.
#
# The deposit is joined on as well. Three quarters of the pairs that survive
# the malignant filter come from GSE180661, so a median taken over all of them
# is very nearly a median over ovarian cancer; without the deposit column that
# is invisible, and the table reads as pan-cancer when it is not.
#
# Run per side, because they answer different questions. The source side is
# the primary compartment the gate selects within, where a depth artefact
# appears as shallow cells rejected. The target side is the metastatic
# compartment the transport lands on, where the same artefact appears as mass
# concentrating on deep cells instead.
#
# What to read, per arm: auc_total_counts and auc_n_genes_by_counts near 0.5,
# and spearman_decision_cost_total_counts near 0. Those are the statistics the
# real data can answer. Nothing here is a detection rate -- the real pairs
# have no ground truth, and an arm cannot be preferred for producing a more
# interesting biological result. The benchmark supplies the other half.
#
# An arm that failed on some pairs is still read on the rest: the stage is
# chained with afterany and the diagnostic reports what is missing rather than
# refusing to run. An arm with no completed pair at all is named in the log,
# because a silently absent arm is a comparison with a hole in it.
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

# Every block that ran, found rather than declared: whatever is on disk is
# what there is to compare, and a block the submitter forgot to mention would
# otherwise vanish from the table without a word.
datasets=()
missing=()
blocks=()
shopt -s nullglob
for block_dir in "$out"/*/; do
    block=$(basename "$block_dir")
    [[ "$block" == "diagnostics" ]] && continue
    [[ "$block" == *__* ]] && {
        echo "block name '$block' contains __, which the pooling splits on" >&2
        exit 2
    }
    seen=0
    for label in "${labels[@]}"; do
        if compgen -G "$block_dir$label/*/scope_$scope/$budget_tag/cell_confidence.csv" \
            > /dev/null; then
            datasets+=(--dataset "${block}__${label}=$block_dir$label")
            seen=$(( seen + 1 ))
        else
            missing+=("$block/$label")
        fi
    done
    (( seen > 0 )) && blocks+=("$block")
    echo "block $block: $seen of ${#labels[@]} arms have a gate"
done
if (( ${#missing[@]} > 0 )); then
    echo "no completed gate for: ${missing[*]}" >&2
fi
if (( ${#datasets[@]} == 0 )); then
    echo "not one arm produced a gate under scope_$scope/$budget_tag" >&2
    exit 2
fi

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

python - "$out" <<'PY'
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
frames = []
for side in ("source", "target"):
    path = root / "diagnostics" / side / "gate_covariate_pair_statistics.csv"
    if not path.exists():
        continue
    table = pd.read_csv(path)
    table.insert(0, "side", side)
    frames.append(table)
if not frames:
    raise SystemExit("no per-side statistics to join")
pooled = pd.concat(frames, ignore_index=True)

# The dataset column holds block__label, because that is what was passed as
# --dataset. Split on the FIRST separator only: a block name is checked for
# __ before submission, while a preprocessing label never contains one.
split = pooled["dataset"].astype(str).str.split("__", n=1, expand=True)
pooled.insert(1, "block", split[0])
pooled.insert(2, "preprocessing", split[1])
pooled = pooled.drop(columns=["dataset"])

# Which call named the malignant compartment for this block, and which pairs
# it covered. Both were written beside the results by the array, so they
# cannot drift from the runs the way a re-specified argument can.
definition = {}
deposit = {}
for block_dir in sorted(root.glob("*/")):
    block = block_dir.name
    record = block_dir / "COMPARTMENT_DEFINITION"
    if record.exists():
        definition[block] = record.read_text(encoding="utf-8").strip()
    manifest_record = block_dir / "MANIFEST_PATH"
    if not manifest_record.exists():
        continue
    manifest_path = Path(manifest_record.read_text(encoding="utf-8").strip())
    if not manifest_path.exists():
        print(f"{block}: manifest {manifest_path} is gone; no deposit column")
        continue
    manifest = pd.read_csv(manifest_path)
    if {"pair_id", "dataset_id"} <= set(manifest.columns):
        deposit.update(dict(zip(manifest["pair_id"].astype(str),
                                manifest["dataset_id"].astype(str))))

pooled["compartment_definition"] = pooled["block"].map(definition).fillna("unrecorded")
pooled["deposit"] = pooled["pair_id"].astype(str).map(deposit).fillna("unknown")
pooled.to_csv(root / "diagnostics" / "factorial_gate_statistics.csv", index=False)
print(f"wrote {len(pooled)} rows to "
      f"{root / 'diagnostics' / 'factorial_gate_statistics.csv'}")

# Named to match 25_diagnose_gate_covariates.py's own COVARIATES, which are
# total_counts, n_genes_by_counts and pct_counts_mitochondrial. A name that
# does not match is reported rather than dropped: a readout quietly missing
# from the comparison is worse than one that is absent loudly, because the
# table still looks complete.
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

summary = (pooled.groupby(["side", "block", "preprocessing"])[readouts]
           .median().reset_index())
group = pooled.groupby(["side", "block", "preprocessing"])
summary.insert(0, "pairs_n", group.size().to_numpy())
# The medians weight every pair alike, and the manifest is trimmed at
# emptiness rather than at a round number, so some pairs carry a gate over a
# handful of cells. These two columns are what makes that visible instead of
# dissolved into the median.
summary["n_cells_median"] = group["n_cells"].median().to_numpy()
summary["n_cells_min"] = group["n_cells"].min().to_numpy()
summary.to_csv(root / "diagnostics" / "factorial_gate_summary.csv", index=False)

# The same medians per deposit. Three quarters of the pairs come from one of
# them, so the pooled number above is close to that deposit's number, and the
# only way to see whether an arm behaves differently elsewhere is to look.
by_deposit = (pooled.groupby(["side", "deposit", "preprocessing"])[readouts]
              .median().reset_index())
deposit_group = pooled.groupby(["side", "deposit", "preprocessing"])
by_deposit.insert(0, "pairs_n", deposit_group.size().to_numpy())
by_deposit["n_cells_median"] = deposit_group["n_cells"].median().to_numpy()
by_deposit["n_cells_min"] = deposit_group["n_cells"].min().to_numpy()
by_deposit.to_csv(
    root / "diagnostics" / "factorial_gate_summary_by_deposit.csv", index=False)

print(f"wrote {len(summary)} rows to "
      f"{root / 'diagnostics' / 'factorial_gate_summary.csv'}")
print(f"wrote {len(by_deposit)} rows to "
      f"{root / 'diagnostics' / 'factorial_gate_summary_by_deposit.csv'}")
print()
print("pairs per deposit, so the composition is on the record:")
print(pooled[pooled["side"].eq("source")]
      .groupby("deposit")["pair_id"].nunique().to_string())
print()
with pd.option_context("display.width", 220, "display.max_columns", 50):
    print(summary.to_string(index=False))
PY

echo "factorial diagnostics finished"
