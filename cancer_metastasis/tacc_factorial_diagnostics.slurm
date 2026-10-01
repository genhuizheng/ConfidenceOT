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
raw raw_cos raw_ds raw_ds_cos raw_noscale raw_noscale_cos raw_noscale_ds raw_noscale_ds_cos \
logcpm logcpm_cos logcpm_ds logcpm_ds_cos logcpm_noscale logcpm_noscale_cos logcpm_noscale_ds logcpm_noscale_ds_cos \
rank256 rank256_cos rank256_ds rank256_ds_cos rank256_noscale rank256_noscale_cos rank256_noscale_ds rank256_noscale_ds_cos \
ranknm256 ranknm256_cos ranknm256_ds ranknm256_ds_cos ranknm256_noscale ranknm256_noscale_cos ranknm256_noscale_ds ranknm256_noscale_ds_cos})
scope=${CONFIDENCEOT_ANALYSIS_SCOPE:-malignant}
budget_tag=${CONFIDENCEOT_FACTORIAL_BUDGET_TAG:-budget_0.95}
method=${CONFIDENCEOT_FACTORIAL_METHOD:-M4-E}

cd "$repo"
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

# Every block that ran, found rather than declared: whatever is on disk is
# what there is to compare, and a block the submitter forgot to mention would
# otherwise vanish from the table without a word.
# Which labels carry read equalisation. Asked of the class that owns the
# label rather than matched against its text, for the same reason everything
# else about a label is.
ds_labels=($(PYTHONPATH="$repo/src" python - "${labels[@]}" <<'PYDS'
import sys

from confidenceot.preprocessing import Preprocessing

print(" ".join(label for label in sys.argv[1:]
                if Preprocessing.from_label(label).equalise_depth))
PYDS
)) || { echo "could not split the labels by equalisation" >&2; exit 2; }

datasets=()
# Equalised arms are kept per block, and so are their depth tables. Each
# block's equalised counts were subsampled from that block's own cells by its
# own run of 27_downsample_counts.py, so each block has its own
# pre-equalisation depth table, and 25_diagnose_gate_covariates.py takes one
# --predownsample-depth for a whole invocation.
#
# This used to be a single scalar overwritten inside the block loop. With one
# block that is harmless. With two it is wrong: the glob visits blocks in
# name order, the last one's table wins, and every other block's equalised
# arms are joined against a table that does not contain their cells -- which
# 25 refuses with "some cells are absent from the pre-downsampling depth
# table". That is how the first run with a prostate block beside the uniform
# one died two minutes in. Had 25 filled the gaps instead of refusing, the
# result would have been worse than a crash: half the equalised arms scored
# against another dataset's depths.
declare -A ds_by_block=()
declare -A depth_by_block=()
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
            # Split by which depth the arm has to be scored against. An
            # equalised arm's stored depth is nearly constant by
            # construction, so its AUC against it is trivially 0.5; the
            # question is whether the gate still tracks the depth the cell
            # had before the correction.
            if [[ " ${ds_labels[*]} " == *" $label "* ]]; then
                ds_by_block[$block]+="$label "
            else
                datasets+=(--dataset "${block}__${label}=$block_dir$label")
            fi
            seen=$(( seen + 1 ))
        else
            missing+=("$block/$label")
        fi
    done
    if [[ -n "${ds_by_block[$block]:-}" && -f "$block_dir/PREDOWNSAMPLE_DEPTH" ]]; then
        depth_by_block[$block]=$(cat "$block_dir/PREDOWNSAMPLE_DEPTH")
    fi
    (( seen > 0 )) && blocks+=("$block")
    echo "block $block: $seen of ${#labels[@]} arms have a gate"
done
if (( ${#missing[@]} > 0 )); then
    echo "no completed gate for: ${missing[*]}" >&2
fi
if (( ${#datasets[@]} == 0 && ${#ds_by_block[@]} == 0 )); then
    echo "not one arm produced a gate under scope_$scope/$budget_tag" >&2
    exit 2
fi
# Per block, because a block with no table must not borrow another's.
for block in "${!ds_by_block[@]}"; do
    if [[ -z "${depth_by_block[$block]:-}" ]]; then
        echo "block $block has equalised arms but recorded no depth table;" >&2
        echo "  they would be scored against a depth the correction flattened" >&2
        exit 2
    fi
    echo "block $block: equalised arms scored against ${depth_by_block[$block]}"
done
ds_blocks=$(printf '%s\n' "${!ds_by_block[@]}" | sort)

# The per-side directories are rebuilt from what is on disk, every run. A
# directory left by a previous layout would otherwise sit beside the new one
# and be read by nothing -- or, under a looser glob, by the pooling below.
rm -rf "$out/diagnostics/source" "$out/diagnostics/target" \
       "$out/diagnostics/source_ds" "$out/diagnostics/target_ds"

for side in source target; do
    if (( ${#datasets[@]} > 0 )); then
        echo "===== $side, original counts ====="
        python cancer_metastasis/25_diagnose_gate_covariates.py \
            "$out/diagnostics/$side" \
            "${datasets[@]}" \
            --scope "scope_$scope" \
            --budget-tag "$budget_tag" \
            --method "$method" \
            --side "$side"
    fi
    for block in $ds_blocks; do
        block_args=()
        for label in ${ds_by_block[$block]}; do
            block_args+=(--dataset "${block}__${label}=$out/$block/$label")
        done
        echo "===== $side, $block, equalised counts, scored on its own pre-equalisation depth ====="
        python cancer_metastasis/25_diagnose_gate_covariates.py \
            "$out/diagnostics/${side}_ds/$block" \
            "${block_args[@]}" \
            --scope "scope_$scope" \
            --budget-tag "$budget_tag" \
            --method "$method" \
            --side "$side" \
            --predownsample-depth "${depth_by_block[$block]}"
    done
done

python - "$out" <<'PY'
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
# One original-counts table per side, and one equalised table per side per
# block, because each block's equalised arms were scored against that
# block's own depth table.
candidates = []
for bare in ("source", "target"):
    candidates.append(
        (bare, False,
         root / "diagnostics" / bare / "gate_covariate_pair_statistics.csv"))
    candidates.extend(
        (bare, True, path) for path in sorted(
            (root / "diagnostics" / f"{bare}_ds").glob(
                "*/gate_covariate_pair_statistics.csv")))
frames = []
for bare, equalised, path in candidates:
    if not path.exists():
        continue
    table = pd.read_csv(path)
    # 25_diagnose_gate_covariates.py already writes a side column, from the
    # --side it was given; the directory adds whether the arm ran on
    # equalised counts. Checked rather than assumed, because the two must
    # agree for the concatenation to mean anything.
    if "side" not in table:
        table.insert(0, "side", bare)
    elif not table["side"].astype(str).eq(bare).all():
        raise SystemExit(
            f"{path} carries sides {sorted(table['side'].unique())} but was "
            f"written for {bare}")
    table["equalised"] = equalised
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

# Deposit to cancer, from the collection's datasets_manifest.csv rather than
# from the accession numbers, which say nothing. The three the advisor asked
# for are ovarian, colorectal and prostate; the rest are named rather than
# folded into "other", because a block of nineteen pairs that quietly
# disappears is the same failure as a pooled median that hides a deposit.
CANCER = {
    "GSE180661": "ovarian",
    "GSE164522": "colorectal", "GSE178318": "colorectal",
    "GSE225857": "colorectal", "GSE315534": "colorectal",
    "GSE271675": "prostate",
    "GSE167036": "breast", "GSE180286": "breast",
    "GSE163558": "gastric",
    "GSE181919": "head_and_neck", "GSE188737": "head_and_neck",
    "HRA000036": "nasopharyngeal",
    "GSE197177": "pancreatic",
    "GSE184362": "thyroid", "GSE198099": "lung",
}
pooled["cancer"] = pooled["deposit"].map(CANCER).fillna("unmapped")

# One depth column to read, with its provenance beside it. An equalised arm
# has a nearly constant stored depth, so auc_total_counts over it is
# trivially 0.5; what 25 computed for those arms against the recorded
# pre-equalisation depth is auc_predownsample_total_counts. Choosing between
# them per row, and saying which was chosen, is the only way the two halves
# of the grid can be read in one column without the choice being invisible.
predownsample = "auc_predownsample_total_counts"
if predownsample in pooled:
    use = pooled["equalised"] & pooled[predownsample].notna()
    pooled["auc_depth"] = pooled[predownsample].where(use, pooled.get("auc_total_counts"))
    pooled["depth_readout"] = np.where(use, "pre-equalisation depth", "stored depth")
else:
    pooled["auc_depth"] = pooled.get("auc_total_counts")
    pooled["depth_readout"] = "stored depth"
    if pooled["equalised"].any():
        print("equalised arms are present but no pre-equalisation depth column "
              "came back from the diagnostic; their auc_depth is the flattened "
              "one and means nothing")
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

# The same, per cancer. This is the level the advisor asked for -- ovarian,
# colorectal, prostate -- and the level at which the composition problem is
# visible: 92 of the 118 evaluable pairs are ovarian, so a pooled median is
# very nearly an ovarian median.
cancer_readouts = [column for column in
                   (*readouts, "auc_depth") if column in pooled]
cancer_group = pooled.groupby(["side", "cancer", "preprocessing"])
by_cancer = cancer_group[cancer_readouts].median().reset_index()
by_cancer.insert(0, "pairs_n", cancer_group.size().to_numpy())
by_cancer["n_cells_median"] = cancer_group["n_cells"].median().to_numpy()
by_cancer.to_csv(
    root / "diagnostics" / "factorial_gate_summary_by_cancer.csv", index=False)
print(f"wrote {len(by_cancer)} rows to "
      f"{root / 'diagnostics' / 'factorial_gate_summary_by_cancer.csv'}")
print()
print("pairs per cancer, so the composition is on the record:")
print(pooled[pooled["side"].eq("source")]
      .groupby("cancer")["pair_id"].nunique().sort_values(ascending=False)
      .to_string())

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
