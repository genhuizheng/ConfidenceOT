"""Diagnose what a stored ConfidenceOT gate actually separates.

The script reads completed ``cell_confidence.csv`` tables and answers three
questions without refitting optimal transport.

1. Does the rejection-budget floor, rather than the calibrated rejection cost,
   define the retained set?  ``sign_rule_retained_fraction`` is the fraction of
   cells that the calibrated decision rule ``decision_cost < rejection_cost``
   would retain.  When that fraction is far below ``retained_fraction``, the
   cardinality floor invented most of the retained set.
2. Do retained cells really carry the lowest decision cost?  The budgeted
   projection ranks cells by the mass-weighted coefficient
   ``partner_mass * (decision_cost - rejection_cost)``, so a retained set built
   mainly by the floor can be ordered by transport mass instead of by cost.
   ``auc_decision_cost`` near 0 means a clean cost ordering; near 0.5 means the
   gate is not ordered by cost at all.
3. Is the gate predictable from per-cell sequencing depth, detected gene count,
   or mitochondrial fraction?  Any ``auc_*`` far from 0.5 for those covariates
   means the gate tracks a technical axis.

``signed_rejection_margin``, ``relative_rejection_margin`` and
``normalized_rejection_score`` are monotone transforms of ``decision_cost``, so
their AUCs are identical to ``auc_decision_cost`` and are not recomputed.

Every statistic is per exact pair.  Dataset rows report the median across pairs
plus a signed-rank test against the AUC 0.5 null, because pairs sharing a
patient or a primary sample are not independent.

AUC orientation: ``retained`` is the positive class, so AUC > 0.5 means
retained cells carry higher values and AUC < 0.5 means they carry lower values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon


COVARIATES = (
    "total_counts",
    "n_genes_by_counts",
    "pct_counts_mitochondrial",
)
REQUIRED = ("method", "side", "observation_id", "retained", "raw_retained")
AUC_TARGETS = ("decision_cost", *COVARIATES)
SUMMARY_EXTRA = (
    "retained_fraction",
    "sign_rule_retained_fraction",
    "forced_in_share_of_retained",
    "sign_rule_concordance",
    "terminal_disagreement_fraction",
    "depth_residual_gate_jaccard",
    "chance_gate_jaccard",
)


def dataset_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"--dataset expects LABEL=PATH, found {value!r}"
        )
    label, path = value.split("=", 1)
    if not label.strip():
        raise argparse.ArgumentTypeError("--dataset label must be non-empty")
    return label.strip(), Path(path.strip())


def rank_auc(values: pd.Series, positive: np.ndarray) -> float:
    """Return the Mann--Whitney AUC, averaging ranks over ties."""
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    label = np.asarray(positive, dtype=bool)
    finite = np.isfinite(numeric)
    numeric, label = numeric[finite], label[finite]
    n_positive = int(label.sum())
    n_negative = int(numeric.size - n_positive)
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    ranks = rankdata(numeric)
    positive_rank_sum = float(ranks[label].sum())
    return (
        positive_rank_sum - n_positive * (n_positive + 1) / 2.0
    ) / (n_positive * n_negative)


def rank_correlation(left: pd.Series | None, right: pd.Series | None) -> float:
    """Return Spearman rho, or NaN when either side is absent or constant."""
    if left is None or right is None:
        return float("nan")
    x = pd.to_numeric(left, errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(right, errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return float(spearmanr(x, y).statistic)


def depth_residual_gate_jaccard(
    table: pd.DataFrame, retained: np.ndarray
) -> float:
    """Jaccard between the gate and a same-size gate on depth-residual cost.

    Ranks are regressed rather than raw values, so the correction is monotone
    and consistent with the Spearman diagnostic.  1.0 means removing the depth
    component selects exactly the same cells; chance overlap for two sets of
    size ``k`` drawn from ``n`` cells is roughly ``k / (2n - k)``, so a value
    near that floor means depth, not the remaining geometry, decides the gate.
    """
    predictors = [
        column for column in ("total_counts", "n_genes_by_counts")
        if column in table
    ]
    if "decision_cost" not in table or not predictors:
        return float("nan")
    cost = pd.to_numeric(table["decision_cost"], errors="coerce").to_numpy(np.float64)
    design = np.column_stack([
        pd.to_numeric(table[column], errors="coerce").to_numpy(np.float64)
        for column in predictors
    ])
    finite = np.isfinite(cost) & np.all(np.isfinite(design), axis=1)
    actual = retained[finite]
    accepted = int(actual.sum())
    if finite.sum() < 10 or accepted == 0 or accepted == actual.size:
        return float("nan")
    response = rankdata(cost[finite])
    ranked = np.column_stack([
        rankdata(design[finite, index]) for index in range(design.shape[1])
    ])
    if np.any(ranked.max(axis=0) == ranked.min(axis=0)):
        return float("nan")
    matrix = np.column_stack([np.ones(ranked.shape[0]), ranked])
    coefficients, *_ = np.linalg.lstsq(matrix, response, rcond=None)
    residual = response - matrix @ coefficients
    corrected = np.zeros(residual.size, dtype=bool)
    corrected[np.argsort(residual, kind="stable")[:accepted]] = True
    union = int((actual | corrected).sum())
    return float(int((actual & corrected).sum()) / union) if union else float("nan")


def confidence_paths(root: Path, scope: str, budget_tag: str | None) -> list[Path]:
    return sorted(root.glob(f"*/{scope}/{budget_tag or '*'}/cell_confidence.csv"))


def group_medians(
    values: pd.Series, retained: np.ndarray, name: str
) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce")
    return {
        f"median_{name}_retained": float(numeric[retained].median()),
        f"median_{name}_rejected": float(numeric[~retained].median()),
    }


def pair_record(
    path: Path, dataset: str, method: str, side: str, scope: str
) -> dict[str, object] | None:
    table = pd.read_csv(path)
    missing = [column for column in REQUIRED if column not in table]
    if missing:
        raise RuntimeError(f"{path} is missing required columns: {missing}")
    table = table.loc[table["method"].eq(method) & table["side"].eq(side)]
    if table.empty:
        return None
    table = table.reset_index(drop=True)
    if table["observation_id"].duplicated().any():
        raise RuntimeError(f"{path} has duplicate observation_id for {method}/{side}")

    retained = table["retained"].astype(bool).to_numpy()
    raw_retained = table["raw_retained"].astype(bool).to_numpy()
    if "budget_overridden" in table:
        overridden = table["budget_overridden"].astype(bool).to_numpy()
    else:
        overridden = raw_retained != retained
    # The floor can only force rejected cells in.  The reverse direction means
    # the terminal re-solve moved the raw gate away from the returned gate, so
    # it is recorded rather than assumed to be empty.
    forced_in = overridden & retained
    forced_out = overridden & ~retained
    record: dict[str, object] = {
        "dataset": dataset,
        "pair_id": path.parents[2].name,
        "budget_tag": path.parents[0].name,
        "scope": scope,
        "method": method,
        "side": side,
        "n_cells": int(retained.size),
        "retained_n": int(retained.sum()),
        "retained_fraction": float(retained.mean()),
        "raw_retained_fraction": float(raw_retained.mean()),
        "forced_in_n": int(forced_in.sum()),
        "forced_in_share_of_retained": (
            float(forced_in.sum() / retained.sum()) if retained.sum() else float("nan")
        ),
        "terminal_disagreement_fraction": float(forced_out.mean()),
    }

    if "decision_cost" in table and "rejection_cost" in table:
        cost = pd.to_numeric(table["decision_cost"], errors="coerce")
        c = pd.to_numeric(table["rejection_cost"], errors="coerce")
        # The calibrated decision rule the rejection cost was chosen to define.
        sign_rule = (cost < c).to_numpy()
        usable = np.isfinite(cost.to_numpy()) & np.isfinite(c.to_numpy())
        record["rejection_cost"] = float(c.iloc[0])
        record["sign_rule_retained_fraction"] = (
            float(sign_rule[usable].mean()) if usable.any() else float("nan")
        )
        record["sign_rule_concordance"] = (
            float((sign_rule[usable] == retained[usable]).mean())
            if usable.any() else float("nan")
        )
    else:
        record["rejection_cost"] = float("nan")
        record["sign_rule_retained_fraction"] = float("nan")
        record["sign_rule_concordance"] = float("nan")

    for column in AUC_TARGETS:
        if column not in table:
            record[f"auc_{column}"] = float("nan")
            continue
        record[f"auc_{column}"] = rank_auc(table[column], retained)
        record.update(group_medians(table[column], retained, column))
    for column in COVARIATES:
        # A strong negative rank correlation means the transport cost itself is
        # a readout of the covariate, upstream of any gate decision.
        record[f"spearman_decision_cost_{column}"] = rank_correlation(
            table.get("decision_cost"), table.get(column)
        )
    record["depth_residual_gate_jaccard"] = depth_residual_gate_jaccard(
        table, retained
    )
    record["chance_gate_jaccard"] = (
        float(retained.sum() / (2 * retained.size - retained.sum()))
        if retained.size else float("nan")
    )
    if "decision_cost" in table:
        # The raw gate is a sign test on the coefficient, so its cost AUC is
        # near 0 by construction.  A large gap to auc_decision_cost isolates
        # the budgeted projection as the source of the disagreement.
        record["auc_raw_decision_cost"] = rank_auc(table["decision_cost"], raw_retained)
    return record


def null_value(statistic: str) -> float:
    """Return the no-effect reference for a statistic, or NaN when undefined."""
    if statistic.startswith("auc_"):
        return 0.5
    if statistic.startswith("spearman_"):
        return 0.0
    return float("nan")


def summarize(pairs: pd.DataFrame) -> pd.DataFrame:
    statistics = [
        column for column in pairs.columns
        if (
            column.startswith(("auc_", "spearman_", "median_"))
            or column in SUMMARY_EXTRA
        )
        and pairs[column].notna().any()
    ]
    rows = []
    for dataset, table in pairs.groupby("dataset", sort=True):
        for column in statistics:
            values = pd.to_numeric(table[column], errors="coerce").dropna()
            null = null_value(column)
            p_value = float("nan")
            fraction_above = float("nan")
            if np.isfinite(null) and len(values):
                fraction_above = float(values.gt(null).mean())
                nonzero = (values - null)[(values - null).ne(0.0)]
                if len(nonzero) >= 6:
                    try:
                        p_value = float(wilcoxon(nonzero).pvalue)
                    except ValueError:
                        p_value = float("nan")
            rows.append({
                "dataset": dataset,
                "statistic": column,
                "pair_n": int(len(values)),
                "median": float(values.median()) if len(values) else float("nan"),
                "q25": float(values.quantile(0.25)) if len(values) else float("nan"),
                "q75": float(values.quantile(0.75)) if len(values) else float("nan"),
                "null_value": null,
                "fraction_above_null": fraction_above,
                "signed_rank_p_vs_null": p_value,
            })
    order = {name: index for index, name in enumerate(SUMMARY_EXTRA)}
    frame = pd.DataFrame(rows)
    frame["sort_key"] = frame["statistic"].map(
        lambda name: order.get(name, len(order))
    )
    return frame.sort_values(
        ["dataset", "sort_key", "statistic"], kind="stable"
    ).drop(columns="sort_key")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--dataset", type=dataset_argument, action="append", required=True,
        metavar="LABEL=OT_ROOT",
        help="Dataset label and the OT root holding <pair_id>/<scope>/<budget_tag>",
    )
    parser.add_argument("--scope", default="scope_malignant")
    parser.add_argument(
        "--budget-tag", default=None,
        help="Exact budget directory name; omit to accept every completed tag",
    )
    parser.add_argument("--method", default="M4-E")
    parser.add_argument("--side", default="source", choices=("source", "target"))
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    records = []
    inventory = []
    for dataset, root in args.dataset:
        paths = confidence_paths(root, args.scope, args.budget_tag)
        inventory.append({
            "dataset": dataset, "ot_root": str(root), "confidence_file_n": len(paths),
        })
        if not paths:
            raise RuntimeError(
                f"{dataset}: no cell_confidence.csv under {root} with "
                f"scope={args.scope} budget_tag={args.budget_tag}"
            )
        for path in paths:
            record = pair_record(path, dataset, args.method, args.side, args.scope)
            if record is not None:
                records.append(record)
    if not records:
        raise RuntimeError(f"No rows matched method={args.method} side={args.side}")

    pairs = pd.DataFrame(records).sort_values(["dataset", "pair_id"], kind="stable")
    # Statistics whose source column was absent are dropped from the summary,
    # so report availability explicitly rather than letting a row vanish.
    availability = {
        column: sorted(
            pairs.loc[pairs[f"auc_{column}"].notna(), "dataset"].unique().tolist()
        )
        for column in AUC_TARGETS
        if f"auc_{column}" in pairs
    }
    duplicated = pairs.duplicated(["dataset", "pair_id"], keep=False)
    if duplicated.any():
        # Several budget tags completed for one pair; a summary median over
        # them would silently mix caps.
        raise RuntimeError(
            "Multiple budget tags matched for these pairs; pass --budget-tag: "
            + ", ".join(
                sorted(pairs.loc[duplicated, "pair_id"].astype(str).unique())[:10]
            )
        )
    summary = summarize(pairs)
    pairs.to_csv(args.output_root / "gate_covariate_pair_statistics.csv", index=False)
    summary.to_csv(args.output_root / "gate_covariate_dataset_summary.csv", index=False)
    report = {
        "method": args.method,
        "side": args.side,
        "scope": args.scope,
        "budget_tag": args.budget_tag or "any_completed",
        "inventory": inventory,
        "pair_n": int(len(pairs)),
        "auc_positive_class": "retained",
        "datasets_with_usable_column": availability,
        "interpretation": {
            "sign_rule_retained_fraction": (
                "Fraction retained by the calibrated rule decision_cost < "
                "rejection_cost. Far below retained_fraction means the "
                "cardinality floor, not the rejection cost, built the gate."
            ),
            "sign_rule_concordance": (
                "Agreement between the returned gate and that calibrated rule. "
                "1.0 means the gate is exactly the rule."
            ),
            "forced_in_share_of_retained": (
                "Share of retained cells the raw gate wanted to reject."
            ),
            "terminal_disagreement_fraction": (
                "Cells the raw gate retained but the returned gate rejected. "
                "Expected near 0; a larger value means the terminal re-solve "
                "disagrees with the gate it returned."
            ),
            "auc_decision_cost": (
                "Near 0 means retained cells carry the lowest decision cost, as "
                "intended. Near 0.5 means the gate is not ordered by cost."
            ),
            "auc_total_counts": (
                "Departure from 0.5 means the gate tracks sequencing depth."
            ),
            "spearman_decision_cost_total_counts": (
                "Rank correlation between transport cost and depth. A strong "
                "negative value means the cost geometry itself reads depth, "
                "upstream of the gate."
            ),
            "depth_residual_gate_jaccard": (
                "Overlap between the gate and a same-size gate built on "
                "depth-residual cost. Compare against chance_gate_jaccard: a "
                "value near chance means depth decides which cells are kept."
            ),
        },
    }
    (args.output_root / "gate_covariate_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
