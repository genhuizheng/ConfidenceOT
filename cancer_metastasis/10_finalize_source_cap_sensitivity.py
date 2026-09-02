"""Create a biological and numerical report for asymmetric source-cap sensitivity.

The first run passed to ``09_compare_asymmetric_caps.py`` is treated as the
baseline.  This script never selects a cap merely because it lowers rejection;
it quantifies target-cell turnover and origin-ranking stability at several
anatomical resolutions.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GROUP = ["dataset_id", "patient_id", "target_sample", "analysis_scope"]


def safe_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def normalized_site(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return re.sub(r"_+", "_", text)


def collapse_laterality(value: object) -> str:
    tokens = [
        token for token in normalized_site(value).split("_")
        if token not in {"left", "right", "bilateral"}
    ]
    return "_".join(tokens)


def anatomical_compartment(value: object) -> str:
    """Conservative broad class used only as a sensitivity interpretation."""
    site = collapse_laterality(value)
    if (
        "adnexa" in site
        or "ovary" in site
        or "ovarian" in site
        or "fallopian" in site
    ):
        return "tubo_ovarian_compartment"
    return site


def rank_candidates(metrics: pd.DataFrame) -> pd.DataFrame:
    candidate_n = metrics.groupby(["run_label", *GROUP])["source_sample"].transform(
        "nunique"
    )
    ranked = metrics[candidate_n >= 2].copy()
    ranked = ranked.sort_values(
        [
            "run_label", *GROUP, "target_final_rejection_rate",
            "target_mean_rejection_score", "source_final_rejection_rate",
            "transported_mass", "source_sample",
        ],
        ascending=[True] * (len(GROUP) + 1) + [True, True, True, False, True],
        kind="stable",
    )
    ranked["primary_rank"] = ranked.groupby(["run_label", *GROUP]).cumcount() + 1
    return ranked


def winner_table(ranked: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "run_label", *GROUP, "source_sample", "pair_id",
        "target_final_rejection_rate", "target_mean_rejection_score",
        "source_final_rejection_rate", "transported_mass",
    ]
    top = ranked[ranked["primary_rank"].eq(1)][columns].copy()
    second = ranked[ranked["primary_rank"].eq(2)][
        ["run_label", *GROUP, "source_sample", "target_final_rejection_rate",
         "target_mean_rejection_score"]
    ].rename(columns={
        "source_sample": "second_source_sample",
        "target_final_rejection_rate": "second_target_rejection_rate",
        "target_mean_rejection_score": "second_target_rejection_score",
    })
    top = top.merge(second, on=["run_label", *GROUP], how="left", validate="one_to_one")
    top["margin_to_second_target_rejection_rate"] = (
        top["second_target_rejection_rate"] - top["target_final_rejection_rate"]
    )
    top["margin_to_second_target_rejection_score"] = (
        top["second_target_rejection_score"] - top["target_mean_rejection_score"]
    )
    top["winner_laterality_collapsed"] = top["source_sample"].map(collapse_laterality)
    top["winner_anatomical_compartment"] = top["source_sample"].map(
        anatomical_compartment
    )
    return top


def winner_stability(winners: pd.DataFrame, baseline_label: str) -> pd.DataFrame:
    baseline = winners[winners["run_label"].eq(baseline_label)][
        [*GROUP, "source_sample", "winner_laterality_collapsed",
         "winner_anatomical_compartment", "target_final_rejection_rate",
         "target_mean_rejection_score"]
    ].rename(columns={
        "source_sample": "baseline_source_sample",
        "winner_laterality_collapsed": "baseline_laterality_collapsed",
        "winner_anatomical_compartment": "baseline_anatomical_compartment",
        "target_final_rejection_rate": "baseline_target_rejection_rate",
        "target_mean_rejection_score": "baseline_target_rejection_score",
    })
    result = winners.merge(baseline, on=GROUP, how="left", validate="many_to_one")
    result["exact_winner_matches_baseline"] = (
        result["source_sample"] == result["baseline_source_sample"]
    )
    result["laterality_collapsed_matches_baseline"] = (
        result["winner_laterality_collapsed"]
        == result["baseline_laterality_collapsed"]
    )
    result["anatomical_compartment_matches_baseline"] = (
        result["winner_anatomical_compartment"]
        == result["baseline_anatomical_compartment"]
    )
    result["target_rejection_delta_from_baseline"] = (
        result["target_final_rejection_rate"]
        - result["baseline_target_rejection_rate"]
    )
    result["target_score_delta_from_baseline"] = (
        result["target_mean_rejection_score"]
        - result["baseline_target_rejection_score"]
    )
    return result


def origin_group_robustness(
    stability: pd.DataFrame, baseline_label: str, recommended_label: str
) -> pd.DataFrame:
    base = stability[stability["run_label"].eq(baseline_label)][
        [*GROUP, "source_sample", "winner_laterality_collapsed",
         "winner_anatomical_compartment",
         "margin_to_second_target_rejection_rate",
         "margin_to_second_target_rejection_score"]
    ].rename(columns={
        "source_sample": "baseline_winner",
        "winner_laterality_collapsed": "baseline_laterality_collapsed",
        "winner_anatomical_compartment": "baseline_anatomical_compartment",
        "margin_to_second_target_rejection_rate": "baseline_margin_rate",
        "margin_to_second_target_rejection_score": "baseline_margin_score",
    })
    result = base.copy()
    alternatives = [label for label in stability["run_label"].unique() if label != baseline_label]
    for label in alternatives:
        tag = safe_label(label)
        use = stability[stability["run_label"].eq(label)][
            [*GROUP, "source_sample", "exact_winner_matches_baseline",
             "laterality_collapsed_matches_baseline",
             "anatomical_compartment_matches_baseline",
             "target_rejection_delta_from_baseline"]
        ].rename(columns={
            "source_sample": f"{tag}_winner",
            "exact_winner_matches_baseline": f"{tag}_exact_match",
            "laterality_collapsed_matches_baseline": f"{tag}_laterality_match",
            "anatomical_compartment_matches_baseline": f"{tag}_compartment_match",
            "target_rejection_delta_from_baseline": f"{tag}_target_rejection_delta",
        })
        result = result.merge(use, on=GROUP, how="left", validate="one_to_one")

    rec = safe_label(recommended_label)
    result["recommended_range_exact_robust"] = result[f"{rec}_exact_match"]
    result["recommended_range_laterality_robust"] = result[f"{rec}_laterality_match"]
    result["recommended_range_compartment_robust"] = result[f"{rec}_compartment_match"]
    exact_columns = [f"{safe_label(label)}_exact_match" for label in alternatives]
    laterality_columns = [f"{safe_label(label)}_laterality_match" for label in alternatives]
    compartment_columns = [f"{safe_label(label)}_compartment_match" for label in alternatives]
    result["all_caps_exact_robust"] = result[exact_columns].fillna(False).all(axis=1)
    result["all_caps_laterality_robust"] = result[laterality_columns].fillna(False).all(axis=1)
    result["all_caps_compartment_robust"] = result[compartment_columns].fillna(False).all(axis=1)
    result["interpretation"] = np.select(
        [
            result["recommended_range_exact_robust"],
            result["recommended_range_laterality_robust"],
            result["recommended_range_compartment_robust"],
        ],
        [
            "exact_sampling_site_robust",
            "laterality_ambiguous_but_site_robust",
            "fine_site_ambiguous_but_broad_compartment_robust",
        ],
        default="anatomical_compartment_sensitive",
    )
    return result


def pair_deltas(metrics: pd.DataFrame, baseline_label: str) -> pd.DataFrame:
    value_columns = [
        "source_final_rejection_rate", "target_final_rejection_rate",
        "source_mean_rejection_score", "target_mean_rejection_score",
        "transported_mass", "rejection_cost",
    ]
    value_columns = [column for column in value_columns if column in metrics]
    keys = ["pair_id", *GROUP, "source_sample"]
    baseline = metrics[metrics["run_label"].eq(baseline_label)][keys + value_columns]
    baseline = baseline.rename(
        columns={column: f"baseline_{column}" for column in value_columns}
    )
    compared = metrics.merge(baseline, on=keys, how="left", validate="many_to_one")
    for column in value_columns:
        compared[f"delta_{column}"] = (
            compared[column] - compared[f"baseline_{column}"]
        )
        compared[f"absolute_delta_{column}"] = compared[f"delta_{column}"].abs()
    return compared


def weighted_gate_summary(gates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (label, side), table in gates.groupby(["run_label", "side"], sort=False):
        weights = table["cell_n"].to_numpy()
        row = {"run_label": label, "side": side, "cell_occurrence_n": int(weights.sum())}
        for column in (
            "gate_changed_fraction", "newly_retained_fraction",
            "newly_rejected_fraction", "mean_score_delta",
        ):
            row[column] = float(np.average(table[column], weights=weights))
        row["median_pair_gate_changed_fraction"] = float(table["gate_changed_fraction"].median())
        row["maximum_pair_gate_changed_fraction"] = float(table["gate_changed_fraction"].max())
        rows.append(row)
    return pd.DataFrame(rows)


def origin_stability_summary(
    stability: pd.DataFrame, baseline_label: str
) -> pd.DataFrame:
    rows = []
    for label, table in stability.groupby("run_label", sort=False):
        rows.append({
            "run_label": label,
            "group_n": len(table),
            "exact_winner_match_rate": float(
                table["exact_winner_matches_baseline"].mean()
            ),
            "laterality_collapsed_match_rate": float(
                table["laterality_collapsed_matches_baseline"].mean()
            ),
            "anatomical_compartment_match_rate": float(
                table["anatomical_compartment_matches_baseline"].mean()
            ),
            "mean_target_rejection_delta": float(
                table["target_rejection_delta_from_baseline"].mean()
            ),
            "mean_absolute_target_rejection_delta": float(
                table["target_rejection_delta_from_baseline"].abs().mean()
            ),
            "is_baseline": label == baseline_label,
        })
    return pd.DataFrame(rows)


def pair_delta_summary(delta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, table in delta.groupby("run_label", sort=False):
        target = table["delta_target_final_rejection_rate"].abs()
        source = table["delta_source_final_rejection_rate"].abs()
        rows.append({
            "run_label": label,
            "pair_n": int(table["pair_id"].nunique()),
            "mean_absolute_source_rejection_delta": float(source.mean()),
            "p90_absolute_source_rejection_delta": float(source.quantile(0.90)),
            "maximum_absolute_source_rejection_delta": float(source.max()),
            "mean_absolute_target_rejection_delta": float(target.mean()),
            "p90_absolute_target_rejection_delta": float(target.quantile(0.90)),
            "maximum_absolute_target_rejection_delta": float(target.max()),
            "target_pair_fraction_changed_over_5pp": float((target > 0.05).mean()),
            "target_pair_fraction_changed_over_10pp": float((target > 0.10).mean()),
        })
    return pd.DataFrame(rows)


def winner_margin_summary(winners: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, table in winners.groupby("run_label", sort=False):
        rate_margin = table["margin_to_second_target_rejection_rate"]
        score_margin = table["margin_to_second_target_rejection_score"]
        rows.append({
            "run_label": label,
            "group_n": len(table),
            "median_rate_margin": float(rate_margin.median()),
            "median_score_margin": float(score_margin.median()),
            "rate_margin_at_most_2pp_fraction": float((rate_margin <= 0.02).mean()),
            "rate_margin_at_most_5pp_fraction": float((rate_margin <= 0.05).mean()),
            "exact_rate_and_score_tie_fraction": float(
                ((rate_margin == 0) & (score_margin == 0)).mean()
            ),
        })
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_figures(
    summary: pd.DataFrame,
    gate_summary: pd.DataFrame,
    stability: pd.DataFrame,
    pair_delta: pd.DataFrame,
    baseline_label: str,
    output_dir: Path,
) -> None:
    labels = summary["run_label"].tolist()
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.bar(x - width / 2, 100 * summary["source_weighted_rejection_rate"], width, label="Source")
    ax.bar(x + width / 2, 100 * summary["target_weighted_rejection_rate"], width, label="Target")
    ax.set(xticks=x, xticklabels=labels, ylabel="Weighted rejected cells (%)")
    ax.set_title("Observed rejection under asymmetric source caps")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, output_dir, "01_overall_rejection_by_cap")

    gate_plot = gate_summary.copy()
    gate_plot["key"] = gate_plot["run_label"] + "\n" + gate_plot["side"].str.title()
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.bar(gate_plot["key"], 100 * gate_plot["gate_changed_fraction"], color="#5b4b8a")
    ax.set(ylabel="Cells with changed binary gate (%)", title="Cell-level gate turnover from baseline")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, output_dir, "02_cell_gate_turnover")

    alt = stability[stability["run_label"].ne(baseline_label)]
    stability_summary = alt.groupby("run_label", sort=False).agg(
        exact=("exact_winner_matches_baseline", "mean"),
        laterality_collapsed=("laterality_collapsed_matches_baseline", "mean"),
        anatomical_compartment=("anatomical_compartment_matches_baseline", "mean"),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    width = 0.24
    sx = np.arange(len(stability_summary))
    for index, (column, label) in enumerate((
        ("exact", "Exact sample"),
        ("laterality_collapsed", "Left/right collapsed"),
        ("anatomical_compartment", "Broad compartment"),
    )):
        ax.bar(sx + (index - 1) * width, 100 * stability_summary[column], width, label=label)
    ax.set(
        xticks=sx, xticklabels=stability_summary["run_label"], ylim=(0, 105),
        ylabel="Origin winner agreement with baseline (%)",
        title="Origin stability depends on biological resolution",
    )
    ax.legend(frameon=False, loc="lower left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, output_dir, "03_origin_stability_by_resolution")

    alt_delta = pair_delta[pair_delta["run_label"].ne(baseline_label)]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    values = [
        alt_delta.loc[alt_delta["run_label"].eq(label), "delta_target_final_rejection_rate"] * 100
        for label in alt_delta["run_label"].unique()
    ]
    boxplot_labels = alt_delta["run_label"].unique()
    try:
        # Matplotlib >=3.9 renamed ``labels`` to ``tick_labels``.
        ax.boxplot(values, tick_labels=boxplot_labels, showfliers=False)
    except TypeError:
        # Retain compatibility with older TACC environments.
        ax.boxplot(values, labels=boxplot_labels, showfliers=False)
    ax.axhline(0, color="0.3", linewidth=1)
    ax.set(
        ylabel="Pair-level target rejection change (percentage points)",
        title="Target effects despite a fixed target cap",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, output_dir, "04_target_pair_rejection_deltas")


def markdown_report(
    summary: pd.DataFrame,
    gate_summary: pd.DataFrame,
    stability: pd.DataFrame,
    robustness: pd.DataFrame,
    baseline_label: str,
    recommended_label: str,
) -> str:
    base = summary.set_index("run_label").loc[baseline_label]
    rec = summary.set_index("run_label").loc[recommended_label]
    rec_stability = stability[stability["run_label"].eq(recommended_label)]
    rec_gates = gate_summary[gate_summary["run_label"].eq(recommended_label)].set_index("side")
    exact_n = int(rec_stability["exact_winner_matches_baseline"].sum())
    laterality_n = int(rec_stability["laterality_collapsed_matches_baseline"].sum())
    compartment_n = int(rec_stability["anatomical_compartment_matches_baseline"].sum())
    group_n = len(rec_stability)
    interpretation_counts = robustness["interpretation"].value_counts().to_dict()
    return f"""# GSE180661 asymmetric source-cap sensitivity

## Scope

- Malignant-cell-only ConfidenceOT M4-E analysis.
- {int(base['pair_n'])} exact primary-to-metastasis pairs.
- {group_n} metastatic targets with at least two candidate primary samples.
- Baseline: `{baseline_label}`; recommended sensitivity setting: `{recommended_label}`.
- The target cap is held at {rec['target_rejection_budget_cap']:.2f}, but target gates are re-estimated jointly and are not frozen.

## Main numerical findings

- Source weighted rejection changes from {100 * base['source_weighted_rejection_rate']:.2f}% to {100 * rec['source_weighted_rejection_rate']:.2f}%.
- Target weighted rejection changes from {100 * base['target_weighted_rejection_rate']:.2f}% to {100 * rec['target_weighted_rejection_rate']:.2f}%.
- At cell-occurrence level, {100 * rec_gates.loc['source', 'gate_changed_fraction']:.2f}% of source gates and {100 * rec_gates.loc['target', 'gate_changed_fraction']:.2f}% of target gates change.
- Exact winner stability is {exact_n}/{group_n} ({100 * exact_n / group_n:.2f}%).
- Laterality-collapsed winner stability is {laterality_n}/{group_n} ({100 * laterality_n / group_n:.2f}%).
- Broad anatomical-compartment stability is {compartment_n}/{group_n} ({100 * compartment_n / group_n:.2f}%).

## Biological interpretation

- Recommended-range classifications: `{json.dumps(interpretation_counts, sort_keys=True)}`.
- Exact-site changes that disappear after removing left/right labels represent laterality ambiguity, not evidence for a different organ of origin.
- Absolute rejection rates must not be interpreted as a known fraction of non-lineage cells: the cap is a safety/coverage constraint, not biological ground truth.
- A stable winner indicates expression-state compatibility across cap settings. It does not establish clonal ancestry without orthogonal CNV, mutation, or lineage evidence.

## Decision

- Preserve the baseline as the primary prespecified analysis.
- Use `{recommended_label}` as the main sensitivity analysis.
- Report both exact-site and laterality-collapsed origin stability.
- Treat broader cap changes as a stress-test boundary rather than as a replacement primary setting.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("comparison_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--recommended-label", default="source090")
    args = parser.parse_args()

    report_path = args.comparison_dir / "asymmetric_cap_report.json"
    with report_path.open(encoding="utf-8") as handle:
        comparison_report = json.load(handle)
    baseline_label = comparison_report["baseline_label"]

    summary = pd.read_csv(args.comparison_dir / "asymmetric_cap_summary.csv")
    metrics = pd.read_csv(args.comparison_dir / "asymmetric_cap_pair_metrics.csv")
    gates = pd.read_csv(args.comparison_dir / "asymmetric_cap_cell_gate_stability.csv")
    if args.recommended_label not in set(summary["run_label"]):
        raise ValueError(f"Unknown recommended label: {args.recommended_label}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranked = rank_candidates(metrics)
    winners = winner_table(ranked)
    stability = winner_stability(winners, baseline_label)
    robustness = origin_group_robustness(
        stability, baseline_label, args.recommended_label
    )
    delta = pair_deltas(metrics, baseline_label)
    gate_summary = weighted_gate_summary(gates)
    stability_summary = origin_stability_summary(stability, baseline_label)
    delta_summary = pair_delta_summary(delta)
    margin_summary = winner_margin_summary(winners)

    summary.to_csv(args.output_dir / "overall_cap_summary.csv", index=False)
    gate_summary.to_csv(args.output_dir / "cell_gate_turnover_summary.csv", index=False)
    stability_summary.to_csv(
        args.output_dir / "origin_stability_summary.csv", index=False
    )
    delta_summary.to_csv(args.output_dir / "pair_level_delta_summary.csv", index=False)
    margin_summary.to_csv(args.output_dir / "winner_margin_summary.csv", index=False)
    ranked.to_csv(args.output_dir / "origin_candidate_rankings_all_caps.csv", index=False)
    stability.to_csv(args.output_dir / "origin_winner_stability_long.csv", index=False)
    robustness.to_csv(args.output_dir / "origin_group_robustness.csv", index=False)
    delta.to_csv(args.output_dir / "pair_level_cap_deltas.csv", index=False)
    gates.sort_values(
        ["gate_changed_fraction", "run_label", "side"],
        ascending=[False, True, True],
    ).to_csv(args.output_dir / "most_sensitive_cell_gate_pairs.csv", index=False)
    delta[delta["run_label"].ne(baseline_label)].sort_values(
        ["absolute_delta_target_final_rejection_rate", "run_label"],
        ascending=[False, True],
    ).to_csv(args.output_dir / "most_sensitive_target_metric_pairs.csv", index=False)
    stability[
        stability["run_label"].ne(baseline_label)
        & ~stability["exact_winner_matches_baseline"]
    ].to_csv(args.output_dir / "changed_exact_origin_winners.csv", index=False)
    stability[
        stability["run_label"].ne(baseline_label)
        & ~stability["anatomical_compartment_matches_baseline"]
    ].to_csv(args.output_dir / "changed_anatomical_compartment_winners.csv", index=False)

    make_figures(
        summary, gate_summary, stability, delta, baseline_label, args.output_dir
    )
    report = markdown_report(
        summary, gate_summary, stability, robustness,
        baseline_label, args.recommended_label,
    )
    (args.output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Comprehensive sensitivity analysis: {args.output_dir}")


if __name__ == "__main__":
    main()
