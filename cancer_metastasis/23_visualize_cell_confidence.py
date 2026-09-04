"""Visualize cell-level ConfidenceOT scores without requiring an expression embedding."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STATUS_ORDER = ["retained", "rejected", "site_or_cap_discordant"]
STATUS_LABEL = {
    "retained": "Retained",
    "rejected": "Rejected",
    "site_or_cap_discordant": "Discordant",
}
STATUS_COLOR = {
    "retained": "#2b8cbe",
    "rejected": "#d7301f",
    "site_or_cap_discordant": "#969696",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cell_confidence_root", type=Path)
    parser.add_argument("four_state_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--method", default="M4-E")
    parser.add_argument("--annotation", default="Ovarian.cancer.cell")
    return parser.parse_args()


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def read_pair_scores(root: Path, method: str, annotation: str) -> pd.DataFrame:
    keep = [
        "method", "side", "observation_id", "sample_id", "annotation",
        "rejected", "raw_rejected", "normalized_rejection_score", "budget_overridden",
    ]
    rows = []
    for path in sorted(root.glob("**/cell_confidence.csv")):
        table = pd.read_csv(path, usecols=keep)
        table = table.loc[
            table["method"].eq(method) & table["annotation"].eq(annotation)
        ].copy()
        if not table.empty:
            table["pair_id"] = path.parents[2].name
            rows.append(table)
    if not rows:
        raise FileNotFoundError("No matching cell-confidence rows were found")
    return pd.concat(rows, ignore_index=True)


def read_consensus(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("patients/*/four_state_cell_classification.csv.gz")):
        table = pd.read_csv(path)
        table["patient_id"] = path.parent.name.split("_", 1)[-1]
        rows.append(table)
    if not rows:
        raise FileNotFoundError("No four-state cell classification files were found")
    return pd.concat(rows, ignore_index=True)


def summarize_cells(scores: pd.DataFrame, consensus: pd.DataFrame) -> pd.DataFrame:
    side_map = {"source": "primary", "target": "metastasis"}
    scores = scores.assign(side=scores["side"].map(side_map))
    summary = scores.groupby(["side", "observation_id"], as_index=False).agg(
        pair_n=("pair_id", "nunique"),
        mean_pair_rejection_score=("normalized_rejection_score", "mean"),
        sd_pair_rejection_score=("normalized_rejection_score", "std"),
        minimum_pair_rejection_score=("normalized_rejection_score", "min"),
        maximum_pair_rejection_score=("normalized_rejection_score", "max"),
        final_rejection_frequency=("rejected", "mean"),
        raw_rejection_frequency=("raw_rejected", "mean"),
        budget_override_frequency=("budget_overridden", "mean"),
    )
    summary["sd_pair_rejection_score"] = summary["sd_pair_rejection_score"].fillna(0)
    columns = [
        "side", "observation_id", "sample", "patient_id", "consensus_status",
        "observed_pair_n", "expected_pair_n", "mean_baseline_rejection_score",
    ]
    return consensus[columns].merge(
        summary, on=["side", "observation_id"], how="left", validate="one_to_one"
    )


def distribution_axis(ax, cells: pd.DataFrame, side: str) -> None:
    subset = cells.loc[cells["side"].eq(side)]
    values = [
        subset.loc[subset["consensus_status"].eq(status), "mean_baseline_rejection_score"]
        .dropna().to_numpy()
        for status in STATUS_ORDER
    ]
    violin = ax.violinplot(values, showmedians=True, showextrema=False, widths=0.8)
    for body, status in zip(violin["bodies"], STATUS_ORDER):
        body.set_facecolor(STATUS_COLOR[status])
        body.set_edgecolor("none")
        body.set_alpha(0.72)
    violin["cmedians"].set_color("black")
    violin["cmedians"].set_linewidth(1.1)
    for index, values_for_status in enumerate(values, 1):
        if len(values_for_status):
            ax.text(index, 1.015, f"n={len(values_for_status):,}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0.5, color="#555555", linestyle="--", linewidth=0.9)
    ax.set_xticks(range(1, 4), [STATUS_LABEL[value] for value in STATUS_ORDER])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Mean cell-level rejection score")
    ax.set_title(side.capitalize())
    ax.grid(axis="y", alpha=0.18)


def distribution_figures(cells: pd.DataFrame, output: Path) -> None:
    for side in ("primary", "metastasis"):
        fig, ax = plt.subplots(figsize=(6.0, 4.8))
        distribution_axis(ax, cells, side)
        fig.tight_layout()
        save(fig, output, f"07_{side}_cell_confidence_distribution")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, side in zip(axes, ("primary", "metastasis")):
        distribution_axis(ax, cells, side)
    fig.suptitle("Cell-level ConfidenceOT score by consensus state", fontsize=15)
    fig.tight_layout()
    save(fig, output, "07_cell_confidence_distribution_combined")


def landscape_axis(ax, cells: pd.DataFrame, side: str) -> None:
    subset = cells.loc[cells["side"].eq(side)].copy()
    rng = np.random.default_rng(20260904)
    if len(subset) > 60000:
        subset = subset.iloc[rng.choice(len(subset), 60000, replace=False)]
    for status in STATUS_ORDER:
        use = subset.loc[subset["consensus_status"].eq(status)]
        ax.scatter(
            use["mean_pair_rejection_score"], use["sd_pair_rejection_score"],
            s=5, alpha=0.22, linewidth=0, color=STATUS_COLOR[status],
            label=STATUS_LABEL[status], rasterized=True,
        )
    ax.axvline(0.5, color="#555555", linestyle="--", linewidth=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Mean pair-specific rejection score")
    ax.set_ylabel("Across-pair score SD")
    ax.set_title(side.capitalize())
    ax.grid(alpha=0.15)
    ax.legend(frameon=False, markerscale=2.5, fontsize=8)


def landscape_figures(cells: pd.DataFrame, output: Path) -> None:
    for side in ("primary", "metastasis"):
        fig, ax = plt.subplots(figsize=(6.0, 4.8))
        landscape_axis(ax, cells, side)
        fig.tight_layout()
        save(fig, output, f"08_{side}_cell_confidence_landscape")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, side in zip(axes, ("primary", "metastasis")):
        landscape_axis(ax, cells, side)
    fig.suptitle("Confidence strength and cross-pair stability", fontsize=15)
    fig.tight_layout()
    save(fig, output, "08_cell_confidence_landscape_combined")


def patient_heatmap(cells: pd.DataFrame, output: Path) -> None:
    table = cells.copy()
    table["state"] = table["side"] + "_" + table["consensus_status"]
    states = [f"{side}_{status}" for side in ("primary", "metastasis") for status in STATUS_ORDER]
    matrix = table.pivot_table(
        index="patient_id", columns="state", values="mean_baseline_rejection_score", aggfunc="median"
    ).reindex(columns=states)
    matrix.to_csv(output / "patient_median_cell_confidence.csv")
    fig, ax = plt.subplots(figsize=(9.2, max(6.0, 0.25 * len(matrix))))
    image = ax.imshow(matrix.to_numpy(), cmap="magma", vmin=0, vmax=1, aspect="auto")
    labels = [value.replace("_", " ").title() for value in states]
    ax.set_xticks(range(len(states)), labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(matrix)), matrix.index, fontsize=7)
    ax.set_title("Patient-level median ConfidenceOT rejection score")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("Median cell-level rejection score")
    fig.tight_layout()
    save(fig, output, "09_patient_median_cell_confidence_heatmap")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False})
    scores = read_pair_scores(args.cell_confidence_root, args.method, args.annotation)
    consensus = read_consensus(args.four_state_root)
    cells = summarize_cells(scores, consensus)
    cells.to_csv(args.output_root / "cell_confidence_summary.csv.gz", index=False)
    distribution_figures(cells, args.output_root)
    landscape_figures(cells, args.output_root)
    patient_heatmap(cells, args.output_root)
    print(f"Cell confidence rows: {len(cells):,}")
    print(f"Figures: {args.output_root}")


if __name__ == "__main__":
    main()
