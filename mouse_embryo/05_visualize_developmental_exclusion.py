"""Visualize developmental exclusion and Traditional-OT forced matches.

The primary biological object is the calibrated M4-R gate.  Balanced OT is
used as a counterfactual: where would it force ConfidenceOT-rejected spatial
bins to transport if rejection were forbidden?
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_CONFIDENCE_METHOD = "Calibrated | M4-R / UOT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared_pair", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--confidence-method", default=DEFAULT_CONFIDENCE_METHOD)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def method_file(root: Path, method: str) -> Path:
    safe = method.lower().replace(" ", "_").replace("/", "_").replace("|", "_")
    return root / "methods" / f"{safe}.npz"


def grouped_flow(
    coupling: np.ndarray,
    row_labels: np.ndarray,
    column_labels: np.ndarray,
    row_categories: list[str],
    column_categories: list[str],
) -> np.ndarray:
    matrix = np.zeros((len(row_categories), len(column_categories)), dtype=float)
    for row, row_label in enumerate(row_categories):
        row_mask = row_labels == row_label
        for column, column_label in enumerate(column_categories):
            matrix[row, column] = coupling[np.ix_(row_mask, column_labels == column_label)].sum()
    totals = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0)


def draw_heatmap(
    axis: plt.Axes,
    matrix: np.ndarray,
    rows: list[str],
    columns: list[str],
    title: str,
) -> plt.AxesImage:
    image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    axis.set_yticks(range(len(rows)), rows, fontsize=8)
    axis.set_xticks(range(len(columns)), columns, rotation=55, ha="right", fontsize=8)
    axis.set_title(title, fontsize=13, pad=10)
    return image


def plot_mirror_rejection(
    population: pd.DataFrame,
    method: str,
    destination: Path,
    dpi: int,
) -> None:
    selected = population[population["method"].eq(method)]
    source = selected[selected.side.eq("source")].set_index("annotation")
    target = selected[selected.side.eq("target")].set_index("annotation")
    annotations = sorted(
        set(source.index).union(target.index),
        key=lambda value: max(
            float(source.loc[value, "mean_rejection_signal"]) if value in source.index else -1,
            float(target.loc[value, "mean_rejection_signal"]) if value in target.index else -1,
        ),
    )
    y = np.arange(len(annotations))
    source_values = np.asarray([
        float(source.loc[value, "mean_rejection_signal"]) if value in source.index else np.nan
        for value in annotations
    ])
    target_values = np.asarray([
        float(target.loc[value, "mean_rejection_signal"]) if value in target.index else np.nan
        for value in annotations
    ])
    figure, axis = plt.subplots(figsize=(12, max(7, 0.42 * len(annotations) + 2)))
    axis.barh(y, -np.nan_to_num(source_values), color="#c43c39", label="Source: candidate disappearance")
    axis.barh(y, np.nan_to_num(target_values), color="#3478b8", label="Target: candidate emergence")
    for row, value in enumerate(source_values):
        if np.isfinite(value) and value > 0:
            axis.text(-value - 0.015, row, f"{value:.0%}", ha="right", va="center", fontsize=8)
    for row, value in enumerate(target_values):
        if np.isfinite(value) and value > 0:
            axis.text(value + 0.015, row, f"{value:.0%}", ha="left", va="center", fontsize=8)
    axis.axvline(0, color="#555555", linewidth=0.8)
    axis.set_yticks(y, annotations)
    axis.set_xlim(-1.02, 1.02)
    axis.set_xticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1],
                    ["100%", "75%", "50%", "25%", "0", "25%", "50%", "75%", "100%"])
    axis.set_xlabel("Fraction of spatial bins rejected by calibrated M4-R")
    axis.set_title("Candidate disappearance and emergence by annotated population", fontsize=15)
    axis.legend(loc="lower right", frameon=False)
    axis.spines[["top", "right", "left"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_spatial_candidates(
    prepared: np.lib.npyio.NpzFile,
    source_rejected: np.ndarray,
    target_rejected: np.ndarray,
    destination: Path,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 6))
    panels = (
        (axes[0], prepared["source_spatial"], prepared["source_background_spatial"],
         source_rejected, "Source: candidate disappearance", "#c43c39"),
        (axes[1], prepared["target_spatial"], prepared["target_background_spatial"],
         target_rejected, "Target: candidate emergence", "#3478b8"),
    )
    for axis, coordinates, background, rejected, title, color in panels:
        axis.scatter(background[:, 0], background[:, 1], s=5, c="#d0d0d0", alpha=0.55,
                     linewidths=0, label="Cavity (pre-analysis exclusion)", rasterized=True)
        axis.scatter(coordinates[~rejected, 0], coordinates[~rejected, 1], s=7, c="#a9a9a9",
                     alpha=0.45, linewidths=0, label="Retained", rasterized=True)
        axis.scatter(coordinates[rejected, 0], coordinates[rejected, 1], s=13, c=color,
                     alpha=0.9, linewidths=0, label="ConfidenceOT rejected", rasterized=True)
        axis.set_title(title, fontsize=14)
        axis.set_xlabel("Spatial x")
        axis.set_ylabel("Spatial y")
        axis.set_aspect("equal", adjustable="datalim")
        axis.invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    figure.suptitle("Expression-only ConfidenceOT projected back to embryo space", fontsize=16)
    figure.subplots_adjust(bottom=0.16, top=0.88, wspace=0.20)
    figure.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_transition_comparison(
    balanced: np.ndarray,
    retained: np.ndarray,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
    destination: Path,
    dpi: int,
) -> None:
    source_categories = sorted(np.unique(source_labels))
    target_categories = sorted(np.unique(target_labels))
    traditional = grouped_flow(
        balanced, source_labels, target_labels, source_categories, target_categories
    )
    confidence = grouped_flow(
        retained, source_labels, target_labels, source_categories, target_categories
    )
    figure, axes = plt.subplots(1, 2, figsize=(18, 7.5), sharey=True)
    image = draw_heatmap(
        axes[0], traditional, source_categories, target_categories,
        "Traditional OT: all bins are forced to transition",
    )
    draw_heatmap(
        axes[1], confidence, source_categories, target_categories,
        "ConfidenceOT: retained-to-retained transitions only",
    )
    axes[0].set_ylabel("Source annotation")
    for axis in axes:
        axis.set_xlabel("Target annotation")
    figure.colorbar(image, ax=axes, label="Source-conditional transport probability",
                    fraction=0.025, pad=0.02)
    figure.subplots_adjust(left=0.11, right=0.92, bottom=0.24, top=0.88, wspace=0.16)
    figure.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_forced_matches(
    balanced: np.ndarray,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
    source_rejected: np.ndarray,
    target_rejected: np.ndarray,
    destination: Path,
    table_destination: Path,
    dpi: int,
) -> None:
    rejected_source_categories = sorted(np.unique(source_labels[source_rejected]))
    rejected_target_categories = sorted(np.unique(target_labels[target_rejected]))
    source_categories = sorted(np.unique(source_labels))
    target_categories = sorted(np.unique(target_labels))
    source_forced = grouped_flow(
        balanced[source_rejected], source_labels[source_rejected], target_labels,
        rejected_source_categories, target_categories,
    )
    target_forced = grouped_flow(
        balanced[:, target_rejected].T, target_labels[target_rejected], source_labels,
        rejected_target_categories, source_categories,
    )
    rows = []
    for row, source_label in enumerate(rejected_source_categories):
        for column, target_label in enumerate(target_categories):
            rows.append({
                "rejection_side": "source", "rejected_annotation": source_label,
                "traditional_forced_partner_annotation": target_label,
                "conditional_probability": source_forced[row, column],
            })
    for row, target_label in enumerate(rejected_target_categories):
        for column, source_label in enumerate(source_categories):
            rows.append({
                "rejection_side": "target", "rejected_annotation": target_label,
                "traditional_forced_partner_annotation": source_label,
                "conditional_probability": target_forced[row, column],
            })
    pd.DataFrame(rows).to_csv(table_destination, index=False)
    figure, axes = plt.subplots(1, 2, figsize=(19, 8))
    image = draw_heatmap(
        axes[0], source_forced, rejected_source_categories, target_categories,
        "Traditional OT destinations for source disappearance candidates",
    )
    axes[0].set_ylabel("ConfidenceOT-rejected source annotation")
    axes[0].set_xlabel("Traditional OT forced target annotation")
    draw_heatmap(
        axes[1], target_forced, rejected_target_categories, source_categories,
        "Traditional OT precursors for target emergence candidates",
    )
    axes[1].set_ylabel("ConfidenceOT-rejected target annotation")
    axes[1].set_xlabel("Traditional OT forced source annotation")
    figure.colorbar(image, ax=axes, label="Forced-match conditional probability",
                    fraction=0.025, pad=0.02)
    figure.subplots_adjust(left=0.13, right=0.92, bottom=0.23, top=0.88, wspace=0.35)
    figure.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared = np.load(args.prepared_pair, allow_pickle=False)
    population = pd.read_csv(args.run_dir / "population_rejection.csv")
    metrics = pd.read_csv(args.run_dir / "method_metrics.csv")
    if args.confidence_method not in set(metrics.method):
        raise KeyError(f"Confidence method not found: {args.confidence_method}")
    confidence = np.load(method_file(args.run_dir, args.confidence_method), allow_pickle=False)
    traditional = np.load(method_file(args.run_dir, "Balanced OT"), allow_pickle=False)
    source_rejected = confidence["source_rejection_signal"] >= 0.5
    target_rejected = confidence["target_rejection_signal"] >= 0.5
    source_labels = prepared["source_labels"].astype(str)
    target_labels = prepared["target_labels"].astype(str)
    plot_mirror_rejection(
        population, args.confidence_method, args.output_dir / "population_disappearance_emergence.png",
        args.dpi,
    )
    plot_spatial_candidates(
        prepared, source_rejected, target_rejected,
        args.output_dir / "spatial_disappearance_emergence.png", args.dpi,
    )
    plot_transition_comparison(
        traditional["coupling"], confidence["retained_analysis_coupling"],
        source_labels, target_labels, args.output_dir / "traditional_vs_confidence_transitions.png",
        args.dpi,
    )
    plot_forced_matches(
        traditional["coupling"], source_labels, target_labels, source_rejected, target_rejected,
        args.output_dir / "traditional_forced_matches.png",
        args.output_dir / "traditional_forced_matches.csv", args.dpi,
    )
    summary = {
        "confidence_method": args.confidence_method,
        "source_rejected_n": int(source_rejected.sum()),
        "target_rejected_n": int(target_rejected.sum()),
        "source_total_n": int(source_rejected.size),
        "target_total_n": int(target_rejected.size),
    }
    pd.DataFrame([summary]).to_csv(args.output_dir / "developmental_exclusion_summary.csv", index=False)
    print(pd.Series(summary).to_string())
    print(f"Biological comparison figures: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
