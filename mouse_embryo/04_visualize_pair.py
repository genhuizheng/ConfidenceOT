"""Create PPT-ready transition and rejection figures for one MOSTA OT run."""

from __future__ import annotations

import argparse
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared_pair", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def method_file(root: Path, method: str) -> Path:
    safe = method.lower().replace(" ", "_").replace("/", "_").replace("|", "_")
    return root / "methods" / f"{safe}.npz"


def plot_rejection_heatmap(metrics, rejection, destination: Path, dpi: int) -> None:
    methods = metrics["method"].tolist()
    source_labels = sorted(rejection.loc[rejection.side == "source", "annotation"].unique())
    target_labels = sorted(rejection.loc[rejection.side == "target", "annotation"].unique())
    columns = [("source", value) for value in source_labels] + [("target", value) for value in target_labels]
    values = np.zeros((len(methods), len(columns)))
    lookup = rejection.set_index(["method", "side", "annotation"])["mean_rejection_signal"]
    for row, method in enumerate(methods):
        for column, key in enumerate(columns):
            values[row, column] = lookup.get((method, *key), np.nan)
    width = max(13, 0.42 * len(columns) + 4)
    figure, axis = plt.subplots(figsize=(width, 0.55 * len(methods) + 2.8))
    image = axis.imshow(values, vmin=0, vmax=1, cmap="magma", aspect="auto")
    axis.set_yticks(range(len(methods)), methods, fontsize=10)
    axis.set_xticks(range(len(columns)), [f"{side.title()}: {label}" for side, label in columns],
                    rotation=55, ha="right", fontsize=9)
    axis.set_title("Population-level rejection signal", fontsize=15, pad=12)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isfinite(value):
                axis.text(column, row, f"{value:.2f}", ha="center", va="center",
                          fontsize=7, color="black" if value > 0.65 else "white")
    figure.colorbar(image, ax=axis, label="Mean rejection signal", fraction=0.025, pad=0.02)
    figure.tight_layout()
    figure.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_transitions(metrics, transitions, destination: Path, dpi: int) -> None:
    methods = metrics["method"].tolist()
    source_labels = sorted(transitions["source_annotation"].unique())
    target_labels = sorted(transitions["target_annotation"].unique())
    n_columns = 3 if len(methods) > 4 else 2
    n_rows = ceil(len(methods) / n_columns)
    figure, axes = plt.subplots(n_rows, n_columns, figsize=(6.3 * n_columns, 5.5 * n_rows), squeeze=False)
    last_image = None
    for axis, method in zip(axes.ravel(), methods):
        subset = transitions[transitions.method == method]
        matrix = subset.pivot(index="source_annotation", columns="target_annotation",
                              values="source_conditional_probability").reindex(
                                  index=source_labels, columns=target_labels, fill_value=0
                              ).to_numpy()
        last_image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        axis.set_title(method, fontsize=12)
        axis.set_yticks(range(len(source_labels)), source_labels, fontsize=7)
        axis.set_xticks(range(len(target_labels)), target_labels, rotation=55, ha="right", fontsize=7)
        axis.set_xlabel("Target annotation")
        axis.set_ylabel("Source annotation")
    for axis in axes.ravel()[len(methods):]:
        axis.axis("off")
    if last_image is not None:
        figure.colorbar(last_image, ax=axes.ravel().tolist(), label="Source-conditional transport probability",
                        fraction=0.015, pad=0.01)
    figure.suptitle("Annotation-level transport transitions", fontsize=16)
    figure.subplots_adjust(left=0.10, right=0.92, bottom=0.10, top=0.92, wspace=0.38, hspace=0.45)
    figure.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_rejection_maps(prepared, metrics, run_dir: Path, destination: Path, dpi: int, *, space: str) -> None:
    methods = metrics["method"].tolist()
    if space == "spatial":
        source_xy, target_xy = prepared["source_spatial"], prepared["target_spatial"]
        backgrounds = (prepared["source_background_spatial"], prepared["target_background_spatial"])
        axis_names = ("Spatial x", "Spatial y")
        title = "Spatial localization of rejection signal"
    else:
        source_xy, target_xy = prepared["source_pca"][:, :2], prepared["target_pca"][:, :2]
        backgrounds = (np.empty((0, 2)), np.empty((0, 2)))
        axis_names = ("PC1", "PC2")
        title = "PCA localization of rejection signal"
    figure, axes = plt.subplots(len(methods), 2, figsize=(10, max(4, 3.4 * len(methods))), squeeze=False)
    last = None
    for row, method in enumerate(methods):
        fitted = np.load(method_file(run_dir, method), allow_pickle=False)
        for column, (coordinates, background, signal, side) in enumerate((
            (source_xy, backgrounds[0], fitted["source_rejection_signal"], "Source"),
            (target_xy, backgrounds[1], fitted["target_rejection_signal"], "Target"),
        )):
            axis = axes[row, column]
            if background.size:
                axis.scatter(background[:, 0], background[:, 1], s=3, c="#d3d3d3", alpha=0.6,
                             linewidths=0, rasterized=True)
            last = axis.scatter(coordinates[:, 0], coordinates[:, 1], c=signal, s=7,
                                cmap="magma", vmin=0, vmax=1, linewidths=0, rasterized=True)
            axis.set_title(f"{method} — {side}", fontsize=10)
            axis.set_xlabel(axis_names[0])
            axis.set_ylabel(axis_names[1])
            axis.set_aspect("equal", adjustable="datalim")
            if space == "spatial":
                axis.invert_yaxis()
    figure.suptitle(title, fontsize=16)
    if last is not None:
        figure.colorbar(last, ax=axes.ravel().tolist(), label="Rejection signal", fraction=0.018, pad=0.015)
    figure.subplots_adjust(left=0.08, right=0.90, bottom=0.04, top=0.96, wspace=0.25, hspace=0.38)
    figure.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared = np.load(args.prepared_pair, allow_pickle=False)
    metrics = pd.read_csv(args.run_dir / "method_metrics.csv")
    rejection = pd.read_csv(args.run_dir / "population_rejection.csv")
    transitions = pd.read_csv(args.run_dir / "population_transitions.csv")
    plot_rejection_heatmap(metrics, rejection, args.output_dir / "population_rejection.png", args.dpi)
    plot_transitions(metrics, transitions, args.output_dir / "population_transitions.png", args.dpi)
    plot_rejection_maps(prepared, metrics, args.run_dir, args.output_dir / "spatial_rejection.png", args.dpi,
                        space="spatial")
    plot_rejection_maps(prepared, metrics, args.run_dir, args.output_dir / "pca_rejection.png", args.dpi,
                        space="pca")
    print(metrics.to_string(index=False))
    print(f"Figures: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
