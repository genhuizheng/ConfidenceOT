"""Audit MOSTA h5ad sections without loading expression matrices into memory."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SAMPLE_PATTERN = re.compile(r"^(E(?P<day>\d+(?:\.\d+)?))_E(?P<embryo>\d+)S(?P<section>\d+)\.MOSTA\.h5ad$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create replicate-preserving MOSTA sample/QC tables and spatial figures."
    )
    parser.add_argument("data_root", type=Path, help="Directory containing *.MOSTA.h5ad files.")
    parser.add_argument("output_root", type=Path, help="Directory for generated tables and figures.")
    parser.add_argument(
        "--stages", nargs="+", default=("E9.5", "E10.5", "E11.5"),
        help="Developmental stages to include.",
    )
    parser.add_argument(
        "--representative-section", default="E1S1",
        help="Section identifier plotted once per stage when available.",
    )
    return parser.parse_args()


def sample_metadata(path: Path) -> dict[str, object]:
    match = SAMPLE_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Unrecognized MOSTA filename: {path.name}")
    return {
        "sample": path.name.removesuffix(".MOSTA.h5ad"),
        "stage": f"E{match.group('day')}",
        "embryo": f"E{match.group('embryo')}",
        "section": f"S{match.group('section')}",
        "embryo_section": f"E{match.group('embryo')}S{match.group('section')}",
        "stage_day": float(match.group("day")),
        "embryo_number": int(match.group("embryo")),
        "section_number": int(match.group("section")),
    }


def discover_sections(data_root: Path, stages: set[str]) -> list[Path]:
    records: list[tuple[tuple[float, int, int], Path]] = []
    for path in data_root.glob("*.MOSTA.h5ad"):
        try:
            metadata = sample_metadata(path)
        except ValueError:
            continue
        if metadata["stage"] not in stages:
            continue
        key = (
            float(metadata["stage_day"]),
            int(metadata["embryo_number"]),
            int(metadata["section_number"]),
        )
        records.append((key, path))
    return [path for _, path in sorted(records)]


def audit_sections(
    paths: list[Path],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    inventory_rows: list[dict[str, object]] = []
    composition: dict[str, pd.Series] = {}
    spatial_records: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for path in paths:
        metadata = sample_metadata(path)
        data = ad.read_h5ad(path, backed="r")
        required_obs = {"annotation", "total_counts", "n_genes_by_counts"}
        missing = sorted(required_obs - set(data.obs.columns))
        if missing:
            data.file.close()
            raise KeyError(f"{path.name} is missing obs columns: {missing}")
        if "count" not in data.layers or "spatial" not in data.obsm:
            data.file.close()
            raise KeyError(f"{path.name} must contain layers['count'] and obsm['spatial'].")

        labels = data.obs["annotation"].astype(str).replace("nan", "Unannotated")
        counts = labels.value_counts()
        sample = str(metadata["sample"])
        composition[sample] = counts
        spatial_records[sample] = (
            np.asarray(data.obsm["spatial"], dtype=np.float64),
            labels.to_numpy(),
        )
        inventory_rows.append({
            **metadata,
            "file_name": path.name,
            "file_bytes": path.stat().st_size,
            "n_spatial_bins": data.n_obs,
            "n_genes": data.n_vars,
            "n_annotations": int(counts.size),
            "cavity_n": int(counts.get("Cavity", 0)),
            "cavity_fraction": float(counts.get("Cavity", 0) / data.n_obs),
            "median_total_counts": float(data.obs["total_counts"].median()),
            "median_detected_genes": float(data.obs["n_genes_by_counts"].median()),
        })
        data.file.close()

    return pd.DataFrame(inventory_rows), pd.DataFrame(composition).fillna(0).astype(int), spatial_records


def plot_composition(fractions: pd.DataFrame, destination: Path) -> None:
    height = max(7.0, 0.32 * len(fractions.index))
    figure, axis = plt.subplots(figsize=(15, height))
    fraction_values = fractions.to_numpy()
    observed_maximum = float(fraction_values.max()) if fraction_values.size else 0.0
    maximum = max(0.25, observed_maximum)
    image = axis.imshow(fractions.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=maximum)
    axis.set_xticks(np.arange(len(fractions.columns)), labels=fractions.columns, rotation=60, ha="right")
    axis.set_yticks(np.arange(len(fractions.index)), labels=fractions.index)
    axis.set_xlabel("Embryo section")
    axis.set_ylabel("Annotation")
    axis.set_title("MOSTA annotation composition by embryo section")
    for row in range(fractions.shape[0]):
        for column in range(fractions.shape[1]):
            value = float(fractions.iat[row, column])
            if value >= 0.05:
                axis.text(
                    column, row, f"{100 * value:.0f}%", ha="center", va="center",
                    fontsize=7, color="white" if value > 0.12 else "black",
                )
    figure.colorbar(image, ax=axis, label="Fraction of spatial bins")
    figure.tight_layout()
    figure.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_representatives(
    inventory: pd.DataFrame,
    spatial_records: dict[str, tuple[np.ndarray, np.ndarray]],
    stages: list[str],
    representative_section: str,
    destination: Path,
) -> None:
    selected: list[str] = []
    for stage in stages:
        match = inventory.loc[
            (inventory["stage"] == stage)
            & (inventory["embryo_section"] == representative_section),
            "sample",
        ]
        if not match.empty:
            selected.append(str(match.iloc[0]))
    if not selected:
        return
    all_categories = sorted({
        str(label)
        for sample in selected
        for label in np.unique(spatial_records[sample][1])
        if str(label) != "Cavity"
    })
    palette = plt.get_cmap("tab20")
    colors = {category: palette(index % 20) for index, category in enumerate(all_categories)}
    colors["Cavity"] = "#d3d3d3"
    figure, axes = plt.subplots(1, len(selected), figsize=(6 * len(selected), 6), squeeze=False)
    for axis, sample in zip(axes[0], selected):
        spatial, labels = spatial_records[sample]
        for category in ("Cavity", *all_categories):
            mask = labels == category
            if np.any(mask):
                axis.scatter(
                    spatial[mask, 0], spatial[mask, 1], s=3,
                    color=colors[category], linewidths=0,
                    alpha=0.6 if category == "Cavity" else 0.9,
                )
        axis.set_title(sample)
        axis.set_xlabel("Spatial x")
        axis.set_ylabel("Spatial y")
        axis.set_aspect("equal")
        axis.invert_yaxis()
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=colors[category], label=category)
        for category in ("Cavity", *all_categories)
    ]
    figure.legend(handles=handles, bbox_to_anchor=(1.01, 0.5), loc="center left", frameon=False)
    figure.tight_layout()
    figure.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    paths = discover_sections(args.data_root, set(args.stages))
    if not paths:
        raise FileNotFoundError("No requested MOSTA sections were found.")
    table_root = args.output_root / "tables" / "qc"
    figure_root = args.output_root / "figures" / "qc"
    table_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)
    inventory, counts, spatial_records = audit_sections(paths)
    fractions = counts.div(counts.sum(axis=0), axis=1)
    inventory.to_csv(table_root / "sample_inventory.csv", index=False)
    counts.to_csv(table_root / "annotation_counts.csv")
    fractions.to_csv(table_root / "annotation_fractions.csv")
    plot_composition(fractions, figure_root / "annotation_composition.png")
    plot_representatives(
        inventory, spatial_records, list(args.stages), args.representative_section,
        figure_root / "representative_sections.png",
    )
    print(inventory.to_string(index=False))
    print(f"\nAudited {len(paths)} sections; outputs: {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
