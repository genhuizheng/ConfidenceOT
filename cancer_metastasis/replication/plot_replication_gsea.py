"""Plot independent GSEA results for each author-labelled replication dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASETS = ("GSE181919", "GSE225857")
GSEA_FILE = "primary_compatible_vs_restricted_gsea_all.csv.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replication_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--top-per-direction", type=int, default=12)
    return parser.parse_args()


def display_name(pathway: str) -> str:
    for prefix in ("HALLMARK_", "REACTOME_", "GOBP_", "WP_"):
        if pathway.startswith(prefix):
            pathway = pathway[len(prefix):]
            break
    return (
        pathway.replace("_", " ")
        .title()
        .replace("Tnfa", "TNFα")
        .replace("Nfkb", "NF-κB")
    )


def load_gsea(root: Path, dataset: str) -> pd.DataFrame:
    path = root / dataset / "primary_gsea" / GSEA_FILE
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pd.read_csv(path)
    required = {"pathway", "NES", "fdr"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    table["NES"] = pd.to_numeric(table["NES"], errors="coerce")
    table["fdr"] = pd.to_numeric(table["fdr"], errors="coerce")
    return table.dropna(subset=["pathway", "NES", "fdr"])


def plot_dataset(
    root: Path,
    output_root: Path,
    dataset: str,
    top_per_direction: int,
) -> None:
    table = load_gsea(root, dataset)
    significant = table[table["fdr"].lt(0.05)].copy()
    rejected = significant[significant["NES"].lt(0)].nsmallest(
        top_per_direction, "NES"
    )
    retained = significant[significant["NES"].gt(0)].nlargest(
        top_per_direction, "NES"
    )
    display = pd.concat((rejected, retained), ignore_index=True)
    if display.empty:
        raise RuntimeError(f"No FDR-significant GSEA pathways were found for {dataset}.")

    display["direction"] = np.where(
        display["NES"].gt(0),
        "Source-retained enriched",
        "Source-rejected enriched",
    )
    display["display_pathway"] = display["pathway"].map(display_name)
    display["minus_log10_fdr"] = -np.log10(display["fdr"].clip(lower=1e-300))
    display.to_csv(output_root / f"{dataset}_gsea_display.csv", index=False)

    figure, axis = plt.subplots(figsize=(11.0, 9.2))
    y = np.arange(len(display))
    points = axis.scatter(
        display["NES"], y,
        c=display["minus_log10_fdr"], s=82,
        cmap="viridis", edgecolor="#333333", linewidth=0.4,
    )
    axis.axvline(0, color="#555555", linewidth=0.9)
    axis.set_yticks(y, display["display_pathway"])
    axis.set_xlabel("Normalized enrichment score (NES)")
    axis.set_ylabel("")
    axis.set_title(
        f"{dataset}: pathways associated with the primary malignant-cell OT gate",
        fontsize=14,
    )
    axis.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    axis.set_axisbelow(True)
    limit = max(3.0, float(display["NES"].abs().max()) * 1.12)
    axis.set_xlim(-limit, limit)
    axis.text(
        0.01, 1.012, "Source-rejected enriched",
        transform=axis.transAxes, ha="left", va="bottom",
        color="#2166AC", fontsize=10,
    )
    axis.text(
        0.99, 1.012, "Source-retained enriched",
        transform=axis.transAxes, ha="right", va="bottom",
        color="#B2182B", fontsize=10,
    )
    colorbar = figure.colorbar(points, ax=axis, pad=0.025)
    colorbar.set_label("−log10(GSEA FDR)")
    figure.text(
        0.5, 0.012,
        f"Top {top_per_direction} FDR-significant pathways per direction; "
        "GSEApy prerank using all-gene PyDESeq2 Wald statistics.",
        ha="center", fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 1))

    output = output_root / f"{dataset}_gsea_dotplot"
    figure.savefig(
        output.with_suffix(".png"), dpi=300,
        bbox_inches="tight", facecolor="white",
    )
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        plot_dataset(
            args.replication_root,
            args.output_root,
            dataset,
            args.top_per_direction,
        )
        print(args.output_root / f"{dataset}_gsea_dotplot.png")
        print(args.output_root / f"{dataset}_gsea_dotplot.pdf")


if __name__ == "__main__":
    main()
