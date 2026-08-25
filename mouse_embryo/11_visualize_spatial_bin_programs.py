"""Visualize section-level rejected-bin DEG and optional pathway enrichment.

Every section volcano and every consensus/pathway panel is saved separately,
in addition to candidate-level composite figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


UP_COLOR = "#0072B2"
DOWN_COLOR = "#D55E00"
NEUTRAL_COLOR = "#B8B8B8"


def save(figure: plt.Figure, path: Path, dpi: int) -> None:
    figure.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def section_volcano(axis: plt.Axes, table: pd.DataFrame, top_n: int) -> None:
    fdr = table.fdr.fillna(1).clip(lower=1e-300)
    y = -np.log10(fdr)
    significant = (table.fdr < 0.05) & (table.log2_fold_change.abs() >= 0.25)
    colors = np.where(
        significant & table.log2_fold_change.gt(0), UP_COLOR,
        np.where(significant & table.log2_fold_change.lt(0), DOWN_COLOR, NEUTRAL_COLOR),
    )
    axis.scatter(table.log2_fold_change, y, c=colors, s=10, alpha=0.72, linewidth=0)
    axis.axhline(-np.log10(0.05), color="#666666", linestyle="--", linewidth=0.8)
    axis.axvline(0, color="#888888", linewidth=0.6)
    eligible = table[np.isfinite(table.log2_fold_change) & np.isfinite(table.wilcoxon_score)]
    labels = eligible.nlargest(top_n, "wilcoxon_score")
    for row in labels.itertuples():
        axis.annotate(
            row.gene,
            (row.log2_fold_change, -np.log10(max(row.fdr, 1e-300))),
            xytext=(3, 3), textcoords="offset points", fontsize=7,
        )
    axis.set_xlabel("log2 fold change: rejected vs retained bins")
    axis.set_ylabel("−log10 section FDR")
    axis.grid(alpha=0.13)


def consensus_plot(axis: plt.Axes, table: pd.DataFrame, top_n: int) -> None:
    positive = table.nlargest(top_n, "consensus_rank_score")
    negative = table.nsmallest(top_n, "consensus_rank_score")
    values = pd.concat([negative, positive]).drop_duplicates("gene").sort_values("consensus_rank_score")
    colors = np.where(values.consensus_rank_score >= 0, UP_COLOR, DOWN_COLOR)
    axis.barh(np.arange(len(values)), values.consensus_rank_score, color=colors, alpha=0.88)
    axis.set_yticks(np.arange(len(values)), values.gene, fontsize=8)
    axis.axvline(0, color="#555555", linewidth=0.8)
    axis.set_xlabel("Median section Wilcoxon score × direction consistency")
    axis.set_title("Descriptive across-section consensus")
    axis.grid(axis="x", alpha=0.13)


def pathway_plot(axis: plt.Axes, table: pd.DataFrame, top_n: int) -> None:
    values = table.sort_values(["fdr", "NES"], ascending=[True, False]).head(top_n).sort_values("NES")
    limit = max(float(values.NES.abs().max()), 1.0)
    sizes = 25 + 30 * np.minimum(-np.log10(np.maximum(values.fdr, 1e-300)), 10)
    axis.scatter(
        values.NES, np.arange(len(values)), c=values.NES, s=sizes,
        cmap="coolwarm", vmin=-limit, vmax=limit, edgecolor="#333333", linewidth=0.35,
    )
    axis.set_yticks(np.arange(len(values)), values.pathway.str.replace("_", " "), fontsize=7)
    axis.axvline(0, color="#555555", linewidth=0.8)
    axis.set_xlabel("Normalized enrichment score")
    axis.set_title("Ranked pathway enrichment")
    axis.grid(axis="x", alpha=0.13)


def pathway_consensus_plot(axis: plt.Axes, table: pd.DataFrame, top_n: int) -> None:
    positive = table.nlargest(top_n, "pathway_consensus_score")
    negative = table.nsmallest(top_n, "pathway_consensus_score")
    values = pd.concat([negative, positive]).drop_duplicates("pathway").sort_values(
        "pathway_consensus_score"
    )
    labels = values.pathway.str.replace("_", " ").str.removeprefix("GOBP ")
    colors = np.where(values.pathway_consensus_score >= 0, UP_COLOR, DOWN_COLOR)
    axis.barh(np.arange(len(values)), values.pathway_consensus_score, color=colors, alpha=0.88)
    axis.set_yticks(np.arange(len(values)), labels, fontsize=7)
    axis.axvline(0, color="#555555", linewidth=0.8)
    axis.set_xlabel("Median section NES × direction consistency")
    axis.set_title("Across-section pathway consensus")
    axis.grid(axis="x", alpha=0.13)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deg_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--gsea-root", type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--top-genes", type=int, default=10)
    parser.add_argument("--top-pathways", type=int, default=12)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel_root = args.output_dir / "panels"
    panel_root.mkdir(exist_ok=True)
    completed: list[str] = []

    for candidate_root in sorted(path for path in args.deg_root.iterdir() if path.is_dir()):
        consensus_path = candidate_root / "consensus_deg.csv"
        section_paths = sorted(candidate_root.glob("section_*_bin_deg.csv.gz"))
        if not consensus_path.exists() or not section_paths:
            continue
        consensus = pd.read_csv(consensus_path)
        title = candidate_root.name.replace("_", " ")

        panel_builders: list[tuple[str, object]] = []
        for section_path in section_paths:
            table = pd.read_csv(section_path)
            sample = str(table["sample"].iloc[0])
            panel_builders.append((f"Section {sample}", ("section", table, sample)))
        panel_builders.append(("Across-section consensus", ("consensus", consensus, "consensus")))

        gsea_candidate_root = None if args.gsea_root is None else args.gsea_root / candidate_root.name
        gsea_path = None if gsea_candidate_root is None else gsea_candidate_root / "consensus_gsea_results.csv"
        if gsea_path is not None and gsea_path.exists():
            panel_builders.append(("Consensus-rank GSEA", ("pathway", pd.read_csv(gsea_path), "gsea")))
        pathway_consensus_path = (
            None if gsea_candidate_root is None else gsea_candidate_root / "pathway_consensus.csv"
        )
        if pathway_consensus_path is not None and pathway_consensus_path.exists():
            panel_builders.append((
                "Across-section pathway consensus",
                ("pathway_consensus", pd.read_csv(pathway_consensus_path), "pathway_consensus"),
            ))

        figure, axes = plt.subplots(
            1, len(panel_builders), figsize=(6.2 * len(panel_builders), 5.8), squeeze=False
        )
        for axis, (panel_title, payload) in zip(axes.ravel(), panel_builders):
            kind, table, tag = payload
            if kind == "section":
                section_volcano(axis, table, args.top_genes)
            elif kind == "consensus":
                consensus_plot(axis, table, args.top_genes)
            elif kind == "pathway_consensus":
                pathway_consensus_plot(axis, table, args.top_pathways)
            else:
                pathway_plot(axis, table, args.top_pathways)
            axis.set_title(panel_title)

            single, single_axis = plt.subplots(figsize=(7.2, 6.1))
            if kind == "section":
                section_volcano(single_axis, table, args.top_genes)
            elif kind == "consensus":
                consensus_plot(single_axis, table, args.top_genes)
            elif kind == "pathway_consensus":
                pathway_consensus_plot(single_axis, table, args.top_pathways)
            else:
                pathway_plot(single_axis, table, args.top_pathways)
            single_axis.set_title(f"{title}: {panel_title}", fontsize=10)
            single.tight_layout()
            save(single, panel_root / f"{candidate_root.name}__{tag}", args.dpi)

        if gsea_candidate_root is not None:
            for section_gsea_path in sorted(gsea_candidate_root.glob("section_*_gsea_results.csv")):
                sample = section_gsea_path.name.removeprefix("section_").removesuffix("_gsea_results.csv")
                section_gsea = pd.read_csv(section_gsea_path)
                single, single_axis = plt.subplots(figsize=(8.2, 6.1))
                pathway_plot(single_axis, section_gsea, args.top_pathways)
                single_axis.set_title(f"{title}: section {sample} pathway enrichment", fontsize=10)
                single.tight_layout()
                save(single, panel_root / f"{candidate_root.name}__{sample}__gsea", args.dpi)

        figure.suptitle(title, fontsize=14)
        figure.tight_layout(rect=[0, 0, 1, 0.94])
        save(figure, args.output_dir / candidate_root.name, args.dpi)
        completed.append(candidate_root.name)

    (args.output_dir / "figure_manifest.txt").write_text("\n".join(completed) + "\n")
    print(f"Visualized candidates: {len(completed)}")


if __name__ == "__main__":
    main()
