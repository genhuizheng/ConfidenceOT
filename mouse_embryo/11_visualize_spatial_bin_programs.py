"""Plot hierarchical spatial-bin DEG and GSEA, with every panel saved alone."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save(figure, path: Path, dpi: int):
    figure.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def volcano(axis, table, top_n):
    y = -np.log10(np.maximum(table.conditional_fdr.fillna(1), 1e-300))
    significant = (table.conditional_fdr < 0.05) & (table.section_direction_consistency >= 0.75)
    colors = np.where(significant & (table.meta_effect > 0), "#0072b2",
                      np.where(significant, "#d55e00", "#b8b8b8"))
    axis.scatter(table.meta_effect, y, c=colors, s=9, alpha=0.72, linewidth=0)
    axis.axhline(-np.log10(0.05), color="#777777", linestyle="--", linewidth=0.7)
    score = np.abs(table.meta_z.to_numpy())
    for index in np.argsort(score)[-top_n:]:
        axis.annotate(table.iloc[index].gene, (table.iloc[index].meta_effect, y.iloc[index]),
                      xytext=(3, 3), textcoords="offset points", fontsize=7)
    axis.set_xlabel("Rejected − retained mean log-expression effect")
    axis.set_ylabel("−log10 conditional FDR")
    axis.grid(alpha=0.13)


def pathway_plot(axis, table, top_n):
    values = table.sort_values(["padj", "NES"], ascending=[True, False]).head(top_n).sort_values("NES")
    limit = max(abs(values.NES).max(), 1)
    sizes = 25 + 30 * np.minimum(-np.log10(np.maximum(values.padj, 1e-300)), 10)
    axis.scatter(values.NES, np.arange(len(values)), c=values.NES, s=sizes,
                 cmap="coolwarm", vmin=-limit, vmax=limit, edgecolor="#333333", linewidth=0.35)
    axis.set_yticks(np.arange(len(values)), values.pathway.str.replace("_", " "), fontsize=7)
    axis.axvline(0, color="#555555", linewidth=0.8)
    axis.set_xlabel("GSEA normalized enrichment score")
    axis.grid(axis="x", alpha=0.13)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deg_root", type=Path)
    parser.add_argument("gsea_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--top-genes", type=int, default=12)
    parser.add_argument("--top-pathways", type=int, default=12)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel_root = args.output_dir / "panels"
    panel_root.mkdir(exist_ok=True)
    completed = []
    for candidate_root in sorted(args.deg_root.iterdir()):
        deg_path = candidate_root / "hierarchical_meta_deg.csv"
        gsea_path = args.gsea_root / candidate_root.name / "fgsea_results.csv"
        if not deg_path.exists() or not gsea_path.exists():
            continue
        deg, gsea = pd.read_csv(deg_path), pd.read_csv(gsea_path)
        scope = deg.inference_scope.iloc[0].replace("_", " ")
        title = candidate_root.name.replace("_", " ") + f" ({scope})"
        figure, axes = plt.subplots(1, 2, figsize=(15, 6.3))
        volcano(axes[0], deg, args.top_genes)
        pathway_plot(axes[1], gsea, args.top_pathways)
        axes[0].set_title("Within-annotation spatial-bin DEG")
        axes[1].set_title("Meta-ranked pathway enrichment")
        figure.suptitle(title, fontsize=14)
        figure.tight_layout(rect=[0, 0, 1, 0.94])
        save(figure, args.output_dir / candidate_root.name, args.dpi)

        figure, axis = plt.subplots(figsize=(7.2, 6.1))
        volcano(axis, deg, args.top_genes)
        axis.set_title(title, fontsize=10)
        figure.tight_layout()
        save(figure, panel_root / f"{candidate_root.name}__deg", args.dpi)

        figure, axis = plt.subplots(figsize=(8.2, 6.1))
        pathway_plot(axis, gsea, args.top_pathways)
        axis.set_title(title, fontsize=10)
        figure.tight_layout()
        save(figure, panel_root / f"{candidate_root.name}__gsea", args.dpi)
        completed.append(candidate_root.name)
    (args.output_dir / "figure_manifest.txt").write_text("\n".join(completed) + "\n")
    print(f"Visualized candidates: {len(completed)}")


if __name__ == "__main__":
    main()
