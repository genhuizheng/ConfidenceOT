"""Run all-gene GSEA and plot the two primary within-primary figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_gmt(path: Path) -> dict[str, list[str]]:
    pathways = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                raise ValueError(f"Malformed GMT line {line_number}: {path}")
            pathways[fields[0]] = list(dict.fromkeys(fields[2:]))
    if not pathways:
        raise RuntimeError(f"No pathways read from {path}")
    return pathways


def collection_name(term: str) -> str:
    upper = term.upper()
    if upper.startswith("HALLMARK_"):
        return "Hallmark"
    if upper.startswith(("GOBP_", "GO_BP_")):
        return "GO Biological Process"
    if upper.startswith("REACTOME_"):
        return "Reactome"
    if upper.startswith(("WP_", "WIKIPATHWAYS_")):
        return "WikiPathways"
    return "Other"


def clean_pathway(value: str) -> str:
    for prefix in ("HALLMARK_", "GOBP_", "GO_BP_", "REACTOME_", "WP_"):
        if value.upper().startswith(prefix):
            value = value[len(prefix):]
            break
    return value.replace("_", " ").title()


def leading_edge_n(value) -> int:
    if pd.isna(value):
        return 0
    text = str(value)
    if "/" in text:
        try:
            return int(text.split("/", 1)[0])
        except ValueError:
            pass
    return len([item for item in text.replace(";", ";").split(";") if item])


def plot_volcano(deg: pd.DataFrame, leading: pd.DataFrame, output: Path,
                 patient_n: int, pair_n: int) -> None:
    table = deg.copy()
    table["plot_fdr"] = table["fdr"].clip(lower=np.nextafter(0, 1))
    table["minus_log10_fdr"] = -np.log10(table["plot_fdr"])
    significant = table["fdr"].lt(0.05) & table["log2_fold_change"].abs().ge(1.0)
    colors = np.where(
        significant & table["log2_fold_change"].gt(0), "#C43C39",
        np.where(significant & table["log2_fold_change"].lt(0), "#2C6BAA", "#C9C9C9"),
    )
    figure, axis = plt.subplots(figsize=(9.2, 6.8))
    axis.scatter(table["log2_fold_change"], table["minus_log10_fdr"],
                 c=colors, s=9, alpha=0.65, linewidths=0, rasterized=True)
    axis.axvline(-1, color="#666666", linestyle="--", linewidth=0.8)
    axis.axvline(1, color="#666666", linestyle="--", linewidth=0.8)
    axis.axhline(-np.log10(0.05), color="#666666", linestyle="--", linewidth=0.8)
    label = pd.concat([
        leading[leading["log2_fold_change"].gt(0)].nlargest(10, "absolute_wald_statistic"),
        leading[leading["log2_fold_change"].lt(0)].nlargest(10, "absolute_wald_statistic"),
    ]).drop_duplicates("gene")
    for _, row in label.iterrows():
        axis.annotate(str(row["gene"]),
                      (row["log2_fold_change"], -np.log10(max(row["fdr"], 1e-300))),
                      xytext=(3, 3), textcoords="offset points", fontsize=7)
    axis.set_xlabel("log2 fold change: metastasis-compatible / primary-restricted")
    axis.set_ylabel("-log10(FDR)")
    axis.set_title("Primary malignant-cell programs associated with metastasis compatibility")
    axis.text(0.01, 0.99, f"{patient_n} patients · {pair_n} primary–metastasis pairs",
              transform=axis.transAxes, va="top", fontsize=9)
    figure.tight_layout()
    figure.savefig(output / "01_primary_compatible_vs_restricted_deg_volcano.png", dpi=300)
    figure.savefig(output / "01_primary_compatible_vs_restricted_deg_volcano.pdf")
    plt.close(figure)


def plot_gsea(result: pd.DataFrame, output: Path, top_n: int) -> pd.DataFrame:
    significant = result[result["fdr"].lt(0.05)].copy()
    positive = significant[significant["NES"].gt(0)].sort_values(
        ["fdr", "NES"], ascending=[True, False]
    ).head(top_n)
    negative = significant[significant["NES"].lt(0)].sort_values(
        ["fdr", "NES"], ascending=[True, True]
    ).head(top_n)
    selected = pd.concat([negative, positive], ignore_index=True)
    if selected.empty:
        selected = pd.concat([
            result[result["NES"].lt(0)].nsmallest(top_n, "fdr"),
            result[result["NES"].gt(0)].nsmallest(top_n, "fdr"),
        ], ignore_index=True)
    selected["display_pathway"] = selected["pathway"].map(clean_pathway)
    selected["minus_log10_fdr"] = -np.log10(selected["fdr"].clip(lower=1e-300))
    selected = selected.sort_values("NES").reset_index(drop=True)
    y = np.arange(len(selected))
    figure_height = max(5.5, 0.32 * len(selected) + 1.8)
    figure, axis = plt.subplots(figsize=(10.5, figure_height))
    scatter = axis.scatter(
        selected["NES"], y,
        s=30 + 5 * np.sqrt(selected["leading_edge_n"].clip(lower=1)),
        c=selected["minus_log10_fdr"], cmap="viridis", edgecolor="#333333",
        linewidth=0.35,
    )
    axis.axvline(0, color="#555555", linewidth=0.8)
    axis.set_yticks(y, selected["display_pathway"])
    axis.set_xlabel("Normalized enrichment score (NES)")
    axis.set_title("Pathways associated with primary-tumor metastasis compatibility")
    axis.text(0.01, 1.01, "Primary-restricted enriched ←    → Metastasis-compatible enriched",
              transform=axis.transAxes, fontsize=9)
    colorbar = figure.colorbar(scatter, ax=axis, pad=0.02)
    colorbar.set_label("-log10(GSEA FDR)")
    figure.tight_layout()
    figure.savefig(output / "02_primary_compatible_vs_restricted_gsea_dotplot.png", dpi=300)
    figure.savefig(output / "02_primary_compatible_vs_restricted_gsea_dotplot.pdf")
    plt.close(figure)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pydeseq2_root", type=Path)
    parser.add_argument("human_pathways_gmt", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--minimum-size", type=int, default=10)
    parser.add_argument("--maximum-size", type=int, default=500)
    parser.add_argument("--top-pathways-per-direction", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    deg = pd.read_csv(
        args.pydeseq2_root / "primary_compatible_vs_restricted_all_genes.csv.gz"
    )
    leading_path = (
        args.pydeseq2_root / "primary_compatible_vs_restricted_leading_lfc_1.csv"
    )
    leading = pd.read_csv(leading_path) if leading_path.exists() else deg.iloc[0:0]
    rank = deg[["gene", "wald_statistic"]].replace([np.inf, -np.inf], np.nan).dropna()
    rank = rank.drop_duplicates("gene").sort_values(
        ["wald_statistic", "gene"], ascending=[False, True], kind="stable"
    )
    pathways = read_gmt(args.human_pathways_gmt)
    prerank = gp.prerank(
        rnk=rank, gene_sets=pathways, min_size=args.minimum_size,
        max_size=args.maximum_size, permutation_num=args.permutations,
        threads=args.threads, seed=args.seed, outdir=None, no_plot=True,
        verbose=False,
    ).res2d
    result = prerank.rename(columns={
        "Term": "pathway", "ES": "enrichment_score", "NOM p-val": "p_value",
        "FDR q-val": "fdr", "FWER p-val": "fwer",
        "Lead_genes": "leading_edge_genes", "Tag %": "tag_fraction",
        "Gene %": "rank_fraction",
    }).copy()
    for column in ("enrichment_score", "NES", "p_value", "fdr", "fwer"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["collection"] = result["pathway"].map(collection_name)
    result["direction"] = np.where(
        result["NES"].ge(0), "metastasis_compatible_enriched",
        "primary_restricted_enriched",
    )
    result["leading_edge_n"] = result.get(
        "tag_fraction", result.get("leading_edge_genes", "")
    ).map(leading_edge_n)
    result = result.sort_values(["fdr", "NES"], ascending=[True, False])
    result.to_csv(args.output_root / "primary_compatible_vs_restricted_gsea_all.csv.gz",
                  index=False, compression="gzip")

    report_path = args.pydeseq2_root / "pydeseq2_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    plot_volcano(deg, leading, args.output_root,
                 patient_n=int(report["patient_n"]), pair_n=int(report["pair_n"]))
    selected = plot_gsea(result, args.output_root, args.top_pathways_per_direction)
    selected.to_csv(args.output_root / "primary_compatible_vs_restricted_gsea_display.csv",
                    index=False)
    summary = {
        "engine": "GSEApy prerank",
        "ranking_metric": "all-gene PyDESeq2 Wald statistic",
        "ranked_gene_n": len(rank),
        "tested_pathway_n": len(result),
        "fdr_005_pathway_n": int(result["fdr"].lt(0.05).sum()),
        "positive_direction": "putative metastasis-compatible primary malignant cells",
        "negative_direction": "putative primary-restricted primary malignant cells",
        "ot_genes_filtered": False,
    }
    (args.output_root / "gsea_report.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
