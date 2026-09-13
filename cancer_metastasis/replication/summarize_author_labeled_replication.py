"""Compare neutral M4-E gate-associated signals across author-labelled datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


DATASETS = ("GSE181919", "GSE225857")
DEG_FILE = "primary_compatible_vs_restricted_all_genes.csv.gz"
GSEA_FILE = "primary_compatible_vs_restricted_gsea_all.csv.gz"


def effect_comparison(root: Path) -> pd.DataFrame:
    tables = []
    for dataset in DATASETS:
        path = root / dataset / "primary_deg" / DEG_FILE
        if not path.is_file():
            continue
        table = pd.read_csv(path)[["gene", "log2_fold_change", "wald_statistic", "fdr"]]
        table = table.rename(columns={
            column: f"{dataset}_{column}" for column in table.columns if column != "gene"
        })
        tables.append(table)
    if len(tables) != len(DATASETS):
        return pd.DataFrame()
    result = tables[0].merge(tables[1], on="gene", how="inner", validate="one_to_one")
    a, b = DATASETS
    result["same_log2fc_direction"] = (
        np.sign(result[f"{a}_log2_fold_change"])
        == np.sign(result[f"{b}_log2_fold_change"])
    )
    result["significant_lfc_0p5_both"] = (
        result[f"{a}_fdr"].lt(0.05)
        & result[f"{b}_fdr"].lt(0.05)
        & result[f"{a}_log2_fold_change"].abs().ge(0.5)
        & result[f"{b}_log2_fold_change"].abs().ge(0.5)
    )
    result["minimum_absolute_log2fc"] = np.minimum(
        result[f"{a}_log2_fold_change"].abs(),
        result[f"{b}_log2_fold_change"].abs(),
    )
    return result.sort_values(
        ["significant_lfc_0p5_both", "minimum_absolute_log2fc"],
        ascending=[False, False], kind="stable",
    )


def pathway_comparison(root: Path) -> pd.DataFrame:
    tables = []
    for dataset in DATASETS:
        path = root / dataset / "primary_gsea" / GSEA_FILE
        if not path.is_file():
            continue
        table = pd.read_csv(path)[["pathway", "NES", "fdr"]].rename(columns={
            "NES": f"{dataset}_NES", "fdr": f"{dataset}_fdr",
        })
        tables.append(table)
    if len(tables) != len(DATASETS):
        return pd.DataFrame()
    result = tables[0].merge(tables[1], on="pathway", how="inner", validate="one_to_one")
    a, b = DATASETS
    result["same_nes_direction"] = np.sign(result[f"{a}_NES"]) == np.sign(result[f"{b}_NES"])
    result["fdr_005_both"] = result[f"{a}_fdr"].lt(0.05) & result[f"{b}_fdr"].lt(0.05)
    result["minimum_absolute_nes"] = np.minimum(
        result[f"{a}_NES"].abs(), result[f"{b}_NES"].abs()
    )
    return result.sort_values(
        ["fdr_005_both", "minimum_absolute_nes"],
        ascending=[False, False], kind="stable",
    )


def scatter(table: pd.DataFrame, x: str, y: str, output: Path, title: str,
            axis_label: str) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 6.6))
    axis.hexbin(table[x], table[y], gridsize=60, mincnt=1, bins="log", cmap="viridis")
    axis.axhline(0, color="#777777", linewidth=0.7)
    axis.axvline(0, color="#777777", linewidth=0.7)
    axis.set_xlabel(f"GSE181919 {axis_label}")
    axis.set_ylabel(f"GSE225857 {axis_label}")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output.with_suffix(".png"), dpi=300)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replication_root", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for dataset in DATASETS:
        summary_path = args.replication_root / dataset / "manifest" / "malignant_manifest_summary.json"
        if summary_path.is_file():
            manifest_rows.append(json.loads(summary_path.read_text(encoding="utf-8")))
    pd.DataFrame(manifest_rows).to_csv(
        args.output_root / "dataset_malignant_pair_coverage.csv", index=False
    )

    genes = effect_comparison(args.replication_root)
    pathways = pathway_comparison(args.replication_root)
    if not genes.empty:
        genes.to_csv(args.output_root / "cross_dataset_gene_effects.csv.gz",
                     index=False, compression="gzip")
        genes[genes["significant_lfc_0p5_both"] & genes["same_log2fc_direction"]].to_csv(
            args.output_root / "cross_dataset_concordant_genes_lfc_0p5.csv", index=False
        )
        scatter(
            genes, "GSE181919_log2_fold_change", "GSE225857_log2_fold_change",
            args.output_root / "cross_dataset_gene_effect_correlation",
            "Replication of M4-E primary gate-associated gene effects",
            "log2 fold change (retained/rejected)",
        )
    if not pathways.empty:
        pathways.to_csv(args.output_root / "cross_dataset_pathway_effects.csv.gz",
                        index=False, compression="gzip")
        pathways[pathways["fdr_005_both"] & pathways["same_nes_direction"]].to_csv(
            args.output_root / "cross_dataset_concordant_pathways.csv", index=False
        )
        scatter(
            pathways, "GSE181919_NES", "GSE225857_NES",
            args.output_root / "cross_dataset_pathway_effect_correlation",
            "Replication of M4-E primary gate-associated pathways",
            "GSEA NES",
        )

    a, b = DATASETS
    gene_rho = (
        float(spearmanr(genes[f"{a}_log2_fold_change"], genes[f"{b}_log2_fold_change"]).statistic)
        if len(genes) else None
    )
    pathway_rho = (
        float(spearmanr(pathways[f"{a}_NES"], pathways[f"{b}_NES"]).statistic)
        if len(pathways) else None
    )
    report = {
        "datasets": list(DATASETS),
        "comparison": "primary malignant M4-E source-retained versus source-rejected",
        "biological_labels_assigned_a_priori": False,
        "shared_gene_n": int(len(genes)),
        "gene_log2fc_spearman": gene_rho,
        "concordant_significant_gene_n": int(
            (genes.get("significant_lfc_0p5_both", False) & genes.get("same_log2fc_direction", False)).sum()
        ) if len(genes) else 0,
        "shared_pathway_n": int(len(pathways)),
        "pathway_nes_spearman": pathway_rho,
        "concordant_significant_pathway_n": int(
            (pathways.get("fdr_005_both", False) & pathways.get("same_nes_direction", False)).sum()
        ) if len(pathways) else 0,
    }
    (args.output_root / "replication_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
