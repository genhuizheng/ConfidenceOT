"""Run and compare paired primary-cell PyDESeq2 across M4-E initializations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cancer_metastasis.gse180661.primary_deg import descriptive_metrics, run_pydeseq2


def configuration_name(strategy: str, replicate: int) -> str:
    if strategy == "all_one":
        return "all_one"
    if strategy == "all_zero_projected_to_budget_floor":
        return "projected_zero"
    return f"random_{replicate:02d}"


def load_inputs(root: Path):
    count_tables = []
    metadata_tables = []
    gene_tables = []
    for marker in sorted(root.glob("groups/*/PSEUDOBULK_READY")):
        directory = marker.parent
        counts = pd.read_csv(directory / "pseudobulk_raw_counts.csv.gz", index_col=0)
        metadata = pd.read_csv(directory / "pseudobulk_sample_metadata.csv")
        if list(counts.index.astype(str)) != list(metadata["sample_id"].astype(str)):
            raise RuntimeError(f"Sample order mismatch in {directory}")
        count_tables.append(counts)
        metadata_tables.append(metadata)
        gene_tables.append(pd.read_csv(directory / "pseudobulk_gene_metadata.csv.gz"))
    if not count_tables:
        raise RuntimeError(f"No initialization pseudobulks under {root}")
    genes = sorted(set().union(*(set(table.columns) for table in count_tables)))
    counts = pd.concat(
        [table.reindex(columns=genes, fill_value=0) for table in count_tables]
    ).astype(np.int64)
    metadata = pd.concat(metadata_tables, ignore_index=True).set_index("sample_id")
    metadata = metadata.loc[counts.index]
    gene_metadata = pd.concat(gene_tables, ignore_index=True).groupby(
        "gene", as_index=False
    ).agg(gene_used_in_ot_representation=("used_for_ot", "max"))
    return counts, metadata, gene_metadata


def complete_configuration(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    strategy: str,
    replicate: int,
    minimum_cells: int,
):
    selected = metadata[
        metadata["strategy"].eq(strategy) & metadata["replicate"].eq(replicate)
    ].copy()
    complete = selected.groupby("pair_id").filter(
        lambda table: len(table) == 2
        and set(table["comparison_status"]) == {"case", "reference"}
        and table["cell_n"].min() >= minimum_cells
    )
    return counts.loc[complete.index], complete


def decorate(result, counts, metadata, gene_metadata):
    metrics = descriptive_metrics(counts, metadata).rename(columns={
        "compatible_mean_cpm": "retained_mean_cpm",
        "restricted_mean_cpm": "rejected_mean_cpm",
    })
    result = result.merge(metrics, on="gene", validate="one_to_one")
    result = result.merge(gene_metadata, on="gene", how="left", validate="one_to_one")
    result["gene_used_in_ot_representation"] = result[
        "gene_used_in_ot_representation"
    ].fillna(False).astype(bool)
    result["patient_direction_consistency"] = np.where(
        result["log2_fold_change"].ge(0),
        result["positive_patient_fraction"],
        result["negative_patient_fraction"],
    )
    result["direction"] = np.select(
        [result["log2_fold_change"].gt(0), result["log2_fold_change"].lt(0)],
        ["primary_retained_enriched", "primary_rejected_enriched"],
        default="no_direction",
    )
    return result.sort_values(["fdr", "wald_statistic"], ascending=[True, False])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pseudobulk_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--minimum-cells-per-state", type=int, default=20)
    parser.add_argument("--minimum-total-count", type=int, default=10)
    parser.add_argument("--n-cpus", type=int, default=16)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    counts, metadata, gene_metadata = load_inputs(args.pseudobulk_root)
    configurations = (
        metadata[["strategy", "replicate"]].drop_duplicates()
        .sort_values(["strategy", "replicate"])
    )
    summaries = []
    results = {}
    for item in configurations.itertuples(index=False):
        strategy, replicate = str(item.strategy), int(item.replicate)
        name = configuration_name(strategy, replicate)
        selected_counts, selected_metadata = complete_configuration(
            counts, metadata, strategy, replicate, args.minimum_cells_per_state
        )
        keep = selected_counts.sum(axis=0).ge(args.minimum_total_count)
        selected_counts = selected_counts.loc[:, keep]
        result = run_pydeseq2(selected_counts, selected_metadata, args.n_cpus)
        result = decorate(result, selected_counts, selected_metadata, gene_metadata)
        directory = args.output_root / "configurations" / name
        directory.mkdir(parents=True, exist_ok=True)
        result.to_csv(directory / "primary_retained_vs_rejected_all_genes.csv.gz",
                      index=False, compression="gzip")
        result[["gene", "wald_statistic"]].dropna().to_csv(
            directory / "primary_retained_vs_rejected_all_genes.rnk",
            sep="\t", header=False, index=False,
        )
        for threshold in (0.5, 1.0):
            leading = result[
                result["fdr"].lt(0.05)
                & result["log2_fold_change"].abs().ge(threshold)
                & result["detected_patient_fraction"].ge(0.25)
                & result["patient_direction_consistency"].ge(0.70)
            ]
            leading.to_csv(directory / f"leading_genes_lfc_{str(threshold).replace('.', 'p')}.csv", index=False)
        summaries.append({
            "configuration": name,
            "strategy": strategy,
            "replicate": replicate,
            "pair_n": int(selected_metadata["pair_id"].nunique()),
            "patient_n": int(selected_metadata["patient_id"].nunique()),
            "tested_gene_n": len(result),
            "fdr_0p05_lfc_0p5_retained_n": int((result["fdr"].lt(0.05) & result["log2_fold_change"].ge(0.5)).sum()),
            "fdr_0p05_lfc_0p5_rejected_n": int((result["fdr"].lt(0.05) & result["log2_fold_change"].le(-0.5)).sum()),
            "fdr_0p05_lfc_1p0_retained_n": int((result["fdr"].lt(0.05) & result["log2_fold_change"].ge(1.0)).sum()),
            "fdr_0p05_lfc_1p0_rejected_n": int((result["fdr"].lt(0.05) & result["log2_fold_change"].le(-1.0)).sum()),
        })
        results[name] = result.set_index("gene")

    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output_root / "initialization_deg_summary.csv", index=False)
    common_genes = sorted(set.intersection(*(set(table.index) for table in results.values())))
    effects = pd.DataFrame({
        name: table.loc[common_genes, "log2_fold_change"] for name, table in results.items()
    })
    effects.index.name = "gene"
    effects.to_csv(args.output_root / "initialization_log2fc_matrix.csv.gz", compression="gzip")
    correlation = effects.corr(method="spearman")
    correlation.to_csv(args.output_root / "initialization_log2fc_spearman.csv")

    stability_rows = []
    for gene in common_genes:
        retained_n = 0
        rejected_n = 0
        significant_n = 0
        for table in results.values():
            row = table.loc[gene]
            significant = bool(pd.notna(row["fdr"]) and row["fdr"] < 0.05
                               and abs(row["log2_fold_change"]) >= 0.5)
            if significant:
                significant_n += 1
                retained_n += int(row["log2_fold_change"] > 0)
                rejected_n += int(row["log2_fold_change"] < 0)
        stability_rows.append({
            "gene": gene,
            "configuration_n": len(results),
            "significant_configuration_n": significant_n,
            "retained_enriched_configuration_n": retained_n,
            "rejected_enriched_configuration_n": rejected_n,
            "direction_consistent_when_significant": retained_n == 0 or rejected_n == 0,
        })
    stability = pd.DataFrame(stability_rows).sort_values(
        ["significant_configuration_n", "gene"], ascending=[False, True]
    )
    stability.to_csv(args.output_root / "initialization_gene_stability_lfc_0p5.csv", index=False)

    figure, axis = plt.subplots(figsize=(8.2, 7.0))
    image = axis.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(range(len(correlation)), correlation.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(correlation)), correlation.index)
    for row in range(len(correlation)):
        for column in range(len(correlation)):
            axis.text(column, row, f"{correlation.iloc[row, column]:.2f}",
                      ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axis, label="Spearman correlation of log2 fold change")
    axis.set_title("Primary retained-versus-rejected DEG stability across initializations")
    figure.tight_layout()
    figure.savefig(args.output_root / "initialization_deg_effect_correlation.png", dpi=300)
    figure.savefig(args.output_root / "initialization_deg_effect_correlation.pdf")
    plt.close(figure)

    report = {
        "engine": "PyDESeq2",
        "design": "~pair_id + comparison_status",
        "contrast": "primary retained versus primary rejected within each exact primary-metastasis pair",
        "configuration_n": len(summary),
        "all_genes_including_ot_representation_genes": True,
        "selection_warning": "Do not select an initialization by DEG yield; assess effect-direction and gene-set stability across initializations.",
    }
    (args.output_root / "initialization_deg_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
