"""Run all-gene paired PyDESeq2 for primary-only pair pseudobulks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CONTRAST = "primary_metastasis_compatible_vs_primary_restricted"


def load_pair_pseudobulks(root: Path):
    count_tables = []
    metadata_tables = []
    gene_tables = []
    for marker in sorted(root.glob("groups/*/PSEUDOBULK_READY")):
        directory = marker.parent
        counts = pd.read_csv(directory / "pseudobulk_raw_counts.csv.gz", index_col=0)
        metadata = pd.read_csv(directory / "pseudobulk_sample_metadata.csv")
        genes = pd.read_csv(directory / "pseudobulk_gene_metadata.csv.gz")
        if list(counts.index.astype(str)) != list(metadata["sample_id"].astype(str)):
            raise RuntimeError(f"Sample order mismatch in {directory}")
        count_tables.append(counts)
        metadata_tables.append(metadata)
        genes["group_directory"] = str(directory)
        gene_tables.append(genes)
    if not count_tables:
        raise RuntimeError(f"No pair pseudobulks found under {root}")
    all_genes = sorted(set().union(*(set(table.columns) for table in count_tables)))
    counts = pd.concat(
        [table.reindex(columns=all_genes, fill_value=0) for table in count_tables]
    ).astype(np.int64)
    metadata = pd.concat(metadata_tables, ignore_index=True).set_index("sample_id")
    metadata = metadata.loc[counts.index]
    if counts.index.duplicated().any():
        raise RuntimeError("Duplicate pseudobulk sample IDs")
    pair_sizes = metadata.groupby("pair_id").size()
    status_complete = metadata.groupby("pair_id")["comparison_status"].agg(
        lambda values: set(values) == {"case", "reference"}
    )
    valid_pairs = set(pair_sizes.index[(pair_sizes == 2) & status_complete])
    keep = metadata["pair_id"].isin(valid_pairs)
    counts = counts.loc[keep]
    metadata = metadata.loc[keep]
    genes = pd.concat(gene_tables, ignore_index=True).groupby("gene", as_index=False).agg(
        gene_used_in_ot_representation=("used_for_ot", "max"),
        evaluated_pair_n=("group_directory", "nunique"),
    )
    return counts, metadata, genes


def run_pydeseq2(counts: pd.DataFrame, metadata: pd.DataFrame, n_cpus: int):
    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.default_inference import DefaultInference
        from pydeseq2.ds import DeseqStats
    except ImportError as error:
        raise RuntimeError("pydeseq2>=0.5,<0.6 is required") from error
    inference = DefaultInference(n_cpus=n_cpus)
    design = metadata.copy()
    design["pair_id"] = design["pair_id"].astype("category")
    design["comparison_status"] = pd.Categorical(
        design["comparison_status"], categories=["reference", "case"]
    )
    dds = DeseqDataSet(
        counts=counts,
        metadata=design,
        design="~pair_id + comparison_status",
        refit_cooks=True,
        inference=inference,
        quiet=False,
    )
    dds.deseq2()
    statistics = DeseqStats(
        dds,
        contrast=["comparison_status", "case", "reference"],
        alpha=0.05,
        cooks_filter=True,
        independent_filter=True,
        inference=inference,
        quiet=False,
    )
    statistics.summary()
    result = statistics.results_df.reset_index().rename(columns={
        "index": "gene", "baseMean": "base_mean",
        "log2FoldChange": "log2_fold_change",
        "lfcSE": "lfc_standard_error", "stat": "wald_statistic",
        "pvalue": "p_value", "padj": "fdr",
    })
    if "gene" not in result:
        result = result.rename(columns={result.columns[0]: "gene"})
    return result


def descriptive_metrics(counts: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    library = counts.sum(axis=1).replace(0, np.nan)
    cpm = counts.div(library, axis=0) * 1e6
    log_cpm = np.log2(cpm + 1.0)
    case_ids = metadata.index[metadata["comparison_status"].eq("case")]
    reference_ids = metadata.index[metadata["comparison_status"].eq("reference")]
    pair_effects = []
    for pair, table in metadata.groupby("pair_id", sort=True):
        case = table.index[table["comparison_status"].eq("case")]
        reference = table.index[table["comparison_status"].eq("reference")]
        if len(case) != 1 or len(reference) != 1:
            continue
        delta = log_cpm.loc[case[0]] - log_cpm.loc[reference[0]]
        delta.name = (str(table["patient_id"].iloc[0]), str(pair))
        pair_effects.append(delta)
    effects = pd.DataFrame(pair_effects)
    effects.index = pd.MultiIndex.from_tuples(
        effects.index, names=["patient_id", "pair_id"]
    )
    patient_effects = effects.groupby(level="patient_id").median()
    detected = counts.gt(0).groupby(metadata["patient_id"]).max()
    return pd.DataFrame({
        "gene": counts.columns,
        "compatible_mean_cpm": cpm.loc[case_ids].mean(axis=0).to_numpy(),
        "restricted_mean_cpm": cpm.loc[reference_ids].mean(axis=0).to_numpy(),
        "detected_patient_fraction": detected.mean(axis=0).to_numpy(),
        "positive_patient_fraction": patient_effects.gt(0).mean(axis=0).to_numpy(),
        "negative_patient_fraction": patient_effects.lt(0).mean(axis=0).to_numpy(),
        "pair_n": len(effects),
        "patient_n": len(patient_effects),
    })


def leading_table(result: pd.DataFrame, threshold: float) -> pd.DataFrame:
    selected = result[
        result["fdr"].lt(0.05)
        & result["log2_fold_change"].abs().ge(threshold)
        & result["detected_patient_fraction"].ge(0.25)
        & result["patient_direction_consistency"].ge(0.70)
    ].copy()
    selected["direction"] = np.where(
        selected["log2_fold_change"].gt(0),
        "metastasis_compatible_enriched",
        "primary_restricted_enriched",
    )
    return selected.sort_values(
        ["direction", "absolute_wald_statistic", "absolute_log2_fold_change"],
        ascending=[True, False, False], kind="stable",
    )


def tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pair_pseudobulk_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--minimum-total-count", type=int, default=10)
    parser.add_argument("--n-cpus", type=int, default=16)
    parser.add_argument("--absolute-log2-fold-change-thresholds", type=float,
                        nargs="+", default=[0.5, 1.0])
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    counts, metadata, ot_genes = load_pair_pseudobulks(args.pair_pseudobulk_root)
    keep = counts.sum(axis=0).ge(args.minimum_total_count)
    counts = counts.loc[:, keep]
    metrics = descriptive_metrics(counts, metadata)
    result = run_pydeseq2(counts, metadata, args.n_cpus)
    result = result.merge(metrics, on="gene", how="left", validate="one_to_one")
    result = result.merge(ot_genes, on="gene", how="left", validate="one_to_one")
    result["gene_used_in_ot_representation"] = result[
        "gene_used_in_ot_representation"
    ].fillna(False).astype(bool)
    result["direction"] = np.select(
        [result["log2_fold_change"].gt(0), result["log2_fold_change"].lt(0)],
        ["metastasis_compatible_enriched", "primary_restricted_enriched"],
        default="no_direction",
    )
    result["patient_direction_consistency"] = np.where(
        result["log2_fold_change"].ge(0),
        result["positive_patient_fraction"], result["negative_patient_fraction"],
    )
    result["absolute_wald_statistic"] = result["wald_statistic"].abs()
    result["absolute_log2_fold_change"] = result["log2_fold_change"].abs()
    result = result.sort_values(["fdr", "absolute_wald_statistic"],
                                ascending=[True, False], kind="stable")
    result.to_csv(args.output_root / "primary_compatible_vs_restricted_all_genes.csv.gz",
                  index=False, compression="gzip")
    for threshold in sorted(set(args.absolute_log2_fold_change_thresholds)):
        selected = leading_table(result, threshold)
        selected.to_csv(
            args.output_root / f"primary_compatible_vs_restricted_leading_lfc_{tag(threshold)}.csv",
            index=False,
        )
    result[["gene", "wald_statistic"]].dropna().sort_values(
        "wald_statistic", ascending=False
    ).to_csv(args.output_root / "primary_compatible_vs_restricted_all_genes.rnk",
             sep="\t", header=False, index=False)
    counts.to_csv(args.output_root / "pseudobulk_raw_counts.csv.gz", compression="gzip")
    metadata.to_csv(args.output_root / "pseudobulk_sample_metadata.csv")
    report = {
        "contrast": CONTRAST,
        "engine": "PyDESeq2",
        "design": "~pair_id + comparison_status",
        "positive_direction": "putative metastasis-compatible primary malignant cells",
        "negative_direction": "putative primary-restricted primary malignant cells",
        "pair_n": int(metadata["pair_id"].nunique()),
        "patient_n": int(metadata["patient_id"].nunique()),
        "pseudobulk_column_n": len(metadata),
        "tested_gene_n": len(result),
        "all_genes_including_ot_representation_genes": True,
        "metastatic_cells_in_pseudobulk": False,
        "independence_warning": (
            "Pairs sharing a patient or primary sample are correlated; use the planned "
            "patient-aware sensitivity analyses for biological inference."
        ),
    }
    (args.output_root / "pydeseq2_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
