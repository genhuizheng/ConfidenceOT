"""Paired patient-level PyDESeq2 for three robust target-cell contrasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def filter_deg(
    table: pd.DataFrame,
    *,
    maximum_fdr: float,
    minimum_absolute_log2_fold_change: float,
) -> pd.DataFrame:
    result = table[
        table["fdr"].lt(maximum_fdr)
        & table["log2_fold_change"].abs().ge(minimum_absolute_log2_fold_change)
    ].copy()
    result["direction"] = np.where(
        result["log2_fold_change"].gt(0), "case_enriched", "reference_enriched"
    )
    return result.sort_values(
        ["fdr", "log2_fold_change"], ascending=[True, False], kind="stable"
    )


def threshold_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def load_patient_pseudobulk(
    group_root: Path,
    minimum_cells_per_status: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    count_tables = []
    metadata_tables = []
    ot_gene_tables = []
    for ready in sorted(group_root.glob("groups/*/PSEUDOBULK_READY")):
        directory = ready.parent
        counts = pd.read_csv(directory / "pseudobulk_raw_counts.csv.gz", index_col=0)
        metadata = pd.read_csv(directory / "pseudobulk_sample_metadata.csv")
        genes = pd.read_csv(directory / "pseudobulk_gene_metadata.csv.gz")
        if list(counts.index.astype(str)) != list(metadata["sample_id"].astype(str)):
            raise RuntimeError(f"Pseudobulk sample mismatch in {directory}")
        count_tables.append(counts)
        metadata_tables.append(metadata)
        genes["group_directory"] = str(directory)
        ot_gene_tables.append(genes)
    if not count_tables:
        raise RuntimeError(f"No PSEUDOBULK_READY groups found under {group_root}")

    all_genes = sorted(set().union(*(set(table.columns) for table in count_tables)))
    counts = pd.concat(
        [table.reindex(columns=all_genes, fill_value=0) for table in count_tables],
        axis=0,
    )
    metadata = pd.concat(metadata_tables, ignore_index=True).set_index("sample_id")
    metadata = metadata.loc[counts.index]
    required = {"contrast", "comparison_status"}
    missing = required.difference(metadata.columns)
    if missing:
        raise RuntimeError(
            "Pseudobulk metadata predates the three-contrast workflow; rerun "
            f"cancer_metastasis/11_run_robust_target_deg.py. Missing: {sorted(missing)}"
        )
    counts = counts.groupby(
        [metadata["patient_id"], metadata["contrast"], metadata["comparison_status"]],
        sort=True,
    ).sum()
    counts.index.names = ["patient_id", "contrast", "comparison_status"]
    cell_counts = metadata.groupby(
        ["patient_id", "contrast", "comparison_status"], sort=True
    ).agg(cell_n=("cell_n", "sum"), target_group_n=("group_id", "nunique"))
    pair_levels = ["patient_id", "contrast"]
    eligible = cell_counts["cell_n"].ge(minimum_cells_per_status).groupby(level=pair_levels).all()
    complete_status = cell_counts.groupby(level=pair_levels).size().eq(2)
    eligible_pairs = set(eligible.index[eligible & complete_status])
    keep_rows = [
        (patient, contrast) in eligible_pairs
        for patient, contrast, _ in counts.index
    ]
    counts = counts.loc[keep_rows]
    cell_counts = cell_counts.loc[keep_rows]
    counts.index = [
        f"{patient}__{contrast}__{status}" for patient, contrast, status in counts.index
    ]
    sample_metadata = cell_counts.reset_index()
    sample_metadata.index = [
        f"{patient}__{contrast}__{status}"
        for patient, contrast, status in zip(
            sample_metadata["patient_id"], sample_metadata["contrast"],
            sample_metadata["comparison_status"]
        )
    ]
    sample_metadata = sample_metadata.loc[counts.index]

    ot = pd.concat(ot_gene_tables, ignore_index=True)
    ot_summary = ot.groupby("gene", as_index=False).agg(
        used_for_ot_anywhere=("used_for_ot", "max"),
        evaluated_group_n=("group_directory", "nunique"),
    )
    return counts.astype(np.int64), sample_metadata, ot_summary


def run_pydeseq2(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    n_cpus: int,
) -> pd.DataFrame:
    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.default_inference import DefaultInference
        from pydeseq2.ds import DeseqStats
    except ImportError as error:
        raise RuntimeError("pydeseq2>=0.5,<0.6 is required") from error

    inference = DefaultInference(n_cpus=n_cpus)
    metadata = metadata.copy()
    metadata["patient_id"] = metadata["patient_id"].astype("category")
    metadata["comparison_status"] = pd.Categorical(
        metadata["comparison_status"], categories=["reference", "case"],
    )
    dds = DeseqDataSet(
        counts=counts,
        metadata=metadata,
        design="~patient_id + comparison_status",
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
        "index": "gene",
        "baseMean": "base_mean",
        "log2FoldChange": "log2_fold_change",
        "lfcSE": "log2_fold_change_se",
        "pvalue": "p_value",
        "padj": "fdr",
        "stat": "wald_statistic",
    })
    if "gene" not in result:
        result = result.rename(columns={result.columns[0]: "gene"})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group_output_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--minimum-cells-per-patient-status", type=int, default=20)
    parser.add_argument("--minimum-total-count", type=int, default=10)
    parser.add_argument("--n-cpus", type=int, default=16)
    parser.add_argument("--maximum-fdr", type=float, default=0.05)
    parser.add_argument(
        "--absolute-log2-fold-change-thresholds",
        type=float,
        nargs="+",
        default=[0.5, 1.0],
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    counts, metadata, ot_genes = load_patient_pseudobulk(
        args.group_output_root, args.minimum_cells_per_patient_status
    )
    keep = counts.sum(axis=0).ge(args.minimum_total_count)
    counts = counts.loc[:, keep]
    metadata.to_csv(args.output_dir / "pseudobulk_sample_metadata.csv")
    counts.to_csv(args.output_dir / "pseudobulk_raw_counts.csv.gz", compression="gzip")

    contrast_reports = []
    contrasts_root = args.output_dir / "contrasts"
    contrasts_root.mkdir(exist_ok=True)
    for contrast, contrast_metadata in metadata.groupby("contrast", sort=True, observed=True):
        contrast_metadata = contrast_metadata.copy()
        contrast_counts = counts.loc[contrast_metadata.index]
        destination = contrasts_root / str(contrast)
        destination.mkdir(exist_ok=True)
        result = run_pydeseq2(
            contrast_counts, contrast_metadata, n_cpus=args.n_cpus
        ).merge(ot_genes, on="gene", how="left", validate="one_to_one")
        result["used_for_ot_anywhere"] = result["used_for_ot_anywhere"].fillna(False)
        result.to_csv(destination / "pydeseq2_all_gene_discovery.csv", index=False)
        non_ot = result[~result["used_for_ot_anywhere"]].copy()
        non_ot.to_csv(destination / "pydeseq2_non_ot_gene_validation.csv", index=False)
        effect_filter_reports = []
        for threshold in sorted(set(args.absolute_log2_fold_change_thresholds)):
            if threshold < 0:
                raise ValueError("Absolute log2 fold-change thresholds must be non-negative")
            filtered = filter_deg(
                result,
                maximum_fdr=args.maximum_fdr,
                minimum_absolute_log2_fold_change=threshold,
            )
            filtered_non_ot = filter_deg(
                non_ot,
                maximum_fdr=args.maximum_fdr,
                minimum_absolute_log2_fold_change=threshold,
            )
            tag = threshold_tag(threshold)
            filtered.to_csv(
                destination / f"pydeseq2_all_gene_fdr_005_abs_log2fc_{tag}.csv",
                index=False,
            )
            filtered_non_ot.to_csv(
                destination / f"pydeseq2_non_ot_gene_fdr_005_abs_log2fc_{tag}.csv",
                index=False,
            )
            effect_filter_reports.append({
                "maximum_fdr": args.maximum_fdr,
                "minimum_absolute_log2_fold_change": threshold,
                "all_gene_n": len(filtered),
                "non_ot_gene_n": len(filtered_non_ot),
                "non_ot_case_enriched_n": int(
                    filtered_non_ot["direction"].eq("case_enriched").sum()
                ),
                "non_ot_reference_enriched_n": int(
                    filtered_non_ot["direction"].eq("reference_enriched").sum()
                ),
            })
        for label, table in (("all_gene", result), ("non_ot_gene", non_ot)):
            table[["gene", "wald_statistic"]].dropna().sort_values(
                "wald_statistic", ascending=False
            ).to_csv(destination / f"pydeseq2_{label}_wald.rnk",
                     sep="\t", header=False, index=False)
        contrast_metadata.to_csv(destination / "pseudobulk_sample_metadata.csv")
        contrast_counts.to_csv(destination / "pseudobulk_raw_counts.csv.gz", compression="gzip")
        contrast_reports.append({
            "contrast": str(contrast),
            "patient_n": int(contrast_metadata["patient_id"].nunique()),
            "pseudobulk_sample_n": len(contrast_metadata),
            "tested_gene_n": len(result),
            "non_ot_validation_gene_n": len(non_ot),
            "fdr_005_all_gene_n": int(result["fdr"].lt(0.05).sum()),
            "fdr_005_non_ot_gene_n": int(non_ot["fdr"].lt(0.05).sum()),
            "effect_filters": effect_filter_reports,
        })
    report = {
        "deg_engine": "PyDESeq2",
        "design": "~patient_id + comparison_status",
        "contrasts": contrast_reports,
        "count_input": "raw integer counts summed across target groups within patient, contrast, and comparison status",
        "gsea_rank": "PyDESeq2 Wald statistic",
        "gsea_prefilter": "none; all finite Wald statistics are retained for preranked GSEA",
    }
    (args.output_dir / "pydeseq2_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    print(f"PyDESeq2 outputs: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
