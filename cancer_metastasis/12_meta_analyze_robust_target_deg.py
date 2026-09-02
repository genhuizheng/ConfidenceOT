"""Patient-level meta-DEG for cap-robust metastatic malignant-cell states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def bh_adjust(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.full(values.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return result
    order = valid[np.argsort(values[valid], kind="stable")]
    adjusted = values[order] * len(order) / np.arange(1, len(order) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result[order] = np.minimum(adjusted, 1.0)
    return result


def meta_table(patient_contrasts: pd.DataFrame, minimum_patients: int) -> pd.DataFrame:
    rows = []
    for gene, table in patient_contrasts.groupby("gene", sort=False):
        values = table["mean_log_expression_difference"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        patient_n = len(values)
        p_value = np.nan
        if patient_n >= minimum_patients and np.any(values != 0):
            try:
                p_value = float(stats.wilcoxon(values, zero_method="wilcox").pvalue)
            except ValueError:
                p_value = 1.0
        median = float(np.median(values)) if patient_n else np.nan
        sign = np.sign(median)
        consistency = float(np.mean(np.sign(values) == sign)) if sign != 0 else 0.0
        rows.append({
            "gene": gene,
            "patient_n": patient_n,
            "median_mean_log_expression_difference": median,
            "mean_mean_log_expression_difference": float(np.mean(values)) if patient_n else np.nan,
            "direction_consistency": consistency,
            "patient_level_wilcoxon_p_value": p_value,
        })
    result = pd.DataFrame(rows)
    result["fdr"] = bh_adjust(result["patient_level_wilcoxon_p_value"].to_numpy())
    signed_significance = -np.log10(np.maximum(result["fdr"], 1e-300))
    result["gsea_rank_score"] = (
        np.sign(result["median_mean_log_expression_difference"])
        * signed_significance
        * result["direction_consistency"]
    )
    result["inference_unit"] = "patient"
    result["interpretation"] = "robust_rejected_minus_robust_retained_target_malignant_state"
    return result.sort_values(
        ["fdr", "direction_consistency", "median_mean_log_expression_difference"],
        ascending=[True, False, False], kind="stable",
    )


def volcano(table: pd.DataFrame, output: Path) -> None:
    plot = table[np.isfinite(table["fdr"])].copy()
    x = plot["median_mean_log_expression_difference"].to_numpy()
    y = -np.log10(np.maximum(plot["fdr"].to_numpy(), 1e-300))
    significant = (plot["fdr"] < 0.05) & (plot["direction_consistency"] >= 0.70)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(x[~significant], y[~significant], s=8, c="0.75", alpha=0.55, linewidths=0)
    ax.scatter(x[significant], y[significant], s=12, c=np.where(x[significant] > 0, "#c44e52", "#4c72b0"), alpha=0.8, linewidths=0)
    candidates = plot.assign(abs_rank=plot["gsea_rank_score"].abs()).nlargest(14, "abs_rank")
    for _, row in candidates.iterrows():
        ax.text(
            row["median_mean_log_expression_difference"],
            -np.log10(max(row["fdr"], 1e-300)),
            row["gene"], fontsize=7,
        )
    ax.axvline(0, color="0.3", linewidth=1)
    ax.axhline(-np.log10(0.05), color="0.5", linewidth=1, linestyle="--")
    ax.set(
        xlabel="Median patient-level mean log-expression difference\n(rejected − retained)",
        ylabel="−log10(FDR)",
        title="Cap-robust metastatic malignant-state meta-DEG",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "meta_deg_volcano.png", dpi=240, bbox_inches="tight")
    fig.savefig(output / "meta_deg_volcano.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group_output_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--minimum-patients", type=int, default=5)
    parser.add_argument("--minimum-site-patients", type=int, default=3)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = []
    contrasts = []
    for path in sorted(args.group_output_root.glob("groups/*/diagnostics.json")):
        with path.open(encoding="utf-8") as handle:
            record = json.load(handle)
        record["group_directory"] = str(path.parent)
        diagnostics.append(record)
        contrast_path = path.parent / "non_ot_gene_contrasts.csv.gz"
        if record.get("status") == "complete" and contrast_path.is_file():
            contrasts.append(pd.read_csv(contrast_path))
    diagnostics_table = pd.DataFrame(diagnostics)
    diagnostics_table.to_csv(args.output_dir / "group_deg_diagnostics.csv", index=False)
    if not contrasts:
        raise RuntimeError("No completed non-OT gene contrast tables were found")
    long = pd.concat(contrasts, ignore_index=True)
    long.to_csv(
        args.output_dir / "non_ot_gene_contrasts_all_groups.csv.gz",
        index=False, compression="gzip",
    )

    # Each patient receives equal weight. Multiple target samples from one
    # patient are median-collapsed before the pan-metastatic signed-rank test.
    patient = long.groupby(["patient_id", "gene"], as_index=False).agg(
        mean_log_expression_difference=("mean_log_expression_difference", "median"),
        target_group_n=("group_id", "nunique"),
    )
    overall = meta_table(patient, args.minimum_patients)
    overall.to_csv(args.output_dir / "meta_deg_all_targets_non_ot.csv", index=False)
    overall[["gene", "gsea_rank_score"]].dropna().to_csv(
        args.output_dir / "meta_deg_all_targets_non_ot.rnk",
        sep="\t", header=False, index=False,
    )

    site_tables = []
    site_counts = (
        long[["patient_id", "target_sample"]].drop_duplicates()
        .groupby("target_sample")["patient_id"].nunique()
    )
    rank_root = args.output_dir / "site_gsea_ranks"
    rank_root.mkdir(exist_ok=True)
    for site, patient_n in site_counts.items():
        if patient_n < args.minimum_site_patients:
            continue
        use = long[long["target_sample"].eq(site)].groupby(
            ["patient_id", "gene"], as_index=False
        ).agg(mean_log_expression_difference=("mean_log_expression_difference", "median"))
        table = meta_table(use, args.minimum_site_patients)
        table.insert(0, "target_sample", site)
        site_tables.append(table)
        safe_site = "_".join(str(site).replace("/", "_").split())
        table[["gene", "gsea_rank_score"]].dropna().to_csv(
            rank_root / f"{safe_site}.rnk", sep="\t", header=False, index=False
        )
    if site_tables:
        pd.concat(site_tables, ignore_index=True).to_csv(
            args.output_dir / "meta_deg_by_target_site_non_ot.csv.gz",
            index=False, compression="gzip",
        )
    pd.DataFrame({"target_sample": site_counts.index, "patient_n": site_counts.values}).to_csv(
        args.output_dir / "target_site_patient_counts.csv", index=False
    )
    volcano(overall, args.output_dir)
    report = {
        "completed_group_n": int((diagnostics_table["status"] == "complete").sum()),
        "skipped_group_n": int((diagnostics_table["status"] != "complete").sum()),
        "unique_patient_n": int(long["patient_id"].nunique()),
        "target_group_n": int(long["group_id"].nunique()),
        "tested_non_ot_gene_n": len(overall),
        "fdr_005_direction_consistency_070_gene_n": int(
            ((overall["fdr"] < 0.05) & (overall["direction_consistency"] >= 0.70)).sum()
        ),
        "primary_inference_unit": "patient; multiple metastatic targets median-collapsed within patient",
        "primary_gene_scope": "genes not used in either baseline or source090 winner OT representation",
        "biological_claim": "genes enriched or depleted in cap-robust target malignant cells not explained by the selected primary expression state",
    }
    (args.output_dir / "meta_deg_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    print(f"Meta-DEG outputs: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
