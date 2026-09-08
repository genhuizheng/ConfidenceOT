"""Summarize primary malignant-cell DEG robustness across source caps."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def cap_slug(cap: float) -> str:
    return f"{cap:.2f}".replace(".", "p")


def pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def weighted_mean(table: pd.DataFrame, value: str, weight: str) -> float:
    use = table[[value, weight]].dropna()
    return float(np.average(use[value], weights=use[weight])) if len(use) else np.nan


def load_metrics(root: Path) -> pd.DataFrame:
    tables = []
    for path in sorted(root.glob("*/scope_malignant/*/pair_metrics.csv")):
        table = pd.read_csv(path)
        tables.append(table[table["method"].eq("M4-E")])
    if not tables:
        raise FileNotFoundError(f"No M4-E pair metrics under {root}")
    return pd.concat(tables, ignore_index=True)


def load_pseudobulk_metadata(root: Path) -> tuple[pd.DataFrame, int, int]:
    ready = sorted(root.glob("groups/*/PSEUDOBULK_READY"))
    skipped = sorted(root.glob("groups/*/PSEUDOBULK_SKIPPED"))
    tables = [pd.read_csv(path.parent / "pseudobulk_sample_metadata.csv") for path in ready]
    if not tables:
        raise RuntimeError(f"No DEG-ready pseudobulks under {root}")
    return pd.concat(tables, ignore_index=True), len(ready), len(skipped)


def load_deg(root: Path, cap: float) -> pd.DataFrame:
    path = root / "primary_compatible_vs_restricted_all_genes.csv.gz"
    table = pd.read_csv(path)
    table["source_cap"] = cap
    return table


def selected(table: pd.DataFrame, threshold: float) -> pd.Series:
    return (
        table["fdr"].lt(0.05)
        & table["log2_fold_change"].abs().ge(threshold)
        & table["detected_patient_fraction"].ge(0.25)
        & table["patient_direction_consistency"].ge(0.70)
    )


def summarize_cap(
    cap: float, metrics: pd.DataFrame, metadata: pd.DataFrame,
    ready_n: int, skipped_n: int, deg: pd.DataFrame,
) -> dict[str, object]:
    by_state = metadata.set_index("sample_id")
    case = by_state[by_state["comparison_status"].eq("case")]
    reference = by_state[by_state["comparison_status"].eq("reference")]
    raw_source = metrics["source_raw_rejection_rate"]
    row: dict[str, object] = {
        "source_cap": cap,
        "pair_n": int(metrics["pair_id"].nunique()),
        "patient_n": int(metrics["patient_id"].nunique()),
        "deg_ready_pair_n": ready_n,
        "deg_skipped_pair_n": skipped_n,
        "source_weighted_rejection_rate": weighted_mean(
            metrics, "source_final_rejection_rate", "source_analyzed_n"
        ),
        "target_weighted_rejection_rate": weighted_mean(
            metrics, "target_final_rejection_rate", "target_analyzed_n"
        ),
        "source_cap_binding_pair_fraction": float(raw_source.ge(cap - 1e-9).mean()),
        "compatible_cell_n_median": float(case["cell_n"].median()),
        "restricted_cell_n_median": float(reference["cell_n"].median()),
        "compatible_cell_fraction_median": float(
            case.set_index("pair_id")["cell_n"].div(
                metadata.groupby("pair_id")["cell_n"].sum()
            ).median()
        ),
        "compatible_counts_per_cell_median": float(
            (case["library_size"] / case["cell_n"]).median()
        ),
        "restricted_counts_per_cell_median": float(
            (reference["library_size"] / reference["cell_n"]).median()
        ),
        "tested_gene_n": len(deg),
    }
    for threshold in (0.5, 1.0):
        use = deg[selected(deg, threshold)]
        tag = str(threshold).replace(".", "p")
        row[f"compatible_enriched_lfc_{tag}_n"] = int(
            use["log2_fold_change"].gt(0).sum()
        )
        row[f"restricted_enriched_lfc_{tag}_n"] = int(
            use["log2_fold_change"].lt(0).sum()
        )
    return row


def gene_stability(long: pd.DataFrame, caps: list[float]) -> pd.DataFrame:
    long = long.copy()
    long["strict_lfc_0p5"] = selected(long, 0.5)
    long["strict_lfc_1p0"] = selected(long, 1.0)
    rows = []
    for gene, table in long.groupby("gene", sort=False):
        positive = table["log2_fold_change"].gt(0)
        negative = table["log2_fold_change"].lt(0)
        strict05 = table["strict_lfc_0p5"]
        strict10 = table["strict_lfc_1p0"]
        rows.append({
            "gene": gene,
            "cap_n": int(table["source_cap"].nunique()),
            "median_log2_fold_change": float(table["log2_fold_change"].median()),
            "minimum_log2_fold_change": float(table["log2_fold_change"].min()),
            "maximum_log2_fold_change": float(table["log2_fold_change"].max()),
            "positive_cap_n": int(positive.sum()),
            "negative_cap_n": int(negative.sum()),
            "compatible_strict_lfc_0p5_cap_n": int((strict05 & positive).sum()),
            "restricted_strict_lfc_0p5_cap_n": int((strict05 & negative).sum()),
            "compatible_strict_lfc_1p0_cap_n": int((strict10 & positive).sum()),
            "restricted_strict_lfc_1p0_cap_n": int((strict10 & negative).sum()),
            "minimum_patient_direction_consistency": float(
                table["patient_direction_consistency"].min()
            ),
            "maximum_fdr": float(table["fdr"].max()),
        })
    result = pd.DataFrame(rows)
    n = len(caps)
    result["same_positive_direction_all_caps"] = result["positive_cap_n"].eq(n)
    result["same_negative_direction_all_caps"] = result["negative_cap_n"].eq(n)
    result["strict_lfc_0p5_all_caps"] = (
        result["compatible_strict_lfc_0p5_cap_n"].eq(n)
        | result["restricted_strict_lfc_0p5_cap_n"].eq(n)
    )
    result["strict_lfc_1p0_all_caps"] = (
        result["compatible_strict_lfc_1p0_cap_n"].eq(n)
        | result["restricted_strict_lfc_1p0_cap_n"].eq(n)
    )
    return result.sort_values(
        ["strict_lfc_0p5_all_caps", "minimum_patient_direction_consistency",
         "median_log2_fold_change"],
        ascending=[False, False, False], kind="stable",
    )


def plot_summary(summary: pd.DataFrame, output: Path) -> None:
    plt = pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes[0, 0].plot(summary["source_cap"], summary["source_weighted_rejection_rate"],
                    marker="o", label="Primary")
    axes[0, 0].plot(summary["source_cap"], summary["target_weighted_rejection_rate"],
                    marker="o", label="Metastasis")
    axes[0, 0].set(ylabel="Weighted rejection rate", xlabel="Source cap")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].plot(summary["source_cap"], summary["compatible_cell_fraction_median"],
                    marker="o", color="#2878B5")
    axes[0, 1].set(ylabel="Median compatible primary-cell fraction", xlabel="Source cap")
    axes[1, 0].plot(summary["source_cap"], summary["source_cap_binding_pair_fraction"],
                    marker="o", color="#C82423")
    axes[1, 0].set(ylabel="Fraction of pairs hitting source cap", xlabel="Source cap")
    axes[1, 1].plot(summary["source_cap"], summary["compatible_enriched_lfc_0p5_n"],
                    marker="o", label="Compatible enriched")
    axes[1, 1].plot(summary["source_cap"], summary["restricted_enriched_lfc_0p5_n"],
                    marker="o", label="Restricted enriched")
    axes[1, 1].set(ylabel="Strict DEG count (|LFC| >= 0.5)", xlabel="Source cap")
    axes[1, 1].legend(frameon=False)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.savefig(output / "source_cap_sensitivity_summary.png", dpi=300)
    fig.savefig(output / "source_cap_sensitivity_summary.pdf")
    plt.close(fig)


def plot_volcanoes(long: pd.DataFrame, caps: list[float], output: Path) -> None:
    plt = pyplot()
    ncols = 3
    nrows = math.ceil(len(caps) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for index, cap in enumerate(caps):
        table = long[long["source_cap"].eq(cap)].copy()
        sig = selected(table, 0.5)
        colors = np.where(
            sig & table["log2_fold_change"].gt(0), "#D62728",
            np.where(sig & table["log2_fold_change"].lt(0), "#2878B5", "#BDBDBD"),
        )
        x = table["log2_fold_change"].clip(-8, 8)
        y = -np.log10(table["fdr"].clip(lower=1e-300))
        ax = axes.flat[index]
        ax.scatter(x, y, c=colors, s=4, alpha=0.65, linewidths=0)
        ax.axvline(0, color="black", lw=0.6)
        ax.set(title=f"Source cap {cap:.2f}", xlabel="log2FC: compatible / restricted",
               ylabel="-log10(FDR)")
        single, single_ax = plt.subplots(figsize=(6, 5))
        single_ax.scatter(x, y, c=colors, s=5, alpha=0.65, linewidths=0)
        single_ax.axvline(0, color="black", lw=0.6)
        single_ax.set(title=f"Primary DEG, source cap {cap:.2f}",
                      xlabel="log2FC: compatible / restricted", ylabel="-log10(FDR)")
        single.tight_layout()
        single.savefig(output / f"volcano_source_cap_{cap_slug(cap)}.png", dpi=300)
        single.savefig(output / f"volcano_source_cap_{cap_slug(cap)}.pdf")
        plt.close(single)
    for ax in axes.flat[len(caps):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output / "volcano_all_source_caps.png", dpi=300)
    fig.savefig(output / "volcano_all_source_caps.pdf")
    plt.close(fig)


def plot_gene_heatmap(long: pd.DataFrame, stability: pd.DataFrame,
                      caps: list[float], output: Path) -> None:
    plt = pyplot()
    candidates = stability[
        stability["strict_lfc_0p5_all_caps"]
    ].copy()
    if candidates.empty:
        candidates = stability[
            stability[["compatible_strict_lfc_0p5_cap_n",
                       "restricted_strict_lfc_0p5_cap_n"]].max(axis=1).ge(
                math.ceil(0.6 * len(caps))
            )
        ].copy()
    candidates["effect"] = candidates["median_log2_fold_change"].abs()
    genes = candidates.nlargest(40, "effect")["gene"].tolist()
    if not genes:
        return
    matrix = long[long["gene"].isin(genes)].pivot(
        index="gene", columns="source_cap", values="log2_fold_change"
    ).reindex(index=genes, columns=caps)
    fig, ax = plt.subplots(figsize=(7, max(5, 0.23 * len(genes))))
    image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    ax.set_xticks(range(len(caps)), [f"{cap:.2f}" for cap in caps])
    ax.set_yticks(range(len(genes)), genes, fontsize=7)
    ax.set(xlabel="Source rejection cap", ylabel="Gene")
    fig.colorbar(image, ax=ax, label="log2FC: compatible / restricted")
    fig.tight_layout()
    fig.savefig(output / "cross_cap_gene_effect_heatmap.png", dpi=300)
    fig.savefig(output / "cross_cap_gene_effect_heatmap.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ot_base", type=Path)
    parser.add_argument("pseudobulk_base", type=Path)
    parser.add_argument("deg_base", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--source-caps", type=float, nargs="+",
                        default=[0.50, 0.70, 0.75, 0.85, 0.90, 0.95])
    parser.add_argument("--target-cap", type=float, default=0.95)
    args = parser.parse_args()
    caps = sorted(set(args.source_caps))
    args.output_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    deg_tables = []
    for cap in caps:
        slug = cap_slug(cap)
        metrics = load_metrics(args.ot_base / f"source_cap_{slug}")
        metadata, ready_n, skipped_n = load_pseudobulk_metadata(
            args.pseudobulk_base / f"source_cap_{slug}"
        )
        deg = load_deg(args.deg_base / f"source_cap_{slug}", cap)
        summary_rows.append(summarize_cap(cap, metrics, metadata, ready_n, skipped_n, deg))
        deg_tables.append(deg)
    summary = pd.DataFrame(summary_rows).sort_values("source_cap")
    long = pd.concat(deg_tables, ignore_index=True)
    stability = gene_stability(long, caps)
    summary.to_csv(args.output_root / "source_cap_summary.csv", index=False)
    long.to_csv(args.output_root / "all_genes_all_caps_long.csv.gz", index=False,
                compression="gzip")
    stability.to_csv(args.output_root / "gene_cross_cap_stability.csv", index=False)
    stability[stability["strict_lfc_0p5_all_caps"]].to_csv(
        args.output_root / "genes_strictly_significant_at_all_caps_lfc_0p5.csv", index=False
    )
    stability[stability["strict_lfc_1p0_all_caps"]].to_csv(
        args.output_root / "genes_strictly_significant_at_all_caps_lfc_1p0.csv", index=False
    )
    plot_summary(summary, args.output_root)
    plot_volcanoes(long, caps, args.output_root)
    plot_gene_heatmap(long, stability, caps, args.output_root)
    report = {
        "source_caps": caps,
        "target_cap_fixed": args.target_cap,
        "contrast": "primary malignant retained versus primary malignant rejected within each exact primary-metastasis pair",
        "positive_log2fc": "putative metastasis-compatible primary malignant cells",
        "negative_log2fc": "putative primary-restricted primary malignant cells",
        "deg_engine": "PyDESeq2",
        "design": "~pair_id + comparison_status",
        "all_genes_including_ot_representation_genes": True,
        "strict_gene_definition": "FDR < 0.05, |log2FC| >= threshold, detected-patient fraction >= 0.25, patient direction consistency >= 0.70",
        "cap_selection_rule": "No cap is selected by maximizing metastasis-related DEG or pathway yield; interpret only jointly with cap binding, group size, QC, and cross-cap direction stability.",
        "pair_dependence_warning": "Repeated primary samples across metastatic partners are correlated; patient direction consistency is reported and confirmatory inference requires a patient-aware model or independent cohort.",
        "strict_lfc_0p5_all_caps_gene_n": int(stability["strict_lfc_0p5_all_caps"].sum()),
        "strict_lfc_1p0_all_caps_gene_n": int(stability["strict_lfc_1p0_all_caps"].sum()),
    }
    (args.output_root / "source_cap_deg_sensitivity_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
