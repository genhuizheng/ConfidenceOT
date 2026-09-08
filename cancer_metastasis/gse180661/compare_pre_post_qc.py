"""Compare matched ConfidenceOT, DEG, and GSEA results before and after cell QC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def completed_pair_directories(root: Path) -> dict[str, Path]:
    return {
        path.parents[2].name: path.parent
        for path in root.glob("*/scope_malignant/*/SUCCESS")
    }


def correlation(left: pd.Series, right: pd.Series) -> dict[str, float]:
    valid = left.notna() & right.notna()
    if valid.sum() < 3:
        return {"n": int(valid.sum()), "pearson": np.nan, "spearman": np.nan}
    return {
        "n": int(valid.sum()),
        "pearson": float(pearsonr(left[valid], right[valid]).statistic),
        "spearman": float(spearmanr(left[valid], right[valid]).statistic),
    }


def compare_cells(
    pre_root: Path, post_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pre_dirs = completed_pair_directories(pre_root)
    post_dirs = completed_pair_directories(post_root)
    pair_rows = []
    gate_rows = []
    qc_rows = []
    for pair in sorted(set(pre_dirs) & set(post_dirs)):
        pre = pd.read_csv(pre_dirs[pair] / "cell_confidence.csv")
        post = pd.read_csv(post_dirs[pair] / "cell_confidence.csv")
        pre = pre[pre["method"].eq("M4-E")]
        post = post[post["method"].eq("M4-E")]
        qc = pd.read_csv(post_dirs[pair] / "cell_qc.csv.gz")
        for side in ("source", "target"):
            pre_side = pre[pre["side"].eq(side)].copy()
            post_side = post[post["side"].eq(side)].copy()
            pre_index = pre_side.set_index("observation_id")
            post_index = post_side.set_index("observation_id")
            shared = pre_index.index.intersection(post_index.index)
            agreement = (
                pre_index.loc[shared, "retained"].astype(bool).to_numpy()
                == post_index.loc[shared, "retained"].astype(bool).to_numpy()
            )
            qc_side = qc[qc["side"].eq(side)]
            pair_rows.append({
                "pair_id": pair,
                "side": side,
                "pre_analyzed_n": len(pre_side),
                "post_analyzed_n": len(post_side),
                "qc_evaluated_n": len(qc_side),
                "qc_pass_n": int(qc_side["qc_pass"].astype(bool).sum()),
                "qc_retained_fraction": float(qc_side["qc_pass"].astype(bool).mean()),
                "shared_cell_n": len(shared),
                "gate_agreement_fraction": float(np.mean(agreement)) if len(shared) else np.nan,
                "pre_rejection_rate": float(pre_side["rejected"].astype(bool).mean()),
                "post_rejection_rate": float(post_side["rejected"].astype(bool).mean()),
            })
            if len(shared):
                frame = pd.DataFrame({
                    "pair_id": pair,
                    "side": side,
                    "observation_id": shared,
                    "pre_retained": pre_index.loc[shared, "retained"].astype(bool).to_numpy(),
                    "post_retained": post_index.loc[shared, "retained"].astype(bool).to_numpy(),
                    "pre_score": pre_index.loc[shared, "normalized_rejection_score"].to_numpy(),
                    "post_score": post_index.loc[shared, "normalized_rejection_score"].to_numpy(),
                })
                gate_rows.append(frame)
            qc_index = qc_side.set_index("observation_id")
            for phase, classified in (("pre_qc", pre_index), ("post_qc", post_index)):
                ids = classified.index.intersection(qc_index.index)
                if not len(ids):
                    continue
                values = qc_index.loc[ids, [
                    "total_counts", "n_genes_by_counts",
                    "pct_counts_mitochondrial", "qc_pass",
                ]].reset_index()
                values.insert(0, "phase", phase)
                values.insert(0, "side", side)
                values.insert(0, "pair_id", pair)
                values["state"] = np.where(
                    classified.loc[ids, "retained"].astype(bool).to_numpy(),
                    "retained", "rejected",
                )
                qc_rows.append(values)
    return (
        pd.DataFrame(pair_rows),
        pd.concat(gate_rows, ignore_index=True),
        pd.concat(qc_rows, ignore_index=True),
    )


def load_deg(root: Path, suffix: str) -> pd.DataFrame:
    table = pd.read_csv(root / "primary_compatible_vs_restricted_all_genes.csv.gz")
    return table[["gene", "log2_fold_change", "wald_statistic", "fdr"]].rename(columns={
        "log2_fold_change": f"log2_fold_change_{suffix}",
        "wald_statistic": f"wald_statistic_{suffix}",
        "fdr": f"fdr_{suffix}",
    })


def load_gsea(root: Path, suffix: str) -> pd.DataFrame:
    table = pd.read_csv(root / "primary_compatible_vs_restricted_gsea_all.csv.gz")
    return table[["collection", "pathway", "NES", "fdr"]].rename(columns={
        "NES": f"NES_{suffix}", "fdr": f"fdr_{suffix}",
    })


def scatter(left, right, xlabel, ylabel, output, labels=None) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 6.6))
    axis.scatter(left, right, s=8, alpha=0.35, color="#476D9C", linewidths=0)
    lower = float(np.nanmin([left.min(), right.min()]))
    upper = float(np.nanmax([left.max(), right.max()]))
    axis.plot([lower, upper], [lower, upper], "--", color="#555555", linewidth=0.8)
    if labels is not None:
        distance = np.abs(left - right)
        for index in np.argsort(-distance.to_numpy())[:12]:
            axis.annotate(str(labels.iloc[index]), (left.iloc[index], right.iloc[index]), fontsize=7)
    axis.set(xlabel=xlabel, ylabel=ylabel)
    figure.tight_layout()
    figure.savefig(output.with_suffix(".png"), dpi=300)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def plot_qc(qc: pd.DataFrame, side: str, output: Path) -> pd.DataFrame:
    metrics = [
        ("total_counts", "Median nCount", True),
        ("n_genes_by_counts", "Median nFeature", True),
        ("pct_counts_mitochondrial", "Median mitochondrial %", False),
    ]
    pair_medians = (
        qc[qc["side"].eq(side)]
        .groupby(["pair_id", "phase", "state"], as_index=False)[
            [item[0] for item in metrics]
        ]
        .median()
    )
    order = [
        ("pre_qc", "rejected"), ("pre_qc", "retained"),
        ("post_qc", "rejected"), ("post_qc", "retained"),
    ]
    labels = ["Pre\nRejected", "Pre\nRetained", "Post\nRejected", "Post\nRetained"]
    colors = ["#4C78A8", "#E45756", "#72A0C1", "#F28E85"]
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))
    for axis, (metric, ylabel, log_scale) in zip(axes, metrics):
        values = [
            pair_medians.loc[
                pair_medians["phase"].eq(phase) & pair_medians["state"].eq(state),
                metric,
            ].dropna().to_numpy()
            for phase, state in order
        ]
        boxes = axis.boxplot(values, tick_labels=labels, showfliers=False, patch_artist=True)
        for patch, color in zip(boxes["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        if log_scale:
            axis.set_yscale("log")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(f"{side.title()} malignant-cell QC before and after filtering")
    figure.tight_layout()
    figure.savefig(output.with_suffix(".png"), dpi=300)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)
    return pair_medians


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pre_ot_root", type=Path)
    parser.add_argument("post_ot_root", type=Path)
    parser.add_argument("pre_deg_root", type=Path)
    parser.add_argument("post_deg_root", type=Path)
    parser.add_argument("pre_gsea_root", type=Path)
    parser.add_argument("post_gsea_root", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    pair, gates, qc = compare_cells(args.pre_ot_root, args.post_ot_root)
    pair.to_csv(args.output_root / "pair_level_pre_post_qc.csv", index=False)
    gates.to_csv(args.output_root / "shared_cell_gate_changes.csv.gz", index=False,
                 compression="gzip")
    qc.to_csv(args.output_root / "cell_qc_by_gate.csv.gz", index=False,
              compression="gzip")
    source_qc = plot_qc(qc, "source", args.output_root / "03_primary_qc_pre_vs_post")
    target_qc = plot_qc(qc, "target", args.output_root / "04_metastasis_qc_pre_vs_post")
    pd.concat([
        source_qc.assign(side="source"), target_qc.assign(side="target")
    ], ignore_index=True).to_csv(
        args.output_root / "pair_median_qc_by_gate.csv", index=False
    )

    deg = load_deg(args.pre_deg_root, "pre_qc").merge(
        load_deg(args.post_deg_root, "post_qc"), on="gene", validate="one_to_one"
    )
    deg.to_csv(args.output_root / "deg_pre_post_qc.csv.gz", index=False,
               compression="gzip")
    scatter(
        deg["log2_fold_change_pre_qc"], deg["log2_fold_change_post_qc"],
        "Pre-QC log2 fold change", "Post-QC log2 fold change",
        args.output_root / "01_deg_log2fc_pre_vs_post_qc", deg["gene"],
    )

    gsea = load_gsea(args.pre_gsea_root, "pre_qc").merge(
        load_gsea(args.post_gsea_root, "post_qc"),
        on=["collection", "pathway"], validate="one_to_one",
    )
    gsea.to_csv(args.output_root / "gsea_pre_post_qc.csv.gz", index=False,
                compression="gzip")
    hallmark = gsea[gsea["collection"].eq("Hallmark")].reset_index(drop=True)
    hallmark.to_csv(args.output_root / "hallmark_pre_post_qc.csv", index=False)
    scatter(
        hallmark["NES_pre_qc"], hallmark["NES_post_qc"],
        "Pre-QC Hallmark NES", "Post-QC Hallmark NES",
        args.output_root / "02_hallmark_nes_pre_vs_post_qc", hallmark["pathway"],
    )

    deg_cor = correlation(deg["log2_fold_change_pre_qc"], deg["log2_fold_change_post_qc"])
    gsea_cor = correlation(gsea["NES_pre_qc"], gsea["NES_post_qc"])
    hallmark_cor = correlation(hallmark["NES_pre_qc"], hallmark["NES_post_qc"])
    pre_significant = set(gsea.loc[gsea["fdr_pre_qc"].lt(0.05), "pathway"])
    post_significant = set(gsea.loc[gsea["fdr_post_qc"].lt(0.05), "pathway"])
    report = {
        "pair_n": int(pair["pair_id"].nunique()),
        "median_qc_retained_fraction_by_side": pair.groupby("side")[
            "qc_retained_fraction"
        ].median().to_dict(),
        "median_gate_agreement_by_side": pair.groupby("side")[
            "gate_agreement_fraction"
        ].median().to_dict(),
        "deg_log2fc_correlation": deg_cor,
        "gsea_nes_correlation": gsea_cor,
        "hallmark_nes_correlation": hallmark_cor,
        "pre_qc_significant_pathway_n": len(pre_significant),
        "post_qc_significant_pathway_n": len(post_significant),
        "significant_pathway_jaccard": (
            len(pre_significant & post_significant) / len(pre_significant | post_significant)
            if pre_significant | post_significant else np.nan
        ),
    }
    (args.output_root / "pre_post_qc_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
