"""Create publication-oriented figures for the four-state malignant analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CORE_CONTRASTS = [
    "primary_rejected_vs_primary_retained",
    "metastasis_rejected_vs_metastasis_retained",
    "metastasis_retained_vs_primary_retained",
    "metastasis_rejected_vs_primary_retained",
    "metastasis_rejected_vs_primary_rejected",
]

DISPLAY = {
    "primary_rejected_vs_primary_retained": "Primary rejected vs retained",
    "metastasis_rejected_vs_metastasis_retained": "Metastasis rejected vs retained",
    "metastasis_retained_vs_primary_retained": "Metastasis retained vs primary retained",
    "metastasis_rejected_vs_primary_retained": "Metastasis rejected vs primary retained",
    "metastasis_rejected_vs_primary_rejected": "Metastasis rejected vs primary rejected",
    "primary_retained_vs_primary_nonmalignant": "Primary retained vs non-malignant",
    "primary_rejected_vs_primary_nonmalignant": "Primary rejected vs non-malignant",
    "metastasis_retained_vs_metastasis_nonmalignant": "Metastasis retained vs non-malignant",
    "metastasis_rejected_vs_metastasis_nonmalignant": "Metastasis rejected vs non-malignant",
}

COLORS = {"case": "#c43c4e", "reference": "#3274a1", "neutral": "#b7b7b7"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("four_state_root", type=Path)
    parser.add_argument("pydeseq2_root", type=Path)
    parser.add_argument("gsea_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--maximum-fdr", type=float, default=0.05)
    parser.add_argument("--minimum-absolute-log2fc", type=float, default=0.5)
    return parser.parse_args()


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def diagnostics_table(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("patients/*/diagnostics.json")):
        with path.open(encoding="utf-8") as handle:
            report = json.load(handle)
        row = {"patient_id": report["patient_id"], "selected_pair_n": report["selected_pair_n"]}
        row.update(report["state_cell_n"])
        source = report["excluded_source_status_counts"]
        target = report["excluded_target_status_counts"]
        row["primary_discordant"] = source.get("site_or_cap_discordant", 0) + source.get(
            "incomplete_pair_coverage", 0
        )
        row["metastasis_discordant"] = target.get("site_or_cap_discordant", 0) + target.get(
            "incomplete_pair_coverage", 0
        )
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No patient diagnostics under {root}")
    return pd.DataFrame(rows)


def state_fraction_table(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in table.iterrows():
        for side in ("primary", "metastasis"):
            values = {
                "retained": row[f"{side}_retained"],
                "rejected": row[f"{side}_rejected"],
                "discordant": row[f"{side}_discordant"],
            }
            denominator = sum(values.values())
            for state, value in values.items():
                rows.append({
                    "patient_id": row["patient_id"], "side": side, "state": state,
                    "cell_n": int(value), "fraction": value / denominator if denominator else np.nan,
                })
    return pd.DataFrame(rows)


def plot_state_fraction_axis(ax, fractions: pd.DataFrame, side: str) -> None:
    states = ["retained", "rejected", "discordant"]
    colors = ["#2b8cbe", "#d7301f", "#969696"]
    subset = fractions[fractions["side"].eq(side)]
    values = [subset.loc[subset["state"].eq(state), "fraction"].dropna() for state in states]
    box = ax.boxplot(values, patch_artist=True, showfliers=False, widths=0.58)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    rng = np.random.default_rng(20260903)
    for index, (series, color) in enumerate(zip(values, colors), 1):
        ax.scatter(rng.normal(index, 0.045, len(series)), series, s=16, color=color,
                   edgecolor="white", linewidth=0.3, alpha=0.85)
    ax.set_xticks(range(1, 4), ["Retained", "Rejected", "Discordant"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of malignant cells")
    ax.set_title(f"{side.capitalize()} gate composition")
    ax.grid(axis="y", alpha=0.2)


def state_figures(diagnostics: pd.DataFrame, output: Path) -> None:
    fractions = state_fraction_table(diagnostics)
    diagnostics.to_csv(output / "four_state_patient_cell_counts.csv", index=False)
    fractions.to_csv(output / "four_state_patient_cell_fractions.csv", index=False)
    for side in ("primary", "metastasis"):
        fig, ax = plt.subplots(figsize=(5.8, 4.5))
        plot_state_fraction_axis(ax, fractions, side)
        save(fig, output, f"01_{side}_gate_fractions")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)
    for ax, side in zip(axes, ("primary", "metastasis")):
        plot_state_fraction_axis(ax, fractions, side)
    fig.suptitle("ConfidenceOT four-state malignant-cell composition", fontsize=15)
    fig.tight_layout()
    save(fig, output, "01_four_state_gate_fractions_combined")


def read_deg(root: Path, contrast: str) -> pd.DataFrame:
    return pd.read_csv(
        root / "contrasts" / contrast / "pydeseq2_non_ot_gene_validation.csv"
    )


def volcano_axis(ax, table: pd.DataFrame, title: str, maximum_fdr: float,
                 minimum_absolute_lfc: float, label_n: int = 10) -> None:
    data = table.copy()
    data = data.loc[
        np.isfinite(data["log2_fold_change"])
        & np.isfinite(data["fdr"])
    ].copy()
    if data.empty:
        ax.text(
            0.5, 0.52,
            "FDR unavailable\n(no valid Wald tests)",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="#555555",
        )
        ax.set_axis_off()
        ax.set_title(f"{title}\nnot statistically evaluable", fontsize=10)
        return
    data["plot_fdr"] = data["fdr"].clip(lower=1e-300)
    data["minus_log10_fdr"] = -np.log10(data["plot_fdr"])
    significant = data["fdr"].lt(maximum_fdr) & data["log2_fold_change"].abs().ge(
        minimum_absolute_lfc
    )
    direction = np.where(
        significant & data["log2_fold_change"].gt(0), "case",
        np.where(significant, "reference", "neutral"),
    )
    for label in ("neutral", "reference", "case"):
        keep = direction == label
        ax.scatter(data.loc[keep, "log2_fold_change"], data.loc[keep, "minus_log10_fdr"],
                   s=9 if label == "neutral" else 15, color=COLORS[label],
                   alpha=0.42 if label == "neutral" else 0.75, linewidth=0)
    ax.axvline(-minimum_absolute_lfc, color="#777777", lw=0.8, ls="--")
    ax.axvline(minimum_absolute_lfc, color="#777777", lw=0.8, ls="--")
    ax.axhline(-np.log10(maximum_fdr), color="#777777", lw=0.8, ls="--")
    labels = data.loc[significant].sort_values(
        ["fdr", "log2_fold_change"], ascending=[True, False]
    ).head(label_n)
    labels = labels.sort_values(["log2_fold_change", "minus_log10_fdr"])
    for label_index, (_, row) in enumerate(labels.iterrows()):
        right_side = row["log2_fold_change"] >= 0
        ax.annotate(str(row["gene"]), (row["log2_fold_change"], row["minus_log10_fdr"]),
                    xytext=(4 if right_side else -4, 4 + 7 * (label_index % 3)),
                    textcoords="offset points", fontsize=7,
                    ha="left" if right_side else "right")
    ax.set_xlabel("log2 fold change (case/reference)")
    ax.set_ylabel("−log10 FDR")
    ax.set_title(f"{title}\nvalidated genes: {int(significant.sum()):,}", fontsize=10)
    ax.grid(alpha=0.14)


def volcano_figures(args: argparse.Namespace) -> pd.DataFrame:
    contrasts = CORE_CONTRASTS[:4]
    summary = []
    for contrast in CORE_CONTRASTS:
        table = read_deg(args.pydeseq2_root, contrast)
        significant = table["fdr"].lt(args.maximum_fdr) & table["log2_fold_change"].abs().ge(
            args.minimum_absolute_log2fc
        )
        summary.append({
            "contrast": contrast,
            "patient_n": pd.read_csv(
                args.pydeseq2_root / "contrasts" / contrast / "pseudobulk_sample_metadata.csv"
            )["patient_id"].nunique(),
            "validated_deg_n": int(significant.sum()),
            "case_enriched_n": int((significant & table["log2_fold_change"].gt(0)).sum()),
            "reference_enriched_n": int((significant & table["log2_fold_change"].lt(0)).sum()),
        })
        fig, ax = plt.subplots(figsize=(6.2, 4.8))
        volcano_axis(ax, table, DISPLAY[contrast], args.maximum_fdr,
                     args.minimum_absolute_log2fc, 12)
        save(fig, args.output_root, f"02_{contrast}_volcano")
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, contrast in zip(axes.ravel(), contrasts):
        volcano_axis(ax, read_deg(args.pydeseq2_root, contrast), DISPLAY[contrast],
                     args.maximum_fdr, args.minimum_absolute_log2fc, 7)
    fig.suptitle("Patient-paired non-OT-gene validation", fontsize=16)
    fig.tight_layout()
    save(fig, args.output_root, "02_core_non_ot_volcano_combined")
    result = pd.DataFrame(summary)
    result.to_csv(args.output_root / "core_non_ot_deg_summary.csv", index=False)
    return result


def hallmark_table(gsea_root: Path) -> pd.DataFrame:
    rows = []
    for contrast in CORE_CONTRASTS:
        path = gsea_root / "contrasts" / contrast / "non_ot_gene_validation_gsea_results.csv"
        table = pd.read_csv(path)
        table = table[table["collection"].eq("Hallmark")].copy()
        table["contrast"] = contrast
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def hallmark_figures(table: pd.DataFrame, output: Path, maximum_fdr: float) -> None:
    table.to_csv(output / "core_hallmark_gsea.csv", index=False)
    preferred = [
        "HALLMARK_TNFA_SIGNALING_VIA_NFKB", "HALLMARK_INTERFERON_GAMMA_RESPONSE",
        "HALLMARK_INTERFERON_ALPHA_RESPONSE", "HALLMARK_INFLAMMATORY_RESPONSE",
        "HALLMARK_IL6_JAK_STAT3_SIGNALING", "HALLMARK_G2M_CHECKPOINT",
        "HALLMARK_E2F_TARGETS", "HALLMARK_MITOTIC_SPINDLE",
        "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION", "HALLMARK_TGF_BETA_SIGNALING",
        "HALLMARK_HYPOXIA", "HALLMARK_ANGIOGENESIS", "HALLMARK_P53_PATHWAY",
        "HALLMARK_APOPTOSIS", "HALLMARK_MTORC1_SIGNALING",
        "HALLMARK_OXIDATIVE_PHOSPHORYLATION", "HALLMARK_MYC_TARGETS_V2",
    ]
    present = [value for value in preferred if value in set(table["pathway"])]
    nes = table.pivot(index="pathway", columns="contrast", values="NES").reindex(
        index=present, columns=CORE_CONTRASTS
    )
    fdr = table.pivot(index="pathway", columns="contrast", values="fdr").reindex(
        index=present, columns=CORE_CONTRASTS
    )
    values = nes.where(fdr.lt(maximum_fdr), 0.0)
    limit = max(2.0, float(np.nanmax(np.abs(values.to_numpy()))))
    fig, ax = plt.subplots(figsize=(11.5, 8.2))
    image = ax.imshow(values.to_numpy(), cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_yticks(range(len(values)), [value.replace("HALLMARK_", "").replace("_", " ")
                                             for value in values.index], fontsize=8)
    ax.set_xticks(range(len(values.columns)), [DISPLAY[value] for value in values.columns],
                  rotation=35, ha="right", fontsize=8)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if fdr.iloc[i, j] < maximum_fdr:
                ax.text(j, i, f"{nes.iloc[i, j]:.1f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(values.iloc[i, j]) > limit * 0.58 else "black")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label("NES (non-significant pathways shown as 0)")
    ax.set_title("Hallmark programs across four-state malignant contrasts")
    fig.tight_layout()
    save(fig, output, "03_core_hallmark_gsea_heatmap")

    for contrast in CORE_CONTRASTS:
        subset = table[table["contrast"].eq(contrast) & table["fdr"].lt(maximum_fdr)].copy()
        subset = subset.assign(abs_nes=subset["NES"].abs()).nlargest(18, "abs_nes").sort_values("NES")
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=(7.2, 6.0))
        colors = np.where(subset["NES"].gt(0), COLORS["case"], COLORS["reference"])
        sizes = np.clip(-np.log10(subset["fdr"].clip(lower=1e-300)), 1, 20) * 15
        ax.scatter(subset["NES"], range(len(subset)), s=sizes, c=colors, alpha=0.82)
        ax.set_yticks(range(len(subset)), subset["pathway"].str.replace("HALLMARK_", "", regex=False)
                       .str.replace("_", " ", regex=False), fontsize=8)
        ax.axvline(0, color="#777777", lw=0.8)
        ax.set_xlabel("Normalized enrichment score")
        ax.set_title(DISPLAY[contrast])
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        save(fig, output, f"03_{contrast}_hallmark_dotplot")


def matrix_figure(matrix: pd.DataFrame, output: Path, stem: str, title: str,
                  colorbar_label: str) -> None:
    limit = max(1.0, float(np.nanmax(np.abs(matrix.to_numpy()))))
    fig, ax = plt.subplots(figsize=(10.5, max(4.5, 0.31 * len(matrix))))
    image = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_yticks(range(len(matrix)), matrix.index, fontsize=8)
    ax.set_xticks(range(len(matrix.columns)), [DISPLAY.get(value, value) for value in matrix.columns],
                  rotation=35, ha="right", fontsize=8)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label(colorbar_label)
    ax.set_title(title)
    fig.tight_layout()
    save(fig, output, stem)


def gene_effect_figures(args: argparse.Namespace) -> None:
    effects = {}
    selected: list[str] = []
    for contrast in CORE_CONTRASTS[:4]:
        table = read_deg(args.pydeseq2_root, contrast).set_index("gene")
        effects[contrast] = table["log2_fold_change"]
        if contrast in {"metastasis_rejected_vs_metastasis_retained",
                        "metastasis_rejected_vs_primary_retained"}:
            hit = table[table["fdr"].lt(args.maximum_fdr) & table["log2_fold_change"].abs().ge(
                args.minimum_absolute_log2fc
            )].sort_values("fdr").head(15)
            selected.extend(hit.index.astype(str))
    selected = list(dict.fromkeys(selected))[:28]
    matrix = pd.DataFrame(effects).reindex(selected)
    matrix.to_csv(args.output_root / "core_key_gene_log2fc.csv")
    matrix_figure(matrix, args.output_root, "04_core_key_gene_log2fc_heatmap",
                  "Validated gene effects across core contrasts", "PyDESeq2 log2 fold change")

    markers = ["EPCAM", "KRT7", "KRT8", "KRT18", "KRT19", "MUC1", "MSLN", "PAX8",
               "WFDC2", "MKI67", "VIM", "COL1A1", "PTPRC", "CD3D", "LST1"]
    background = [
        "primary_retained_vs_primary_nonmalignant",
        "primary_rejected_vs_primary_nonmalignant",
        "metastasis_retained_vs_metastasis_nonmalignant",
        "metastasis_rejected_vs_metastasis_nonmalignant",
    ]
    marker_effects = {}
    for contrast in background:
        table = pd.read_csv(
            args.pydeseq2_root / "contrasts" / contrast / "pydeseq2_all_gene_discovery.csv"
        ).set_index("gene")
        marker_effects[contrast] = table["log2_fold_change"]
    marker_matrix = pd.DataFrame(marker_effects).reindex(markers)
    marker_matrix.to_csv(args.output_root / "malignant_identity_marker_log2fc.csv")
    matrix_figure(marker_matrix, args.output_root, "05_malignant_identity_marker_heatmap",
                  "Malignant identity relative to matched non-malignant background",
                  "PyDESeq2 log2 fold change")


def patient_gene_effects(root: Path, contrast: str, genes: list[str]) -> pd.DataFrame:
    rows = []
    for patient_root in sorted(root.glob("patients/*")):
        metadata_path = patient_root / "pseudobulk_sample_metadata.csv"
        counts_path = patient_root / "pseudobulk_raw_counts.csv.gz"
        if not metadata_path.exists() or not counts_path.exists():
            continue
        metadata = pd.read_csv(metadata_path)
        metadata = metadata.loc[metadata["contrast"].eq(contrast)]
        if set(metadata["comparison_status"]) != {"case", "reference"}:
            continue
        counts = pd.read_csv(counts_path)
        counts = counts.loc[counts["sample_id"].isin(metadata["sample_id"])]
        if len(counts) != 2:
            continue
        numeric = counts.drop(columns="sample_id")
        library_size = numeric.sum(axis=1).replace(0, np.nan)
        available = [gene for gene in genes if gene in numeric.columns]
        log_cpm = np.log2(numeric[available].div(library_size, axis=0) * 1e6 + 1)
        log_cpm.index = counts["sample_id"].to_numpy()
        status = metadata.set_index("comparison_status")["sample_id"]
        effect = log_cpm.loc[status["case"]] - log_cpm.loc[status["reference"]]
        effect.name = str(metadata["patient_id"].iloc[0])
        rows.append(effect)
    if not rows:
        return pd.DataFrame(columns=genes)
    return pd.DataFrame(rows).reindex(columns=genes)


def patient_consistency_figure(args: argparse.Namespace) -> None:
    contrast = "metastasis_rejected_vs_metastasis_retained"
    table = read_deg(args.pydeseq2_root, contrast)
    hits = table.loc[
        table["fdr"].lt(args.maximum_fdr)
        & table["log2_fold_change"].abs().ge(args.minimum_absolute_log2fc)
    ].sort_values(["fdr", "log2_fold_change"], ascending=[True, False])
    genes = hits["gene"].astype(str).head(12).tolist()
    matrix = patient_gene_effects(args.four_state_root, contrast, genes)
    matrix.to_csv(args.output_root / "metastasis_rejected_patient_logcpm_effects.csv")
    if matrix.empty:
        return
    consistency = (matrix.gt(0).sum(axis=0) / matrix.notna().sum(axis=0)).fillna(0)
    labels = [f"{gene}\n{int(round(100 * consistency[gene]))}% positive" for gene in matrix.columns]
    limit = max(1.0, float(np.nanpercentile(np.abs(matrix.to_numpy()), 98)))
    fig, ax = plt.subplots(figsize=(11.5, max(6.0, 0.24 * len(matrix))))
    image = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_yticks(range(len(matrix)), matrix.index, fontsize=7)
    ax.set_xticks(range(len(matrix.columns)), labels, rotation=40, ha="right", fontsize=8)
    ax.set_xlabel("Gene and fraction of patients with higher expression in rejected cells")
    ax.set_ylabel("Patient")
    ax.set_title("Patient-level consistency: metastasis rejected vs retained malignant cells")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label("Patient pseudobulk log2(CPM + 1) difference")
    fig.tight_layout()
    save(fig, args.output_root, "06_metastasis_rejected_patient_gene_consistency")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False})
    diagnostics = diagnostics_table(args.four_state_root)
    state_figures(diagnostics, args.output_root)
    deg_summary = volcano_figures(args)
    hallmark = hallmark_table(args.gsea_root)
    hallmark_figures(hallmark, args.output_root, args.maximum_fdr)
    gene_effect_figures(args)
    patient_consistency_figure(args)
    report = {
        "evidence_status": "preliminary",
        "patient_n": int(diagnostics["patient_id"].nunique()),
        "core_contrasts": CORE_CONTRASTS,
        "volcano_effect_filter": {
            "maximum_fdr": args.maximum_fdr,
            "minimum_absolute_log2_fold_change": args.minimum_absolute_log2fc,
        },
        "separate_panels_saved": True,
        "deg_summary": deg_summary.to_dict("records"),
        "interpretation_limit": (
            "Primary retained is the across-partner consensus state; pair-specific sensitivity "
            "must be reported separately."
        ),
    }
    (args.output_root / "figure_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    print(f"Figures: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
