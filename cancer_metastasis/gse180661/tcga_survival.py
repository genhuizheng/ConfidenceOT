"""Project frozen primary-cell programs into TCGA-OV and analyze survival."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


def tcga_patient_id(value: str) -> str:
    return str(value).strip().upper()[:12]


def is_primary_tumor(value: str) -> bool:
    fields = str(value).strip().upper().split("-")
    return len(fields) >= 4 and fields[3][:2] == "01"


def read_expression(path: Path, gene_column: str, gene_map_table: Path | None = None,
                    gene_map_id_column: str = "id",
                    gene_map_symbol_column: str = "gene") -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t")
    if gene_column not in table:
        raise KeyError(f"Expression table lacks {gene_column!r}")
    table[gene_column] = table[gene_column].astype(str)
    if gene_map_table is not None:
        mapping = pd.read_csv(gene_map_table, sep="\t", dtype=str)
        required = {gene_map_id_column, gene_map_symbol_column}
        if not required.issubset(mapping.columns):
            raise KeyError(
                f"Gene map lacks columns: {sorted(required - set(mapping.columns))}"
            )
        mapping = mapping[[gene_map_id_column, gene_map_symbol_column]].dropna()
        mapping = mapping.drop_duplicates(gene_map_id_column)
        symbol_by_id = mapping.set_index(gene_map_id_column)[gene_map_symbol_column]
        table[gene_column] = table[gene_column].map(symbol_by_id)
        table = table[table[gene_column].notna()].copy()
        if table.empty:
            raise RuntimeError("No expression identifiers mapped to gene symbols")
    table[gene_column] = table[gene_column].astype(str).str.upper()
    numeric = table.drop(columns=gene_column).apply(pd.to_numeric, errors="coerce")
    numeric.index = table[gene_column]
    numeric = numeric.groupby(level=0).mean()
    primary = [column for column in numeric if is_primary_tumor(column)]
    if not primary:
        raise RuntimeError("No TCGA Primary Solid Tumor barcodes (sample type 01) found")
    numeric = numeric[primary]
    by_patient: dict[str, list[str]] = {}
    for column in numeric:
        by_patient.setdefault(tcga_patient_id(column), []).append(column)
    selected = [sorted(columns)[0] for columns in by_patient.values()]
    numeric = numeric[selected]
    numeric.columns = [tcga_patient_id(column) for column in numeric.columns]
    return numeric


def expression_on_log2_tpm_scale(expression: pd.DataFrame, scale: str) -> pd.DataFrame:
    if expression.min().min() < 0:
        raise ValueError("Expression table must contain non-negative values")
    if scale == "raw_tpm":
        return np.log2(expression + 1.0)
    if scale == "log2_tpm_plus_one":
        return expression
    raise ValueError(f"Unsupported expression scale: {scale}")


def frozen_signature(deg: pd.DataFrame, lfc: float, fdr: float,
                     detection: float, consistency: float):
    selected = deg[
        deg["fdr"].lt(fdr)
        & deg["log2_fold_change"].abs().ge(lfc)
        & deg["detected_patient_fraction"].ge(detection)
        & deg["patient_direction_consistency"].ge(consistency)
    ].copy()
    positive = sorted(selected.loc[
        selected["log2_fold_change"].gt(0), "gene"
    ].astype(str).str.upper().unique())
    negative = sorted(selected.loc[
        selected["log2_fold_change"].lt(0), "gene"
    ].astype(str).str.upper().unique())
    if not positive or not negative:
        raise RuntimeError(
            f"Frozen signature requires both directions; positive={len(positive)}, "
            f"negative={len(negative)}"
        )
    return positive, negative, selected


def zscore_rows(expression: pd.DataFrame) -> pd.DataFrame:
    mean = expression.mean(axis=1)
    std = expression.std(axis=1, ddof=0).replace(0, np.nan)
    return expression.sub(mean, axis=0).div(std, axis=0)


def read_gmt(path: Path) -> dict[str, list[str]]:
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 3:
                result[fields[0]] = [gene.upper() for gene in fields[2:]]
    return result


def hallmark_scores(expression: pd.DataFrame, selected_pathways: list[str],
                    gmt: Path, threads: int) -> pd.DataFrame:
    all_sets = read_gmt(gmt)
    gene_sets = {name: all_sets[name] for name in selected_pathways if name in all_sets}
    if not gene_sets:
        return pd.DataFrame(index=expression.columns)
    analysis = gp.ssgsea(
        data=expression, gene_sets=gene_sets, sample_norm_method="rank",
        min_size=10, max_size=500, permutation_num=0, threads=threads,
        outdir=None, no_plot=True, verbose=False,
    ).res2d
    required = {"Name", "Term", "NES"}
    if not required.issubset(analysis.columns):
        raise RuntimeError(f"Unexpected GSEApy ssGSEA columns: {list(analysis.columns)}")
    scores = analysis.pivot(index="Name", columns="Term", values="NES")
    scores.index = scores.index.astype(str).map(tcga_patient_id)
    return scores.apply(pd.to_numeric, errors="coerce")


def prepare_clinical(path: Path, patient_column: str, endpoints: dict[str, tuple[str, str]]):
    clinical = pd.read_csv(path, sep=None, engine="python")
    if patient_column not in clinical:
        raise KeyError(f"Clinical table lacks {patient_column!r}")
    clinical["patient_id"] = clinical[patient_column].map(tcga_patient_id)
    if clinical["patient_id"].duplicated().any():
        clinical = clinical.sort_values("patient_id").drop_duplicates("patient_id")
    for endpoint, (time_column, event_column) in endpoints.items():
        if time_column not in clinical or event_column not in clinical:
            raise KeyError(f"Clinical table lacks {endpoint} columns")
        clinical[f"{endpoint}_time"] = pd.to_numeric(clinical[time_column], errors="coerce")
        clinical[f"{endpoint}_event"] = pd.to_numeric(clinical[event_column], errors="coerce")
    return clinical.set_index("patient_id")


def encode_covariates(table: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    if not covariates:
        return pd.DataFrame(index=table.index)
    missing = [value for value in covariates if value not in table]
    if missing:
        raise KeyError(f"Clinical table lacks requested covariates: {missing}")
    result = table[covariates].copy()
    numeric = []
    categorical = []
    for column in result:
        converted = pd.to_numeric(result[column], errors="coerce")
        if converted.notna().mean() >= 0.9:
            result[column] = converted
            numeric.append(column)
        else:
            categorical.append(column)
    if categorical:
        result = pd.get_dummies(result, columns=categorical, drop_first=True, dtype=float)
    return result.apply(pd.to_numeric, errors="coerce")


def cox_one(table: pd.DataFrame, score: str, endpoint: str,
            covariates: pd.DataFrame, model: str) -> dict:
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test

    columns = [f"{endpoint}_time", f"{endpoint}_event", score]
    analysis = table[columns].join(covariates, how="left")
    analysis = analysis.replace([np.inf, -np.inf], np.nan).dropna()
    analysis = analysis[analysis[f"{endpoint}_time"].gt(0)]
    analysis[score] = (
        analysis[score] - analysis[score].mean()
    ) / analysis[score].std(ddof=0)
    fitter = CoxPHFitter()
    fitter.fit(analysis, duration_col=f"{endpoint}_time",
               event_col=f"{endpoint}_event", show_progress=False)
    row = fitter.summary.loc[score]
    ph = proportional_hazard_test(fitter, analysis, time_transform="rank")
    return {
        "score": score, "endpoint": endpoint, "model": model,
        "patient_n": len(analysis),
        "event_n": int(analysis[f"{endpoint}_event"].sum()),
        "hazard_ratio_per_sd": float(row["exp(coef)"]),
        "ci_95_lower": float(row["exp(coef) lower 95%"]),
        "ci_95_upper": float(row["exp(coef) upper 95%"]),
        "p_value": float(row["p"]),
        "concordance_index": float(fitter.concordance_index_),
        "ph_test_p_value": float(ph.summary.loc[score, "p"]),
    }


def km_plot(table: pd.DataFrame, output: Path):
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    use = table[["OS_time", "OS_event", "MCS"]].dropna()
    use = use[use["OS_time"].gt(0)]
    cutoff = float(use["MCS"].median())
    high = use["MCS"].ge(cutoff)
    test = logrank_test(
        use.loc[high, "OS_time"], use.loc[~high, "OS_time"],
        event_observed_A=use.loc[high, "OS_event"],
        event_observed_B=use.loc[~high, "OS_event"],
    )
    figure, axis = plt.subplots(figsize=(7.5, 6))
    for mask, label, color in (
        (high, "High MCS", "#C43C39"), (~high, "Low MCS", "#2C6BAA")
    ):
        fitter = KaplanMeierFitter(label=f"{label} (n={int(mask.sum())})")
        fitter.fit(use.loc[mask, "OS_time"] / 30.4375,
                   event_observed=use.loc[mask, "OS_event"])
        fitter.plot_survival_function(ax=axis, ci_show=True, color=color)
    axis.set_xlabel("Overall survival (months)")
    axis.set_ylabel("Survival probability")
    axis.set_title("TCGA-OV overall survival by frozen MCS")
    axis.text(0.98, 0.98, f"Median cutoff={cutoff:.3f}\nLog-rank p={test.p_value:.3g}",
              transform=axis.transAxes, ha="right", va="top")
    figure.tight_layout()
    figure.savefig(output / "01_tcga_ov_mcs_os_kaplan_meier.png", dpi=300)
    figure.savefig(output / "01_tcga_ov_mcs_os_kaplan_meier.pdf")
    plt.close(figure)


def forest_plot(results: pd.DataFrame, output: Path, maximum_pathways: int):
    use = results[results["model"].eq("univariable")].copy()
    mcs = use[use["score"].eq("MCS")]
    pathways = use[~use["score"].eq("MCS")].sort_values("fdr").head(maximum_pathways)
    use = pd.concat([mcs, pathways]).sort_values(["endpoint", "hazard_ratio_per_sd"])
    labels = use["endpoint"] + " | " + use["score"].str.replace("HALLMARK_", "", regex=False)
    y = np.arange(len(use))
    figure, axis = plt.subplots(figsize=(9, max(5, 0.34 * len(use) + 2)))
    axis.errorbar(
        use["hazard_ratio_per_sd"], y,
        xerr=np.vstack([
            use["hazard_ratio_per_sd"] - use["ci_95_lower"],
            use["ci_95_upper"] - use["hazard_ratio_per_sd"],
        ]), fmt="o", color="#333333", ecolor="#777777", capsize=2,
    )
    axis.axvline(1, color="#777777", linestyle="--", linewidth=0.8)
    axis.set_yticks(y, labels)
    axis.set_xscale("log")
    axis.set_xlabel("Hazard ratio per 1-SD score increase (95% CI)")
    axis.set_title("TCGA-OV survival associations of frozen primary-tumor programs")
    figure.tight_layout()
    figure.savefig(output / "02_tcga_ov_mcs_hallmark_cox_forest.png", dpi=300)
    figure.savefig(output / "02_tcga_ov_mcs_hallmark_cox_forest.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expression_table", type=Path,
                        help="Gene-by-sample TCGA-OV TPM table")
    parser.add_argument("clinical_table", type=Path)
    parser.add_argument("deg_table", type=Path)
    parser.add_argument("gsea_table", type=Path)
    parser.add_argument("hallmark_gmt", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--gene-column", default="gene")
    parser.add_argument("--gene-map-table", type=Path)
    parser.add_argument("--gene-map-id-column", default="id")
    parser.add_argument("--gene-map-symbol-column", default="gene")
    parser.add_argument(
        "--expression-scale", choices=["raw_tpm", "log2_tpm_plus_one"],
        default="raw_tpm",
    )
    parser.add_argument("--patient-column", default="bcr_patient_barcode")
    parser.add_argument("--os-time-column", default="OS.time")
    parser.add_argument("--os-event-column", default="OS")
    parser.add_argument("--pfi-time-column", default="PFI.time")
    parser.add_argument("--pfi-event-column", default="PFI")
    parser.add_argument("--covariates", nargs="*", default=[])
    parser.add_argument("--signature-log2fc", type=float, default=1.0)
    parser.add_argument("--signature-fdr", type=float, default=0.05)
    parser.add_argument("--minimum-detected-patient-fraction", type=float, default=0.25)
    parser.add_argument("--minimum-patient-direction-consistency", type=float, default=0.70)
    parser.add_argument("--hallmark-fdr", type=float, default=0.05)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--maximum-forest-pathways", type=int, default=20)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    expression = read_expression(
        args.expression_table, args.gene_column, args.gene_map_table,
        args.gene_map_id_column, args.gene_map_symbol_column,
    )
    log_expression = expression_on_log2_tpm_scale(expression, args.expression_scale)
    deg = pd.read_csv(args.deg_table)
    positive, negative, selected_genes = frozen_signature(
        deg, args.signature_log2fc, args.signature_fdr,
        args.minimum_detected_patient_fraction,
        args.minimum_patient_direction_consistency,
    )
    available_positive = [gene for gene in positive if gene in log_expression.index]
    available_negative = [gene for gene in negative if gene in log_expression.index]
    if not available_positive or not available_negative:
        raise RuntimeError("TCGA expression has no usable genes in one signature direction")
    z = zscore_rows(log_expression)
    mcs = z.loc[available_positive].mean(axis=0) - z.loc[available_negative].mean(axis=0)
    scores = pd.DataFrame({"MCS": mcs})

    gsea = pd.read_csv(args.gsea_table)
    selected_pathways = sorted(gsea.loc[
        gsea["collection"].eq("Hallmark") & gsea["fdr"].lt(args.hallmark_fdr),
        "pathway",
    ].astype(str).unique())
    pathway_scores = hallmark_scores(log_expression, selected_pathways,
                                      args.hallmark_gmt, args.threads)
    scores = scores.join(pathway_scores, how="left")
    clinical = prepare_clinical(
        args.clinical_table, args.patient_column,
        {"OS": (args.os_time_column, args.os_event_column),
         "PFI": (args.pfi_time_column, args.pfi_event_column)},
    )
    analysis = clinical.join(scores, how="inner")
    covariates = encode_covariates(analysis, args.covariates)
    score_columns = ["MCS"] + list(pathway_scores.columns)
    rows = []
    for endpoint in ("OS", "PFI"):
        for score in score_columns:
            rows.append(cox_one(analysis, score, endpoint,
                                pd.DataFrame(index=analysis.index), "univariable"))
            if len(covariates.columns):
                rows.append(cox_one(analysis, score, endpoint, covariates, "adjusted"))
    results = pd.DataFrame(rows)
    results["fdr"] = np.nan
    for (endpoint, model), indices in results.groupby(["endpoint", "model"]).groups.items():
        pathway_indices = [index for index in indices if results.loc[index, "score"] != "MCS"]
        if pathway_indices:
            results.loc[pathway_indices, "fdr"] = multipletests(
                results.loc[pathway_indices, "p_value"], method="fdr_bh"
            )[1]
    results.to_csv(args.output_root / "tcga_ov_survival_models.csv", index=False)
    analysis.to_csv(args.output_root / "tcga_ov_expression_clinical_scores.csv.gz",
                    compression="gzip")
    selected_genes.to_csv(args.output_root / "frozen_mcs_genes.csv", index=False)
    pd.DataFrame({
        "direction": ["positive"] * len(positive) + ["negative"] * len(negative),
        "gene": positive + negative,
        "available_in_tcga": [gene in log_expression.index for gene in positive + negative],
    }).to_csv(args.output_root / "frozen_mcs_gene_availability.csv", index=False)
    gsea[gsea["pathway"].isin(selected_pathways)].to_csv(
        args.output_root / "frozen_hallmark_pathways.csv", index=False
    )
    km_plot(analysis, args.output_root)
    forest_plot(results, args.output_root, args.maximum_forest_pathways)
    report = {
        "cohort": "TCGA-OV primary solid tumor",
        "expression_scale": args.expression_scale,
        "gene_map": str(args.gene_map_table) if args.gene_map_table else None,
        "patient_n": len(analysis),
        "positive_signature_gene_n": len(positive),
        "negative_signature_gene_n": len(negative),
        "positive_available_gene_n": len(available_positive),
        "negative_available_gene_n": len(available_negative),
        "frozen_hallmark_pathway_n": len(selected_pathways),
        "primary_endpoint": "OS", "secondary_endpoint": "PFI",
        "optimized_survival_cutoff": False,
        "km_cutoff": "prespecified cohort median MCS",
        "interpretation": "prognostic association, not metastatic lineage evidence",
    }
    (args.output_root / "tcga_ov_survival_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
