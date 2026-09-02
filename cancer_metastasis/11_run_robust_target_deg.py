"""Within-target DEG for cap-robust ConfidenceOT malignant-cell states.

For one multi-primary patient/target group, target malignant cells rejected in
both the baseline-winning fit and the source-cap sensitivity-winning fit are
compared with cells retained in both fits.  Discordant gates are excluded.
The all-gene result is exploratory; genes used in either OT representation are
flagged and excluded from the primary non-circular validation table.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from common import expression_kind, expression_matrix, gene_keys, load_exact_side


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_csv", type=Path)
    parser.add_argument("baseline_root", type=Path)
    parser.add_argument("sensitivity_root", type=Path)
    parser.add_argument("robustness_csv", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--sensitivity-label", default="source090")
    parser.add_argument("--malignant-annotation", default="Ovarian.cancer.cell")
    parser.add_argument("--minimum-rejected-cells", type=int, default=20)
    parser.add_argument("--minimum-retained-cells", type=int, default=20)
    parser.add_argument("--minimum-gene-cells", type=int, default=3)
    parser.add_argument("--target-sum", type=float, default=10_000.0)
    return parser.parse_args()


def pair_id(patient: str, source: str, target: str) -> str:
    return f"{patient}__{source}__{target}"


def one_result_file(root: Path, pair: str, name: str) -> Path:
    matches = sorted((root / pair).glob(f"scope_malignant/*/{name}"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {name} for {pair} under {root}; found {len(matches)}")
    return matches[0]


def read_target_gate(root: Path, pair: str, prefix: str) -> pd.DataFrame:
    path = one_result_file(root, pair, "cell_confidence.csv")
    table = pd.read_csv(
        path,
        usecols=[
            "method", "side", "observation_id", "rejected",
            "normalized_rejection_score", "signed_rejection_margin",
        ],
    )
    table = table[table["method"].eq("M4-E") & table["side"].eq("target")].copy()
    table = table.drop(columns=["method", "side"]).rename(columns={
        "rejected": f"{prefix}_rejected",
        "normalized_rejection_score": f"{prefix}_rejection_score",
        "signed_rejection_margin": f"{prefix}_signed_margin",
    })
    if table["observation_id"].duplicated().any():
        raise RuntimeError(f"Duplicate target observation IDs in {path}")
    return table


def read_hvg(root: Path, pair: str) -> set[str]:
    path = one_result_file(root, pair, "run.json")
    with path.open(encoding="utf-8") as handle:
        run = json.load(handle)
    return {str(value) for value in run.get("hvg", [])}


def valid_symbol(values: np.ndarray) -> np.ndarray:
    missing = {"", "na", "n/a", "nan", "none", "null", "<na>"}
    return ~pd.Series(values.astype(str)).str.strip().str.lower().isin(missing).to_numpy()


def aggregate_gene_symbols(
    matrix: sparse.csr_matrix,
    data,
    ot_features: set[str],
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    keys = gene_keys(data)
    symbols = np.asarray(data.var_names.astype(str), dtype=str)
    if "gene_symbol" in data.var:
        candidate = data.var["gene_symbol"].astype(str).str.strip().to_numpy(dtype=str)
        symbols = np.where(valid_symbol(candidate), candidate, symbols)
    symbols = np.asarray([value.strip() for value in symbols], dtype=str)
    # Stable first-seen ordering, while summing duplicate symbols rather than
    # silently dropping their counts.
    lookup: dict[str, int] = {}
    inverse = np.empty(len(symbols), dtype=np.int64)
    ordered: list[str] = []
    for index, symbol in enumerate(symbols):
        if symbol not in lookup:
            lookup[symbol] = len(ordered)
            ordered.append(symbol)
        inverse[index] = lookup[symbol]
    aggregation = sparse.csr_matrix(
        (np.ones(len(symbols)), (np.arange(len(symbols)), inverse)),
        shape=(len(symbols), len(ordered)),
    )
    collapsed = (matrix @ aggregation).tocsr()
    used = np.zeros(len(ordered), dtype=bool)
    for index, key in enumerate(keys):
        if str(key) in ot_features:
            used[inverse[index]] = True
    return collapsed, np.asarray(ordered, dtype=str), used


def normalize_expression(
    matrix: sparse.csr_matrix, kind: str, target_sum: float
) -> tuple[sparse.csr_matrix, str]:
    matrix = matrix.astype(np.float64).tocsr(copy=True)
    label = str(kind).lower()
    if "log-normalized" in label or "log normalized" in label:
        return matrix, "stored_log_normalized_expression"
    if "normalized" in label and "raw" not in label:
        if matrix.data.size and matrix.data.min() < 0:
            return matrix, "stored_normalized_expression_used_as_provided"
        matrix.data = np.log1p(matrix.data)
        return matrix, "stored_normalized_expression_then_log1p"
    library = np.asarray(matrix.sum(axis=1)).ravel()
    if np.any(library <= 0):
        raise ValueError("A selected malignant cell has zero expression library size")
    matrix = matrix.multiply((target_sum / library)[:, None]).tocsr()
    matrix.data = np.log1p(matrix.data)
    return matrix, f"raw_counts_library_normalized_to_{target_sum:g}_then_log1p"


def scanpy_deg(
    expression: sparse.csr_matrix,
    genes: np.ndarray,
    rejected_n: int,
) -> pd.DataFrame:
    try:
        import scanpy as sc
    except (ImportError, AttributeError) as error:
        raise RuntimeError("scanpy>=1.10,<2 is required for malignant-cell DEG") from error
    labels = np.where(np.arange(expression.shape[0]) < rejected_n, "rejected", "retained")
    analysis = ad.AnnData(
        X=expression,
        obs=pd.DataFrame(
            {"confidence_status": pd.Categorical(labels)},
            index=[f"cell_{index}" for index in range(len(labels))],
        ),
        var=pd.DataFrame(index=pd.Index(genes, name="gene")),
    )
    sc.tl.rank_genes_groups(
        analysis,
        groupby="confidence_status",
        groups=["rejected"],
        reference="retained",
        method="wilcoxon",
        corr_method="benjamini-hochberg",
        tie_correct=True,
        pts=True,
        n_genes=analysis.n_vars,
        use_raw=False,
    )
    result = sc.get.rank_genes_groups_df(analysis, group="rejected").rename(columns={
        "names": "gene", "scores": "wilcoxon_score",
        "logfoldchanges": "log2_fold_change", "pvals": "p_value",
        "pvals_adj": "fdr",
    })
    return result.drop(
        columns=["pct_nz_group", "pct_nz_reference", "pct_nz_rest"],
        errors="ignore",
    )


def main() -> None:
    args = parse_args()
    robustness = pd.read_csv(args.robustness_csv).sort_values(
        ["dataset_id", "patient_id", "target_sample"], kind="stable"
    ).reset_index(drop=True)
    if args.index < 0 or args.index >= len(robustness):
        raise IndexError(f"--index {args.index} outside 0..{len(robustness) - 1}")
    row = robustness.iloc[args.index]
    sensitivity_column = f"{re.sub(r'[^A-Za-z0-9]+', '_', args.sensitivity_label).lower()}_winner"
    if sensitivity_column not in row.index:
        raise KeyError(f"Missing robustness column: {sensitivity_column}")
    patient = str(row["patient_id"])
    target = str(row["target_sample"])
    baseline_source = str(row["baseline_winner"])
    sensitivity_source = str(row[sensitivity_column])
    baseline_pair = pair_id(patient, baseline_source, target)
    sensitivity_pair = pair_id(patient, sensitivity_source, target)
    group_id = f"{patient}__{target}"
    output = args.output_root / "groups" / f"{args.index:03d}_{group_id}"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "SUCCESS").is_file():
        print(f"SKIP completed group={group_id}")
        return

    baseline_gate = read_target_gate(args.baseline_root, baseline_pair, "baseline")
    sensitivity_gate = read_target_gate(
        args.sensitivity_root, sensitivity_pair, args.sensitivity_label
    )
    cells = baseline_gate.merge(
        sensitivity_gate, on="observation_id", how="inner", validate="one_to_one"
    )
    cells["robust_status"] = np.select(
        [
            cells["baseline_rejected"] & cells[f"{args.sensitivity_label}_rejected"],
            ~cells["baseline_rejected"] & ~cells[f"{args.sensitivity_label}_rejected"],
        ],
        ["robust_rejected", "robust_retained"],
        default="cap_discordant",
    )

    manifest = pd.read_csv(args.manifest_csv)
    match = manifest[manifest["pair_id"].eq(baseline_pair)]
    if len(match) != 1:
        raise RuntimeError(f"Manifest did not uniquely resolve {baseline_pair}")
    manifest_row = match.iloc[0]
    target_paths = json.loads(str(manifest_row["target_h5ads_json"]))
    data = load_exact_side(target_paths, target)
    annotations = None
    for column in ("cell_type", "annotation", "celltype", "cell_type_final"):
        if column in data.obs:
            annotations = data.obs[column].astype(str).to_numpy()
            break
    if annotations is None:
        raise KeyError("Target H5AD has no cell-type annotation column")
    data = data[annotations == args.malignant_annotation].copy()
    lookup = {str(value): index for index, value in enumerate(data.obs_names)}
    missing = cells.loc[~cells["observation_id"].astype(str).isin(lookup), "observation_id"]
    if len(missing):
        raise KeyError(f"{len(missing)} target gate IDs were absent from the malignant H5AD subset")

    rejected_ids = cells.loc[
        cells["robust_status"].eq("robust_rejected"), "observation_id"
    ].astype(str).tolist()
    retained_ids = cells.loc[
        cells["robust_status"].eq("robust_retained"), "observation_id"
    ].astype(str).tolist()
    diagnostics = {
        "group_index": args.index,
        "group_id": group_id,
        "dataset_id": str(row["dataset_id"]),
        "patient_id": patient,
        "target_sample": target,
        "baseline_winner": baseline_source,
        "sensitivity_winner": sensitivity_source,
        "baseline_pair_id": baseline_pair,
        "sensitivity_pair_id": sensitivity_pair,
        "baseline_target_gate_n": len(baseline_gate),
        "sensitivity_target_gate_n": len(sensitivity_gate),
        "overlapping_target_cell_n": len(cells),
        "robust_rejected_n": len(rejected_ids),
        "robust_retained_n": len(retained_ids),
        "cap_discordant_n": int(cells["robust_status"].eq("cap_discordant").sum()),
        "exact_winner_robust": bool(row["recommended_range_exact_robust"]),
        "laterality_robust": bool(row["recommended_range_laterality_robust"]),
    }
    cells.to_csv(output / "target_cell_robust_classification.csv.gz", index=False)
    if (
        len(rejected_ids) < args.minimum_rejected_cells
        or len(retained_ids) < args.minimum_retained_cells
    ):
        diagnostics["status"] = "insufficient_robust_cells"
        (output / "diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2), encoding="utf-8"
        )
        (output / "SKIPPED").write_text("insufficient robust cells\n", encoding="utf-8")
        print(diagnostics, flush=True)
        return

    ordered_ids = rejected_ids + retained_ids
    indices = np.asarray([lookup[value] for value in ordered_ids], dtype=np.int64)
    matrix = sparse.csr_matrix(expression_matrix(data)[indices], dtype=np.float64)
    ot_features = read_hvg(args.baseline_root, baseline_pair) | read_hvg(
        args.sensitivity_root, sensitivity_pair
    )
    matrix, genes, used_for_ot = aggregate_gene_symbols(matrix, data, ot_features)
    detected = np.asarray(matrix.getnnz(axis=0)).ravel()
    keep = detected >= args.minimum_gene_cells
    matrix, genes, used_for_ot = matrix[:, keep], genes[keep], used_for_ot[keep]
    expression, normalization = normalize_expression(
        matrix, expression_kind(data), args.target_sum
    )
    rejected_n = len(rejected_ids)
    retained_n = len(retained_ids)
    rejected_mean = np.asarray(expression[:rejected_n].mean(axis=0)).ravel()
    retained_mean = np.asarray(expression[rejected_n:].mean(axis=0)).ravel()
    rejected_fraction = np.asarray(expression[:rejected_n].getnnz(axis=0)).ravel() / rejected_n
    retained_fraction = np.asarray(expression[rejected_n:].getnnz(axis=0)).ravel() / retained_n
    annotation = pd.DataFrame({
        "gene": genes,
        "used_for_ot": used_for_ot,
        "rejected_mean_log_expression": rejected_mean,
        "retained_mean_log_expression": retained_mean,
        "mean_log_expression_difference": rejected_mean - retained_mean,
        "rejected_expression_fraction": rejected_fraction,
        "retained_expression_fraction": retained_fraction,
    })
    deg = scanpy_deg(expression, genes, rejected_n).merge(
        annotation, on="gene", how="left", validate="one_to_one"
    )
    for key in ("group_id", "patient_id", "target_sample"):
        deg[key] = diagnostics[key]
    deg["comparison"] = "cap_robust_rejected_vs_cap_robust_retained_target_malignant_cells"
    deg["inference_scope"] = "within_patient_target_descriptive_cell_level_deg"
    deg.to_csv(output / "all_gene_exploratory_deg.csv.gz", index=False, compression="gzip")
    validation = deg[~deg["used_for_ot"]].copy()
    validation["validation_scope"] = "gene_not_used_in_either_winner_ot_representation"
    validation.to_csv(
        output / "non_ot_gene_validation_deg.csv.gz", index=False, compression="gzip"
    )
    annotation[~annotation["used_for_ot"]].assign(
        group_id=group_id,
        patient_id=patient,
        target_sample=target,
        rejected_n=rejected_n,
        retained_n=retained_n,
    ).to_csv(output / "non_ot_gene_contrasts.csv.gz", index=False, compression="gzip")

    diagnostics.update({
        "status": "complete",
        "tested_gene_n": len(deg),
        "non_ot_validation_gene_n": len(validation),
        "ot_feature_symbol_n": int(used_for_ot.sum()),
        "normalization": normalization,
        "primary_validation": "non-OT genes; rejected versus retained within the same target",
        "all_gene_result": "exploratory because OT feature reuse can induce selection circularity",
    })
    (output / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    (output / "SUCCESS").write_text("complete\n", encoding="utf-8")
    print(diagnostics, flush=True)


if __name__ == "__main__":
    main()
