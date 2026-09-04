"""Prepare primary-only pseudobulks for one primary--metastasis OT pair.

The metastatic malignant sample is only the reference used by ConfidenceOT.
Both pseudobulk profiles contain primary malignant cells exclusively.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from cancer_metastasis.common import expression_matrix, gene_keys, load_exact_side


CONTRAST = "primary_metastasis_compatible_vs_primary_restricted"


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def paths_for(row: pd.Series, side: str) -> list[str]:
    column = f"{side}_h5ads_json"
    if column in row and pd.notna(row[column]):
        return [str(value) for value in json.loads(str(row[column]))]
    return [str(row[f"{side}_h5ad"])]


def annotation_values(data) -> np.ndarray:
    for column in ("cell_type", "annotation", "celltype", "cell_type_final"):
        if column in data.obs:
            return data.obs[column].astype(str).to_numpy()
    raise KeyError("H5AD has no cell-type annotation column")


def gene_symbols(data) -> np.ndarray:
    symbols = np.asarray(data.var_names.astype(str), dtype=str)
    if "gene_symbol" in data.var:
        candidate = data.var["gene_symbol"].astype(str).str.strip().to_numpy(dtype=str)
        missing = pd.Series(candidate).str.lower().isin(
            {"", "na", "n/a", "nan", "none", "null", "<na>"}
        ).to_numpy()
        symbols = np.where(~missing, candidate, symbols)
    return np.asarray([value.strip() for value in symbols], dtype=str)


def collapsed_raw_counts(data, ot_features: set[str]):
    matrix = sparse.csr_matrix(expression_matrix(data), dtype=np.float64)
    if matrix.data.size and (
        matrix.data.min() < 0 or not np.allclose(matrix.data, np.round(matrix.data))
    ):
        raise ValueError("Primary pseudobulk requires non-negative integer raw counts")
    symbols = gene_symbols(data)
    keys = gene_keys(data)
    lookup: dict[str, int] = {}
    inverse = np.empty(len(symbols), dtype=np.int64)
    ordered: list[str] = []
    for index, symbol in enumerate(symbols):
        if not symbol:
            symbol = str(keys[index])
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
    for index, (key, symbol) in enumerate(zip(keys, symbols)):
        if str(key) in ot_features or str(symbol) in ot_features:
            used[inverse[index]] = True
    return collapsed, np.asarray(ordered, dtype=str), used


def one_result_directory(root: Path, pair_id: str, budget_tag: str | None) -> Path:
    scope = root / pair_id / "scope_malignant"
    if budget_tag:
        candidates = [scope / budget_tag]
    else:
        candidates = sorted(path.parent for path in scope.glob("*/SUCCESS"))
    candidates = [path for path in candidates if (path / "SUCCESS").is_file()]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one completed malignant result for {pair_id}; found {len(candidates)}"
        )
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_csv", type=Path)
    parser.add_argument("malignant_ot_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--budget-tag")
    parser.add_argument("--method", default="M4-E")
    parser.add_argument("--malignant-annotation", default="Ovarian.cancer.cell")
    parser.add_argument("--minimum-cells-per-state", type=int, default=20)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest_csv)
    if args.index < 0 or args.index >= len(manifest):
        raise IndexError(f"--index outside 0..{len(manifest) - 1}")
    row = manifest.iloc[args.index]
    pair = str(row["pair_id"])
    output = args.output_root / "groups" / f"{args.index:03d}_{safe_name(pair)}"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "PSEUDOBULK_READY").is_file():
        print(f"SKIP completed pair={pair}")
        return

    result_dir = one_result_directory(args.malignant_ot_root, pair, args.budget_tag)
    confidence = pd.read_csv(result_dir / "cell_confidence.csv")
    confidence = confidence[
        confidence["method"].eq(args.method) & confidence["side"].eq("source")
    ].copy()
    if confidence.empty or confidence["observation_id"].duplicated().any():
        raise RuntimeError(f"Invalid source confidence rows for {pair}")

    with (result_dir / "run.json").open(encoding="utf-8") as handle:
        run = json.load(handle)
    ot_features = {str(value) for value in run.get("hvg", [])}
    source = load_exact_side(paths_for(row, "source"), str(row["source_sample"]))
    malignant = annotation_values(source) == args.malignant_annotation
    source = source[malignant].copy()
    matrix, genes, used_for_ot = collapsed_raw_counts(source, ot_features)
    lookup = {str(value): index for index, value in enumerate(source.obs_names)}
    missing = [
        value for value in confidence["observation_id"].astype(str) if value not in lookup
    ]
    if missing:
        raise KeyError(f"{len(missing)} ConfidenceOT source IDs absent from primary H5AD")

    definitions = [
        ("case", "putative_metastasis_compatible_primary", True),
        ("reference", "putative_primary_restricted", False),
    ]
    count_rows = []
    metadata_rows = []
    for comparison_status, state, retained in definitions:
        ids = confidence.loc[
            confidence["retained"].astype(bool).eq(retained), "observation_id"
        ].astype(str).tolist()
        indices = np.asarray([lookup[value] for value in ids], dtype=np.int64)
        values = np.asarray(matrix[indices].sum(axis=0)).ravel().astype(np.int64)
        sample_id = f"{pair}__{comparison_status}"
        count_rows.append(pd.Series(values, index=genes, name=sample_id))
        metastatic_site = next(
            (
                str(row[column]) for column in
                ("target_site", "metastatic_site", "target_sample")
                if column in row and pd.notna(row[column])
            ),
            str(row["target_sample"]),
        )
        metadata_rows.append({
            "sample_id": sample_id,
            "dataset_id": str(row.get("dataset_id", "")),
            "patient_id": str(row["patient_id"]),
            "pair_id": pair,
            "group_id": pair,
            "primary_sample": str(row["source_sample"]),
            "metastatic_sample": str(row["target_sample"]),
            "metastatic_site": metastatic_site,
            "contrast": CONTRAST,
            "comparison_status": comparison_status,
            "state": state,
            "cell_n": len(indices),
            "library_size": int(values.sum()),
            "analysis_compartment": "primary_malignant_cells_only",
        })

    counts = pd.DataFrame(count_rows).fillna(0).astype(np.int64)
    metadata = pd.DataFrame(metadata_rows)
    counts.to_csv(output / "pseudobulk_raw_counts.csv.gz", compression="gzip")
    metadata.to_csv(output / "pseudobulk_sample_metadata.csv", index=False)
    pd.DataFrame({
        "gene": genes,
        "used_for_ot": used_for_ot,
    }).to_csv(output / "pseudobulk_gene_metadata.csv.gz", index=False, compression="gzip")
    confidence.to_csv(output / "primary_cell_classification.csv.gz", index=False,
                      compression="gzip")

    state_cells = dict(zip(metadata["state"], metadata["cell_n"]))
    ready = all(value >= args.minimum_cells_per_state for value in state_cells.values())
    report = {
        "pair_id": pair,
        "patient_id": str(row["patient_id"]),
        "primary_sample": str(row["source_sample"]),
        "metastatic_sample": str(row["target_sample"]),
        "method": args.method,
        "contrast": CONTRAST,
        "state_cell_n": state_cells,
        "primary_malignant_h5ad_n": int(source.n_obs),
        "confidenceot_analyzed_source_n": int(len(confidence)),
        "all_genes_preserved": True,
        "metastatic_cells_in_pseudobulk": False,
        "ready": ready,
    }
    (output / "diagnostics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    marker = "PSEUDOBULK_READY" if ready else "PSEUDOBULK_SKIPPED"
    (output / marker).write_text(f"{marker.lower()}\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
