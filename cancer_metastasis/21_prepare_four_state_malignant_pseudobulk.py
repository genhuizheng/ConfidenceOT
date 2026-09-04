"""Prepare patient-level four-state malignant pseudobulks for GSE180661.

ConfidenceOT gates both sides of every selected primary--metastasis pair.  This
workflow therefore retains four malignant states: primary retained/rejected
and metastasis retained/rejected.  A primary cell can occur in several pair
fits; it is counted once and enters the main analysis only when its cap-robust
gate agrees across every selected metastatic partner in which that primary
sample participates.  Target cells are likewise required to agree between the
baseline and source-cap sensitivity fits.  Cap- or site-discordant cells are
saved for audit but excluded from the main pseudobulks.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from common import expression_matrix, gene_keys, load_exact_side


MALIGNANT = "Ovarian.cancer.cell"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_csv", type=Path)
    parser.add_argument("baseline_root", type=Path)
    parser.add_argument("sensitivity_root", type=Path)
    parser.add_argument("robustness_csv", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--index", type=int, help="Patient index")
    parser.add_argument("--print-patient-count", action="store_true")
    parser.add_argument("--sensitivity-label", default="source090")
    parser.add_argument("--malignant-annotation", default=MALIGNANT)
    parser.add_argument("--include-exact-winner-unstable", action="store_true")
    return parser.parse_args()


def pair_id(patient: str, source: str, target: str) -> str:
    return f"{patient}__{source}__{target}"


def analysis_groups(
    manifest: pd.DataFrame, robustness: pd.DataFrame, sensitivity_label: str
) -> pd.DataFrame:
    keys = ["dataset_id", "patient_id", "target_sample"]
    sensitivity_column = (
        f"{re.sub(r'[^A-Za-z0-9]+', '_', sensitivity_label).lower()}_winner"
    )
    rows: list[dict] = []
    for values, table in manifest.groupby(keys, sort=True, dropna=False):
        record = dict(zip(keys, values))
        sources = sorted(table["source_sample"].astype(str).unique())
        record["candidate_primary_n"] = len(sources)
        if len(sources) == 1:
            record.update({
                "baseline_winner": sources[0],
                sensitivity_column: sources[0],
                "recommended_range_exact_robust": True,
                "origin_group_type": "single_primary_compatibility",
            })
        else:
            match = robustness.copy()
            for key, value in record.items():
                if key in keys:
                    match = match[match[key].astype(str).eq(str(value))]
            if len(match) != 1:
                raise RuntimeError(f"Could not uniquely resolve origin group: {record}")
            selected = match.iloc[0]
            record.update({
                "baseline_winner": str(selected["baseline_winner"]),
                sensitivity_column: str(selected[sensitivity_column]),
                "recommended_range_exact_robust": bool(
                    selected["recommended_range_exact_robust"]
                ),
                "origin_group_type": "multi_primary_origin_ranking",
            })
        rows.append(record)
    return pd.DataFrame(rows).sort_values(keys, kind="stable").reset_index(drop=True)


def one_result_file(root: Path, pair: str, name: str) -> Path:
    matches = sorted((root / pair).glob(f"scope_malignant/*/{name}"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {name} for {pair} under {root}; found {len(matches)}")
    return matches[0]


def read_gate(root: Path, pair: str, side: str, prefix: str) -> pd.DataFrame:
    path = one_result_file(root, pair, "cell_confidence.csv")
    table = pd.read_csv(
        path,
        usecols=["method", "side", "observation_id", "rejected",
                 "normalized_rejection_score", "signed_rejection_margin"],
    )
    table = table[table["method"].eq("M4-E") & table["side"].eq(side)].copy()
    table = table.drop(columns=["method", "side"]).rename(columns={
        "rejected": f"{prefix}_rejected",
        "normalized_rejection_score": f"{prefix}_rejection_score",
        "signed_rejection_margin": f"{prefix}_signed_margin",
    })
    if table["observation_id"].duplicated().any():
        raise RuntimeError(f"Duplicate {side} observation IDs in {path}")
    return table


def read_hvg(root: Path, pair: str) -> set[str]:
    with one_result_file(root, pair, "run.json").open(encoding="utf-8") as handle:
        return {str(value) for value in json.load(handle).get("hvg", [])}


def cap_robust_gate(baseline: pd.DataFrame, sensitivity: pd.DataFrame) -> pd.DataFrame:
    joined = baseline.merge(
        sensitivity, on="observation_id", how="inner", validate="one_to_one"
    )
    sensitivity_rejected = next(
        column for column in joined if column.endswith("_rejected")
        and column != "baseline_rejected"
    )
    joined["pair_robust_status"] = np.select(
        [
            joined["baseline_rejected"] & joined[sensitivity_rejected],
            ~joined["baseline_rejected"] & ~joined[sensitivity_rejected],
        ],
        ["rejected", "retained"],
        default="cap_discordant",
    )
    return joined


def consensus_classification(
    records: pd.DataFrame, expected_occurrences: dict[str, int]
) -> pd.DataFrame:
    """Collapse repeated pair occurrences without duplicating biological cells."""
    rows = []
    for entity_id, table in records.groupby("entity_id", sort=False):
        sample = str(table["sample"].iloc[0])
        statuses = set(table["pair_robust_status"])
        observed = int(table["pair_id"].nunique())
        expected = int(expected_occurrences[sample])
        if observed != expected:
            status = "incomplete_pair_coverage"
        elif statuses == {"retained"}:
            status = "retained"
        elif statuses == {"rejected"}:
            status = "rejected"
        else:
            status = "site_or_cap_discordant"
        rows.append({
            "entity_id": entity_id,
            "sample": sample,
            "observation_id": str(table["observation_id"].iloc[0]),
            "consensus_status": status,
            "observed_pair_n": observed,
            "expected_pair_n": expected,
            "mean_baseline_rejection_score": float(
                table["baseline_rejection_score"].mean()
            ),
        })
    return pd.DataFrame(rows)


def annotation_values(data) -> np.ndarray:
    for column in ("cell_type", "annotation", "celltype", "cell_type_final"):
        if column in data.obs:
            return data.obs[column].astype(str).to_numpy()
    raise KeyError("H5AD has no cell-type annotation column")


def gene_symbols(data) -> np.ndarray:
    symbols = np.asarray(data.var_names.astype(str), dtype=str)
    if "gene_symbol" in data.var:
        candidate = data.var["gene_symbol"].astype(str).str.strip().to_numpy(dtype=str)
        valid = ~pd.Series(candidate).str.lower().isin(
            {"", "na", "n/a", "nan", "none", "null", "<na>"}
        ).to_numpy()
        symbols = np.where(valid, candidate, symbols)
    return np.asarray([value.strip() for value in symbols], dtype=str)


def collapse_gene_symbols(data, ot_features: set[str]):
    matrix = sparse.csr_matrix(expression_matrix(data), dtype=np.float64)
    if matrix.data.size and (
        matrix.data.min() < 0 or not np.allclose(matrix.data, np.round(matrix.data))
    ):
        raise ValueError("Patient pseudobulk requires non-negative integer raw counts")
    symbols = gene_symbols(data)
    keys = gene_keys(data)
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


def add_vector(destination: defaultdict[str, int], genes: np.ndarray, values: np.ndarray):
    for gene, value in zip(genes, values):
        if value:
            destination[str(gene)] += int(value)


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest_csv)
    robustness = pd.read_csv(args.robustness_csv)
    groups = analysis_groups(manifest, robustness, args.sensitivity_label)
    if not args.include_exact_winner_unstable:
        groups = groups[groups["recommended_range_exact_robust"]].copy()
    patients = sorted(groups["patient_id"].astype(str).unique())
    if args.print_patient_count:
        print(len(patients))
        return
    if args.index is None:
        raise ValueError("--index is required unless --print-patient-count is used")
    if args.index < 0 or args.index >= len(patients):
        raise IndexError(f"--index {args.index} outside 0..{len(patients) - 1}")
    patient = patients[args.index]
    patient_groups = groups[groups["patient_id"].astype(str).eq(patient)].copy()
    sensitivity_column = (
        f"{re.sub(r'[^A-Za-z0-9]+', '_', args.sensitivity_label).lower()}_winner"
    )
    output = args.output_root / "patients" / f"{args.index:03d}_{patient}"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "PSEUDOBULK_READY").is_file():
        print(f"SKIP completed patient={patient}")
        return

    source_records: list[pd.DataFrame] = []
    target_records: list[pd.DataFrame] = []
    source_paths: dict[str, list[str]] = {}
    target_paths: dict[str, list[str]] = {}
    source_pair_counts: dict[str, int] = defaultdict(int)
    target_pair_counts: dict[str, int] = defaultdict(int)
    ot_features: set[str] = set()
    pair_rows = []
    for _, group in patient_groups.iterrows():
        source = str(group["baseline_winner"])
        sensitivity_source = str(group[sensitivity_column])
        if source != sensitivity_source:
            raise RuntimeError("Exact-unstable winner entered the main four-state analysis")
        target = str(group["target_sample"])
        pair = pair_id(patient, source, target)
        match = manifest[manifest["pair_id"].astype(str).eq(pair)]
        if len(match) != 1:
            raise RuntimeError(f"Manifest did not uniquely resolve {pair}")
        row = match.iloc[0]
        source_paths.setdefault(source, json.loads(str(row["source_h5ads_json"])))
        target_paths.setdefault(target, json.loads(str(row["target_h5ads_json"])))
        source_pair_counts[source] += 1
        target_pair_counts[target] += 1
        for side, sample, records in (
            ("source", source, source_records), ("target", target, target_records)
        ):
            gate = cap_robust_gate(
                read_gate(args.baseline_root, pair, side, "baseline"),
                read_gate(args.sensitivity_root, pair, side, args.sensitivity_label),
            )
            gate.insert(0, "pair_id", pair)
            gate.insert(1, "sample", sample)
            gate.insert(2, "entity_id", sample + "::" + gate["observation_id"].astype(str))
            records.append(gate)
        ot_features |= read_hvg(args.baseline_root, pair)
        ot_features |= read_hvg(args.sensitivity_root, pair)
        pair_rows.append({"pair_id": pair, "source_sample": source,
                          "target_sample": target})

    source_long = pd.concat(source_records, ignore_index=True)
    target_long = pd.concat(target_records, ignore_index=True)
    source_cells = consensus_classification(source_long, source_pair_counts)
    target_cells = consensus_classification(target_long, target_pair_counts)
    source_cells.insert(0, "side", "primary")
    target_cells.insert(0, "side", "metastasis")
    classification = pd.concat([source_cells, target_cells], ignore_index=True)
    classification.to_csv(
        output / "four_state_cell_classification.csv.gz", index=False, compression="gzip"
    )

    counts: dict[str, defaultdict[str, int]] = {
        state: defaultdict(int) for state in (
            "primary_retained", "primary_rejected", "primary_nonmalignant",
            "metastasis_retained", "metastasis_rejected", "metastasis_nonmalignant",
        )
    }
    cell_n = {state: 0 for state in counts}
    used_for_ot: set[str] = set()
    for side, paths_by_sample, cells, prefix in (
        ("primary", source_paths, source_cells, "primary"),
        ("metastasis", target_paths, target_cells, "metastasis"),
    ):
        for sample, paths in paths_by_sample.items():
            data = load_exact_side(paths, sample)
            annotations = annotation_values(data)
            matrix, genes, used = collapse_gene_symbols(data, ot_features)
            used_for_ot.update(genes[used])
            lookup = {str(value): index for index, value in enumerate(data.obs_names)}
            sample_cells = cells[cells["sample"].eq(sample)]
            for status in ("retained", "rejected"):
                ids = sample_cells.loc[
                    sample_cells["consensus_status"].eq(status), "observation_id"
                ].astype(str).tolist()
                missing = [value for value in ids if value not in lookup]
                if missing:
                    raise KeyError(f"{len(missing)} {side} gate IDs absent from {sample}")
                indices = np.asarray([lookup[value] for value in ids], dtype=np.int64)
                values = np.asarray(matrix[indices].sum(axis=0)).ravel()
                state = f"{prefix}_{status}"
                add_vector(counts[state], genes, values)
                cell_n[state] += len(indices)
            background = np.flatnonzero(annotations != args.malignant_annotation)
            values = np.asarray(matrix[background].sum(axis=0)).ravel()
            state = f"{prefix}_nonmalignant"
            add_vector(counts[state], genes, values)
            cell_n[state] += len(background)

    core = [
        ("primary_rejected_vs_primary_retained", "primary_rejected", "primary_retained"),
        ("metastasis_rejected_vs_metastasis_retained", "metastasis_rejected", "metastasis_retained"),
        ("metastasis_retained_vs_primary_retained", "metastasis_retained", "primary_retained"),
        ("metastasis_rejected_vs_primary_retained", "metastasis_rejected", "primary_retained"),
        ("metastasis_rejected_vs_primary_rejected", "metastasis_rejected", "primary_rejected"),
        ("metastasis_retained_vs_primary_rejected", "metastasis_retained", "primary_rejected"),
    ]
    background = [
        ("primary_retained_vs_primary_nonmalignant", "primary_retained", "primary_nonmalignant"),
        ("primary_rejected_vs_primary_nonmalignant", "primary_rejected", "primary_nonmalignant"),
        ("metastasis_retained_vs_metastasis_nonmalignant", "metastasis_retained", "metastasis_nonmalignant"),
        ("metastasis_rejected_vs_metastasis_nonmalignant", "metastasis_rejected", "metastasis_nonmalignant"),
    ]
    genes = sorted(set().union(*(set(value) for value in counts.values())))
    records = []
    vectors = []
    for family, definitions in (("malignant_four_state", core),
                                ("same_compartment_background", background)):
        for contrast, case, reference in definitions:
            for comparison_status, state in (("case", case), ("reference", reference)):
                sample_id = f"{patient}__{contrast}__{comparison_status}"
                records.append({
                    "sample_id": sample_id, "patient_id": patient,
                    "group_id": patient,
                    "contrast": contrast, "comparison_status": comparison_status,
                    "cell_set": state, "cell_n": cell_n[state],
                    "contrast_family": family,
                    "selected_pair_n": len(pair_rows),
                })
                vectors.append([counts[state].get(gene, 0) for gene in genes])
    pseudobulk = pd.DataFrame(
        np.asarray(vectors, dtype=np.int64),
        index=pd.Index([record["sample_id"] for record in records], name="sample_id"),
        columns=genes,
    )
    pseudobulk.to_csv(output / "pseudobulk_raw_counts.csv.gz", compression="gzip")
    pd.DataFrame(records).to_csv(output / "pseudobulk_sample_metadata.csv", index=False)
    pd.DataFrame({
        "gene": genes,
        "used_for_ot": [gene in used_for_ot for gene in genes],
    }).to_csv(output / "pseudobulk_gene_metadata.csv.gz", index=False, compression="gzip")
    pd.DataFrame(pair_rows).to_csv(output / "selected_pairs.csv", index=False)
    report = {
        "patient_id": patient,
        "selected_pair_n": len(pair_rows),
        "exact_winner_robust_only": not args.include_exact_winner_unstable,
        "state_cell_n": cell_n,
        "source_consensus_definition": (
            "counted once; same cap-robust gate across every selected metastatic partner"
        ),
        "excluded_source_status_counts": source_cells["consensus_status"].value_counts().to_dict(),
        "excluded_target_status_counts": target_cells["consensus_status"].value_counts().to_dict(),
        "core_contrasts": [value[0] for value in core],
        "background_contrasts": [value[0] for value in background],
    }
    (output / "diagnostics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "PSEUDOBULK_READY").write_text("ready\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
