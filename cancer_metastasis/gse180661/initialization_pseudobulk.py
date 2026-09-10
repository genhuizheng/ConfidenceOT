"""Build primary-malignant pseudobulks for every M4-E initialization run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cancer_metastasis.common import cell_qc_table, load_exact_side
from cancer_metastasis.gse180661.initialization_sensitivity import labels, paths_for, safe_name
from cancer_metastasis.gse180661.primary_pseudobulk import collapsed_raw_counts


def baseline_directory(root: Path, pair_id: str) -> Path:
    candidates = sorted(
        path.parent for path in (root / pair_id / "scope_malignant").glob("*/SUCCESS")
    )
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one baseline result for {pair_id}; found {len(candidates)}")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_csv", type=Path)
    parser.add_argument("baseline_ot_root", type=Path)
    parser.add_argument("initialization_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--minimum-cells-per-state", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--max-observed-cells-per-side", type=int, default=10000)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest_csv)
    row = manifest.iloc[args.index]
    pair_id = str(row["pair_id"])
    name = f"{args.index:03d}_{safe_name(pair_id)}"
    source_dir = args.initialization_root / "pairs" / name
    output = args.output_root / "groups" / name
    output.mkdir(parents=True, exist_ok=True)
    if (output / "PSEUDOBULK_READY").is_file():
        print(f"SKIP completed pair={pair_id}")
        return

    gates = pd.read_csv(source_dir / "source_initialization_gates.csv.gz")
    if gates.duplicated(["strategy", "replicate", "observation_id"]).any():
        raise RuntimeError(f"Duplicate initialization gate rows for {pair_id}")

    baseline = baseline_directory(args.baseline_ot_root, pair_id)
    run = json.loads((baseline / "run.json").read_text(encoding="utf-8"))
    source = load_exact_side(paths_for(row, "source"), str(row["source_sample"]))
    source = source[labels(source) == "Ovarian.cancer.cell"].copy()

    qc = run.get("cell_qc", {})
    if bool(qc.get("applied", False)):
        table = cell_qc_table(
            source,
            minimum_total_counts=int(qc["minimum_total_counts"]),
            minimum_detected_genes=int(qc["minimum_detected_genes"]),
            maximum_mitochondrial_percent=float(qc["maximum_mitochondrial_percent"]),
        )
        source = source[table["qc_pass"].to_numpy()].copy()

    maximum = args.max_observed_cells_per_side
    sample_rng = np.random.default_rng(args.seed + args.index * 65537)
    if maximum > 0 and source.n_obs > maximum:
        selected = np.sort(sample_rng.choice(source.n_obs, maximum, replace=False))
        source = source[selected].copy()

    source_ids = source.obs_names.astype(str).to_numpy()
    if set(source_ids) != set(gates["observation_id"].astype(str)):
        raise RuntimeError(f"Reconstructed source cells differ from initialization gates for {pair_id}")

    matrix, genes, used_for_ot = collapsed_raw_counts(
        source, {str(value) for value in run.get("hvg", [])}
    )
    lookup = {value: index for index, value in enumerate(source_ids)}
    count_rows = []
    metadata_rows = []
    configuration_rows = []

    for (strategy, replicate), gate in gates.groupby(["strategy", "replicate"], sort=True):
        gate = gate.set_index("observation_id").loc[source_ids]
        retained_gate = gate["retained"].astype(bool).to_numpy()
        cell_counts = {}
        for comparison_status, retained in (("case", True), ("reference", False)):
            indices = np.flatnonzero(retained_gate == retained)
            values = np.asarray(matrix[indices].sum(axis=0)).ravel().astype(np.int64)
            sample_id = f"{pair_id}__{strategy}_{int(replicate):02d}__{comparison_status}"
            count_rows.append(pd.Series(values, index=genes, name=sample_id))
            cell_counts[comparison_status] = len(indices)
            metadata_rows.append({
                "sample_id": sample_id,
                "patient_id": str(row["patient_id"]),
                "pair_id": pair_id,
                "primary_sample": str(row["source_sample"]),
                "metastatic_sample": str(row["target_sample"]),
                "strategy": str(strategy),
                "replicate": int(replicate),
                "comparison_status": comparison_status,
                "cell_n": len(indices),
                "library_size": int(values.sum()),
            })
        configuration_rows.append({
            "pair_id": pair_id,
            "strategy": str(strategy),
            "replicate": int(replicate),
            "retained_cell_n": cell_counts["case"],
            "rejected_cell_n": cell_counts["reference"],
            "deg_ready": min(cell_counts.values()) >= args.minimum_cells_per_state,
        })

    pd.DataFrame(count_rows).fillna(0).astype(np.int64).to_csv(
        output / "pseudobulk_raw_counts.csv.gz", compression="gzip"
    )
    pd.DataFrame(metadata_rows).to_csv(output / "pseudobulk_sample_metadata.csv", index=False)
    pd.DataFrame(configuration_rows).to_csv(output / "configuration_qc.csv", index=False)
    pd.DataFrame({"gene": genes, "used_for_ot": used_for_ot}).to_csv(
        output / "pseudobulk_gene_metadata.csv.gz", index=False, compression="gzip"
    )
    (output / "PSEUDOBULK_READY").write_text("complete\n", encoding="utf-8")
    print(pd.DataFrame(configuration_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
