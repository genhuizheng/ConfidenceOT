"""Export stored UMAP coordinates, gate labels and Hallmark scores per cell.

One table feeding two figures. The first shows which cells the gate kept, on the
embedding the original authors published rather than a fresh one:
``24_visualize_four_state_confidence_umap.py`` recomputes neighbours and UMAP
with scanpy, which costs an allocation and yields coordinates nobody has seen,
while the stored ones are free and already familiar from the paper.

The second answers the premise directly. If the retained primary cells really
resemble the metastasis, their pathway scores should sit between the rejected
primary cells and the metastasis, leaning towards the metastasis. Three groups
-- primary retained, primary rejected, metastasis -- against a few Hallmark sets
settles that at the level of single cells, and shows in one panel whether a
pathway that agrees does so for the stated reason: androgen response low *and*
proliferation high in the retained cells would agree with the metastasis on one
axis while contradicting it on another.

Both sides are exported in one run, because a figure comparing primary to
metastasis needs them on one scale and from one pass.

Coordinates come from one of two places:

* ``--umap-obs-columns`` for GSE180661, whose per-cell coordinates survived into
  the depth-equalised H5ADs as ordinary obs columns.
* ``--umap-h5ad`` for GSE271675, whose embedding lives in ``obsm`` of the large
  source object. Only the two coordinate columns and the cell index are read, so
  the fifteen gigabytes never enter memory.

Scores are mean log-CPM over each set's genes present in the matrix, and the
sets come from the same Hallmark GMT the gene-set tests use, so a score and an
enrichment result cannot disagree about what a pathway contains.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

# Python puts this file's own directory on the path, which for a tool in a
# subdirectory is tools/ rather than the package beside it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import expression_matrix, load_exact_side  # noqa: E402


DEFAULT_SETS = (
    "HALLMARK_ANDROGEN_RESPONSE",
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
    "HALLMARK_INTERFERON_GAMMA_RESPONSE",
    "HALLMARK_G2M_CHECKPOINT",
    "HALLMARK_E2F_TARGETS",
)


def read_gmt(path: Path, wanted: tuple[str, ...]) -> dict[str, list[str]]:
    sets: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > 2 and parts[0] in wanted:
                sets[parts[0]] = [gene.strip() for gene in parts[2:] if gene.strip()]
    missing = [name for name in wanted if name not in sets]
    if missing:
        raise RuntimeError(f"{path}: sets not found: {missing}")
    return sets


def read_obsm_umap(path: Path, key: str) -> pd.DataFrame:
    """Read one obsm embedding and the cell index, without loading the matrix."""
    with h5py.File(path, "r") as handle:
        if "obsm" not in handle or key not in handle["obsm"]:
            raise RuntimeError(
                f"{key!r} not in obsm; available: {list(handle.get('obsm', []))}"
            )
        coordinates = np.asarray(handle["obsm"][key][:, :2], dtype=np.float64)
        group = handle["obs"]
        index_key = group.attrs.get("_index", "_index")
        index_key = index_key.decode() if isinstance(index_key, bytes) else str(index_key)
        names = group[index_key][:]
    names = [v.decode() if isinstance(v, bytes) else str(v) for v in names]
    return pd.DataFrame({"observation_id": names,
                         "umap_1": coordinates[:, 0], "umap_2": coordinates[:, 1]})


def symbol_index(data) -> dict[str, int]:
    symbols = np.asarray(data.var_names.astype(str))
    if "gene_symbol" in data.var:
        candidate = data.var["gene_symbol"].astype(str).str.strip().to_numpy(dtype=str)
        valid = ~pd.Series(candidate).str.lower().isin(
            {"", "na", "n/a", "nan", "none", "null", "<na>"}).to_numpy()
        symbols = np.where(valid, candidate, symbols)
    index: dict[str, int] = {}
    for position, gene in enumerate(symbols):
        index.setdefault(str(gene).strip(), position)
    return index


def set_scores(data, sets: dict[str, list[str]]) -> pd.DataFrame:
    matrix = sparse.csr_matrix(expression_matrix(data), dtype=np.float64)
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    totals[totals == 0] = 1.0
    scaled = (sparse.diags(1e6 / totals) @ matrix).tocsr()
    scaled.data = np.log1p(scaled.data)
    index = symbol_index(data)
    scores = {}
    for name, genes in sets.items():
        columns = [index[gene] for gene in genes if gene in index]
        label = name.replace("HALLMARK_", "").lower()
        scores[label] = (
            np.asarray(scaled[:, columns].mean(axis=1)).ravel()
            if columns else np.full(scaled.shape[0], np.nan)
        )
    return pd.DataFrame(scores)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_csv", type=Path)
    parser.add_argument("gate_root", type=Path)
    parser.add_argument("hallmark_gmt", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--scope", default="scope_malignant")
    parser.add_argument("--method", default="M4-E")
    parser.add_argument("--hallmark-set", action="append", dest="sets", default=None)
    parser.add_argument("--umap-obs-columns", nargs=2, default=None, metavar=("X", "Y"))
    parser.add_argument("--umap-h5ad", type=Path, default=None)
    parser.add_argument("--umap-obsm-key", default="X_umap_harmony")
    args = parser.parse_args()

    wanted = tuple(args.sets) if args.sets else DEFAULT_SETS
    sets = read_gmt(args.hallmark_gmt, wanted)
    print(f"gene sets: " + ", ".join(f"{k.replace('HALLMARK_','')}({len(v)})"
                                     for k, v in sets.items()))

    stored = None
    if args.umap_h5ad:
        stored = read_obsm_umap(args.umap_h5ad, args.umap_obsm_key)
        stored = stored.drop_duplicates("observation_id").set_index("observation_id")
        print(f"embedding: {len(stored)} cells from {args.umap_obsm_key}")
    elif not args.umap_obs_columns:
        print("no embedding requested; coordinates will be blank")

    manifest = pd.read_csv(args.manifest_csv)
    frames: list[pd.DataFrame] = []
    # Per side, so a metastasis shared by several primaries is read once.
    for side in ("source", "target"):
        seen: set[tuple[str, str]] = set()
        for row in manifest.to_dict("records"):
            sample = str(row[f"{side}_sample"])
            key = (str(row["patient_id"]), sample)
            if key in seen:
                continue
            matches = sorted(
                args.gate_root.glob(f"{row['pair_id']}/{args.scope}/*/cell_confidence.csv")
            )
            if not matches:
                continue
            gate = pd.read_csv(
                matches[0], usecols=["method", "side", "observation_id", "retained"]
            )
            gate = gate.loc[gate["method"].eq(args.method) & gate["side"].eq(side)]
            if gate.empty:
                continue
            seen.add(key)

            data = load_exact_side(
                [str(v) for v in json.loads(str(row[f"{side}_h5ads_json"]))], sample
            )
            table = set_scores(data, sets)
            table.insert(0, "observation_id", np.asarray(data.obs_names.astype(str)))
            if args.umap_obs_columns:
                for target, column in zip(("umap_1", "umap_2"), args.umap_obs_columns):
                    table[target] = (
                        pd.to_numeric(data.obs[column], errors="coerce").to_numpy()
                        if column in data.obs else np.nan
                    )
            elif stored is not None:
                for target in ("umap_1", "umap_2"):
                    table[target] = table["observation_id"].map(stored[target]).to_numpy()
            else:
                table["umap_1"] = table["umap_2"] = np.nan

            merged = gate.assign(
                observation_id=gate["observation_id"].astype(str)
            ).merge(table, on="observation_id", how="inner")
            merged["patient_id"] = str(row["patient_id"])
            merged["sample_id"] = sample
            merged["side"] = "primary" if side == "source" else "metastasis"
            frames.append(merged.drop(columns=["method"]))
            del data
        print(f"{side}: {len(seen)} samples")

    if not frames:
        raise RuntimeError("No sample produced both a gate and scores")
    cells = pd.concat(frames, ignore_index=True)
    cells["retained"] = cells["retained"].astype(bool)
    # The group a figure splits on: with one-sided rejection the metastatic side
    # is never gated, so it is one group rather than two.
    cells["group"] = np.where(
        cells["side"].eq("metastasis"), "metastasis",
        np.where(cells["retained"], "primary retained", "primary rejected"),
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if args.output_csv.name.endswith(".gz") else None
    cells.to_csv(args.output_csv, index=False, compression=compression)

    placed = cells[["umap_1", "umap_2"]].notna().all(axis=1)
    print(json.dumps({
        "cells": int(len(cells)),
        "with_coordinates": int(placed.sum()),
        "group_counts": {k: int(v) for k, v in cells["group"].value_counts().items()},
        "output": str(args.output_csv),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
