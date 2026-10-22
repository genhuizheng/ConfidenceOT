"""One frozen joint UMAP per patient, for showing the CNV / RNA strategies patient by patient.

    python cancer_metastasis/53_patient_umap.py --cells EXPORT/cells \\
        --manifest PANCANCER_ELIGIBLE.csv --manifest PROSTATE_ELIGIBLE.csv \\
        --out OUT [--datasets GSE180661 ...] [--task-index 0 --task-count 8]

**Cells.** A patient's cells are the union of the reference cells of every pair of that patient in
the 51_strategy_export.py package (cells/<dataset>/<pair>.csv.gz), keyed by cell_uid
(dataset:patient:sample:observation_id, the same in every pair). A primary sample shared by
several pairs therefore enters once; where pairs subsampled a large side differently, the union
holds every cell any pair used.

**Coordinates.** 48_reference_umap.py's recipe, unchanged (its constants are imported): original
counts of each sample's h5ad (the manifests' *_h5ads_json), normalize_total 1e4, log1p, 2,000
Seurat HVGs, scale (max 10), PCA 40 (arpack), 20 neighbours, UMAP, seed 20260904; both sides
stacked. Coordinates compare only within one patient and are never recomputed: a patient with
DONE is skipped.

Writes OUT/patients/<dataset>/<patient>/: patient_umap.csv.gz (cell_uid, dataset_id, patient_id,
side, sample, observation_id, umap_1, umap_2, total_counts, detected_genes, pairs = the pairs the
cell is a reference cell of), reference.json, hvg.txt, DONE (or FAILED with the error).
No OT is run and no gate result is read or written.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import expression_matrix, load_exact_side  # noqa: E402

_spec = importlib.util.spec_from_file_location("reference48", HERE / "48_reference_umap.py")
reference48 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reference48)
SEED, N_HVG, N_PCS, N_NEIGHBORS = reference48.SEED, reference48.N_HVG, reference48.N_PCS, reference48.N_NEIGHBORS
safe = reference48.safe


def sample_sources(paths: list[Path]) -> dict:
    """(dataset, patient, sample) -> h5ads_json, from both sides of every manifest row; a sample
    that two rows give different files for is an error, never a guess. The patient is part of the
    key because sample names repeat between patients (GSE180661 names samples by site)."""
    found: dict = {}
    for path in paths:
        table = pd.read_csv(path)
        for side in ("source", "target"):
            for dataset, patient, sample, files in table[["dataset_id", "patient_id", f"{side}_sample",
                                                          f"{side}_h5ads_json"]].itertuples(index=False):
                key = (str(dataset), str(patient), str(sample))
                if key in found and found[key] != str(files):
                    raise SystemExit(f"{key}: two manifests name different h5ad files")
                found[key] = str(files)
    return found


def patient_cells(cells_root: Path, datasets: list[str] | None) -> pd.DataFrame:
    frames = []
    for path in sorted(cells_root.glob("*/*.csv.gz")):
        if datasets and path.parent.name not in datasets:
            continue
        frames.append(pd.read_csv(path, usecols=["cell_uid", "pair_id", "dataset_id", "patient_id", "side",
                                                 "sample", "observation_id"]))
    table = pd.concat(frames, ignore_index=True)
    for column in ("cell_uid", "dataset_id", "patient_id", "sample", "observation_id"):
        table[column] = table[column].astype(str)
    pairs = table.groupby("cell_uid")["pair_id"].agg(lambda s: ";".join(sorted(set(s)))).rename("pairs")
    cells = table.drop_duplicates("cell_uid").drop(columns="pair_id").merge(pairs, on="cell_uid")
    if cells.groupby("cell_uid")["side"].nunique().max() > 1:
        raise SystemExit("a cell_uid is primary in one pair and metastasis in another")
    return cells


def patient_umap(cells: pd.DataFrame, sources: dict, out: Path) -> dict:
    import anndata as ad
    import scanpy as sc

    pieces = []
    for (dataset, patient, sample), part in cells.groupby(["dataset_id", "patient_id", "sample"], sort=True):
        if (dataset, patient, sample) not in sources:
            raise KeyError(f"{dataset} {patient} {sample}: no manifest row names this sample's h5ad")
        data = load_exact_side(json.loads(sources[(dataset, patient, sample)]), sample)
        data.obs_names = data.obs_names.astype(str)
        wanted = part["observation_id"].tolist()
        missing = sorted(set(wanted) - set(data.obs_names))
        if missing:
            raise KeyError(f"{sample}: {len(missing)} reference cells are absent from the h5ad, e.g. {missing[:3]}")
        data = data[wanted].copy()
        matrix = sparse.csr_matrix(expression_matrix(data), dtype=np.float64)
        if matrix.data.size and (matrix.data.min() < 0 or not np.allclose(matrix.data, np.round(matrix.data))):
            raise ValueError(f"{sample}: the expression matrix is not raw counts")
        piece = ad.AnnData(X=matrix.astype(np.float32), var=pd.DataFrame(index=data.var_names.astype(str)))
        piece.obs_names = part["cell_uid"].tolist()
        piece.obs["total_counts"] = np.asarray(matrix.sum(axis=1)).ravel()
        piece.obs["detected_genes"] = np.asarray((matrix > 0).sum(axis=1)).ravel()
        pieces.append(piece)
    data = ad.concat(pieces, join="inner", merge="same", index_unique=None)
    if data.obs_names.has_duplicates:
        raise RuntimeError("duplicate cell ids across samples")
    genes_shared = int(data.n_vars)
    sc.pp.normalize_total(data, target_sum=1e4)
    sc.pp.log1p(data)
    sc.pp.highly_variable_genes(data, n_top_genes=N_HVG, flavor="seurat", subset=True)
    hvg = list(data.var_names)
    sc.pp.scale(data, max_value=10)
    n_pcs = min(N_PCS, data.n_vars - 1, data.n_obs - 1)
    sc.tl.pca(data, n_comps=n_pcs, svd_solver="arpack", random_state=SEED)
    sc.pp.neighbors(data, n_neighbors=N_NEIGHBORS, n_pcs=n_pcs, random_state=SEED)
    sc.tl.umap(data, random_state=SEED)
    table = cells.set_index("cell_uid").loc[data.obs_names].rename_axis("cell_uid").reset_index()
    table["umap_1"], table["umap_2"] = data.obsm["X_umap"][:, 0], data.obsm["X_umap"][:, 1]
    table["total_counts"] = data.obs["total_counts"].to_numpy()
    table["detected_genes"] = data.obs["detected_genes"].to_numpy()
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "patient_umap.csv.gz", index=False, compression="gzip")
    record = {"dataset_id": str(cells["dataset_id"].iloc[0]), "patient_id": str(cells["patient_id"].iloc[0]),
              "cells": {side: int((table["side"] == side).sum()) for side in ("primary", "metastasis")},
              "samples": {side: sorted(table.loc[table["side"] == side, "sample"].unique().tolist())
                          for side in ("primary", "metastasis")},
              "pairs": sorted(set(";".join(table["pairs"]).split(";"))),
              "genes_shared_by_samples": genes_shared,
              "n_hvg": N_HVG, "n_pcs": n_pcs, "n_neighbors": N_NEIGHBORS, "seed": SEED,
              "hvg_md5": hashlib.md5("\n".join(hvg).encode()).hexdigest(), "scanpy": sc.__version__}
    (out / "reference.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    (out / "hvg.txt").write_text("\n".join(hvg) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", type=Path, required=True, help="the export package's cells/ directory")
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--task-count", type=int, default=1)
    args = parser.parse_args()
    sources = sample_sources(args.manifest)
    cells = patient_cells(args.cells, args.datasets)
    failures = 0
    patients = sorted(cells.groupby(["dataset_id", "patient_id"]).groups)
    for index, (dataset, patient) in enumerate(patients):
        if index % args.task_count != args.task_index:
            continue
        out = args.out / "patients" / dataset / safe(patient)
        if (out / "DONE").exists():
            print(f"skip {dataset} {patient}: DONE", flush=True)
            continue
        try:
            part = cells[(cells["dataset_id"] == dataset) & (cells["patient_id"] == patient)]
            record = patient_umap(part, sources, out)
            (out / "DONE").write_text("done\n", encoding="utf-8")
            (out / "FAILED").unlink(missing_ok=True)
            print(f"{dataset} {patient}: {record['cells']} cells, {len(record['pairs'])} pairs", flush=True)
        except Exception as error:  # noqa: BLE001 - a failed patient is recorded and reported, never hidden
            failures += 1
            out.mkdir(parents=True, exist_ok=True)
            (out / "FAILED").write_text(f"{type(error).__name__}: {error}\n{traceback.format_exc()}", encoding="utf-8")
            print(f"FAILED {dataset} {patient}: {type(error).__name__}: {error}", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
