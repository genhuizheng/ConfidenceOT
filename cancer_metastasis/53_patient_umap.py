"""One frozen joint UMAP per patient, for showing the CNV / RNA strategies patient by patient.

    python cancer_metastasis/53_patient_umap.py --cells EXPORT/cells \\
        --manifest PANCANCER_ELIGIBLE.csv --manifest PROSTATE_ELIGIBLE.csv \\
        --out OUT [--datasets GSE180661 ...] [--task-index 0 --task-count 8]

**Cells.** A patient's cells are the union of the reference cells of every pair of that patient in
the 51_strategy_export.py package (cells/<dataset>/<pair>.csv.gz), keyed by cell_uid
(dataset:patient:sample:observation_id, the same in every pair). A primary sample shared by
several pairs therefore enters once; where pairs subsampled a large side differently, the union
holds every cell any pair used.

**Coordinates, --space raw (default).** 48_reference_umap.py's recipe, unchanged (its constants are
imported): original counts of each sample's h5ad (the manifests' *_h5ads_json), normalize_total
1e4, log1p, 2,000 Seurat HVGs, scale (max 10), PCA 40 (arpack), 20 neighbours, UMAP, seed
20260904; both sides stacked.

**Coordinates, --space gate.** The geometry the gates see: give the depth-equalised manifests
(27_downsample_counts.py's copies, the ones the strategy OT read) and the coordinates are
ranknm256_noscale_ds_cos built by the gate's own code (common.prepare_joint_representation with
Preprocessing.from_label, primary as source and metastasis as target; 2,000 HVGs and 30 PCs, the
02_run_pair.py defaults the strategy runs used), L2-normalised as the cosine cost does, then 20
neighbours and UMAP with the same seed. The recipe is the gate's; it is fitted on the patient's
pooled cells, so it is not any single pair's geometry.

Coordinates compare only within one patient and are never recomputed: a patient with DONE is
skipped.

--group dataset: one UMAP over every patient of a dataset together (OUT/datasets/<dataset>/), same
recipe; patients then separate by their own differences as much as by anything else.

Writes OUT/patients/<dataset>/<patient>/ (or OUT/datasets/<dataset>/): patient_umap.csv.gz (cell_uid, dataset_id, patient_id,
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
from common import expression_matrix, load_exact_side, prepare_joint_representation  # noqa: E402

GATE_LABEL = "ranknm256_noscale_ds_cos"
GATE_N_HVG, GATE_N_PCS = 2000, 30  # 02_run_pair.py's --n-hvg / --n-pcs defaults; the strategy runs pass neither

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
    record = {"space": "raw", "dataset_id": str(cells["dataset_id"].iloc[0]),
              "patient_id": ";".join(sorted(cells["patient_id"].astype(str).unique())),
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


def load_side(cells: pd.DataFrame, sources: dict, missing_rows: list):
    """One side's cells as one AnnData over all its samples, obs_names = cell_uid.

    The equalised h5ads hold the cells 27_ kept for the ds analysis. The export's reference cells
    are the cycling reference UMAP's, which also counted the raw-count arms, so a few reference
    cells are not in the equalised files. No strategy gate (all ds) could see them; they are left
    out of this embedding and listed in missing_rows, never filled in."""
    import anndata as ad

    pieces = []
    for (dataset, patient, sample), part in cells.groupby(["dataset_id", "patient_id", "sample"], sort=True):
        if (dataset, patient, sample) not in sources:
            raise KeyError(f"{dataset} {patient} {sample}: no manifest row names this sample's h5ad")
        data = load_exact_side(json.loads(sources[(dataset, patient, sample)]), sample)
        data.obs_names = data.obs_names.astype(str)
        present = part["observation_id"].isin(set(data.obs_names))
        missing_rows.extend(part.loc[~present, ["cell_uid", "side", "sample", "observation_id"]].to_dict("records"))
        part = part[present]
        data = data[part["observation_id"].tolist()].copy()
        data.obs_names = part["cell_uid"].tolist()
        pieces.append(data)
    if len(pieces) == 1:
        return pieces[0]
    return ad.concat(pieces, join="inner", merge="same", uns_merge="same", index_unique=None)


def gate_umap(cells: pd.DataFrame, sources: dict, out: Path) -> dict:
    import anndata as ad
    import scanpy as sc

    from confidenceot.preprocessing import Preprocessing

    configuration = Preprocessing.from_label(GATE_LABEL)
    missing_rows: list = []
    primary = load_side(cells[cells["side"] == "primary"], sources, missing_rows)
    metastasis = load_side(cells[cells["side"] == "metastasis"], sources, missing_rows)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(missing_rows, columns=["cell_uid", "side", "sample", "observation_id"]).to_csv(
        out / "missing_from_equalised.tsv", sep="\t", index=False)
    source, target, genes, provenance = prepare_joint_representation(
        primary, metastasis, n_hvg=GATE_N_HVG, n_pcs=GATE_N_PCS, seed=SEED,
        representation=configuration.normalisation, rank_top_n=configuration.rank_top_n,
        minimum_detection_rate=configuration.minimum_detection_rate, cost=configuration.cost,
        equalise_depth=configuration.equalise_depth, scale_genes=configuration.scale_genes,
        label_stem=configuration.label_stem)
    if provenance.get("label") != GATE_LABEL:
        raise SystemExit(f"built {provenance.get('label')!r}, not {GATE_LABEL!r}")
    names = list(primary.obs_names) + list(metastasis.obs_names)
    data = ad.AnnData(X=np.zeros((len(names), 1), dtype=np.float32))
    data.obs_names = names
    data.obsm["X_gate"] = np.vstack([np.asarray(source), np.asarray(target)]).astype(np.float32)
    sc.pp.neighbors(data, n_neighbors=N_NEIGHBORS, use_rep="X_gate", random_state=SEED)
    sc.tl.umap(data, random_state=SEED)
    depth = {}
    for side, side_data in (("primary", primary), ("metastasis", metastasis)):
        matrix = sparse.csr_matrix(expression_matrix(side_data))
        depth[side] = (np.asarray(matrix.sum(axis=1)).ravel(), np.asarray((matrix > 0).sum(axis=1)).ravel())
    table = cells.set_index("cell_uid").loc[names].rename_axis("cell_uid").reset_index()
    table["umap_1"], table["umap_2"] = data.obsm["X_umap"][:, 0], data.obsm["X_umap"][:, 1]
    table["total_counts"] = np.concatenate([depth["primary"][0], depth["metastasis"][0]])
    table["detected_genes"] = np.concatenate([depth["primary"][1], depth["metastasis"][1]])
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "patient_umap.csv.gz", index=False, compression="gzip")
    counts = table["total_counts"].to_numpy(float)
    record = {"space": "gate", "label": GATE_LABEL, "dataset_id": str(cells["dataset_id"].iloc[0]),
              "patient_id": ";".join(sorted(cells["patient_id"].astype(str).unique())),
              "cells": {side: int((table["side"] == side).sum()) for side in ("primary", "metastasis")},
              "samples": {side: sorted(table.loc[table["side"] == side, "sample"].unique().tolist())
                          for side in ("primary", "metastasis")},
              "pairs": sorted(set(";".join(table["pairs"]).split(";"))),
              "reference_cells_not_in_equalised_files": {
                  side: sum(r["side"] == side for r in missing_rows) for side in ("primary", "metastasis")},
              "n_hvg": GATE_N_HVG, "n_pcs": GATE_N_PCS, "n_neighbors": N_NEIGHBORS, "seed": SEED,
              # equalised input reads as a near-constant total; recorded so a raw manifest shows
              "total_counts_quantiles": {q: float(np.quantile(counts, q)) for q in (0.05, 0.5, 0.95)},
              "genes_md5": hashlib.md5("\n".join(map(str, genes)).encode()).hexdigest(),
              "provenance": {k: v for k, v in provenance.items() if isinstance(v, (str, int, float, bool))},
              "scanpy": sc.__version__}
    (out / "reference.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    (out / "hvg.txt").write_text("\n".join(map(str, genes)) + "\n", encoding="utf-8")
    return record


REPLICATION_SEED = 20260913  # replication/plot_primary_metastasis_umap.py's defaults, kept as they were


def replication_umap(cells: pd.DataFrame, sources: dict, out: Path) -> dict:
    """The deck's old colorectal UMAP recipe (replication/plot_primary_metastasis_umap.py), unchanged:
    original counts, common.prepare_joint_representation at its defaults (log_cpm, per-gene z-score,
    2,000 HVGs, 30 PCs, squared Euclidean), then umap-learn on the PCs with 20 neighbours,
    min_dist 0.25, Euclidean metric, seed 20260913. Only the cells differ: this unit's reference cells."""
    import umap

    missing_rows: list = []
    primary = load_side(cells[cells["side"] == "primary"], sources, missing_rows)
    metastasis = load_side(cells[cells["side"] == "metastasis"], sources, missing_rows)
    if missing_rows:
        raise KeyError(f"{len(missing_rows)} reference cells are absent from the original h5ads")
    source, target, genes, provenance = prepare_joint_representation(
        primary, metastasis, n_hvg=2000, n_pcs=30, seed=REPLICATION_SEED)
    pcs = np.vstack([np.asarray(source), np.asarray(target)]).astype(np.float32)
    coordinates = umap.UMAP(n_neighbors=min(20, len(pcs) - 1), min_dist=0.25, metric="euclidean",
                            random_state=REPLICATION_SEED).fit_transform(pcs)
    names = list(primary.obs_names) + list(metastasis.obs_names)
    depth = {}
    for side, side_data in (("primary", primary), ("metastasis", metastasis)):
        matrix = sparse.csr_matrix(expression_matrix(side_data))
        depth[side] = (np.asarray(matrix.sum(axis=1)).ravel(), np.asarray((matrix > 0).sum(axis=1)).ravel())
    table = cells.set_index("cell_uid").loc[names].rename_axis("cell_uid").reset_index()
    table["umap_1"], table["umap_2"] = coordinates[:, 0], coordinates[:, 1]
    table["total_counts"] = np.concatenate([depth["primary"][0], depth["metastasis"][0]])
    table["detected_genes"] = np.concatenate([depth["primary"][1], depth["metastasis"][1]])
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "patient_umap.csv.gz", index=False, compression="gzip")
    record = {"space": "replication", "label": provenance.get("label"),
              "dataset_id": str(cells["dataset_id"].iloc[0]),
              "patient_id": ";".join(sorted(cells["patient_id"].astype(str).unique())),
              "cells": {side: int((table["side"] == side).sum()) for side in ("primary", "metastasis")},
              "pairs": sorted(set(";".join(table["pairs"]).split(";"))),
              "n_hvg": 2000, "n_pcs": 30, "n_neighbors": 20, "min_dist": 0.25, "seed": REPLICATION_SEED,
              "genes_md5": hashlib.md5("\n".join(map(str, genes)).encode()).hexdigest(),
              "umap_learn": umap.__version__}
    (out / "reference.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", type=Path, required=True, help="the export package's cells/ directory")
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--task-count", type=int, default=1)
    parser.add_argument("--space", choices=["raw", "gate", "replication"], default="raw",
                        help="raw: 48_'s recipe on original counts; gate: ranknm256_noscale_ds_cos on the "
                             "equalised counts (give the depth-equalised manifests)")
    parser.add_argument("--group", choices=["patient", "dataset"], default="patient",
                        help="one UMAP per patient (OUT/patients/<dataset>/<patient>) or one per dataset, "
                             "every patient's cells together (OUT/datasets/<dataset>)")
    args = parser.parse_args()
    sources = sample_sources(args.manifest)
    cells = patient_cells(args.cells, args.datasets)
    failures = 0
    if args.group == "dataset":
        units = [(dataset, None) for dataset in sorted(cells["dataset_id"].unique())]
    else:
        units = sorted(cells.groupby(["dataset_id", "patient_id"]).groups)
    for index, (dataset, patient) in enumerate(units):
        if index % args.task_count != args.task_index:
            continue
        out = (args.out / "datasets" / dataset if patient is None
               else args.out / "patients" / dataset / safe(patient))
        if (out / "DONE").exists():
            print(f"skip {dataset} {patient or 'all patients'}: DONE", flush=True)
            continue
        try:
            part = cells[(cells["dataset_id"] == dataset) & ((cells["patient_id"] == patient) if patient else True)]
            record = {"gate": gate_umap, "replication": replication_umap}.get(args.space, patient_umap)(part, sources, out)
            (out / "DONE").write_text("done\n", encoding="utf-8")
            (out / "FAILED").unlink(missing_ok=True)
            print(f"{dataset} {patient or 'all patients'}: {record['cells']} cells, {len(record['pairs'])} pairs", flush=True)
        except Exception as error:  # noqa: BLE001 - a failed patient is recorded and reported, never hidden
            failures += 1
            out.mkdir(parents=True, exist_ok=True)
            (out / "FAILED").write_text(f"{type(error).__name__}: {error}\n{traceback.format_exc()}", encoding="utf-8")
            print(f"FAILED {dataset} {patient or 'all patients'}: {type(error).__name__}: {error}", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
