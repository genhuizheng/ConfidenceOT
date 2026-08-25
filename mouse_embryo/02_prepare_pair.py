"""Prepare one replicate-preserving MOSTA section pair for OT.

The script reads raw counts from ``layers['count']``, excludes anatomical
background from model fitting, performs proportional stratified sampling by
annotation, and learns one shared log-normalized HVG/PCA representation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_h5ad", type=Path)
    parser.add_argument("target_h5ad", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--n-per-side", type=int, default=1000)
    parser.add_argument("--n-hvg", type=int, default=2000)
    parser.add_argument("--n-pcs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--exclude-annotation", nargs="+", default=["Cavity"])
    return parser.parse_args()


def stratified_indices(labels: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Proportionally sample while retaining every observed annotation."""
    categories, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    if n >= labels.size:
        return np.arange(labels.size, dtype=np.int64)
    if n < categories.size:
        raise ValueError(f"n_per_side={n} is smaller than {categories.size} annotations.")
    ideal = counts * (n / labels.size)
    allocation = np.floor(ideal).astype(int)
    allocation = np.maximum(allocation, 1)
    allocation = np.minimum(allocation, counts)
    while allocation.sum() > n:
        candidates = np.flatnonzero(allocation > 1)
        index = candidates[np.argmax(allocation[candidates] - ideal[candidates])]
        allocation[index] -= 1
    while allocation.sum() < n:
        candidates = np.flatnonzero(allocation < counts)
        index = candidates[np.argmax(ideal[candidates] - allocation[candidates])]
        allocation[index] += 1
    selected = []
    for category_index, take in enumerate(allocation):
        members = np.flatnonzero(inverse == category_index)
        selected.append(rng.choice(members, size=take, replace=False))
    return np.sort(np.concatenate(selected).astype(np.int64))


def read_side(
    path: Path,
    *,
    n: int,
    excluded: set[str],
    rng: np.random.Generator,
) -> dict[str, object]:
    dataset = ad.read_h5ad(path, backed="r")
    try:
        required_obs = {"annotation"}
        missing = required_obs.difference(dataset.obs.columns)
        if missing or "count" not in dataset.layers or "spatial" not in dataset.obsm:
            raise KeyError(f"{path.name} lacks required MOSTA fields; missing obs={sorted(missing)}")
        labels_all = np.asarray(dataset.obs["annotation"].astype(str).to_numpy(), dtype=str)
        observation_ids_all = np.asarray(dataset.obs_names.astype(str), dtype=str)
        excluded_mask = np.isin(labels_all, list(excluded))
        eligible = np.flatnonzero(~excluded_mask)
        all_spatial = np.asarray(dataset.obsm["spatial"]).astype(np.float64)
        local = stratified_indices(labels_all[eligible], min(n, eligible.size), rng)
        rows = np.sort(eligible[local])
        raw = dataset.layers["count"][rows]
        raw = sparse.csr_matrix(raw, dtype=np.float64)
        return {
            "counts": raw,
            "genes": np.asarray(dataset.var_names.astype(str), dtype=str),
            "labels": labels_all[rows],
            "spatial": all_spatial[rows],
            "background_spatial": all_spatial[excluded_mask],
            "background_labels": labels_all[excluded_mask],
            "background_ids": observation_ids_all[excluded_mask],
            "observation_ids": observation_ids_all[rows],
            "rows": rows,
            "sample": path.name.removesuffix(".MOSTA.h5ad"),
            "path": str(path.resolve()),
            "eligible_n": int(eligible.size),
        }
    finally:
        dataset.file.close()


def common_gene_matrices(source: dict[str, object], target: dict[str, object]):
    source_genes = np.asarray(source["genes"])
    target_genes = np.asarray(target["genes"])
    target_lookup = {gene: index for index, gene in enumerate(target_genes)}
    source_columns = np.asarray([i for i, gene in enumerate(source_genes) if gene in target_lookup])
    common = source_genes[source_columns]
    target_columns = np.asarray([target_lookup[gene] for gene in common])
    if common.size == 0:
        raise ValueError("The two sections have no common genes.")
    return (
        sparse.csr_matrix(source["counts"])[:, source_columns],
        sparse.csr_matrix(target["counts"])[:, target_columns],
        common,
    )


def normalize_log(counts: sparse.csr_matrix) -> sparse.csr_matrix:
    library = np.asarray(counts.sum(axis=1)).ravel()
    if np.any(library <= 0):
        raise ValueError("A selected spatial bin has zero raw library size.")
    normalized = counts.multiply((10_000.0 / library)[:, None]).tocsr()
    normalized.data = np.log1p(normalized.data)
    return normalized


def shared_hvg_pca(
    source: sparse.csr_matrix,
    target: sparse.csr_matrix,
    *,
    n_hvg: int,
    n_pcs: int,
    seed: int,
):
    combined = sparse.vstack([normalize_log(source), normalize_log(target)], format="csr")
    mean = np.asarray(combined.mean(axis=0)).ravel()
    mean_square = np.asarray(combined.power(2).mean(axis=0)).ravel()
    variance = np.maximum(mean_square - mean * mean, 0.0)
    dispersion = variance / np.maximum(mean, 1e-8)
    expressed = np.flatnonzero(mean > 0)
    hvg_n = min(int(n_hvg), expressed.size)
    selected = expressed[np.argsort(dispersion[expressed], kind="stable")[-hvg_n:]]
    dense = combined[:, selected].toarray().astype(np.float32)
    gene_mean = dense.mean(axis=0, dtype=np.float64)
    gene_std = dense.std(axis=0, dtype=np.float64)
    gene_std[gene_std < 1e-8] = 1.0
    dense = np.clip((dense - gene_mean) / gene_std, -10.0, 10.0).astype(np.float32)
    pcs_n = min(int(n_pcs), dense.shape[0] - 1, dense.shape[1])
    model = PCA(n_components=pcs_n, svd_solver="randomized", random_state=seed)
    coordinates = model.fit_transform(dense).astype(np.float32)
    return coordinates, selected, model.explained_variance_ratio_.astype(np.float32)


def main() -> None:
    args = parse_args()
    if min(args.n_per_side, args.n_hvg, args.n_pcs) <= 0:
        raise ValueError("Sampling, HVG, and PCA dimensions must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    excluded = set(args.exclude_annotation)
    source = read_side(args.source_h5ad, n=args.n_per_side, excluded=excluded, rng=rng)
    target = read_side(args.target_h5ad, n=args.n_per_side, excluded=excluded, rng=rng)
    source_counts, target_counts, common_genes = common_gene_matrices(source, target)
    coordinates, hvg_indices, explained = shared_hvg_pca(
        source_counts, target_counts, n_hvg=args.n_hvg, n_pcs=args.n_pcs, seed=args.seed
    )
    n_source = source_counts.shape[0]
    destination = args.output_dir / "prepared_pair.npz"
    np.savez_compressed(
        destination,
        source_pca=coordinates[:n_source], target_pca=coordinates[n_source:],
        source_labels=source["labels"], target_labels=target["labels"],
        source_spatial=source["spatial"], target_spatial=target["spatial"],
        source_background_spatial=source["background_spatial"],
        target_background_spatial=target["background_spatial"],
        source_background_labels=source["background_labels"],
        target_background_labels=target["background_labels"],
        source_background_ids=source["background_ids"],
        target_background_ids=target["background_ids"],
        source_ids=source["observation_ids"], target_ids=target["observation_ids"],
        source_rows=source["rows"], target_rows=target["rows"],
        hvg_genes=common_genes[hvg_indices], explained_variance_ratio=explained,
    )
    # Enforce a portable, pickle-free artifact contract before reporting success.
    with np.load(destination, allow_pickle=False) as verification:
        for key in verification.files:
            verification[key]
    metadata = {
        "source_sample": source["sample"], "target_sample": target["sample"],
        "source_path": source["path"], "target_path": target["path"],
        "source_eligible_n": source["eligible_n"], "target_eligible_n": target["eligible_n"],
        "source_sampled_n": int(n_source), "target_sampled_n": int(target_counts.shape[0]),
        "common_gene_n": int(common_genes.size), "hvg_n": int(hvg_indices.size),
        "pca_n": int(coordinates.shape[1]), "seed": args.seed,
        "excluded_annotations": sorted(excluded),
        "representation": "raw count -> library 1e4 -> log1p -> joint HVG -> scaled joint PCA",
    }
    (args.output_dir / "preparation.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"Prepared pair: {destination.resolve()}")


if __name__ == "__main__":
    main()
