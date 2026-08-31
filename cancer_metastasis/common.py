"""Shared I/O and representation helpers for metastatic cancer analyses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse


def string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def truthy(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "verified"}


def read_file_record(path: Path) -> dict[str, Any]:
    import anndata as ad

    data = ad.read_h5ad(path, backed="r")
    try:
        metadata = dict(data.uns.get("pairing_metadata", {}))
        samples = string_list(metadata.get("samples"))
        counts = metadata.get("n_cells_per_sample", {})
        if not isinstance(counts, dict):
            counts = {}
        return {
            "path": str(path.resolve()),
            "dataset_id": path.parents[1].name,
            "patient_id": str(metadata.get("patient_id", path.parent.name)).strip(),
            "file_kind": str(metadata.get("file_kind", "unknown")),
            "site_class": str(metadata.get("site_class", "")),
            "samples": samples,
            "sample_counts": {str(key): int(value) for key, value in counts.items()},
            "source_pair_ids": string_list(metadata.get("pair_ids_as_source")),
            "target_pair_ids": string_list(metadata.get("pair_ids_as_target")),
            "pairing_verified": truthy(metadata.get("pairing_verified", False)),
            "n_obs": int(data.n_obs),
            "n_vars": int(data.n_vars),
            "obs_columns": [str(value) for value in data.obs.columns],
            "var_columns": [str(value) for value in data.var.columns],
        }
    finally:
        data.file.close()


def resolve_samples(patient: str, pair_id: str, source_samples: list[str], target_samples: list[str]) -> tuple[str, str]:
    matches = [
        (source, target)
        for source in source_samples
        for target in target_samples
        if f"{patient}__{source}__{target}" == pair_id
    ]
    if len(matches) != 1:
        return "", ""
    return matches[0]


def build_pair_manifest(converted_root: Path, minimum_cells: int = 20) -> pd.DataFrame:
    files = [read_file_record(path) for path in sorted(converted_root.glob("*/*/*.h5ad"))]
    source_map: dict[str, list[dict[str, Any]]] = {}
    target_map: dict[str, list[dict[str, Any]]] = {}
    for record in files:
        for pair_id in record["source_pair_ids"]:
            source_map.setdefault(pair_id, []).append(record)
        for pair_id in record["target_pair_ids"]:
            target_map.setdefault(pair_id, []).append(record)
    rows: list[dict[str, Any]] = []
    for pair_id in sorted(set(source_map) | set(target_map)):
        sources, targets = source_map.get(pair_id, []), target_map.get(pair_id, [])
        reasons: list[str] = []
        if not sources:
            reasons.append("source_file_count=0")
        elif len(sources) > 1 and any(item["file_kind"] != "library" for item in sources):
            reasons.append(f"ambiguous_nonlibrary_source_file_count={len(sources)}")
        if not targets:
            reasons.append("target_file_count=0")
        elif len(targets) > 1 and any(item["file_kind"] != "library" for item in targets):
            reasons.append(f"ambiguous_nonlibrary_target_file_count={len(targets)}")
        source = sources[0] if sources else None
        target = targets[0] if targets else None
        patient = source["patient_id"] if source else (target["patient_id"] if target else "")
        dataset = source["dataset_id"] if source else (target["dataset_id"] if target else "")
        source_sample = target_sample = ""
        if source and target:
            if source["dataset_id"] != target["dataset_id"]:
                reasons.append("dataset_mismatch")
            if not patient or patient.lower() in {"nan", "none", "unknown"}:
                reasons.append("missing_patient_id")
            if source["patient_id"] != target["patient_id"]:
                reasons.append("patient_mismatch")
            if any(item["dataset_id"] != source["dataset_id"] for item in sources + targets):
                reasons.append("dataset_mismatch_across_files")
            if any(item["patient_id"] != patient for item in sources + targets):
                reasons.append("patient_mismatch_across_files")
            source_sample, target_sample = resolve_samples(
                patient, pair_id,
                sorted({sample for item in sources for sample in item["samples"]}),
                sorted({sample for item in targets for sample in item["samples"]}),
            )
            if not source_sample or not target_sample:
                reasons.append("pair_samples_not_uniquely_resolved")
        source_n = sum(int(item["sample_counts"].get(source_sample, 0)) for item in sources)
        target_n = sum(int(item["sample_counts"].get(target_sample, 0)) for item in targets)
        if source and source_sample and not source_n:
            source_n = source["n_obs"] if len(source["samples"]) == 1 else 0
        if target and target_sample and not target_n:
            target_n = target["n_obs"] if len(target["samples"]) == 1 else 0
        if source_n < minimum_cells:
            reasons.append(f"source_cells<{minimum_cells}")
        if target_n < minimum_cells:
            reasons.append(f"target_cells<{minimum_cells}")
        rows.append({
            "pair_id": pair_id, "dataset_id": dataset, "patient_id": patient,
            "source_h5ad": source["path"] if source else "",
            "target_h5ad": target["path"] if target else "",
            "source_h5ads_json": json.dumps([item["path"] for item in sources]),
            "target_h5ads_json": json.dumps([item["path"] for item in targets]),
            "source_sample": source_sample, "target_sample": target_sample,
            "source_n": source_n, "target_n": target_n,
            "source_file_n": len(sources), "target_file_n": len(targets),
            "source_file_kind": source["file_kind"] if source else "",
            "target_file_kind": target["file_kind"] if target else "",
            "has_cell_type": bool(source and target and "cell_type" in source["obs_columns"] and "cell_type" in target["obs_columns"]),
            "eligible": not reasons, "skip_reason": ";".join(reasons),
        })
    result = pd.DataFrame(rows)
    if len(result):
        result.insert(0, "pair_index", np.arange(len(result), dtype=int))
        result["eligible_index"] = -1
        eligible = result.index[result["eligible"]]
        result.loc[eligible, "eligible_index"] = np.arange(len(eligible), dtype=int)
    return result


def select_sample(data: Any, sample: str) -> Any:
    if "sample_id" in data.obs:
        mask = data.obs["sample_id"].astype(str).to_numpy() == sample
        if not np.any(mask):
            raise ValueError(f"sample_id={sample!r} has no observations")
        return data[mask].to_memory()
    if sample and len(data.uns.get("pairing_metadata", {}).get("samples", [])) != 1:
        raise ValueError("Multi-sample H5AD lacks obs['sample_id']; exact pair cannot be isolated")
    return data.to_memory()


def load_exact_side(paths: list[str], sample: str):
    """Load one exact sample, combining its sorted-library files conservatively."""
    import anndata as ad

    pieces = []
    expression_kinds = []
    for value in paths:
        backed = ad.read_h5ad(value, backed="r")
        try:
            piece = select_sample(backed, sample)
        finally:
            backed.file.close()
        if "feature_type" in piece.var:
            piece = piece[:, piece.var["feature_type"].astype(str).eq("Gene Expression")].copy()
        elif "feature_types" in piece.var:
            piece = piece[:, piece.var["feature_types"].astype(str).eq("Gene Expression")].copy()
        keys = gene_keys(piece)
        keep = ~pd.Index(keys).duplicated(keep="first")
        piece = piece[:, keep].copy()
        piece.var_names = pd.Index(keys[keep], dtype=str)
        expression_kinds.append(expression_kind(piece))
        pieces.append(piece)
    if not pieces:
        raise ValueError("No H5AD files supplied for one pair side")
    if len(pieces) == 1:
        return pieces[0]
    combined = ad.concat(
        pieces, axis=0, join="inner", merge="same", uns_merge="same", index_unique=None,
    )
    if combined.obs_names.has_duplicates:
        combined.obs_names_make_unique()
    kinds = sorted(set(expression_kinds))
    if len(kinds) != 1:
        raise ValueError(f"Library files disagree on Expression matrix type: {kinds}")
    metadata = dict(combined.uns.get("metadata", {}))
    metadata["Expression matrix type"] = kinds[0]
    combined.uns["metadata"] = metadata
    combined.uns["confidenceot_library_merge"] = {
        "file_n": len(pieces), "gene_join": "inner", "sample_id": sample,
    }
    return combined


def gene_keys(data: Any) -> np.ndarray:
    keys = np.asarray(data.var_names.astype(str), dtype=str)
    for column in ("gene_id", "gene_symbol"):
        if column in data.var:
            values = data.var[column].astype(str).to_numpy()
            valid = ~pd.Series(values).isin(["", "nan", "None"]).to_numpy()
            keys = np.where(valid, values, keys)
            if column == "gene_id":
                break
    return keys


def expression_matrix(data: Any):
    for layer in ("counts", "count"):
        if layer in data.layers:
            return data.layers[layer]
    return data.X


def expression_kind(data: Any) -> str:
    metadata = data.uns.get("metadata", {})
    if hasattr(metadata, "get"):
        value = metadata.get("Expression matrix type", "")
        if str(value).strip():
            return str(value).strip().lower()
    return "unknown"


def prepare_joint_representation(source: Any, target: Any, *, n_hvg: int, n_pcs: int, seed: int):
    source_keys, target_keys = gene_keys(source), gene_keys(target)
    source_first: dict[str, int] = {}
    target_first: dict[str, int] = {}
    for index, key in enumerate(source_keys):
        source_first.setdefault(str(key), index)
    for index, key in enumerate(target_keys):
        target_first.setdefault(str(key), index)
    common = sorted(set(source_first) & set(target_first))
    if len(common) < 2:
        raise ValueError("Fewer than two common genes")
    source_index = [source_first[key] for key in common]
    target_index = [target_first[key] for key in common]
    source_x = sparse.csr_matrix(expression_matrix(source)[:, source_index], dtype=np.float64)
    target_x = sparse.csr_matrix(expression_matrix(target)[:, target_index], dtype=np.float64)
    source_kind, target_kind = expression_kind(source), expression_kind(target)
    def normalize(matrix, kind):
        if "log-normalized" in kind or "log normalized" in kind:
            return matrix, "stored log-normalized expression used without retransformation"
        if "normalized" in kind and "raw" not in kind:
            if matrix.data.size and np.min(matrix.data) < 0:
                return matrix, "stored normalized expression contains negative values; used as provided"
            matrix.data = np.log1p(matrix.data)
            return matrix, "stored normalized expression -> log1p"
        totals = np.asarray(matrix.sum(axis=1)).ravel()
        scaled = sparse.diags(1e4 / np.maximum(totals, 1.0)) @ matrix
        scaled.data = np.log1p(scaled.data)
        return scaled, "raw counts -> library size 1e4 -> log1p"
    source_x, source_transform = normalize(source_x, source_kind)
    target_x, target_transform = normalize(target_x, target_kind)
    joint = sparse.vstack([source_x, target_x], format="csr")
    mean = np.asarray(joint.mean(axis=0)).ravel()
    mean2 = np.asarray(joint.multiply(joint).mean(axis=0)).ravel()
    selected = np.argsort(-(mean2 - mean * mean), kind="stable")[: min(n_hvg, len(common))]
    dense = joint[:, selected].toarray().astype(np.float32)
    dense -= dense.mean(axis=0)
    std = dense.std(axis=0)
    dense /= np.where(std > 1e-8, std, 1.0)
    from sklearn.decomposition import PCA
    components = min(n_pcs, dense.shape[0] - 1, dense.shape[1])
    coordinates = PCA(n_components=components, random_state=seed).fit_transform(dense)
    preprocessing = {
        "source_expression_kind": source_kind,
        "target_expression_kind": target_kind,
        "source_transform": source_transform,
        "target_transform": target_transform,
        "joint_hvg": "top variance after side-specific declared-expression transformation",
        "joint_pca": "centered and gene-scaled PCA",
    }
    return coordinates[: source.n_obs], coordinates[source.n_obs :], [common[index] for index in selected], preprocessing


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if hasattr(value, "__dict__"):
        return {key: json_ready(item) for key, item in value.__dict__.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    return value
