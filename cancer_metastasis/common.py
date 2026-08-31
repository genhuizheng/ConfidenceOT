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
        if len(sources) != 1:
            reasons.append(f"source_file_count={len(sources)}")
        if len(targets) != 1:
            reasons.append(f"target_file_count={len(targets)}")
        source = sources[0] if len(sources) == 1 else None
        target = targets[0] if len(targets) == 1 else None
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
            if not source["pairing_verified"] or not target["pairing_verified"]:
                reasons.append("pair_not_verified")
            source_sample, target_sample = resolve_samples(
                patient, pair_id, source["samples"], target["samples"]
            )
            if not source_sample or not target_sample:
                reasons.append("pair_samples_not_uniquely_resolved")
        source_n = int(source["sample_counts"].get(source_sample, 0)) if source else 0
        target_n = int(target["sample_counts"].get(target_sample, 0)) if target else 0
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
            "source_sample": source_sample, "target_sample": target_sample,
            "source_n": source_n, "target_n": target_n,
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
    source_x = expression_matrix(source)[:, source_index]
    target_x = expression_matrix(target)[:, target_index]
    source_x = sparse.csr_matrix(source_x, dtype=np.float64)
    target_x = sparse.csr_matrix(target_x, dtype=np.float64)
    def normalize(matrix):
        totals = np.asarray(matrix.sum(axis=1)).ravel()
        scaled = sparse.diags(1e4 / np.maximum(totals, 1.0)) @ matrix
        scaled.data = np.log1p(scaled.data)
        return scaled
    source_x, target_x = normalize(source_x), normalize(target_x)
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
    return coordinates[: source.n_obs], coordinates[source.n_obs :], [common[index] for index in selected]


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
