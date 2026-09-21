"""Shared I/O and the AnnData adapter over the library's preprocessing.

Everything from the gene intersection onwards now lives in
``confidenceot.preprocessing``, so the depth screen and this pipeline run
the same code. What stays here is what the library has no business knowing:
h5ad loading, sample selection, gene-key resolution, declared expression
kinds and per-cell QC.

The transforms, the read equalisation, the cost geometry and their
provenance are re-exported below so existing imports keep working.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

# Re-exported so that `from common import unit_rows` and friends keep working
# in the scripts that already do it; the implementations are the library's.
from confidenceot.preprocessing import (  # noqa: F401
    Preprocessing,
    equalise_depth,
    median_pair_scale,
    pearson_residual_block,
    pearson_residual_gene_variance,
    rank_value_encode,
    squared_euclidean,
    unit_rows,
)


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
            values = data.var[column].astype(str).str.strip().to_numpy()
            missing = {"", "na", "n/a", "nan", "none", "null", "<na>"}
            valid = ~pd.Series(values).str.lower().isin(missing).to_numpy()
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


def cell_qc_table(
    data: Any,
    *,
    minimum_total_counts: int,
    minimum_detected_genes: int,
    maximum_mitochondrial_percent: float,
) -> pd.DataFrame:
    """Calculate label-blind cell QC metrics and a transparent pass/fail call."""
    matrix = sparse.csr_matrix(expression_matrix(data), dtype=np.float64)
    if matrix.data.size and np.min(matrix.data) < 0:
        raise ValueError("Cell QC requires non-negative expression values")
    total_counts = np.asarray(matrix.sum(axis=1)).ravel()
    detected_genes = np.asarray((matrix > 0).sum(axis=1)).ravel()
    symbols = np.asarray(data.var_names.astype(str), dtype=str)
    if "gene_symbol" in data.var:
        candidate = data.var["gene_symbol"].astype(str).str.strip().to_numpy(dtype=str)
        valid = ~pd.Series(candidate).str.lower().isin(
            {"", "na", "n/a", "nan", "none", "null", "<na>"}
        ).to_numpy()
        symbols = np.where(valid, candidate, symbols)
    mitochondrial = np.char.startswith(np.char.upper(symbols), "MT-")
    mitochondrial_counts = (
        np.asarray(matrix[:, mitochondrial].sum(axis=1)).ravel()
        if np.any(mitochondrial)
        else np.zeros(data.n_obs, dtype=float)
    )
    mitochondrial_percent = np.divide(
        100.0 * mitochondrial_counts,
        total_counts,
        out=np.zeros_like(total_counts, dtype=float),
        where=total_counts > 0,
    )
    low_counts = total_counts < minimum_total_counts
    low_features = detected_genes < minimum_detected_genes
    high_mitochondrial = mitochondrial_percent > maximum_mitochondrial_percent
    passed = ~(low_counts | low_features | high_mitochondrial)
    reasons = []
    for count_fail, feature_fail, mitochondrial_fail in zip(
        low_counts, low_features, high_mitochondrial
    ):
        values = []
        if count_fail:
            values.append("low_total_counts")
        if feature_fail:
            values.append("low_detected_genes")
        if mitochondrial_fail:
            values.append("high_mitochondrial_percent")
        reasons.append(";".join(values))
    return pd.DataFrame({
        "observation_id": data.obs_names.astype(str),
        "total_counts": total_counts,
        "n_genes_by_counts": detected_genes,
        "pct_counts_mitochondrial": mitochondrial_percent,
        "qc_pass": passed,
        "qc_failure_reason": reasons,
    })


REPRESENTATIONS = ("log_cpm", "rank_value", "rank_no_median",
                   "pearson_residuals")


def _library_normalisation(representation: str, kind: str) -> tuple[str, str | None]:
    """Map a representation and a stored expression kind onto the library's.

    The library's normalisations say what they do to the numbers; an h5ad's
    ``expression_kind`` says what has already been done to them. Only this
    layer knows both, which is why the mapping lives here and not in
    ``confidenceot.preprocessing``.
    """
    if representation != "log_cpm":
        return representation, None
    if "log-normalized" in kind or "log normalized" in kind:
        return "precomputed", "logcpm"
    if "normalized" in kind and "raw" not in kind:
        return "log1p", None
    return "log_cpm", None


def prepare_joint_representation(
    source: Any, target: Any, *, n_hvg: int, n_pcs: int, seed: int,
    representation: str = "log_cpm", rank_top_n: int = 512,
    minimum_detection_rate: float = 0.0, residual_theta: float = 100.0,
    cost: str = "squared_euclidean", equalise_depth: bool = False,
    label_stem: str | None = None,
):
    """Joint representation of one pair, from two AnnData objects.

    The AnnData adapter over :class:`confidenceot.Preprocessing`: it resolves
    gene keys and the stored expression kind, and everything from the gene
    intersection onwards is the library's. The transform, the depth
    equalisation and the cost geometry had been written out separately here and
    in ``scripts/validate_depth_null_specificity.py``, which meant the screen's
    conclusions were measurements of a configuration the production runner
    assembled by hand. ``tests/test_preprocessing.py`` pins the move: the cost
    matrix, its scale and the generator state are unchanged for all eight
    measured configurations.

    ``cost='cosine'`` L2-normalises the coordinates before returning them, so a
    caller cannot reach the cosine configuration by remembering to do it
    afterwards and miss it.

    ``equalise_depth`` is **recorded, not applied**. Read equalisation happens
    upstream in ``27_downsample_counts.py``, at the file level, so that the OT,
    the pseudobulk, the differential expression and the scoring all see one
    matrix and no stage can disagree with another about which counts it used.
    Setting it here is what lets the run carry the configuration's real name:
    without it a depth-equalised cosine rank run would record itself as
    ``rank256_cos`` and be indistinguishable from one on untouched counts.

    One behaviour change: the two sides must now declare the same expression
    kind for *every* representation. The rank and residual paths already
    refused a mismatch; ``log_cpm`` used to transform each side by its own
    declared kind, which puts the two sides of one joint PCA on different
    scales. That was permissive where it should not have been.
    """
    if representation not in REPRESENTATIONS:
        raise ValueError(
            f"Unknown representation {representation!r}; expected one of "
            f"{REPRESENTATIONS}"
        )
    source_kind, target_kind = expression_kind(source), expression_kind(target)
    if source_kind != target_kind:
        raise ValueError(
            f"{representation} needs both sides on one scale; found "
            f"{source_kind!r} and {target_kind!r}"
        )
    normalisation, resolved_stem = _library_normalisation(representation, source_kind)
    configuration = Preprocessing(
        normalisation=normalisation,
        rank_top_n=rank_top_n,
        residual_theta=residual_theta,
        minimum_detection_rate=minimum_detection_rate,
        equalise_depth=equalise_depth,
        n_hvg=n_hvg,
        n_pcs=n_pcs,
        cost=cost,
        label_stem=label_stem or resolved_stem,
    )
    prepared = configuration.representation(
        expression_matrix(source), expression_matrix(target), seed=seed,
        source_genes=gene_keys(source), target_genes=gene_keys(target),
    )
    # The keys the existing run records carry, kept so a new run.json can still
    # be read beside an old one, plus everything the configuration object knows.
    provenance = {
        "source_expression_kind": source_kind,
        "target_expression_kind": target_kind,
        "source_transform": prepared.provenance["transform"],
        "target_transform": prepared.provenance["transform"],
        "representation": representation,
        **prepared.provenance,
        # Set last so it cannot read as though this call did the subsampling.
        "equalise_applied_here": False,
    }
    return prepared.source, prepared.target, prepared.selected_genes, provenance


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
