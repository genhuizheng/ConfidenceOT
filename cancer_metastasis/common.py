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


def _nonzero_median_per_gene(matrix: Any) -> np.ndarray:
    """Median of each gene's nonzero values, as Geneformer defines its scale.

    Zeros are excluded because including them would make the median zero for
    almost every gene in single-cell data, leaving nothing to divide by.
    """
    columns = matrix.tocsc()
    medians = np.ones(columns.shape[1], dtype=np.float64)
    for gene in range(columns.shape[1]):
        start, stop = columns.indptr[gene], columns.indptr[gene + 1]
        if stop > start:
            medians[gene] = float(np.median(columns.data[start:stop]))
    return np.where(medians > 0, medians, 1.0)


def rank_value_encode(joint: Any, top_n: int, use_gene_median: bool = True) -> Any:
    """Encode each cell as its ranking of genes, in Geneformer's formulation.

    Each cell is divided by its own total, each gene by its nonzero median over
    the joint matrix, and the genes are then ranked within the cell.  The two
    divisions are what make the ordering informative: ranking raw expression
    puts the same housekeeping genes on top of every cell, whereas ranking
    ``expression / gene median`` puts whatever is unusually high in *this* cell
    on top.

    The median must be taken over both sides together.  Taken per side it would
    normalise away exactly the cross-side differences the gate exists to
    detect, and the method would be blind by construction.

    Only the top ``top_n`` genes are kept, and the value falls linearly from
    1.0 to 1/top_n across them.  A fixed cut is what makes the encoding
    depth-invariant: cells differ in how many genes they detect, so a variable
    cut would let detection breadth back in, which is the part of the depth
    effect that survives subsampling to a common total -- on GSE180661 the gate
    still tracked original depth at AUC 0.557 among cells all sitting at
    exactly 3,119 counts. ``top_n`` therefore has to stay below the shallowest
    cell's detected-gene count.
    """
    if top_n < 2:
        raise ValueError("top_n must be at least 2")
    totals = np.asarray(joint.sum(axis=1)).ravel()
    scaled = sparse.diags(1.0 / np.maximum(totals, 1e-12)) @ sparse.csr_matrix(joint)
    if use_gene_median:
        medians = _nonzero_median_per_gene(scaled)
        scaled = (scaled @ sparse.diags(1.0 / medians)).tocsr()
    else:
        # The ablation: ranking library-normalised expression directly. Kept so
        # the claim that this is dominated by housekeeping genes, and therefore
        # nearly identical across cells, is measured rather than asserted.
        scaled = scaled.tocsr()

    rows, columns, values = [], [], []
    for cell in range(scaled.shape[0]):
        start, stop = scaled.indptr[cell], scaled.indptr[cell + 1]
        if stop == start:
            continue
        data = scaled.data[start:stop]
        genes = scaled.indices[start:stop]
        keep = min(top_n, data.size)
        # Stable sort so ties break on gene order, making the encoding
        # reproducible rather than dependent on the sort implementation.
        order = np.argsort(-data, kind="stable")[:keep]
        rows.append(np.full(keep, cell, dtype=np.int64))
        columns.append(genes[order])
        values.append(1.0 - np.arange(keep, dtype=np.float64) / float(top_n))
    if not rows:
        raise ValueError("Every cell is empty; nothing to rank")
    return sparse.csr_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=scaled.shape,
    )


REPRESENTATIONS = ("log_cpm", "rank_value", "rank_no_median", "pearson_residuals")


def pearson_residual_gene_variance(
    joint: Any, theta: float, chunk: int = 512
) -> np.ndarray:
    """Per-gene variance of analytic Pearson residuals.

    The residual of a count against the depth-only expectation
    ``mu = cell_total * gene_share`` under a negative binomial with dispersion
    ``theta``.  Its point is that a zero in a deep cell and a zero in a shallow
    cell receive different residuals, which is the part of the depth effect
    that library-size division cannot reach and that subsampling to a common
    total does not remove.

    Computed in gene blocks because the residual matrix is dense -- every entry,
    including the zeros, has a nonzero residual -- and materialising all of it
    for tens of thousands of genes is unnecessary when only the most variable
    are kept.
    """
    totals = np.asarray(joint.sum(axis=1), dtype=np.float64).ravel()
    gene_totals = np.asarray(joint.sum(axis=0), dtype=np.float64).ravel()
    grand = float(gene_totals.sum())
    if grand <= 0:
        raise ValueError("Joint matrix carries no counts")
    share = gene_totals / grand
    limit = np.sqrt(joint.shape[0])
    columns = joint.tocsc()
    variances = np.zeros(joint.shape[1], dtype=np.float64)
    for start in range(0, joint.shape[1], chunk):
        stop = min(start + chunk, joint.shape[1])
        expected = np.outer(totals, share[start:stop])
        scale = np.sqrt(expected + expected * expected / theta)
        block = columns[:, start:stop].toarray()
        residual = (block - expected) / np.maximum(scale, 1e-12)
        np.clip(residual, -limit, limit, out=residual)
        variances[start:stop] = residual.var(axis=0)
    return variances


def pearson_residual_block(joint: Any, genes: np.ndarray, theta: float) -> np.ndarray:
    """Dense analytic Pearson residuals for the chosen genes."""
    totals = np.asarray(joint.sum(axis=1), dtype=np.float64).ravel()
    gene_totals = np.asarray(joint.sum(axis=0), dtype=np.float64).ravel()
    share = gene_totals / float(gene_totals.sum())
    limit = np.sqrt(joint.shape[0])
    block = joint.tocsc()[:, genes].toarray()
    expected = np.outer(totals, share[genes])
    scale = np.sqrt(expected + expected * expected / theta)
    residual = (block - expected) / np.maximum(scale, 1e-12)
    np.clip(residual, -limit, limit, out=residual)
    return residual.astype(np.float32)


def prepare_joint_representation(
    source: Any, target: Any, *, n_hvg: int, n_pcs: int, seed: int,
    representation: str = "log_cpm", rank_top_n: int = 512,
    minimum_detection_rate: float = 0.0, residual_theta: float = 100.0,
):
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
    if representation not in REPRESENTATIONS:
        raise ValueError(
            f"Unknown representation {representation!r}; expected one of {REPRESENTATIONS}"
        )

    # A gene detected in very few cells carries mostly its own zero pattern,
    # and that pattern is what depth writes into the data. Filtering on
    # detection is orthogonal to the transform and composes with any of them.
    detection_note = "no detection-rate filter"
    if minimum_detection_rate > 0.0:
        counted = sparse.vstack([source_x, target_x], format="csr")
        detected = np.asarray((counted > 0).sum(axis=0)).ravel() / counted.shape[0]
        keep = detected >= minimum_detection_rate
        if int(keep.sum()) < 2:
            raise ValueError(
                f"Detection rate >= {minimum_detection_rate} leaves "
                f"{int(keep.sum())} genes"
            )
        source_x = source_x[:, keep]
        target_x = target_x[:, keep]
        common = [gene for gene, flag in zip(common, keep) if flag]
        detection_note = (
            f"genes detected in >= {minimum_detection_rate:.3f} of cells; "
            f"{int(keep.sum())} of {len(keep)} kept"
        )

    if representation == "pearson_residuals":
        if source_kind != target_kind:
            raise ValueError(
                "pearson_residuals needs both sides on one scale; found "
                f"{source_kind!r} and {target_kind!r}"
            )
        counts = sparse.vstack([source_x, target_x], format="csr")
        variances = pearson_residual_gene_variance(counts, residual_theta)
        selected = np.argsort(-variances, kind="stable")[: min(n_hvg, len(common))]
        dense = pearson_residual_block(counts, selected, residual_theta)
        source_transform = target_transform = (
            f"joint analytic Pearson residuals, theta={residual_theta:g}, "
            "genes ranked by residual variance"
        )
    elif representation in ("rank_value", "rank_no_median"):
        # Stacked before any per-side transformation, so the gene medians and
        # the ranking see one corpus. A per-side transform here would defeat
        # the joint median.
        if source_kind != target_kind:
            raise ValueError(
                f"{representation} needs both sides on one scale; found "
                f"{source_kind!r} and {target_kind!r}"
            )
        use_median = representation == "rank_value"
        joint = rank_value_encode(
            sparse.vstack([source_x, target_x], format="csr"),
            rank_top_n, use_gene_median=use_median,
        )
        mean = np.asarray(joint.mean(axis=0)).ravel()
        mean2 = np.asarray(joint.multiply(joint).mean(axis=0)).ravel()
        selected = np.argsort(-(mean2 - mean * mean), kind="stable")[
            : min(n_hvg, len(common))
        ]
        dense = joint[:, selected].toarray().astype(np.float32)
        source_transform = target_transform = (
            f"joint rank encoding, top {rank_top_n} genes per cell, "
            + ("expression divided by each gene's nonzero median over both sides"
               if use_median else
               "expression ranked directly, without the gene-median division")
        )
    else:
        source_x, source_transform = normalize(source_x, source_kind)
        target_x, target_transform = normalize(target_x, target_kind)
        joint = sparse.vstack([source_x, target_x], format="csr")
        mean = np.asarray(joint.mean(axis=0)).ravel()
        mean2 = np.asarray(joint.multiply(joint).mean(axis=0)).ravel()
        selected = np.argsort(-(mean2 - mean * mean), kind="stable")[
            : min(n_hvg, len(common))
        ]
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
        "representation": representation,
        "rank_top_n": rank_top_n if "rank" in representation else None,
        "residual_theta": residual_theta if representation == "pearson_residuals" else None,
        "detection_filter": detection_note,
        "joint_hvg": (
            "top variance of the rank encoding" if representation == "rank_value"
            else "top variance after side-specific declared-expression transformation"
        ),
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
