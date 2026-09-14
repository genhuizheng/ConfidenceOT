"""Equalise sequencing depth before optimal transport, outside the algorithm.

Diagnostics on the stored gates show that ``decision_cost`` is substantially a
depth readout: rho(cost, depth) runs -0.33 to -0.59 across the three datasets,
retained and rejected cells differ in depth by 1.3x to 3.1x, and a simulation
with no biological difference at all reproduces that signature from depth
spread alone.  The transport plan also pairs cells by depth, rho = 0.64 on
reciprocal-dominant edges.

Depth is a property of the data, not of the transport solver, so it is
corrected here rather than inside the representation.  This script reads the
h5ads a pair manifest points at, keeps the cells the analysis would use,
subsamples every cell's reads to one shared depth, and writes new h5ads plus a
manifest that points at them.  The existing pipeline then runs unchanged: the
OT, the pseudobulk, the differential expression and the UCell scoring all see
the same depth-corrected counts, so no stage can disagree with another about
which matrix it used.

Subsampling is exact multivariate hypergeometric sampling of reads without
replacement, the same operation as ``scanpy.pp.downsample_counts``, drawn here
with NumPy so the step carries no extra dependency and stays reproducible from
its seed.  Counts stay non-negative integers, which is what keeps the step
decoupled: the pseudobulk stage requires integer counts, so a transform such as
Pearson residuals could only have lived inside the representation code.

Cells already at or below the target are left alone.  They are the residual
depth gradient and their number is reported.

**Running the pipeline on the output.** Pass the emitted manifest and keep
per-cell QC metrics without re-filtering, because the cells were already
selected here and downsampling lowers the detected-gene count that a second QC
pass would test::

    --cell-qc --minimum-total-counts 0 --minimum-detected-genes 0 \\
    --maximum-mitochondrial-percent 100

The metrics are still computed and merged into ``cell_confidence.csv`` that
way, which is what the gate diagnostics read.  Those diagnostics should use
``predownsample_depth.csv.gz`` for the depth column: after correction the
stored depth is nearly constant, so the meaningful question is whether the gate
still tracks each cell's *original* depth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from cancer_metastasis.common import cell_qc_table, expression_matrix, load_exact_side
from cancer_metastasis.gse180661.primary_pseudobulk import (
    malignant_annotation_mask,
    paths_for,
    safe_name,
)


def unique_samples(manifest: pd.DataFrame) -> list[tuple[str, tuple[str, ...]]]:
    """Return every distinct (sample, source files) the manifest references.

    One sample can appear on either side and in several pairs, so it is loaded
    and written once.
    """
    seen: dict[str, tuple[str, ...]] = {}
    for row in manifest.to_dict("records"):
        series = pd.Series(row)
        for side in ("source", "target"):
            sample = str(row[f"{side}_sample"])
            files = tuple(paths_for(series, side))
            if sample in seen and seen[sample] != files:
                raise RuntimeError(
                    f"Sample {sample!r} is referenced with two different file sets"
                )
            seen[sample] = files
    return sorted(seen.items())


def analysed_subset(sample: str, files: tuple[str, ...], args: argparse.Namespace):
    """Load one sample and keep only the cells the analysis would use."""
    data = load_exact_side(list(files), sample)
    data = data[malignant_annotation_mask(data, args.malignant_annotations)].copy()
    if data.n_obs == 0:
        return data, None
    qc = cell_qc_table(
        data,
        minimum_total_counts=args.minimum_total_counts,
        minimum_detected_genes=args.minimum_detected_genes,
        maximum_mitochondrial_percent=args.maximum_mitochondrial_percent,
    )
    data = data[qc["qc_pass"].to_numpy()].copy()
    return data, qc


def downsample_matrix(
    matrix: sparse.csr_matrix, target: int, rng: np.random.Generator
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Subsample every row's reads to ``target`` without replacement.

    Returns the new matrix, each row's original depth, and a mask of the rows
    that were already at or below the target and so were left untouched.
    """
    matrix = sparse.csr_matrix(matrix, dtype=np.int64)
    original = np.asarray(matrix.sum(axis=1), dtype=np.int64).ravel()
    untouched = original <= target
    rows = []
    for index in range(matrix.shape[0]):
        start, end = matrix.indptr[index], matrix.indptr[index + 1]
        counts = matrix.data[start:end]
        if untouched[index] or counts.size == 0:
            rows.append(counts)
            continue
        # Exact hypergeometric read subsampling, as in
        # scanpy.pp.downsample_counts.
        rows.append(rng.multivariate_hypergeometric(counts, int(target)))
    reduced = sparse.csr_matrix(
        (np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64),
         matrix.indices.copy(), matrix.indptr.copy()),
        shape=matrix.shape,
        dtype=np.int64,
    )
    # Genes that lost all their reads become explicit zeros; drop them so the
    # matrix stays a faithful sparse representation.
    reduced.eliminate_zeros()
    return reduced, original, untouched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_csv", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--malignant-annotation", action="append", dest="malignant_annotations",
        required=True, help="Author malignant label to keep; may be repeated",
    )
    parser.add_argument("--minimum-total-counts", type=int, default=1000)
    parser.add_argument("--minimum-detected-genes", type=int, default=500)
    parser.add_argument("--maximum-mitochondrial-percent", type=float, default=20.0)
    parser.add_argument(
        "--target-quantile", type=float, default=0.10,
        help="Quantile of the pooled post-QC depth distribution to subsample to",
    )
    parser.add_argument(
        "--target-depth", type=int, default=None,
        help="Explicit shared depth; overrides --target-quantile",
    )
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()
    if not 0.0 < args.target_quantile < 1.0:
        raise ValueError("--target-quantile must lie in (0, 1)")
    destination = args.output_root
    (destination / "h5ad").mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest_csv)
    samples = unique_samples(manifest)

    # Pass one establishes one shared target across every analysed cell. A
    # per-file target would leave the sides of a pair at different depths,
    # which is the difference the correction exists to remove.
    pooled: list[np.ndarray] = []
    per_sample_n: dict[str, int] = {}
    for sample, files in samples:
        data, qc = analysed_subset(sample, files, args)
        if data.n_obs == 0:
            per_sample_n[sample] = 0
            continue
        depth = np.asarray(
            sparse.csr_matrix(expression_matrix(data)).sum(axis=1)
        ).ravel()
        pooled.append(depth)
        per_sample_n[sample] = int(data.n_obs)
        del data, qc
    if not pooled:
        raise RuntimeError("No analysed cells found in any referenced sample")
    all_depth = np.concatenate(pooled)
    target = (
        int(args.target_depth) if args.target_depth is not None
        else int(np.quantile(all_depth, args.target_quantile))
    )
    if target < 1:
        raise RuntimeError(f"Target depth resolved to {target}; raise the quantile")

    rows = []
    depth_records = []
    written: dict[str, str] = {}
    for index, (sample, files) in enumerate(samples):
        data, _ = analysed_subset(sample, files, args)
        if data.n_obs == 0:
            continue
        rng = np.random.default_rng(args.seed + 7919 * index)
        reduced, original, untouched = downsample_matrix(
            expression_matrix(data), target, rng
        )
        data.X = reduced
        for layer in ("counts", "count"):
            if layer in data.layers:
                del data.layers[layer]
        data.obs["predownsample_total_counts"] = original
        data.obs["downsample_untouched"] = untouched
        path = destination / "h5ad" / f"{safe_name(sample)}.h5ad"
        data.write_h5ad(path, compression="lzf")
        written[sample] = str(path)
        rows.append({
            "sample_id": sample,
            "analysed_cell_n": int(data.n_obs),
            "already_at_or_below_target_n": int(untouched.sum()),
            "already_at_or_below_target_fraction": float(untouched.mean()),
            "median_depth_before": float(np.median(original)),
            "median_depth_after": float(
                np.median(np.asarray(reduced.sum(axis=1)).ravel())
            ),
            "h5ad": str(path),
        })
        depth_records.append(pd.DataFrame({
            "sample_id": sample,
            "observation_id": data.obs_names.astype(str),
            "predownsample_total_counts": original,
            "downsample_untouched": untouched,
        }))
        del data, reduced

    updated = manifest.copy()
    for side in ("source", "target"):
        column = f"{side}_sample"
        updated[f"{side}_h5ad"] = updated[column].astype(str).map(written)
        json_column = f"{side}_h5ads_json"
        if json_column in updated:
            updated[json_column] = updated[f"{side}_h5ad"].map(
                lambda value: json.dumps([value]) if pd.notna(value) else value
            )
    missing = int(
        updated["source_h5ad"].isna().sum() + updated["target_h5ad"].isna().sum()
    )
    if missing:
        # A sample with no analysed cells leaves its pairs unusable.
        updated = updated[updated["source_h5ad"].notna() & updated["target_h5ad"].notna()]
    manifest_path = destination / "pair_manifest_downsampled.csv"
    updated.to_csv(manifest_path, index=False)
    pd.concat(depth_records, ignore_index=True).to_csv(
        destination / "predownsample_depth.csv.gz", index=False, compression="gzip"
    )
    per_sample = pd.DataFrame(rows)
    per_sample.to_csv(destination / "downsample_per_sample.csv", index=False)

    report = {
        "target_depth": target,
        "target_quantile": None if args.target_depth is not None else args.target_quantile,
        "pooled_analysed_cell_n": int(all_depth.size),
        "pooled_median_depth_before": float(np.median(all_depth)),
        "sample_n": int(len(rows)),
        "pairs_in": int(len(manifest)),
        "pairs_out": int(len(updated)),
        "cells_already_at_or_below_target": int(
            per_sample["already_at_or_below_target_n"].sum()
        ),
        "cells_already_at_or_below_target_fraction": float(
            per_sample["already_at_or_below_target_n"].sum() / all_depth.size
        ),
        "malignant_annotations": args.malignant_annotations,
        "qc_thresholds": {
            "minimum_total_counts": args.minimum_total_counts,
            "minimum_detected_genes": args.minimum_detected_genes,
            "maximum_mitochondrial_percent": args.maximum_mitochondrial_percent,
        },
        "downstream_invocation": (
            "Cells are already selected and downsampling lowers detected genes, "
            "so run 02_run_pair.py with --cell-qc --minimum-total-counts 0 "
            "--minimum-detected-genes 0 --maximum-mitochondrial-percent 100: the "
            "QC metrics are still recorded but nothing is filtered again."
        ),
        "diagnostic_note": (
            "Stored depth is near constant after correction. Join "
            "predownsample_depth.csv.gz to test whether the gate still tracks "
            "each cell's original depth."
        ),
        "manifest": str(manifest_path),
    }
    (destination / "downsample_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    print(per_sample.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
