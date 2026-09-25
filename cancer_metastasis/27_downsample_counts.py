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

from confidenceot import equalise_depth
from cancer_metastasis.common import cell_qc_table, expression_matrix, load_exact_side
from cancer_metastasis.gse180661.primary_pseudobulk import (
    malignant_annotation_mask,
    paths_for,
    safe_name,
)


def unique_samples(
    manifest: pd.DataFrame,
) -> list[tuple[tuple[str, str], tuple[str, ...]]]:
    """Return every distinct ((patient, sample), source files) referenced.

    The sample column holds a site label such as ``left_adnexa``, which is not
    unique across patients, so the patient is part of the key. One such sample
    can still appear on either side and in several pairs, and is loaded and
    written once.
    """
    seen: dict[tuple[str, str], tuple[str, ...]] = {}
    for row in manifest.to_dict("records"):
        series = pd.Series(row)
        patient = str(row["patient_id"])
        for side in ("source", "target"):
            key = (patient, str(row[f"{side}_sample"]))
            files = tuple(paths_for(series, side))
            if key in seen and seen[key] != files:
                raise RuntimeError(
                    f"Patient {key[0]!r} sample {key[1]!r} is referenced with "
                    "two different file sets"
                )
            seen[key] = files
    return sorted(seen.items())


def subtype_mask(data, column: str, excluded: list[str]) -> np.ndarray:
    """Keep cells whose author subtype is not in ``excluded``.

    ``cell_type`` is ``Ovarian.cancer.cell`` for every epithelial cell in the
    sample, because the study's CellAssign run offers nine categories and none
    of them is normal epithelium. The finer ``cell_subtype`` column carries the
    authors' own sub-clustering, including ``Ciliated.cell.1`` and
    ``Ciliated.cell.2`` -- a non-malignant tube lineage -- and ``NA`` for cells
    their clustering left unassigned or placed in patient-specific clusters
    that the paper disregarded by relative entropy.

    That column is what makes the re-run interpretable: ``NA`` cells are 18% of
    this gate's rejected set against 5.9% of its retained set, so a gate fitted
    without them answers a different and better-posed question.
    """
    if column not in data.obs:
        raise RuntimeError(
            f"H5AD has no {column!r} column; found {sorted(data.obs.columns)[:12]}"
        )
    values = data.obs[column].astype(str).str.strip()
    return ~values.isin(set(excluded)).to_numpy()


def analysed_subset(sample: str, files: tuple[str, ...], args: argparse.Namespace):
    """Load one sample and keep only the cells the analysis would use."""
    data = load_exact_side(list(files), sample)
    if args.malignant_column:
        if args.malignant_column not in data.obs:
            raise KeyError(
                f"obs has no {args.malignant_column!r} column; use "
                f"--malignant-annotation for files that predate the uniform "
                f"call. Columns present: {sorted(data.obs.columns)[:12]}")
        keep = data.obs[args.malignant_column].astype(str).to_numpy() == args.malignant_value
    else:
        keep = malignant_annotation_mask(data, args.malignant_annotations)
    data = data[keep].copy()
    if args.excluded_subtypes and data.n_obs:
        data = data[
            subtype_mask(data, args.subtype_column, args.excluded_subtypes)
        ].copy()
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

    The implementation is ``confidenceot.equalise_depth``: the simulation that
    chose this configuration and the stage that applies it to real counts have
    to be the same operation, or the screen's conclusion is about a pipeline
    nobody runs. It is verified bit-identical to the version that produced the
    equalised objects already on disk, so those stay valid and are reused
    rather than regenerated.
    """
    return equalise_depth(
        sparse.csr_matrix(matrix, dtype=np.int64), rng=rng, target=int(target),
        return_untouched=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_csv", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--malignant-column", default=None, metavar="COLUMN",
        help="Select the malignant compartment by this obs column equalling --malignant-value, instead of by the deposit's own labels. One rule for every deposit.")
    parser.add_argument("--malignant-value", default="malignant")
    parser.add_argument(
        "--malignant-annotation", action="append", dest="malignant_annotations", help="Author malignant label to keep; may be repeated",
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
    parser.add_argument(
        "--exclude-subtype", action="append", dest="excluded_subtypes", default=None,
        help="Author cell_subtype value to drop; may be repeated. 'NA' drops "
             "cells their sub-clustering left unassigned, 'Ciliated.cell.1' "
             "and 'Ciliated.cell.2' drop the non-malignant tube lineage.",
    )
    parser.add_argument("--subtype-column", default="cell_subtype")
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()
    if args.malignant_column and args.malignant_annotations:
        parser.error("pass --malignant-column or --malignant-annotation, not both")
    if not args.malignant_column and not args.malignant_annotations:
        parser.error("malignant selection needs --malignant-column or at least "
                     "one --malignant-annotation")
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
    per_sample_n: dict[tuple[str, str], int] = {}
    for (patient, sample), files in samples:
        data, qc = analysed_subset(sample, files, args)
        if data.n_obs == 0:
            per_sample_n[(patient, sample)] = 0
            continue
        depth = np.asarray(
            sparse.csr_matrix(expression_matrix(data)).sum(axis=1)
        ).ravel()
        pooled.append(depth)
        per_sample_n[(patient, sample)] = int(data.n_obs)
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
    written: dict[tuple[str, str], str] = {}
    for index, ((patient, sample), files) in enumerate(samples):
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
        path = destination / "h5ad" / f"{safe_name(patient)}__{safe_name(sample)}.h5ad"
        data.write_h5ad(path, compression="lzf")
        written[(patient, sample)] = str(path)
        rows.append({
            "patient_id": patient,
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
            "patient_id": patient,
            "sample_id": sample,
            "observation_id": data.obs_names.astype(str),
            "predownsample_total_counts": original,
            "downsample_untouched": untouched,
        }))
        del data, reduced

    updated = manifest.copy()
    for side in ("source", "target"):
        column = f"{side}_sample"
        keys = list(zip(updated["patient_id"].astype(str), updated[column].astype(str)))
        updated[f"{side}_h5ad"] = [written.get(key) for key in keys]
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
        "excluded_subtypes": args.excluded_subtypes or [],
        "subtype_column": args.subtype_column,
        # Pass --target-depth to reproduce an earlier run's depth exactly. With
        # a quantile, changing the cell set also changes the target, and two
        # runs then differ in two ways at once.
        "target_depth_was_explicit": args.target_depth is not None,
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
        "malignant_selector": (
            f"{args.malignant_column}=={args.malignant_value}"
            if args.malignant_column else "cell_type in malignant_annotations"),
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
