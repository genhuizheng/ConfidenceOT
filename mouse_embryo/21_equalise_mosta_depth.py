"""Equalise sequencing depth across every MOSTA slice, once, at the file level.

This is the ``_ds`` of ranknm256_noscale_ds_cos as the production pipeline
applies it (cancer_metastasis/27_downsample_counts.py):

* one shared depth for every analysed bin of every slice, the 10th percentile
  of their pooled total counts;
* exact multivariate hypergeometric subsampling of each bin above it, by
  ``confidenceot.equalise_depth``, with 27_'s seed rule (seed + 7919 * index
  over the sorted samples);
* bins at or below the target left untouched and counted.

One target for all slices puts every pair, adjacent or same-stage, at the same
depth, and the pseudobulks later read the counts the OT read.

Analysed bins are every bin except the predefined Cavity annotation, the one
exclusion of this run, carried over from the earlier MOSTA analysis.  There is
no bin QC.

Writes, per slice: equalised/<sample>.h5ad (analysed bins; equalised integer
counts; annotation, x, y, original depth), slices/<sample>_bins.csv.gz (every
bin, Cavity included, with original and equalised depth, for plotting) and
slices/<sample>_genes.txt.gz.  Plus equalisation/report.json and
equalisation/per_slice.csv.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mosta_common import (  # noqa: E402
    EQUALISATION_QUANTILE, EQUALISATION_SEED, EXCLUDED_ANNOTATIONS, Layout,
    read_slices, write_json,
)


def read_counts(path: str):
    """The raw count layer and the bin metadata of one MOSTA file."""
    import anndata as ad

    data = ad.read_h5ad(path, backed="r")
    try:
        missing = [name for name, present in (
            ("obs['annotation']", "annotation" in data.obs),
            ("layers['count']", "count" in data.layers),
            ("obsm['spatial']", "spatial" in data.obsm),
        ) if not present]
        if missing:
            raise KeyError(f"{path} lacks {missing}")
        counts = sparse.csr_matrix(data.layers["count"])
        bin_ids = np.asarray(data.obs_names.astype(str))
        annotation = np.asarray(data.obs["annotation"].astype(str))
        xy = np.asarray(data.obsm["spatial"], dtype=np.float64)[:, :2]
        genes = np.asarray(data.var_names.astype(str))
    finally:
        data.file.close()
    if counts.data.size and not np.all(np.mod(counts.data, 1) == 0):
        raise ValueError(f"{path}: layers['count'] holds non-integer values")
    if np.any(counts.data < 0):
        raise ValueError(f"{path}: layers['count'] holds negative values")
    counts = sparse.csr_matrix(counts, dtype=np.int64)
    counts.eliminate_zeros()
    if len(set(bin_ids)) != len(bin_ids):
        raise ValueError(f"{path}: bin ids repeat")
    return counts, bin_ids, annotation, xy, genes


def depth_pass(task: tuple[str, str]) -> dict:
    sample, path = task
    started = time.perf_counter()
    counts, bin_ids, annotation, xy, genes = read_counts(path)
    return {
        "sample": sample,
        "total": np.asarray(counts.sum(axis=1)).ravel(),
        "detected": np.diff(counts.indptr),
        "analysed": ~np.isin(annotation, EXCLUDED_ANNOTATIONS),
        "n_genes": len(genes),
        "seconds": time.perf_counter() - started,
    }


def write_pass(task: tuple[str, str, int, int, str]) -> dict:
    """Equalise one slice and write its three files."""
    import anndata as ad
    from confidenceot import equalise_depth

    sample, path, index, target, root = task
    layout = Layout(Path(root))
    started = time.perf_counter()
    counts, bin_ids, annotation, xy, genes = read_counts(path)
    analysed = ~np.isin(annotation, EXCLUDED_ANNOTATIONS)
    rows = np.flatnonzero(analysed)
    rng = np.random.default_rng(EQUALISATION_SEED + 7919 * index)
    reduced, original, untouched = equalise_depth(
        counts[rows], rng=rng, target=int(target), return_untouched=True,
    )
    reduced = sparse.csr_matrix(reduced)
    if reduced.data.size and reduced.data.max() > np.iinfo(np.int32).max:
        raise ValueError(f"{sample}: an equalised count does not fit int32")
    reduced = sparse.csr_matrix(reduced, dtype=np.int32)
    obs = pd.DataFrame({
        "annotation": pd.Categorical(annotation[rows]),
        "x": xy[rows, 0], "y": xy[rows, 1],
        "predownsample_total_counts": original.astype(np.int64),
        "predownsample_detected_genes": np.diff(counts[rows].indptr).astype(np.int64),
        "downsample_untouched": untouched,
    }, index=pd.Index(bin_ids[rows], name="bin_id"))
    output = ad.AnnData(X=reduced, obs=obs, var=pd.DataFrame(index=pd.Index(genes, name="gene")))
    output.uns["equalisation"] = {
        "target_depth": int(target), "seed": int(EQUALISATION_SEED + 7919 * index),
        "excluded_annotations": list(EXCLUDED_ANNOTATIONS), "source_file": path,
    }
    layout.equalised.mkdir(parents=True, exist_ok=True)
    staging = layout.equalised / f".{sample}.h5ad.partial"
    output.write_h5ad(staging, compression="lzf")
    staging.replace(layout.equalised_h5ad(sample))

    equalised_total = np.full(len(bin_ids), np.nan)
    equalised_detected = np.full(len(bin_ids), np.nan)
    untouched_all = np.full(len(bin_ids), np.nan)
    equalised_row = np.full(len(bin_ids), -1, dtype=np.int64)
    equalised_total[rows] = np.asarray(reduced.sum(axis=1)).ravel()
    equalised_detected[rows] = np.diff(reduced.indptr)
    untouched_all[rows] = untouched
    equalised_row[rows] = np.arange(len(rows))
    table = pd.DataFrame({
        "bin_id": bin_ids, "x": xy[:, 0], "y": xy[:, 1], "annotation": annotation,
        "analysed": analysed.astype(np.int8), "equalised_row": equalised_row,
        "total_counts_original": np.asarray(counts.sum(axis=1)).ravel(),
        "detected_genes_original": np.diff(counts.indptr),
        "total_counts_equalised": equalised_total,
        "detected_genes_equalised": equalised_detected,
        "untouched_by_equalisation": untouched_all,
    })
    layout.slices.mkdir(parents=True, exist_ok=True)
    table.to_csv(layout.slice_bins(sample), index=False, compression="gzip")
    with gzip.open(layout.slice_genes(sample), "wt", encoding="utf-8") as handle:
        handle.write("\n".join(genes) + "\n")
    detected_after = np.diff(reduced.indptr)
    return {
        "sample": sample,
        "bins_total": int(len(bin_ids)),
        "bins_excluded_cavity": int((~analysed).sum()),
        "bins_analysed": int(len(rows)),
        "genes": int(len(genes)),
        "median_depth_before": float(np.median(original)) if len(rows) else float("nan"),
        "median_depth_after": float(np.median(np.asarray(reduced.sum(axis=1)).ravel())) if len(rows) else float("nan"),
        "untouched_bins": int(untouched.sum()),
        "untouched_fraction": float(untouched.mean()) if len(rows) else float("nan"),
        "bins_detecting_under_256_genes_after": int((detected_after < 256).sum()),
        "median_detected_genes_after": float(np.median(detected_after)) if len(rows) else float("nan"),
        "seed": int(EQUALISATION_SEED + 7919 * index),
        "seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    layout = Layout(args.root)
    slices = read_slices(layout)
    tasks = list(zip(slices["sample"], slices["path"]))
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        depth = {record["sample"]: record for record in pool.map(depth_pass, tasks)}
    pooled = np.concatenate([depth[sample]["total"][depth[sample]["analysed"]] for sample, _ in tasks])
    target = int(np.quantile(pooled, EQUALISATION_QUANTILE))
    if target < 1:
        raise SystemExit(f"target depth resolved to {target}")
    print(f"pooled analysed bins {pooled.size}; median depth {np.median(pooled):.0f}; "
          f"target (q={EQUALISATION_QUANTILE}) {target}", flush=True)
    write_tasks = [(sample, path, index, target, str(layout.root))
                   for index, (sample, path) in enumerate(tasks)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        per_slice = pd.DataFrame(list(pool.map(write_pass, write_tasks)))
    layout.equalisation.mkdir(parents=True, exist_ok=True)
    per_slice.to_csv(layout.equalisation / "per_slice.csv", index=False)
    write_json(layout.equalisation / "report.json", {
        "target_depth": target,
        "target_quantile": EQUALISATION_QUANTILE,
        "target_rule": "quantile of the pooled total counts of every analysed bin of every slice",
        "procedure": "confidenceot.equalise_depth (exact multivariate hypergeometric, "
                     "without replacement), as cancer_metastasis/27_downsample_counts.py",
        "seed_rule": f"{EQUALISATION_SEED} + 7919 * slice index (slices.csv order)",
        "excluded_annotations": list(EXCLUDED_ANNOTATIONS),
        "bin_qc": "none",
        "slices": int(len(per_slice)),
        "bins_total": int(per_slice["bins_total"].sum()),
        "bins_excluded_cavity": int(per_slice["bins_excluded_cavity"].sum()),
        "bins_analysed": int(per_slice["bins_analysed"].sum()),
        "pooled_median_depth_before": float(np.median(pooled)),
        "untouched_bins": int(per_slice["untouched_bins"].sum()),
        "untouched_fraction": float(per_slice["untouched_bins"].sum() / per_slice["bins_analysed"].sum()),
        "bins_detecting_under_256_genes_after": int(per_slice["bins_detecting_under_256_genes_after"].sum()),
        "wall_seconds": time.perf_counter() - started,
    })
    print(per_slice.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
