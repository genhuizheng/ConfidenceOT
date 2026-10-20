"""infercnvpy on one patient unit, from the same inputs the R inferCNV run read.

    python cancer_metastasis/49_infercnvpy_unit.py --unit UNIT_DIR --dataset GSE180661 \\
        --patient SPECTRUM-OV-002 --out CNV_ROOT [--threads 16]

UNIT_DIR is a unit the R inferCNV runs used (infercnv_r/runs_patient/<dataset>/<patient>/,
or a prostate pilot unit): counts.tsv.gz (raw counts, genes x cells), annotations.tsv
(cell, group; the groups prefixed ref_ are the reference), gene_order.tsv (gene, chr,
start, stop) and cell_meta.tsv.gz (cell, source_file, barcode). The cells, the reference
and the genes are those of the R run; only the method changes.

Method: counts normalised to 10,000 per cell and log1p, then infercnvpy.tl.infercnv with
every setting at the package default (window 100 genes, step 10, lfc_clip 3,
dynamic_threshold 1.5, chrX and chrY excluded, chunksize 5000), the reference being the
unit's ref_ groups. The CNV matrix is kept exactly as infercnvpy returns it; nothing is
thresholded or filtered afterwards.

Written to CNV_ROOT/<dataset>/<patient>/:
  cnv_matrix.npz        the full CNV matrix, every cell of the unit (reference included),
                        cells x windows, scipy sparse, as infercnvpy returns it
  cnv_cells.tsv         its rows: cell, group, is_reference, source_file, barcode, total_counts
  cnv_windows.tsv       its columns: window, chr, first_gene, last_gene, start, stop
  cnv.f32, cells.tsv, genes.tsv
                        the observation cells only, in 44_export_cnv_cells.R's format so
                        02_run_pair.py --cnv-root reads it unchanged: float32 cells x windows,
                        value + 1 so that 1 is copy-neutral; a barcode naming two observation
                        cells is left out, as in 44_
  diagnostics.json      descriptive only: cell counts and the median per-cell mean |CNV| of
                        reference and observation cells, overall and within total-count quintiles
  provenance.json, SUCCESS
Refuses to overwrite a unit that has SUCCESS.

--unit-h5ad UNIT.h5ad instead of --unit reads one H5AD holding the same four things: raw
counts in X (cells x genes), obs columns group, source_file and barcode, and var columns
chr, start and stop (built for KP-Tracer by kp_lineage/cnv/01_build_infercnvpy_units.py).
Everything after reading is the same code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def read_counts(path: Path) -> tuple[sp.csr_matrix, list[str], list[str]]:
    """counts.tsv.gz, genes x cells, read in gene blocks into a sparse cells x genes matrix."""
    blocks, genes, cells = [], [], None
    for chunk in pd.read_csv(path, sep="\t", index_col=0, chunksize=250):
        if cells is None:
            cells = [str(c) for c in chunk.columns]
        values = chunk.to_numpy()
        if not np.allclose(values, np.round(values)):
            raise ValueError(f"{path}: not integer counts")
        blocks.append(sp.csr_matrix(values.astype(np.float32)))
        genes.extend(str(g) for g in chunk.index)
    matrix = sp.vstack(blocks).T.tocsr()
    return matrix, genes, cells


def window_table(var: pd.DataFrame, chr_pos: dict, n_windows: int, window: int, step: int) -> pd.DataFrame:
    """The genes each window averages, as infercnvpy's running mean lays them out per chromosome."""
    rows = []
    starts = sorted(chr_pos.items(), key=lambda kv: kv[1])
    for k, (chrom, first) in enumerate(starts):
        last = starts[k + 1][1] if k + 1 < len(starts) else n_windows
        genes = var[var["chromosome"] == chrom].sort_values("start")
        for j in range(last - first):
            lo = j * step
            hi = min(lo + window, len(genes)) - 1
            rows.append({"window": first + j, "chr": chrom, "first_gene": genes.index[lo] if lo < len(genes) else "",
                         "last_gene": genes.index[hi] if 0 <= hi < len(genes) else "",
                         "start": int(genes["start"].iloc[lo]) if lo < len(genes) else -1,
                         "stop": int(genes["end"].iloc[hi]) if 0 <= hi < len(genes) else -1})
    table = pd.DataFrame(rows)
    if len(table) != n_windows:
        raise RuntimeError(f"window table has {len(table)} rows for {n_windows} windows")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    unit_input = parser.add_mutually_exclusive_group(required=True)
    unit_input.add_argument("--unit", type=Path)
    unit_input.add_argument("--unit-h5ad", type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--patient", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

    import anndata as ad
    import infercnvpy as cnv
    import scanpy as sc

    out = args.out / args.dataset / args.patient
    if (out / "SUCCESS").exists():
        sys.exit(f"{out}/SUCCESS exists; not overwritten")
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if args.unit_h5ad is not None:
        unit = args.unit_h5ad
        inputs = {"unit.h5ad": unit}
        if not unit.exists():
            sys.exit(f"{unit} is missing")
        source = ad.read_h5ad(unit)
        for column in ("group", "source_file", "barcode"):
            if column not in source.obs:
                raise ValueError(f"{unit}: obs has no {column} column")
        for column in ("chr", "start", "stop"):
            if column not in source.var:
                raise ValueError(f"{unit}: var has no {column} column")
        counts = source.X.tocsr() if sp.issparse(source.X) else sp.csr_matrix(source.X)
        counts = counts.astype(np.float32)
        if counts.nnz and not np.allclose(counts.data, np.round(counts.data)):
            raise ValueError(f"{unit}: X is not integer counts")
        cells = [str(c) for c in source.obs_names]
        genes = [str(g) for g in source.var_names]
        annotations = pd.DataFrame({"cell": cells, "group": source.obs["group"].astype(str).to_numpy()})
        meta = pd.DataFrame({"cell": cells, "source_file": source.obs["source_file"].astype(str).to_numpy(),
                             "barcode": source.obs["barcode"].astype(str).to_numpy()})
        order = pd.DataFrame({"gene": genes, "chr": source.var["chr"].astype(str).to_numpy(),
                              "start": source.var["start"].to_numpy(), "stop": source.var["stop"].to_numpy()})
        del source
    else:
        unit = args.unit
        inputs = {name: unit / name for name in ("counts.tsv.gz", "annotations.tsv", "gene_order.tsv",
                                                 "cell_meta.tsv.gz")}
        for name, path in inputs.items():
            if not path.exists():
                sys.exit(f"{unit}: {name} is missing")
        annotations = pd.read_csv(inputs["annotations.tsv"], sep="\t", header=None, names=["cell", "group"], dtype=str)
        order = pd.read_csv(inputs["gene_order.tsv"], sep="\t", header=None, dtype=str)
        if order.shape[1] < 4:
            raise ValueError(f"{inputs['gene_order.tsv']}: expected gene, chr, start, stop")
        order = order.iloc[:, :4]
        order.columns = ["gene", "chr", "start", "stop"]
        if order.iloc[0]["start"] in ("start",):
            order = order.iloc[1:]
        meta = pd.read_csv(inputs["cell_meta.tsv.gz"], sep="\t", dtype=str)
        for column in ("cell", "source_file", "barcode"):
            if column not in meta:
                raise ValueError(f"{inputs['cell_meta.tsv.gz']}: no {column} column")
        counts, genes, cells = read_counts(inputs["counts.tsv.gz"])
    print(f"{args.dataset}/{args.patient}: {len(cells)} cells x {len(genes)} genes read", flush=True)
    if set(cells) != set(annotations["cell"]):
        raise ValueError("counts.tsv.gz and annotations.tsv name different cells")
    annotations = annotations.set_index("cell").loc[cells]
    meta = meta.drop_duplicates("cell").set_index("cell").reindex(cells)
    if meta["barcode"].isna().any():
        raise ValueError(f"{int(meta['barcode'].isna().sum())} cells are missing from cell_meta.tsv.gz")
    order = order.drop_duplicates("gene").set_index("gene")
    # Positions can be written as floats ("5239802.0"); read them as numbers, then whole bases.
    for column in ("start", "stop"):
        values = pd.to_numeric(order[column], errors="coerce")
        if values.isna().any() or not np.allclose(values, np.round(values)):
            raise ValueError(f"gene_order.tsv: {column} is not a whole number for "
                             f"{int(values.isna().sum())} genes")
        order[column] = np.round(values).astype(np.int64)
    missing = [g for g in genes if g not in order.index]
    if missing:
        raise ValueError(f"{len(missing)} genes have no position, first {missing[:3]}")
    chrom = order.loc[genes, "chr"].astype(str)
    chrom = chrom.where(chrom.str.startswith("chr"), "chr" + chrom)

    data = ad.AnnData(X=counts, obs=pd.DataFrame({"group": annotations["group"].to_numpy()}, index=cells),
                      var=pd.DataFrame({"chromosome": chrom.to_numpy(),
                                        "start": order.loc[genes, "start"].to_numpy(),
                                        "end": order.loc[genes, "stop"].to_numpy()}, index=genes))
    data.obs["total_counts"] = np.asarray(counts.sum(axis=1)).ravel()
    reference_groups = sorted(g for g in data.obs["group"].unique() if str(g).startswith("ref_"))
    if not reference_groups:
        raise ValueError("annotations.tsv has no ref_ group")
    data.obs["is_reference"] = data.obs["group"].isin(reference_groups)
    sc.pp.normalize_total(data, target_sum=1e4)
    sc.pp.log1p(data)
    params = {"reference_key": "group", "reference_cat": reference_groups, "window_size": 100, "step": 10,
              "lfc_clip": 3, "dynamic_threshold": 1.5, "exclude_chromosomes": ("chrX", "chrY")}
    cnv.tl.infercnv(data, n_jobs=args.threads, **params)
    matrix = data.obsm["X_cnv"]
    matrix = matrix.tocsr() if sp.issparse(matrix) else sp.csr_matrix(matrix)
    chr_pos = {str(k): int(v) for k, v in data.uns["cnv"]["chr_pos"].items()}
    print(f"CNV matrix {matrix.shape[0]} cells x {matrix.shape[1]} windows", flush=True)

    # The full matrix, every cell, as infercnvpy returned it.
    sp.save_npz(out / "cnv_matrix.npz", matrix, compressed=True)
    rows = pd.DataFrame({"cell": cells, "group": data.obs["group"].to_numpy(),
                         "is_reference": data.obs["is_reference"].to_numpy(),
                         "source_file": meta["source_file"].to_numpy(), "barcode": meta["barcode"].to_numpy(),
                         "total_counts": data.obs["total_counts"].to_numpy()})
    rows.to_csv(out / "cnv_cells.tsv", sep="\t", index=False)
    windows = window_table(data.var, chr_pos, matrix.shape[1], params["window_size"], params["step"])
    windows.to_csv(out / "cnv_windows.tsv", sep="\t", index=False)

    # The observation cells in 44_'s format, for 02_run_pair.py --cnv-root.
    observed = np.flatnonzero(~rows["is_reference"].to_numpy())
    barcodes = rows["barcode"].to_numpy()[observed]
    shared = pd.Series(barcodes).duplicated(keep=False).to_numpy()
    keep = observed[~shared]
    dense = (matrix[keep].toarray() + 1.0).astype(np.float32)
    dense.tofile(out / "cnv.f32")
    rows.iloc[keep][["cell", "source_file", "barcode"]].to_csv(out / "cells.tsv", sep="\t", index=False)
    windows.rename(columns={"window": "gene"})[["gene", "chr", "start", "stop"]].to_csv(
        out / "genes.tsv", sep="\t", index=False)

    # Diagnostics: per-cell signal, overall and within total-count quintiles.
    signal = np.asarray(abs(matrix).mean(axis=1)).ravel()
    rows["signal"] = signal
    reference = rows["is_reference"].to_numpy()
    quintile = pd.qcut(rows["total_counts"].rank(method="first"), 5, labels=False)
    by_quintile = []
    for q in range(5):
        inside = quintile.to_numpy() == q
        by_quintile.append({"quintile": q + 1,
                            "total_counts_range": [float(rows["total_counts"][inside].min()),
                                                   float(rows["total_counts"][inside].max())],
                            "observation_median": float(np.median(signal[inside & ~reference])) if (inside & ~reference).any() else None,
                            "reference_median": float(np.median(signal[inside & reference])) if (inside & reference).any() else None,
                            "observation_n": int((inside & ~reference).sum()), "reference_n": int((inside & reference).sum())})
    diagnostics = {"cells": int(len(rows)), "reference_cells": int(reference.sum()),
                   "observation_cells": int((~reference).sum()), "observation_exported": int(len(keep)),
                   "observation_left_out_shared_barcode": int(shared.sum()),
                   "windows": int(matrix.shape[1]), "reference_groups": reference_groups,
                   "signal_median": {"reference": float(np.median(signal[reference])),
                                     "observation": float(np.median(signal[~reference]))},
                   "by_total_count_quintile": by_quintile}
    (out / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    rows[["cell", "is_reference", "total_counts", "signal"]].to_csv(out / "cnv_signal.tsv.gz", sep="\t", index=False,
                                                                     compression="gzip")
    provenance = {"unit": str(unit), "dataset": args.dataset, "patient": args.patient,
                  "inputs_md5": {name: md5(path) for name, path in inputs.items()},
                  "normalisation": "normalize_total(target_sum=1e4), log1p", "infercnv": {
                      k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()},
                  "versions": {"infercnvpy": cnv.__version__, "scanpy": sc.__version__, "anndata": ad.__version__,
                               "numpy": np.__version__},
                  "seconds": round(time.time() - started, 1)}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (out / "SUCCESS").write_text("done\n", encoding="utf-8")
    print(json.dumps(diagnostics["signal_median"]), flush=True)


if __name__ == "__main__":
    main()
