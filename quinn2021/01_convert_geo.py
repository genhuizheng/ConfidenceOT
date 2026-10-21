"""Convert Quinn et al. 2021 (Science, abc1944; GEO GSE161363) to one H5AD per experiment.

    python quinn2021/01_convert_geo.py --raw RAW_DIR --out OUT_DIR

RAW_DIR is GSE161363_RAW.tar extracted as is. GEO holds, per experiment tag (5k, 30k,
10k, 100k, LM0): a 10x matrix (matrix.<tag>.mtx.gz, barcodes.<tag>.tsv.gz,
genes.<tag>.tsv.gz) and the authors' per-cell table (meta.<tag>.tsv.gz); and for every tag
but LM0 the lineage files (alleleTable, character_matrices, trees).

Conversion only, nothing is computed or filtered:
* OUT_DIR/h5ad/<tag>.h5ad: X = the matrix as deposited, cells x genes, sparse; var = the
  genes file (gene_id, gene_symbol; var_names = symbols made unique, the originals kept);
  obs = every column of the authors' meta table, joined on the cell barcode. Cells without
  a meta row are kept, with empty values, and counted; meta rows without a cell are counted.
* OUT_DIR/lineage/<tag>/: the allele table copied, the character-matrix and tree tarballs
  extracted, all unchanged.
* OUT_DIR/conversion_report.json: shapes, whether X holds integer counts, the join counts,
  the meta columns and, for every column with fewer than 100 values, its value counts.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import tarfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp


def find(raw: Path, kind: str, tag: str) -> Path | None:
    hits = sorted(raw.glob(f"GSM*_{kind}.{tag}.*")) + sorted(raw.glob(f"GSM*_{kind}_{tag}.*"))
    return hits[0] if hits else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    (args.out / "h5ad").mkdir(parents=True, exist_ok=True)
    tags = sorted({re.match(r"GSM\d+_matrix\.(.+)\.mtx\.gz$", p.name).group(1)
                   for p in args.raw.glob("GSM*_matrix.*.mtx.gz")})
    report = {"raw": str(args.raw), "experiments": {}}
    for tag in tags:
        paths = {kind: find(args.raw, kind, tag) for kind in ("matrix", "barcodes", "genes", "meta")}
        missing = [k for k, v in paths.items() if v is None]
        if missing:
            raise FileNotFoundError(f"{tag}: no {missing} file")
        barcodes = pd.read_csv(paths["barcodes"], sep="\t", header=None, dtype=str)[0].tolist()
        genes = pd.read_csv(paths["genes"], sep="\t", header=None, dtype=str)
        matrix = scipy.io.mmread(paths["matrix"]).tocsr()
        if matrix.shape == (len(genes), len(barcodes)):
            matrix = matrix.T.tocsr()
        elif matrix.shape != (len(barcodes), len(genes)):
            raise ValueError(f"{tag}: matrix {matrix.shape} fits neither {len(barcodes)} cells nor {len(genes)} genes")
        integer = bool(matrix.nnz == 0 or np.allclose(matrix.data, np.round(matrix.data)))
        meta = pd.read_csv(paths["meta"], sep="\t", index_col=0, dtype=str)
        meta.index = meta.index.astype(str)
        join_rule = "barcode as written in both files"
        if not meta.index.isin(barcodes).any() and "sampleID" in meta:
            # 10k: the matrix names a cell "<tissue>.<16 nt>-<library>", the meta table
            # "<16 nt>-<library>" with the tissue in sampleID (each tissue has its own library
            # number in both files). Rebuild the matrix name from the meta row; used only when no
            # barcode matches as written, and only if it is one-to-one.
            core = meta.index.str.extract(r"^([ACGTN]{16}-\d+)$")[0]
            rebuilt = meta["sampleID"].astype(str).to_numpy() + "." + core.to_numpy()
            if core.notna().all() and not pd.Index(rebuilt).duplicated().any():
                meta.index = pd.Index(rebuilt)
                join_rule = "<sampleID>.<meta barcode as written> rebuilt from the meta row"
        obs = meta.reindex(barcodes)
        obs.index = pd.Index(barcodes, name=None)
        # A cell without a meta row has missing values; as categories they stay missing (not
        # "" or "nan"), and h5py can write them, which it cannot for strings mixed with NaN.
        for column in obs.columns:
            obs[column] = obs[column].astype("category")
        var = pd.DataFrame({"gene_id": genes[0].to_numpy(),
                            "gene_symbol": (genes[1] if genes.shape[1] > 1 else genes[0]).to_numpy()})
        var.index = pd.Index(var["gene_symbol"].to_numpy())
        data = ad.AnnData(X=matrix.astype(np.float32), obs=obs, var=var)
        data.var_names_make_unique()
        data.uns["source"] = {k: v.name for k, v in paths.items()}
        data.write_h5ad(args.out / "h5ad" / f"{tag}.h5ad", compression="lzf")

        lineage = args.out / "lineage" / tag
        copied = []
        for kind in ("alleleTable", "character_matrices", "trees"):
            path = find(args.raw, kind, tag)
            if path is None:
                continue
            lineage.mkdir(parents=True, exist_ok=True)
            if path.name.endswith(".tar.gz"):
                with tarfile.open(path) as archive:
                    for member in archive.getmembers():
                        if member.name.startswith("/") or ".." in Path(member.name).parts:
                            raise ValueError(f"{path}: unsafe member {member.name}")
                    archive.extractall(lineage / kind)
            else:
                shutil.copy2(path, lineage / path.name)
            copied.append(path.name)

        columns = {}
        for column in meta.columns:
            values = meta[column]
            if values.nunique(dropna=True) < 100:
                columns[column] = {str(k): int(v) for k, v in values.value_counts(dropna=False).items()}
            else:
                columns[column] = f"{values.nunique(dropna=True)} distinct values"
        report["experiments"][tag] = {
            "files": {k: v.name for k, v in paths.items()}, "lineage_files": copied,
            "cells": int(data.n_obs), "genes": int(data.n_vars), "nonzero": int(matrix.nnz),
            "integer_counts": integer, "max_value": float(matrix.max()) if matrix.nnz else 0.0,
            "join_rule": join_rule,
            "cells_with_meta": int(meta.index.isin(barcodes).sum()),
            "cells_without_meta": int(len(barcodes) - meta.index.isin(barcodes).sum()),
            "meta_rows_without_cell": int((~meta.index.isin(barcodes)).sum()),
            "meta_columns": list(meta.columns), "meta_values": columns,
            "genes_file_columns": int(genes.shape[1]),
        }
        print(tag, data.shape, "integer" if integer else "NOT integer", flush=True)
    (args.out / "conversion_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
