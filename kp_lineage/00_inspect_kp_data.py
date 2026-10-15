"""Read-only structure report of the KP spatial lineage (Jones, Sun et al.
2026) and KP-Tracer (Yang et al. 2022) downloads, before any analysis is
designed on them.

    python kp_lineage/00_inspect_kp_data.py --spatial DIR --tracer DIR --out REPORT.txt

Nothing is written next to the data and no matrix is loaded whole. Every
h5ad is opened with h5py and reported as: shape, how X and each layer are
stored (dense or CSR, dtype), whether a sample of X's stored values is
integer (raw counts or not), and for every obs column its dtype, the number
of distinct values, and up to --max-levels of them with their counts. var,
obsm, layers and uns are listed by key. The small text tables (metadata,
README, tumour statistics) are printed whole or by their header and value
counts.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import Counter

import h5py
import numpy as np
import pandas as pd


def decode(values):
    return [v.decode() if isinstance(v, bytes) else v for v in values]


def read_column(obs, name):
    """One obs column as a plain array, for both anndata on-disk layouts:
    categorical as a group (categories + codes), or the pre-0.7 layout with
    codes as the dataset and categories under uns/<name>_categories."""
    item = obs[name]
    if isinstance(item, h5py.Group):
        if "categories" in item and "codes" in item:
            categories = np.asarray(decode(item["categories"][()]), dtype=object)
            codes = item["codes"][()]
            out = np.empty(len(codes), dtype=object)
            out[codes >= 0] = categories[codes[codes >= 0]]
            out[codes < 0] = None
            return out
        return None
    values = item[()]
    if "categories" in item.attrs:
        categories = np.asarray(decode(obs.file[item.attrs["categories"]][()]), dtype=object)
        out = np.empty(len(values), dtype=object)
        out[values >= 0] = categories[values[values >= 0]]
        return out
    if values.dtype.kind in "SO":
        return np.asarray(decode(values), dtype=object)
    return values


def describe_matrix(node, sample=200000):
    if isinstance(node, h5py.Dataset):
        flat = node[: min(node.shape[0], max(1, sample // max(1, node.shape[1] if node.ndim > 1 else 1)))]
        values = np.asarray(flat).ravel()
        kind = f"dense {node.shape} {node.dtype}"
    else:
        encoding = node.attrs.get("encoding-type", node.attrs.get("h5sparse_format", "group"))
        if isinstance(encoding, bytes):
            encoding = encoding.decode()
        shape = node.attrs.get("shape", None)
        data = node["data"] if "data" in node else None
        if data is None:
            return f"{encoding} (no data array)"
        values = data[: min(data.shape[0], sample)]
        kind = f"{encoding} shape={tuple(shape) if shape is not None else '?'} nnz={data.shape[0]} {data.dtype}"
    values = values[np.isfinite(values)] if values.dtype.kind == "f" else values
    if values.size == 0:
        return kind + "; sample empty"
    integer = bool(np.all(np.equal(np.mod(values, 1), 0)))
    return (f"{kind}; sample of {values.size}: min {values.min():.4g}, max {values.max():.4g}, "
            f"all integer {integer}")


def describe_h5ad(path, max_levels, write):
    write(f"\n=== {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")
    try:
        f = h5py.File(path, "r")
    except OSError as error:
        write(f"  cannot open: {error}")
        return
    with f:
        write(f"  top-level keys: {sorted(f.keys())}")
        if "X" in f:
            write(f"  X: {describe_matrix(f['X'])}")
        if "raw" in f and "X" in f["raw"]:
            write(f"  raw/X: {describe_matrix(f['raw']['X'])}")
        if "layers" in f:
            for key in f["layers"]:
                write(f"  layers/{key}: {describe_matrix(f['layers'][key])}")
        for group in ("obsm", "varm", "obsp", "uns"):
            if group in f:
                write(f"  {group}: {sorted(f[group].keys())}")
        if "var" in f:
            var = f["var"]
            index_key = var.attrs.get("_index", "_index")
            if isinstance(index_key, bytes):
                index_key = index_key.decode()
            names = decode(var[index_key][:5]) if index_key in var else []
            write(f"  var columns: {[k for k in var.keys() if k != index_key]}; first names {names}")
        if "obs" not in f:
            return
        obs = f["obs"]
        index_key = obs.attrs.get("_index", "_index")
        if isinstance(index_key, bytes):
            index_key = index_key.decode()
        if index_key in obs:
            index = obs[index_key]
            write(f"  n_obs {index.shape[0]}; first obs names {decode(index[:3])}")
        order = obs.attrs.get("column-order", None)
        columns = decode(list(order)) if order is not None and len(order) else [k for k in obs.keys()]
        for name in columns:
            if name == index_key or name not in obs:
                continue
            values = read_column(obs, name)
            if values is None:
                write(f"  obs[{name}]: (unreadable group)")
                continue
            if values.dtype.kind in "fiub":
                finite = values[np.isfinite(values)] if values.dtype.kind == "f" else values
                distinct = np.unique(finite).size if finite.size else 0
                if distinct <= max_levels:
                    counts = Counter(finite.tolist())
                    write(f"  obs[{name}] {values.dtype}: {distinct} values {dict(sorted(counts.items()))}")
                else:
                    write(f"  obs[{name}] {values.dtype}: {distinct} distinct; "
                          f"min {finite.min():.4g} median {np.median(finite):.4g} max {finite.max():.4g}; "
                          f"missing {values.size - finite.size}")
            else:
                counts = Counter(values.tolist())
                shown = dict(counts.most_common(max_levels))
                write(f"  obs[{name}] text: {len(counts)} distinct; top {shown}")


def describe_table(path, write, value_columns_max=60, max_levels=40):
    write(f"\n=== {path}  ({os.path.getsize(path) / 1e6:.2f} MB)")
    sep = "\t" if path.endswith((".tsv", ".txt")) else ","
    try:
        table = pd.read_csv(path, sep=sep, low_memory=False)
    except Exception as error:  # noqa: BLE001 - report and continue
        write(f"  cannot read as a table: {error}")
        return
    write(f"  shape {table.shape}")
    if table.shape[0] <= 80:
        write(table.to_string(max_cols=value_columns_max))
        return
    write(f"  columns: {list(table.columns)}")
    write(table.head(5).to_string(max_cols=value_columns_max))
    for column in table.columns:
        counts = table[column].value_counts(dropna=False)
        if len(counts) <= max_levels:
            write(f"  [{column}] {len(counts)} values: {counts.to_dict()}")
        else:
            write(f"  [{column}] {len(counts)} distinct")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spatial", required=True, help="KPSpatial_Data directory")
    parser.add_argument("--tracer", required=True, help="KPTracer-Data directory")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-levels", type=int, default=40)
    args = parser.parse_args()

    handle = open(args.out, "w", encoding="utf-8")

    def write(line):
        handle.write(line + "\n")
        handle.flush()

    write("# Text files")
    for path in [os.path.join(args.tracer, "README")]:
        write(f"\n=== {path}")
        with open(path, encoding="utf-8", errors="replace") as text:
            write(text.read())
    for path in [os.path.join(args.spatial, "KPSpatial_meta.tsv"),
                 os.path.join(args.tracer, "tumor_statistics.tsv"),
                 os.path.join(args.tracer, "trees", "tumor_list.txt"),
                 os.path.join(args.tracer, "KPTracer_meta.csv"),
                 os.path.join(args.tracer, "plasticity_scores.tsv")]:
        if os.path.exists(path):
            describe_table(path, write, max_levels=args.max_levels)
    for path in sorted(glob.glob(os.path.join(args.spatial, "slideseq", "trees", "C36_09_allele_table.tsv")))[:1]:
        write(f"\n=== {path} (header and first rows)")
        write(pd.read_csv(path, sep="\t", nrows=5).to_string())

    write("\n# h5ad files")
    files = (sorted(glob.glob(os.path.join(args.spatial, "slideseq", "expression", "*.h5ad")))
             + sorted(glob.glob(os.path.join(args.spatial, "slidetags", "expression", "*.h5ad")))
             + sorted(glob.glob(os.path.join(args.spatial, "KP_2022", "*.h5ad")))
             + sorted(glob.glob(os.path.join(args.tracer, "expression", "*.h5ad"))))
    for path in files:
        describe_h5ad(path, args.max_levels, write)
    write("\nDONE")
    handle.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
