"""Summarise chosen obs columns of an H5AD without building the whole frame.

``anndata.read_h5ad(..., backed="r")`` still materialises every obs column, and
this object has eighty of them over 239,430 cells, which a TACC login node's
memory allowance will not grant. Reading the named columns through h5py keeps
the footprint to those columns.

Categorical columns are stored as codes plus categories, and the stored
categories include levels with no cells, because a Seurat export carries unused
factor levels across. A label list is therefore not evidence that such cells
exist; only the counts are.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def read_column(handle: h5py.File, name: str) -> pd.Series:
    node = handle["obs"][name]
    if isinstance(node, h5py.Group) and "categories" in node:
        categories = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in node["categories"][:]
        ]
        codes = np.asarray(node["codes"][:])
        values = np.where(
            codes >= 0, np.asarray(categories, dtype=object)[np.clip(codes, 0, None)], None
        )
        return pd.Series(values, name=name)
    values = node[:]
    if h5py.check_string_dtype(node.dtype) or node.dtype.kind in "OS":
        values = [v.decode() if isinstance(v, bytes) else str(v) for v in values]
    return pd.Series(values, name=name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5ad", type=Path)
    parser.add_argument("--count", action="append", dest="counts", default=None,
                        help="obs column to tabulate; may be repeated")
    parser.add_argument("--cross", action="append", dest="crosses", default=None,
                        metavar="A,B", help="pair of obs columns to cross-tabulate")
    args = parser.parse_args()

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    with h5py.File(args.h5ad, "r") as handle:
        available = sorted(handle["obs"])
        cache: dict[str, pd.Series] = {}

        def column(name: str) -> pd.Series:
            if name not in cache:
                if name not in handle["obs"]:
                    raise SystemExit(f"{name!r} not in obs; available: {available}")
                cache[name] = read_column(handle, name)
            return cache[name]

        shape = handle["X"].attrs.get("shape")
        print(f"{args.h5ad}\nshape: {shape}\n")
        for name in args.counts or []:
            series = column(name)
            print(f"== {name} ({series.nunique(dropna=True)} observed) ==")
            print(series.value_counts(dropna=False).to_string())
            print()
        for pair in args.crosses or []:
            left, right = [part.strip() for part in pair.split(",", 1)]
            print(f"== {left} x {right} ==")
            print(pd.crosstab(column(left), column(right)).to_string())
            print()
        if "var" in handle:
            names = handle["var"].get("_index")
            if names is not None:
                sample = names[: min(6, names.shape[0])]
                sample = [v.decode() if isinstance(v, bytes) else str(v) for v in sample]
                print("var index sample:", sample)
            print("var columns:", sorted(k for k in handle["var"] if not k.startswith("_")))


if __name__ == "__main__":
    main()
