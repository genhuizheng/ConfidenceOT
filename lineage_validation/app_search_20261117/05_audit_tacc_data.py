"""Format audit of the external real datasets downloaded on TACC (2026-11-18) before any unit is defined. Reads headers,
metadata and archive listings only; no counts are transformed and nothing is written next to the data.

    python -I lineage_validation/app_search_20261117/05_audit_tacc_data.py --data $SCRATCH/OT_project/app_search_20261117/data \
        --out $SCRATCH/OT_project/app_search_20261117/audit

Per dataset directory it writes OUT/<dataset>.txt:
  files            name and size of every file
  tar archives     member names and sizes (no extraction)
  .h5ad            shape, X dtype, integer check on the first 2,000 rows, obs columns with dtype and up to 25 levels each,
                   var columns, uns / obsm / layers keys
  csv(.gz) / tsv   header, row count and up to 25 levels of each low-cardinality column (annotation tables only; count
                   matrices and files above 50 MB are read for the header only)
"""

from __future__ import annotations

import argparse
import gzip
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

LEVELS = 25


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def describe_frame(df: pd.DataFrame, w) -> None:
    for c in df.columns:
        s = df[c]
        nun = s.nunique(dropna=False)
        w(f"    {c} [{s.dtype}] unique={nun} missing={int(s.isna().sum())}")
        if nun <= LEVELS:
            w("      " + "; ".join(f"{k}: {v}" for k, v in s.astype(str).value_counts(dropna=False).items()))
        elif pd.api.types.is_numeric_dtype(s):
            w(f"      min {s.min()} median {s.median()} max {s.max()}")


def audit_h5ad(f: Path, w) -> None:
    import anndata as ad
    A = ad.read_h5ad(f, backed="r")
    w(f"  shape {A.shape}")
    X = A.X[: min(2000, A.n_obs)]
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    w(f"  X dtype {X.dtype}; first rows integer: {bool(np.allclose(X, np.round(X)))}; max {X.max()}; row sums median {np.median(X.sum(1))}")
    w(f"  layers {list(A.layers.keys())}; obsm {list(A.obsm.keys())}; uns {list(A.uns.keys())[:30]}")
    w(f"  var columns {list(A.var.columns)}; var_names head {list(A.var_names[:5])}")
    w("  obs:")
    describe_frame(A.obs, w)


def audit_table(f: Path, w) -> None:
    sep = "\t" if ".tsv" in f.name or f.name.endswith(".txt.gz") or f.name.endswith(".txt") else ","
    if f.stat().st_size > 50 * 2**20 or "matrix" in f.name:
        op = gzip.open if f.suffix == ".gz" else open
        with op(f, "rt") as h:
            head = h.readline().rstrip("\n").split(sep)
        w(f"  large table, header only: {len(head)} fields; first {head[:6]}")
        return
    df = pd.read_csv(f, sep=sep, low_memory=False)
    w(f"  rows {len(df)} columns {len(df.columns)}")
    describe_frame(df, w)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for d in sorted(x for x in a.data.iterdir() if x.is_dir()):
        lines: list[str] = []
        w = lines.append
        w(f"# {d.name}")
        for f in sorted(d.iterdir()):
            w(f"\n## {f.name}  ({human(f.stat().st_size)})")
            try:
                if f.suffix == ".tar":
                    with tarfile.open(f) as t:
                        for m in t.getmembers():
                            w(f"  {m.name}  {human(m.size)}")
                elif f.suffix == ".h5ad":
                    audit_h5ad(f, w)
                elif any(f.name.endswith(s) for s in (".csv", ".csv.gz", ".tsv", ".tsv.gz", ".txt.gz")) and f.name != "filelist.txt":
                    audit_table(f, w)
            except Exception as e:  # report and continue: an audit must cover every file
                w(f"  ERROR {type(e).__name__}: {e}")
        (a.out / f"{d.name}.txt").write_text("\n".join(lines) + "\n")
        print(d.name, "audited", flush=True)


if __name__ == "__main__":
    main()
