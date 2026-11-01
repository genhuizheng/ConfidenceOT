"""Arm B step 1: LTL331 analysed cells as Matrix Market for splatEstimate (R in the container cannot read h5ad).

    python realdata_benchmark/splatter/21_export_ltl331_mtx.py --run-root RUN_ROOT --out OUT_DIR [--cells 2000]

Per time point, up to --cells cells (seed 20261031) of the original counts of the analysed cells (inputs/rawqc), every
gene with at least one count among them; and a pooled set of the same size drawn evenly from the 8 time points.
Writes OUT/<name>/{matrix.mtx (genes x cells, integer), genes.tsv, cells.tsv}. splatEstimate on the full 50,492
cells is not needed for its moment estimates and is slow; the subsample size is fixed here, not tuned.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rb_common as rb  # noqa: E402


def write(X: sp.csr_matrix, genes, cells, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    keep = np.asarray((X > 0).sum(0)).ravel() > 0
    scipy.io.mmwrite(str(out / "matrix.mtx"), sp.csr_matrix(X[:, keep]).T.astype(np.int64))
    pd.Series(np.asarray(genes)[keep]).to_csv(out / "genes.tsv", sep="\t", index=False, header=False)
    pd.Series(cells).to_csv(out / "cells.tsv", sep="\t", index=False, header=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cells", type=int, default=2000)
    a = p.parse_args()
    if a.out.exists():
        sys.exit(f"{a.out} exists; not overwritten")
    rng = np.random.default_rng(20261031)
    pooled = []
    files = sorted((a.run_root / "inputs" / "rawqc" / "h5ad").glob("*.h5ad"))
    for f in files:
        x = ad.read_h5ad(f)
        pick = np.sort(rng.choice(x.n_obs, min(a.cells, x.n_obs), replace=False))
        X = sp.csr_matrix(x.X)[pick].astype(np.int64)
        write(X, x.var_names, x.obs_names[pick], a.out / f.stem)
        share = np.sort(rng.choice(x.n_obs, min(a.cells // len(files), x.n_obs), replace=False))
        pooled.append((sp.csr_matrix(x.X)[share].astype(np.int64), x.obs_names[share], x.var_names))
        print(f.stem, len(pick), "cells", flush=True)
    genes = pooled[0][2]
    if any(list(g) != list(genes) for _, _, g in pooled):
        sys.exit("time points differ in genes")
    write(sp.vstack([X for X, _, _ in pooled]).tocsr(), genes, np.concatenate([c for _, c, _ in pooled]), a.out / "pooled")
    print("pooled written")


if __name__ == "__main__":
    main()
