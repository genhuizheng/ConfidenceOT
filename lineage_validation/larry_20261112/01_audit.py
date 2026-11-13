"""LARRY (Weinreb et al. 2020, GSE140802) in vitro state-fate data: audit and raw-count recovery check.

    python -I lineage_validation/larry_20261112/01_audit.py --data results/larry_20261112/data --out results/larry_20261112

The public matrix is total-count normalised. If each cell was only rescaled, the smallest non-zero value of a cell is
its count-1 level and normed / that value is an integer. This script measures that per cell (share of non-zero entries
within 0.02 of an integer after rescaling) and, where it holds, writes the recovered counts.
Writes OUT/audit/{summary.tsv, per_cell.tsv.gz, clones.tsv} and OUT/prep/counts_recovered.h5ad (cells x genes,
integer counts, obs = metadata + clone id).
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


def read_mtx(path, dtype=np.float32, chunk=20_000_000):
    with gzip.open(path, "rt") as f:
        line = f.readline()
        while line.startswith("%"):
            line = f.readline()
        nr, nc, nnz = map(int, line.split())
        rows, cols, vals = [], [], []
        for df in pd.read_csv(f, sep=" ", header=None, names=["r", "c", "v"], chunksize=chunk, dtype={"r": np.int32, "c": np.int32, "v": np.float64}):
            rows.append(df.r.to_numpy() - 1); cols.append(df.c.to_numpy() - 1); vals.append(df.v.to_numpy().astype(dtype))
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nr, nc))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    au, pr = a.out / "audit", a.out / "prep"; au.mkdir(parents=True, exist_ok=True); pr.mkdir(parents=True, exist_ok=True)
    pre = "GSM4185642_stateFate_inVitro_"
    meta = pd.read_csv(a.data / f"{pre}metadata.txt.gz", sep="\t")
    genes = pd.read_csv(a.data / f"{pre}gene_names.txt.gz", header=None)[0].astype(str).to_numpy()
    clones = read_mtx(a.data / f"{pre}clone_matrix.mtx.gz", dtype=np.float32)
    print("metadata", meta.shape, meta.columns.tolist()); print("clone matrix", clones.shape, "genes", len(genes))
    X = read_mtx(a.data / f"{pre}normed_counts.mtx.gz")
    print("counts", X.shape, X.nnz)
    # raw recovery
    X.sort_indices()
    mins = np.array([X.data[X.indptr[i]:X.indptr[i + 1]].min() if X.indptr[i + 1] > X.indptr[i] else np.nan for i in range(X.shape[0])])
    scale = np.repeat(1 / mins, np.diff(X.indptr))
    R = X.data.astype(np.float64) * scale
    frac = np.abs(R - np.round(R)) <= 0.02
    per_cell_int = np.add.reduceat(frac, X.indptr[:-1]) / np.maximum(np.diff(X.indptr), 1)
    tot_norm = np.asarray(X.sum(1)).ravel()
    Rc = sp.csr_matrix((np.round(R).astype(np.float32), X.indices, X.indptr), shape=X.shape)
    tot_raw = np.asarray(Rc.sum(1)).ravel()
    pc = meta.copy(); pc["normed_total"] = tot_norm; pc["recovered_total"] = tot_raw; pc["detected_genes"] = np.diff(X.indptr)
    pc["integer_share"] = per_cell_int; pc["min_nonzero"] = mins
    ncl = np.asarray(clones.sum(1)).ravel(); pc["n_clones"] = ncl
    cl_id = np.full(len(pc), -1); r_, c_ = clones.nonzero(); cl_id[r_] = c_
    pc["clone"] = cl_id
    pc.to_csv(au / "per_cell.tsv.gz", sep="\t", index=False, compression="gzip")
    summ = {"cells": X.shape[0], "genes": X.shape[1], "nnz": int(X.nnz), "normed_total_median": float(np.median(tot_norm)),
            "normed_total_cv": float(np.std(tot_norm) / np.mean(tot_norm)), "integer_share_median": float(np.median(per_cell_int)),
            "integer_share_p05": float(np.percentile(per_cell_int, 5)), "cells_integer_share_ge_0.99": float(np.mean(per_cell_int >= 0.99)),
            "recovered_total_median": float(np.median(tot_raw)), "cells_in_one_clone": int((ncl == 1).sum()), "cells_in_gt1_clone": int((ncl > 1).sum()),
            "clones": clones.shape[1]}
    pd.Series(summ).to_csv(au / "summary.tsv", sep="\t", header=False); print(pd.Series(summ).to_string())
    tcol = [c for c in pc.columns if "Time" in c][0]; pop = [c for c in pc.columns if "Starting" in c][0]; ctc = [c for c in pc.columns if "Cell type" in c][0]
    print(pd.crosstab([pc[pop], pc[tcol]], pc[ctc]).to_string())
    # clones present at day 2 and at day 4/6
    cc = pc[pc.clone >= 0]
    by = cc.groupby("clone")[tcol].agg(lambda s: tuple(sorted(set(s))))
    d2 = cc[cc[tcol] == 2].groupby("clone").size(); late = cc[cc[tcol] > 2].groupby("clone").size()
    both = sorted(set(d2.index) & set(late.index))
    CL = pd.DataFrame({"clone": both, "day2_cells": d2.reindex(both).to_numpy(), "late_cells": late.reindex(both).to_numpy()})
    CL.to_csv(au / "clones.tsv", sep="\t", index=False)
    print(f"clones with day-2 and day-4/6 cells: {len(both)}; day-2 cells in them: {int(CL.day2_cells.sum())}; late cells: {int(CL.late_cells.sum())}")
    print(by.value_counts().head(8).to_string())
    if np.median(per_cell_int) >= 0.99:
        import anndata as ad
        A = ad.AnnData(Rc, obs=pc.reset_index(drop=True).astype({c: str for c in pc.columns if pc[c].dtype == object}), var=pd.DataFrame(index=genes))
        A.obs_names = [f"c{i}" for i in range(A.n_obs)]
        A.write_h5ad(pr / "counts_recovered.h5ad", compression="gzip"); print("wrote counts_recovered.h5ad")


if __name__ == "__main__":
    main()
