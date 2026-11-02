"""Per-cell UCell scores of the authors' marker programs (Sar et al., Genome Med 2026, Additional file 2 Table S2), on
the original counts and on the frozen P10 equalised counts.

    python ltl331_nepc/11_signature_scores.py --table-s2 S2.tsv --converted CONVERTED --run-root RUN_ROOT --out OUT

Programs: every (source, group) of Table S2 with >= 3 genes: canonical_annotation AD / NE / EMT / Proliferation /
Epithelial (the four programs the authors scored with UCell, from Dong et al. 2020, plus the epithelial panel) and
Fig2D_program_panel Luminal / Basal / PRAD / EMT / Neuroendocrine / Proliferation / Ciliary. Single-gene cluster
markers are not scored. The authors define no REST program; none is added.
UCell (Andreatta & Carmona 2021, the authors' v2.6.2 default maxRank 1500): genes ranked within the cell by decreasing
expression with average ties, ranks above maxRank set to maxRank + 1, U = sum of the program's ranks - n(n + 1) / 2,
score = 1 - U / (n x maxRank). Ranks within a cell do not change under per-cell normalisation or log, so counts give
the scores the authors' log-normalised data give. On equalised counts the same formula gives a score at one depth.
Genes of a program absent from the matrix are dropped and listed.
Writes OUT/ (refuses if it exists): CELL_SIGNATURES.tsv.gz (observation_id, counts in {original, p10}, one column per
program), PROGRAM_GENES.tsv (genes used / missing), CLUSTER_SIGNATURES.tsv (median score and the fraction of cells
with ASCL1 / NEUROD1 / AR / REST counts > 0 per seurat cluster, original counts: a check of the state annotation).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata

MAX_RANK = 1500
CHECK_GENES = ["ASCL1", "NEUROD1", "AR", "KLK3", "REST", "SRRM4", "VIM", "MKI67"]


def ucell(X: sp.csr_matrix, cols: dict[str, np.ndarray], max_rank: int = MAX_RANK) -> pd.DataFrame:
    X = sp.csr_matrix(X); G = X.shape[1]
    out = {k: np.empty(X.shape[0]) for k in cols}
    for i in range(X.shape[0]):
        a, b = X.indptr[i], X.indptr[i + 1]
        idx, val = X.indices[a:b], X.data[a:b]
        nz = val > 0; idx, val = idx[nz], val[nz]
        r_nz = rankdata(-val, method="average")
        rank_zero = (len(val) + 1 + G) / 2.0
        pos = dict(zip(idx.tolist(), r_nz.tolist()))
        for k, c in cols.items():
            r = np.array([pos.get(int(j), rank_zero) for j in c])
            r = np.minimum(r, max_rank + 1)
            n = len(c); u = r.sum() - n * (n + 1) / 2
            out[k][i] = 1 - u / (n * max_rank)
    return pd.DataFrame(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table-s2", type=Path, required=True, help="Table S2 as tsv (columns source, group, gene after its header rows)")
    p.add_argument("--converted", type=Path, required=True)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        sys.exit(f"{a.out} exists; not overwritten")
    a.out.mkdir(parents=True)
    raw = pd.read_csv(a.table_s2, sep="\t", header=None, dtype=str)
    start = raw.index[raw[0].eq("source")][0]
    s2 = raw.iloc[start + 1:, :3].set_axis(["source", "group", "gene"], axis=1).dropna()
    s2 = s2[~s2.source.eq("Fig3_cluster_markers")]
    programs = {f"{src}:{grp}": list(dict.fromkeys(g.gene.str.strip())) for (src, grp), g in s2.groupby(["source", "group"], sort=False)}
    programs = {k: v for k, v in programs.items() if len(v) >= 3}
    parts, used, clus = [], [], []
    for f in sorted((a.converted / "h5ad").glob("*.h5ad")):
        x = ad.read_h5ad(f)
        names = pd.Index(x.var["gene_symbol"].astype(str) if "gene_symbol" in x.var else x.var_names.astype(str))
        first = pd.Series(np.arange(len(names)), index=names); first = first[~first.index.duplicated()]
        cols = {k: first.reindex(v).dropna().astype(int).to_numpy() for k, v in programs.items()}
        if not used:
            used = [{"program": k, "genes": len(v), "used": len(cols[k]), "missing": ", ".join(g for g in v if g not in first.index)} for k, v in programs.items()]
        sc = ucell(x.X, cols).assign(observation_id=x.obs_names.astype(str).to_numpy(), counts="original")
        parts.append(sc)
        X = sp.csr_matrix(x.X)
        det = {g: np.asarray((X[:, first[g]] > 0).todense()).ravel() for g in CHECK_GENES if g in first.index}
        clus.append(pd.DataFrame({"cluster": x.obs.seurat_clusters.astype(int).to_numpy(), **{f"{g}_detected": v for g, v in det.items()},
                                  **{k: sc[k].to_numpy() for k in programs}}))
        dsf = a.run_root / "inputs" / "ds" / "h5ad" / f"LTL331__{f.stem}.h5ad"
        if dsf.exists():
            y = ad.read_h5ad(dsf)
            yn = pd.Index(y.var["gene_symbol"].astype(str) if "gene_symbol" in y.var else y.var_names.astype(str))
            fy = pd.Series(np.arange(len(yn)), index=yn); fy = fy[~fy.index.duplicated()]
            cy = {k: fy.reindex(v).dropna().astype(int).to_numpy() for k, v in programs.items()}
            parts.append(ucell(y.X, cy).assign(observation_id=y.obs_names.astype(str).to_numpy(), counts="p10"))
        print(f.stem, "done", flush=True)
    cells = pd.concat(parts, ignore_index=True)
    cells.to_csv(a.out / "CELL_SIGNATURES.tsv.gz", sep="\t", index=False, compression="gzip")
    pd.DataFrame(used).to_csv(a.out / "PROGRAM_GENES.tsv", sep="\t", index=False)
    c = pd.concat(clus)
    agg = {**{k: (k, "median") for k in programs}, **{col: (col, "mean") for col in c.columns if col.endswith("_detected")}}
    c.groupby("cluster").agg(cells=("cluster", "size"), **agg).reset_index().to_csv(a.out / "CLUSTER_SIGNATURES.tsv", sep="\t", index=False)
    print(pd.read_csv(a.out / "CLUSTER_SIGNATURES.tsv", sep="\t").round(3).to_string(index=False))
    print(pd.DataFrame(used).to_string(index=False))


if __name__ == "__main__":
    main()
