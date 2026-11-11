"""PLAN_v2.md section 5 on TACC: DEG and hallmark GSEA of discordant vs concordant cells, from the raw counts the v3
RNA-OT fits used (realdata_benchmark/v3_grid/worker.py Inputs).

    python lineage_validation/rna_lineage_discordance_20261110/60_deg_gsea.py --groups GROUPS.tsv \
        --worklist /scratch/10119/ghzheng/OT_project/v3_grid_20261104/kp/WORKLIST.tsv \
        --gmt-human h.all.v2025.1.Hs.symbols.gmt --gmt-mouse mh.all.2025.1.Mm.symbols.gmt --out OUT

GROUPS.tsv (from 55_define_groups.py, locally): contrast_id, unit_id, cell, group (case / reference). Per contrast:
normalize_total 1e4, log1p of the unit's raw counts; genes detected in >= 10 cells of the contrast; scanpy Wilcoxon
case vs reference; gseapy prerank on the Wilcoxon score, hallmark only (human for Quinn, the MSigDB mouse hallmark for
KP), 1,000 permutations, seed 20261110, set size 15-500. One tumour per site: effects are read, not p-values.
Writes OUT/deg/<contrast>.tsv.gz and OUT/gsea_hallmark.tsv. Refuses an existing OUT.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--groups", type=Path, required=True)
    p.add_argument("--worklist", type=Path, help="TACC mode: read the counts with the v3 worker Inputs")
    p.add_argument("--counts", type=Path, help="local mode: the h5ad files of 58_export_counts.py")
    p.add_argument("--symbol-map", type=Path, help="gene_ids -> gene_symbol (TSV), for units whose genes are Ensembl ids "
                                                    "(Quinn); unmapped genes are dropped, a repeated symbol keeps its first id")
    p.add_argument("--gmt-human", type=Path, required=True)
    p.add_argument("--gmt-mouse", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--downsample", action="store_true",
                   help="sensitivity: per contrast, binomially thin every cell to the contrast's 10th-percentile total "
                        "count (scanpy downsample_counts, seed 20261110) before normalising, so depth differs less")
    a = p.parse_args()
    if a.out.exists():
        sys.exit(f"{a.out} exists")
    (a.out / "deg").mkdir(parents=True)
    import anndata as ad
    import gseapy
    import scanpy as sc
    G = pd.read_csv(a.groups, sep="\t", dtype=str)
    if a.counts is None:  # TACC mode
        sys.path.insert(0, str(REPO / "realdata_benchmark" / "v3_grid"))
        spec = importlib.util.spec_from_file_location("v3worker", REPO / "realdata_benchmark" / "v3_grid" / "worker.py")
        w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)
        wl = pd.read_csv(a.worklist, sep="\t", dtype=str); inputs = w.Inputs()
    gsea = []
    for u, gu in G.groupby("unit_id", sort=False):
        if a.counts is not None:  # local mode: 58_export_counts.py
            A = ad.read_h5ad(a.counts / f"{u}.h5ad")
        else:
            row = wl[(wl.unit_id == u) & (wl.key == "logcpm|none")].iloc[0]
            v = inputs.get(row)
            A = ad.concat([v["S"], v["T"]], join="inner", index_unique=None)
        A.obs_names = A.obs_names.astype(str)
        if str(A.var_names[0]).startswith("ENSG"):
            if a.symbol_map is None:
                sys.exit(f"{u}: Ensembl gene ids; pass --symbol-map")
            mp = pd.read_csv(a.symbol_map, sep="\t").drop_duplicates("gene_ids").set_index("gene_ids").gene_symbol
            sym = pd.Series(A.var_names.astype(str)).map(mp)
            keep = (sym.notna() & ~sym.duplicated()).to_numpy()
            print(f"{u}: {keep.mean():.3f} of {A.n_vars} Ensembl ids mapped to unique symbols", flush=True)
            A = A[:, keep].copy(); A.var_names = sym[keep].to_numpy()
        gmt = a.gmt_mouse if u.startswith("L1__") or "KPTracer" in u else a.gmt_human
        for cid, g in gu.groupby("contrast_id", sort=False):
            x = A[g.cell.to_numpy()].copy()          # raw counts; normalising per cell is the same per contrast
            if a.downsample:
                tot = np.asarray(x.X.sum(1)).ravel()
                sc.pp.downsample_counts(x, counts_per_cell=int(np.percentile(tot, 10)), random_state=20261110)
            sc.pp.normalize_total(x, target_sum=1e4); sc.pp.log1p(x)
            x.obs["group"] = g.set_index("cell").group.reindex(x.obs_names).to_numpy()
            x = x[:, np.asarray((x.X > 0).sum(0)).ravel() >= 10].copy()
            sc.tl.rank_genes_groups(x, "group", groups=["case"], reference="reference", method="wilcoxon")
            deg = sc.get.rank_genes_groups_df(x, "case")
            deg.to_csv(a.out / "deg" / f"{cid}.tsv.gz", sep="\t", index=False, compression="gzip")
            rnk = deg[["names", "scores"]].drop_duplicates("names").set_index("names")["scores"].sort_values(ascending=False)
            res = gseapy.prerank(rnk=rnk, gene_sets=str(gmt), permutation_num=1000, seed=20261110, min_size=15, max_size=500,
                                 threads=4, outdir=None, verbose=False).res2d
            res = res.rename(columns={"Term": "term", "NES": "nes", "FDR q-val": "fdr", "Lead_genes": "lead_genes"})
            res["contrast_id"] = cid; res["unit_id"] = u
            res["case_n"] = int((x.obs.group == "case").sum()); res["reference_n"] = int((x.obs.group == "reference").sum())
            gsea.append(res[["contrast_id", "unit_id", "case_n", "reference_n", "term", "nes", "fdr", "lead_genes"]])
            print(f"{cid}: {res.case_n.iloc[0]} vs {res.reference_n.iloc[0]}, {x.n_vars} genes", flush=True)
    pd.concat(gsea).to_csv(a.out / "gsea_hallmark.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
