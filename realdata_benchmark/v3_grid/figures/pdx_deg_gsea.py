"""LTL331 (PDX) v3 grid: rejected vs retained cells within each published state, cell-level Wilcoxon DEG and hallmark
prerank GSEA, for the configurations shown cell by cell.

    python realdata_benchmark/v3_grid/figures/pdx_deg_gsea.py --data results/v3_grid_20261104/figures/pdx/data \
        --rawqc results/ltl331_nepc/tacc_20261031/ltl331_confidenceot/inputs/rawqc/h5ad \
        --depth results/ltl331_nepc/tacc_20261031/ltl331_confidenceot/inputs/ds/predownsample_depth.csv.gz \
        --gmt results/reference/msigdb_human/h.all.v2025.1.Hs.symbols.gmt \
        --out results/v3_grid_20261104/pdx_deg_gsea_20261105 [--config splatter_l5_best baseline logcpm]

The design of ltl331_nepc/06_deg_gsea.py (fixed 2026-10-31), with the v3 gates and the 8 published states of Sar et
al. 2026 in place of the coarse cell types:
  unit       configuration x pair x side x state, M4-E gate (gates_<cfg>.tsv.gz of pdx_prepare.py); every unit with
             >= MIN_CELLS rejected and >= MIN_CELLS retained cells is analysed, none is chosen;
  contrast   rejected vs retained (positive score / log fold change / NES = higher in rejected);
  expression the gated cells' original counts (inputs/rawqc, the cells the OT run used), normalize_total 1e4, log1p;
             genes detected in >= 10 cells of the unit;
  test       scanpy rank_genes_groups Wilcoxon. One tumour per time point and no replicate, so cells are the units and
             the p-values are pseudo-replicated: read effects (scores, log fold changes, NES), not significance;
  GSEA       gseapy prerank on the Wilcoxon score, hallmark only, 1,000 permutations, seed 20261031, set size 15-500.
Every unit records the median pre-equalisation depth of its rejected and retained cells and the AUC of depth for
rejected vs retained, so the depth difference behind each contrast is visible.
Writes OUT/units.tsv (every unit, analysed or why not), OUT/deg/<cfg>/<pair>__<side>__<state>.tsv.gz and
OUT/gsea_hallmark.tsv (every analysed unit x hallmark set).
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import anndata as ad
import gseapy
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pdx_prepare import STATES  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--rawqc", type=Path, required=True)
    p.add_argument("--depth", type=Path, required=True)
    p.add_argument("--gmt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", nargs="+", default=["splatter_l5_best", "logcpm"])
    p.add_argument("--min-cells", type=int, default=20)
    a = p.parse_args()
    if a.out.exists():
        sys.exit(f"{a.out} exists; not overwritten")
    if not a.gmt.name.startswith("h.all"):
        sys.exit("hallmark only: pass the h.all GMT")
    a.out.mkdir(parents=True)

    state = pd.read_csv(a.data / "cells.tsv.gz", sep="\t", index_col="cell", usecols=["cell", "state"]).state
    depth = pd.read_csv(a.depth).set_index("observation_id")["predownsample_total_counts"]
    expr = {}

    def expression(sample: str) -> ad.AnnData:
        if sample not in expr:
            x = ad.read_h5ad(a.rawqc / f"{sample}.h5ad")
            sc.pp.normalize_total(x, target_sum=1e4); sc.pp.log1p(x)
            expr[sample] = x
        return expr[sample]

    units, gsea = [], []
    for cfg in a.config:
        g = pd.read_csv(a.data / f"gates_{cfg}.tsv.gz", sep="\t", dtype={"cell": str})
        g["state"] = g.cell.map(state)
        if g.state.isna().any():
            sys.exit(f"{cfg}: {int(g.state.isna().sum())} gated cells without a state")
        for (pair, side), gp in g.groupby(["pair", "side"], sort=False):
            a_, b_ = pair.replace("LTL331__", "").split("__")
            sample = a_ if side == "source" else b_
            for st in STATES:
                u = gp[gp.state == st]
                if u.empty:
                    continue
                rj = ~u.retained.astype(bool).to_numpy()
                rec = {"config": cfg, "pair": pair.replace("LTL331__", ""), "side": side, "state": st, "sample_id": sample,
                       "n": len(u), "rejected_n": int(rj.sum()), "retained_n": int((~rj).sum())}
                dv = u.cell.map(depth).to_numpy(float)
                rec["median_depth_rejected"] = float(np.median(dv[rj])) if rj.any() else np.nan
                rec["median_depth_retained"] = float(np.median(dv[~rj])) if (~rj).any() else np.nan
                rec["auc_depth_rejected"] = float(roc_auc_score(rj, dv)) if 0 < rj.mean() < 1 else np.nan
                if min(rec["rejected_n"], rec["retained_n"]) < a.min_cells:
                    units.append({**rec, "status": f"skipped: < {a.min_cells} cells in a group"}); continue
                x = expression(sample)[u.cell.to_numpy()].copy()
                x.obs["group"] = np.where(rj, "rejected", "retained")
                x = x[:, np.asarray((x.X > 0).sum(0)).ravel() >= 10].copy()
                sc.tl.rank_genes_groups(x, "group", groups=["rejected"], reference="retained", method="wilcoxon", pts=True)
                deg = sc.get.rank_genes_groups_df(x, "rejected")
                name = f"{rec['pair']}__{side}__{st.replace(' ', '_')}"
                (a.out / "deg" / cfg).mkdir(parents=True, exist_ok=True)
                deg.to_csv(a.out / "deg" / cfg / f"{name}.tsv.gz", sep="\t", index=False, compression="gzip")
                rnk = deg[["names", "scores"]].drop_duplicates("names").set_index("names")["scores"].sort_values(ascending=False)
                res = gseapy.prerank(rnk=rnk, gene_sets=str(a.gmt), permutation_num=1000, seed=20261031, min_size=15,
                                     max_size=500, threads=4, outdir=None, verbose=False).res2d
                res = res.rename(columns={"Term": "term", "NES": "nes", "ES": "es", "FDR q-val": "fdr", "NOM p-val": "nominal_p",
                                          "Lead_genes": "lead_genes"})
                for k, v in rec.items():
                    res[k] = v
                gsea.append(res[list(rec) + ["term", "es", "nes", "nominal_p", "fdr", "lead_genes"]])
                units.append({**rec, "genes_tested": x.n_vars, "status": "analysed"})
                print(f"{cfg} {name}: {rec['rejected_n']} rejected / {rec['retained_n']} retained, {x.n_vars} genes", flush=True)
            pd.DataFrame(units).to_csv(a.out / "units.tsv", sep="\t", index=False)  # progress, rewritten each pair
    pd.DataFrame(units).to_csv(a.out / "units.tsv", sep="\t", index=False)
    pd.concat(gsea).to_csv(a.out / "gsea_hallmark.tsv", sep="\t", index=False)
    print(f"{sum(u['status'] == 'analysed' for u in units)} units analysed of {len(units)}")


if __name__ == "__main__":
    main()
