"""LTL331 (PDX) v3 grid: DEG and hallmark GSEA between the retained cells of one state grouped by where the coupling
sends them (source side: dominant fate at the next time point) or where it brings them from (target side: dominant
origin at the previous time point). ConfidenceOT logCPM only (user, 2026-11-05); a balanced OT logCPM arm comes later.

    python realdata_benchmark/v3_grid/figures/pdx_fate_deg_gsea.py --grid results/v3_grid_20261104/ltl331 \
        --data results/v3_grid_20261104/figures/pdx/data \
        --rawqc results/ltl331_nepc/tacc_20261031/ltl331_confidenceot/inputs/rawqc/h5ad \
        --ds results/ltl331_nepc/tacc_20261031/ltl331_confidenceot/inputs/ds/h5ad \
        --depth results/ltl331_nepc/tacc_20261031/ltl331_confidenceot/inputs/ds/predownsample_depth.csv.gz \
        --gmt results/reference/msigdb_human/h.all.v2025.1.Hs.symbols.gmt \
        --out results/v3_grid_20261104/pdx_fate_deg_gsea_20261105 [--config logcpm] [--min-cells 20]

Design (the 06_ design, with groups by coupling instead of by gate):
  cells      the retained x retained block of the M4-E coupling (couplings_2cfg/); rejected cells have no fate and
             take no part;
  fate       source cell: the target state receiving the largest share of its row mass; target cell: the source state
             sending the largest share of its column mass (argmax, no threshold; the share is recorded per cell);
  unit       configuration x pair x side x own state;
  contrast   within a unit, every fate group of >= MIN_CELLS cells against the reference group: the cells whose fate
             is their own state (they stay) if that group has >= MIN_CELLS cells, else the largest group. Positive
             score / log fold change / NES = higher in the fate group;
  expression rawqc (the gated cells' original counts, primary) and ds (the same cells' equalised counts, 27_, 2,817
             UMIs, so depth cannot differ between the groups), each normalize_total 1e4, log1p; genes detected in
             >= 10 cells of the contrast;
  test       scanpy rank_genes_groups Wilcoxon; one tumour per time point, so the p-values are pseudo-replicated:
             read effects (scores, log fold changes, NES), not significance;
  GSEA       gseapy prerank on the Wilcoxon score, hallmark only, 1,000 permutations, seed 20261031, set size 15-500.
Every contrast records the median pre-equalisation depth of both groups and the AUC of depth (fate group vs reference).
Writes OUT/fates_<cfg>.tsv.gz (every retained cell: pair, side, cell, own state, fate, fate share),
OUT/contrasts.tsv (every unit and group, analysed or why not), OUT/deg/<expr>/<cfg>/<contrast>.tsv.gz and
OUT/gsea_hallmark.tsv (expression x contrast x hallmark set).
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
from pdx_prepare import SHOWN, STATES  # noqa: E402


def fates(grid: Path, gates: pd.DataFrame, state: pd.Series, cid: str) -> pd.DataFrame:
    enc, _, scale, cost, pca = cid.split("|")
    sub = f"{enc}__none/{enc}__none__{scale}__{cost}__{pca}/coupling_m4e.npz"
    out = []
    for unit in sorted(gates.pair.unique()):
        z = np.load(grid / "couplings_2cfg" / unit / sub, allow_pickle=False)
        P = z["coupling"].astype(np.float64)
        src, tgt = z["source_ids"].astype(str), z["target_ids"].astype(str)
        gu = gates[gates.pair == unit]
        rs = gu[gu.side == "source"].set_index("cell").retained.astype(bool).reindex(src).to_numpy()
        rt = gu[gu.side == "target"].set_index("cell").retained.astype(bool).reindex(tgt).to_numpy()
        Q = P[np.ix_(rs, rt)]
        for side, own, other, M in (("source", src[rs], tgt[rt], Q), ("target", tgt[rt], src[rs], Q.T)):
            other_state = state.reindex(other).to_numpy()
            by = np.stack([M[:, other_state == y].sum(1) for y in STATES], 1)
            tot = by.sum(1)
            k = by.argmax(1)
            out.append(pd.DataFrame({"pair": unit.replace("LTL331__", ""), "side": side, "cell": own,
                                     "state": state.reindex(own).to_numpy(), "fate": np.asarray(STATES)[k],
                                     "fate_share": np.divide(by[np.arange(len(k)), k], tot, out=np.full(len(k), np.nan),
                                                             where=tot > 0)}))
    return pd.concat(out, ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--rawqc", type=Path, required=True)
    p.add_argument("--ds", type=Path, required=True)
    p.add_argument("--depth", type=Path, required=True)
    p.add_argument("--gmt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", nargs="+", default=["logcpm"])
    p.add_argument("--min-cells", type=int, default=20)
    a = p.parse_args()
    if a.out.exists():
        sys.exit(f"{a.out} exists; not overwritten")
    if not a.gmt.name.startswith("h.all"):
        sys.exit("hallmark only: pass the h.all GMT")
    a.out.mkdir(parents=True)

    state = pd.read_csv(a.data / "cells.tsv.gz", sep="\t", index_col="cell", usecols=["cell", "state"]).state
    depth = pd.read_csv(a.depth).set_index("observation_id")["predownsample_total_counts"]
    roots = {"rawqc": a.rawqc, "ds": a.ds}
    expr = {}

    def expression(kind: str, sample: str) -> ad.AnnData:
        if (kind, sample) not in expr:
            x = ad.read_h5ad(roots[kind] / (f"{sample}.h5ad" if kind == "rawqc" else f"LTL331__{sample}.h5ad"))
            sc.pp.normalize_total(x, target_sum=1e4); sc.pp.log1p(x)
            expr[kind, sample] = x
        return expr[kind, sample]

    rows, gsea = [], []
    for cfg in a.config:
        g = pd.read_csv(a.data / f"gates_{cfg}.tsv.gz", sep="\t", dtype={"cell": str})
        f = fates(a.grid, g, state, SHOWN[cfg])
        f.to_csv(a.out / f"fates_{cfg}.tsv.gz", sep="\t", index=False, compression="gzip")
        for (pair, side, st), u in f.groupby(["pair", "side", "state"], sort=False):
            a_, b_ = pair.split("__")
            sample = a_ if side == "source" else b_
            n = u.fate.value_counts()
            big = n[n >= a.min_cells]
            ref = st if st in big.index else (big.index[0] if len(big) else None)
            for fate, k in n.items():
                rec = {"config": cfg, "pair": pair, "side": side, "state": st, "sample_id": sample, "fate": fate,
                       "reference": ref, "fate_n": int(k), "reference_n": int(n.get(ref, 0)) if ref else 0}
                if ref is None or fate == ref or k < a.min_cells:
                    why = ("reference" if fate == ref else f"skipped: < {a.min_cells} cells" if k < a.min_cells
                           else "skipped: no group reaches the minimum")
                    rows.append({**rec, "status": why}); continue
                cells = u[u.fate.isin([fate, ref])]
                grp = (cells.fate == fate).to_numpy()
                dv = cells.cell.map(depth).to_numpy(float)
                rec.update(median_depth_fate=float(np.median(dv[grp])), median_depth_reference=float(np.median(dv[~grp])),
                           auc_depth_fate=float(roc_auc_score(grp, dv)),
                           median_fate_share=float(cells.fate_share[grp].median()))
                name = f"{pair}__{side}__{st.replace(' ', '_')}__to__{fate.replace(' ', '_')}"
                for kind in roots:
                    x = expression(kind, sample)[cells.cell.to_numpy()].copy()
                    x.obs["group"] = np.where(grp, "fate", "reference")
                    x = x[:, np.asarray((x.X > 0).sum(0)).ravel() >= 10].copy()
                    sc.tl.rank_genes_groups(x, "group", groups=["fate"], reference="reference", method="wilcoxon")
                    deg = sc.get.rank_genes_groups_df(x, "fate")
                    (a.out / "deg" / kind / cfg).mkdir(parents=True, exist_ok=True)
                    deg.to_csv(a.out / "deg" / kind / cfg / f"{name}.tsv.gz", sep="\t", index=False, compression="gzip")
                    rnk = deg[["names", "scores"]].drop_duplicates("names").set_index("names")["scores"]
                    res = gseapy.prerank(rnk=rnk.sort_values(ascending=False), gene_sets=str(a.gmt), permutation_num=1000,
                                         seed=20261031, min_size=15, max_size=500, threads=4, outdir=None,
                                         verbose=False).res2d
                    res = res.rename(columns={"Term": "term", "NES": "nes", "ES": "es", "FDR q-val": "fdr",
                                              "NOM p-val": "nominal_p", "Lead_genes": "lead_genes"})
                    for c, v in rec.items():
                        res[c] = v
                    res["expression"] = kind
                    gsea.append(res[["expression"] + list(rec) + ["term", "es", "nes", "nominal_p", "fdr", "lead_genes"]])
                rows.append({**rec, "status": "analysed"})
                print(f"{cfg} {name}: {k} vs {rec['reference_n']} ({ref}), depth AUC {rec['auc_depth_fate']:.2f}", flush=True)
    pd.DataFrame(rows).to_csv(a.out / "contrasts.tsv", sep="\t", index=False)
    if gsea:
        pd.concat(gsea).to_csv(a.out / "gsea_hallmark.tsv", sep="\t", index=False)
    print(f"{sum(r['status'] == 'analysed' for r in rows)} contrasts analysed")


if __name__ == "__main__":
    main()
