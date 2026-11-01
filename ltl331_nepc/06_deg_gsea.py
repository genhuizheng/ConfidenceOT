"""Rejected vs retained cells within each of the authors' cell types: cell-level Wilcoxon DEG and hallmark prerank GSEA
(local, after the TACC download).

    python ltl331_nepc/06_deg_gsea.py --run-root RUN_ROOT --converted CONVERTED --gmt h.all.v2025.1.Hs.symbols.gmt \
        --out OUT [--labels ranknm256_noscale_ds_cos logcpm] [--method M4-E] [--min-cells 20]

Design, fixed before any DEG was looked at (2026-10-31):
  unit       label x pair x side x coarse_celltype, M4-E gate; every unit with >= MIN_CELLS rejected and >= MIN_CELLS
             retained cells is analysed, none is chosen;
  contrast   rejected vs retained (positive score / log fold change = higher in rejected);
  expression the original counts of the gated cells (inputs/rawqc), normalize_total 1e4, log1p; genes detected in
             >= 10 cells of the unit. --expression ds (added 2026-10-31 in the validity audit, a sensitivity analysis,
             not a replacement) reads the same cells' equalised counts (inputs/ds, 27_, 2,817 UMIs) instead, so that
             depth cannot differ between the two groups; the primary analysis stays rawqc;
  test       scanpy rank_genes_groups Wilcoxon. One tumour per time point and no replicate, so cells are the units and
             the p-values are pseudo-replicated: report effects (scores, log fold changes, NES), not significance;
  GSEA       gseapy prerank on the Wilcoxon score, hallmark only, 1,000 permutations, seed 20261031, set size 15-500.
Every unit also records the median pre-equalisation depth of its rejected and retained cells and the AUC of that
depth for rejected vs retained, so the depth difference behind each contrast is visible.
Writes OUT/units.tsv (every unit, analysed or why not), OUT/deg/<label>/<pair>__<side>__<type>.tsv.gz and
OUT/gsea_hallmark.tsv (every unit x hallmark set).
"""

from __future__ import annotations

import argparse
import glob
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
TYPES = ["PRAD", "Basal", "Other", "NE"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--converted", type=Path, required=True)
    p.add_argument("--gmt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--labels", nargs="+", default=["ranknm256_noscale_ds_cos", "logcpm"])
    p.add_argument("--method", default="M4-E")
    p.add_argument("--min-cells", type=int, default=20)
    p.add_argument("--expression", choices=("rawqc", "ds"), default="rawqc")
    a = p.parse_args()
    if a.out.exists():
        sys.exit(f"{a.out} exists; not overwritten")
    if not a.gmt.name.startswith("h.all"):
        sys.exit("hallmark only: pass the h.all GMT")
    a.out.mkdir(parents=True)

    types = []
    for f in sorted((a.converted / "h5ad").glob("*.h5ad")):
        x = ad.read_h5ad(f, backed="r"); types.append(x.obs["coarse_celltype"].astype(str)); x.file.close()
    types = pd.concat(types)
    depth = pd.read_csv(a.run_root / "inputs" / "ds" / "predownsample_depth.csv.gz").set_index("observation_id")["predownsample_total_counts"]
    expr = {}

    def expression(sample: str) -> ad.AnnData:
        if sample not in expr:
            name = f"{sample}.h5ad" if a.expression == "rawqc" else f"LTL331__{sample}.h5ad"
            x = ad.read_h5ad(a.run_root / "inputs" / a.expression / "h5ad" / name)
            sc.pp.normalize_total(x, target_sum=1e4); sc.pp.log1p(x)
            expr[sample] = x
        return expr[sample]

    units, gsea = [], []
    for label in a.labels:
        for f in sorted(glob.glob(str(a.run_root / "runs" / "acc0.99" / "ltl331" / label / "*" / "scope_malignant" / "*" / "cell_confidence.csv"))):
            pair = Path(f).parents[2].name.replace("LTL331__", "")
            c = pd.read_csv(f, usecols=["method", "side", "observation_id", "sample_id", "rejected"])
            c = c[c.method == a.method]
            c["coarse_celltype"] = c.observation_id.map(types)
            for (side, t), u in c.groupby(["side", "coarse_celltype"]):
                rec = {"label": label, "pair": pair, "side": side, "coarse_celltype": t, "sample_id": u.sample_id.iat[0],
                       "n": len(u), "rejected_n": int(u.rejected.sum()), "retained_n": int((~u.rejected).sum()),
                       "expression": a.expression}
                dv = u.observation_id.map(depth).to_numpy(float); rj = u.rejected.to_numpy(bool)
                rec["median_depth_rejected"] = float(np.median(dv[rj])) if rj.any() else np.nan
                rec["median_depth_retained"] = float(np.median(dv[~rj])) if (~rj).any() else np.nan
                rec["auc_depth_rejected"] = float(roc_auc_score(rj, dv)) if 0 < rj.mean() < 1 else np.nan
                if min(rec["rejected_n"], rec["retained_n"]) < a.min_cells:
                    units.append({**rec, "status": f"skipped: < {a.min_cells} cells in a group"}); continue
                x = expression(rec["sample_id"])[u.observation_id.to_numpy()].copy()
                x.obs["group"] = np.where(u.rejected.to_numpy(bool), "rejected", "retained")
                x = x[:, np.asarray((x.X > 0).sum(0)).ravel() >= 10].copy()
                sc.tl.rank_genes_groups(x, "group", groups=["rejected"], reference="retained", method="wilcoxon", pts=True)
                deg = sc.get.rank_genes_groups_df(x, "rejected")
                name = f"{pair}__{side}__{t}"
                (a.out / "deg" / label).mkdir(parents=True, exist_ok=True)
                deg.to_csv(a.out / "deg" / label / f"{name}.tsv.gz", sep="\t", index=False, compression="gzip")
                rnk = deg[["names", "scores"]].drop_duplicates("names").set_index("names")["scores"].sort_values(ascending=False)
                res = gseapy.prerank(rnk=rnk, gene_sets=str(a.gmt), permutation_num=1000, seed=20261031, min_size=15,
                                     max_size=500, threads=4, outdir=None, verbose=False).res2d
                res = res.rename(columns={"Term": "term", "NES": "nes", "ES": "es", "FDR q-val": "fdr", "NOM p-val": "nominal_p",
                                          "Lead_genes": "lead_genes"})
                for k, v in rec.items():
                    res[k] = v
                gsea.append(res[list(rec) + ["term", "es", "nes", "nominal_p", "fdr", "lead_genes"]])
                units.append({**rec, "genes_tested": x.n_vars, "status": "analysed"})
                print(f"{label} {name}: {rec['rejected_n']} rejected / {rec['retained_n']} retained, {x.n_vars} genes", flush=True)
    pd.DataFrame(units).to_csv(a.out / "units.tsv", sep="\t", index=False)
    pd.concat(gsea).to_csv(a.out / "gsea_hallmark.tsv", sep="\t", index=False)
    print(f"{sum(u['status'] == 'analysed' for u in units)} units analysed of {len(units)}")


if __name__ == "__main__":
    main()
