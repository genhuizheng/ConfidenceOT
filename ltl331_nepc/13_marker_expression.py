"""Marker expression behind the paper's mesenchymal-like (cluster 17) and neural-crest statements: per cluster / state and
per time point, the share of cells detecting each gene and the mean log1p(CP10k). Genes: the authors' Table S2 panels
(Fig. 2D EMT, luminal, PRAD, NE; Fig. 3 cluster-17 markers COL3A1, CAV1) and the developmental regulators named in the
text (MSX1, PAX7) with other neural-crest genes as context. Descriptive only; no OT.

    python ltl331_nepc/13_marker_expression.py --rawqc results/ltl331_nepc/tacc_20261031/ltl331_confidenceot/inputs/rawqc/h5ad \
        --out results/ltl331_nepc/marker_expression_20261117
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STATE = {**{c: "AR-high PRAD" for c in (11, 15)}, **{c: "AR-low PRAD" for c in (0, 2, 4, 5, 8, 12, 16)}, 6: "pre-EMT",
         17: "intermediate EMT-like", 10: "NEPC ASCL1+", 13: "early NEPC ASCL1-", **{c: "terminal NEPC ASCL1-" for c in (1, 7, 9)},
         **{c: "NEPC other" for c in (3, 14)}}
GENES = {
    "EMT (Fig 2D)": ["VIM", "CDH2", "FN1", "ZEB1", "ZEB2", "TWIST1", "SNAI1", "SNAI2", "CAV1", "COL1A1", "COL3A1", "CDH11", "DCN"],
    "luminal / epithelial": ["EPCAM", "KRT8", "KRT18", "CDH1"],
    "PRAD / AR": ["AR", "KLK3", "NKX3-1", "FOXA1", "TMPRSS2"],
    "NE": ["ASCL1", "CHGA", "SYP", "INSM1", "NEUROD1", "POU3F2", "SOX2"],
    "neural crest / developmental (text)": ["MSX1", "MSX2", "PAX7", "PAX3", "SOX9", "SOX10", "FOXD3", "TFAP2A", "TFAP2B", "ZIC1", "ZIC2"],
}
ORDER = ["AR-high PRAD", "AR-low PRAD", "pre-EMT", "intermediate EMT-like", "NEPC ASCL1+", "early NEPC ASCL1-", "terminal NEPC ASCL1-", "NEPC other"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rawqc", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        sys.exit(f"{a.out} exists; not overwritten")
    a.out.mkdir(parents=True)
    import anndata as ad
    want = [g for v in GENES.values() for g in v]
    rows = []
    for f in sorted(glob.glob(str(a.rawqc / "*.h5ad"))):
        A = ad.read_h5ad(f)
        X = A.X.tocsr() if hasattr(A.X, "tocsr") else A.X
        tot = np.asarray(X.sum(1)).ravel()
        have = [g for g in want if g in A.var_names]
        Y = A[:, have].X; Y = Y.toarray() if hasattr(Y, "toarray") else np.asarray(Y)
        L = np.log1p(Y / tot[:, None] * 1e4)
        d = pd.DataFrame(L, columns=have); d["detected_any"] = 0
        d["cluster"] = A.obs.seurat_clusters.astype(int).to_numpy(); d["week"] = Path(f).stem.split("_")[0]
        for g in have:
            d[f"det_{g}"] = Y[:, have.index(g)] > 0
        rows.append(d)
        print(Path(f).stem, A.n_obs, "missing genes:", sorted(set(want) - set(have)), flush=True)
    D = pd.concat(rows, ignore_index=True); D["state"] = D.cluster.map(STATE)
    have = [g for g in want if g in D.columns]
    out = []
    for key, grp in (("state", D.groupby("state")), ("cluster", D.groupby("cluster")), ("week x state", D.groupby(["week", "state"]))):
        for k, h in grp:
            for g in have:
                out.append({"by": key, "group": k if not isinstance(k, tuple) else " | ".join(map(str, k)), "cells": len(h), "gene": g,
                            "panel": next(n for n, v in GENES.items() if g in v),
                            "detected_share": float(h[f"det_{g}"].mean()), "mean_log1p_cp10k": float(h[g].mean())})
    T = pd.DataFrame(out); T.to_csv(a.out / "marker_expression.tsv", sep="\t", index=False)
    s = T[T.by == "state"].pivot_table(index=["panel", "gene"], columns="group", values="detected_share").reindex(columns=ORDER)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    print((s * 100).round(0).to_string())
    w = T[(T.by == "week x state") & T.group.str.endswith("intermediate EMT-like")]
    print(w.pivot_table(index="gene", columns="group", values="detected_share").mul(100).round(0).to_string())


if __name__ == "__main__":
    main()
