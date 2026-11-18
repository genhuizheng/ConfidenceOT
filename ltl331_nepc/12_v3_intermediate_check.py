"""The paper's transient mesenchymal (intermediate EMT-like, cluster 17) readout of STATE_BIOLOGY_REPORT.md section 2 / 4,
recomputed on the frozen-calibration-v3 grid (296 configurations, rawqc counts, no DS). Same state map and the same
definitions: M4-E; coupling enrichment = mass x total / (row mass x column mass) in the retained block; retention of a
state against the rest of its side. Named rows: Splatter L5 best and logCPM + PCA; every configuration is summarised
(median, IQR, share of configurations above 1 / above the rest).

    python ltl331_nepc/12_v3_intermediate_check.py --v3 results/v3_grid_20261104/ltl331 \
        --rawqc results/ltl331_nepc/tacc_20261031/ltl331_confidenceot/inputs/rawqc/h5ad --out results/ltl331_nepc/v3_intermediate_check_20261117
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
NAMED = {"L5 best": "ranknm256-midrank|none|scale|cos|pca50", "logCPM + PCA": "logcpm|none|noscale|sqeuc|pca30"}
INT = "intermediate EMT-like"


def pair_name(u):
    w = [x.split("_")[0][1:] for x in u.split("__")[1:]]
    return f"{int(w[0])}->{int(w[1])}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--v3", type=Path, required=True)
    p.add_argument("--rawqc", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        sys.exit(f"{a.out} exists; not overwritten")
    a.out.mkdir(parents=True)
    import anndata as ad
    # coupling enrichment
    C = pd.read_csv(a.v3 / "COUPLING_CLUSTERS_v3tacc.tsv.gz", sep="\t")
    C = C[(C.method == "M4-E") & (C.block == "retained")].copy()
    C["src"] = C.source_cluster.astype(int).map(STATE); C["tgt"] = C.target_cluster.astype(int).map(STATE)
    S = C.groupby(["unit_id", "config_id", "src", "tgt"]).mass.sum().reset_index()
    tot = S.groupby(["unit_id", "config_id"]).mass.transform("sum")
    row = S.groupby(["unit_id", "config_id", "src"]).mass.transform("sum"); col = S.groupby(["unit_id", "config_id", "tgt"]).mass.transform("sum")
    S["enrichment"] = S.mass * tot / (row * col); S["p_target_given_source"] = S.mass / row; S["pair"] = S.unit_id.map(pair_name)
    S.to_csv(a.out / "state_coupling_all_configs.tsv.gz", sep="\t", index=False, compression="gzip")
    sel = S[(S.src == INT) | (S.tgt == INT)]
    summ = sel.groupby(["pair", "src", "tgt"]).agg(configs=("enrichment", "size"), median=("enrichment", "median"),
                                                   q25=("enrichment", lambda v: v.quantile(.25)), q75=("enrichment", lambda v: v.quantile(.75)),
                                                   share_above_1=("enrichment", lambda v: float((v > 1).mean())),
                                                   p_median=("p_target_given_source", "median")).reset_index()
    for nm, cid in NAMED.items():
        x = sel[sel.config_id == cid].set_index(["pair", "src", "tgt"]).enrichment
        summ[nm] = summ.set_index(["pair", "src", "tgt"]).index.map(x.to_dict().get)
    summ.to_csv(a.out / "intermediate_coupling.tsv", sep="\t", index=False)
    # retention of the intermediate state against the rest of its side
    obs = pd.concat([ad.read_h5ad(f, backed="r").obs[["seurat_clusters"]].assign(sample=Path(f).stem)
                     for f in sorted(glob.glob(str(a.rawqc / "*.h5ad")))])
    obs["state"] = obs.seurat_clusters.astype(int).map(STATE)
    key = pd.Series(obs.state.to_numpy(), index=obs["sample"].astype(str) + "|" + obs.index.astype(str))
    R = []
    for f in sorted(glob.glob(str(a.v3 / "CELL_GATES_v3tacc" / "*.tsv.gz"))):
        u = Path(f).name.replace(".tsv.gz", "")
        g = pd.read_csv(f, sep="\t", usecols=["config_id", "method", "side", "observation_id", "sample_id", "retained"])
        g = g[g.method == "M4-E"]
        g["state"] = (g.sample_id.astype(str) + "|" + g.observation_id.astype(str)).map(key)
        for (cid, side), h in g.groupby(["config_id", "side"]):
            m = h.state == INT
            if m.sum() == 0:
                continue
            R.append({"pair": pair_name(u), "config_id": cid, "side": side, "intermediate_cells": int(m.sum()),
                      "unmapped": int(h.state.isna().sum()), "retained_intermediate": float(h.retained[m].mean()),
                      "retained_rest": float(h.retained[~m].mean())})
    R = pd.DataFrame(R); R["difference"] = R.retained_intermediate - R.retained_rest
    R.to_csv(a.out / "intermediate_retention_all_configs.tsv", sep="\t", index=False)
    rs = R.groupby(["pair", "side"]).agg(cells=("intermediate_cells", "first"), configs=("difference", "size"), median_diff=("difference", "median"),
                                         median_retained=("retained_intermediate", "median"),
                                         share_above_rest=("difference", lambda v: float((v > 0).mean()))).reset_index()
    for nm, cid in NAMED.items():
        x = R[R.config_id == cid].set_index(["pair", "side"])
        rs[f"{nm} diff"] = rs.set_index(["pair", "side"]).index.map(x.difference.to_dict().get)
        rs[f"{nm} retained"] = rs.set_index(["pair", "side"]).index.map(x.retained_intermediate.to_dict().get)
    rs.to_csv(a.out / "intermediate_retention.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(summ.round(3).to_string(index=False)); print(rs.round(3).to_string(index=False))
    print("unmapped gate cells:", int(R.unmapped.max()) if len(R) else None)


if __name__ == "__main__":
    main()
