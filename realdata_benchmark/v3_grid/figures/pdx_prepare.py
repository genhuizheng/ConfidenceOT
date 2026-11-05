"""LTL331 (PDX) v3 grid: one table of cells, gates, state-level transitions and matched links for the figures.

    python realdata_benchmark/v3_grid/figures/pdx_prepare.py --grid results/v3_grid_20261104/ltl331 \
        --converted results/ltl331_nepc/tacc_20261031/data/ltl331_GSE297328/converted/h5ad \
        --embeddings .../AUTHOR_EMBEDDINGS_LTL331.npz --out results/v3_grid_20261104/figures/pdx/data

Inputs: the v3 collect output (CELL_GATES_v3tacc/<pair>.tsv.gz, every configuration), the M4-E couplings of the two
configurations shown cell by cell (couplings_2cfg/), the converted obs (the authors' seurat_clusters, coarse_celltype)
and the authors' UMAP of their joint object (display only; nothing is fitted on it).
Writes:
  cells.tsv.gz             every converted LTL331 cell: sample, week, cluster, state, coarse type, author UMAP x / y
  gates_<cfg>.tsv.gz       the two shown configurations: pair, side, cell, retained (M4-E)
  state_rejected_all.tsv   every configuration x pair x side x state: cells, rejected cells (M4-E)
  transitions_<cfg>.tsv    pair x source state x target state: coupling mass on the retained x retained block, and
                           the same over all cells
  links_<cfg>.tsv.gz       every retained source cell: its retained target of largest coupling mass and that mass,
                           and the row's retained mass
The state map is the published one of Sar et al. 2026 (memory: ltl331-states-and-author-harmony).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

SHOWN = {"baseline": "ranknm256-production|none|noscale|cos|pca30", "splatter_l5_best": "ranknm256-midrank|none|scale|cos|pca50",
         "logcpm": "logcpm|none|noscale|cos|pca30"}
STATE = {11: "AR-high PRAD", 15: "AR-high PRAD", 0: "AR-low PRAD", 2: "AR-low PRAD", 4: "AR-low PRAD", 5: "AR-low PRAD",
         8: "AR-low PRAD", 12: "AR-low PRAD", 16: "AR-low PRAD", 6: "pre-EMT", 17: "intermediate EMT-like",
         10: "NEPC ASCL1+", 13: "early NEPC ASCL1-", 1: "terminal NEPC ASCL1-", 7: "terminal NEPC ASCL1-",
         9: "terminal NEPC ASCL1-", 3: "NEPC other", 14: "NEPC other"}
STATES = ["AR-high PRAD", "AR-low PRAD", "pre-EMT", "intermediate EMT-like", "NEPC ASCL1+", "early NEPC ASCL1-",
          "terminal NEPC ASCL1-", "NEPC other"]


def pair_samples(unit: str) -> tuple[str, str]:
    m = re.match(r"LTL331__(w\d+_[^_]+_\d+(?:_R)?)__(w\d+_[^_]+_\d+(?:_R)?)$", unit)
    if not m:
        raise SystemExit(f"cannot parse {unit}")
    return m.group(1), m.group(2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid", type=Path, required=True)
    p.add_argument("--converted", type=Path, required=True)
    p.add_argument("--embeddings", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    obs = []
    for f in sorted(a.converted.glob("*.h5ad")):
        o = ad.read_h5ad(f, backed="r").obs[["sample_id", "time_weeks", "seurat_clusters", "coarse_celltype"]].copy()
        o.index = o.index.astype(str); obs.append(o)
    cells = pd.concat(obs)
    cells["cluster"] = cells.seurat_clusters.astype(int)
    cells["state"] = cells.cluster.map(STATE)
    if cells.state.isna().any():
        raise SystemExit(f"clusters without a state: {sorted(cells.cluster[cells.state.isna()].unique())}")
    z = np.load(a.embeddings, allow_pickle=False)
    emb = pd.DataFrame(z["X_umap_harmony_dataset"], index=pd.Index(z["cellid"].astype(str)), columns=["umap_x", "umap_y"])
    cells = cells.join(emb, how="left")
    print(f"{len(cells)} cells, {int(cells.umap_x.isna().sum())} without author UMAP")
    cells.index.name = "cell"
    cells.drop(columns="seurat_clusters").to_csv(a.out / "cells.tsv.gz", sep="\t", compression="gzip")
    state_of = cells.state

    summary, gates = [], {k: [] for k in SHOWN}
    inv = {v: k for k, v in SHOWN.items()}
    for f in sorted((a.grid / "CELL_GATES_v3tacc").glob("*.tsv.gz")):
        unit = f.name[:-len(".tsv.gz")]
        print("gates", unit, flush=True)
        for ch in pd.read_csv(f, sep="\t", usecols=["config_id", "method", "side", "observation_id", "retained"],
                              dtype={"observation_id": str}, chunksize=2_000_000):
            ch = ch[ch.method == "M4-E"]
            ch["state"] = ch.observation_id.map(state_of)
            ch["rejected"] = ~ch.retained.astype(bool)
            s = ch.groupby(["config_id", "side", "state"]).rejected.agg(["size", "sum"]).reset_index()
            s.insert(0, "pair", unit); summary.append(s)
            for cid, key in inv.items():
                g = ch[ch.config_id == cid]
                if len(g):
                    gates[key].append(g.assign(pair=unit)[["pair", "side", "observation_id", "retained"]])
    s = pd.concat(summary).groupby(["pair", "config_id", "side", "state"], as_index=False)[["size", "sum"]].sum()
    s.columns = ["pair", "config_id", "side", "state", "cells", "rejected"]
    s.to_csv(a.out / "state_rejected_all.tsv", sep="\t", index=False)

    for key, cid in SHOWN.items():
        g = pd.concat(gates[key]).rename(columns={"observation_id": "cell"})
        g.to_csv(a.out / f"gates_{key}.tsv.gz", sep="\t", index=False, compression="gzip")
        tr, links = [], []
        enc, _, scale, cost, pca = cid.split("|")
        sub = f"{enc}__none/{enc}__none__{scale}__{cost}__{pca}/coupling_m4e.npz"
        if not all((a.grid / "couplings_2cfg" / u / sub).exists() for u in g.pair.unique()):
            print(f"{key}: couplings not all local, gates only", flush=True)
            continue
        for unit in sorted(g.pair.unique()):
            zc = np.load(a.grid / "couplings_2cfg" / unit / sub, allow_pickle=False)
            P = zc["coupling"].astype(np.float64)
            src, tgt = zc["source_ids"].astype(str), zc["target_ids"].astype(str)
            gu = g[g.pair == unit]
            rs = gu[gu.side == "source"].set_index("cell").retained.astype(bool).reindex(src).to_numpy()
            rt = gu[gu.side == "target"].set_index("cell").retained.astype(bool).reindex(tgt).to_numpy()
            if np.isnan(rs.astype(float)).any() or np.isnan(rt.astype(float)).any():
                raise SystemExit(f"{unit}: coupling ids not in the gates")
            ss, st_ = state_of.reindex(src).to_numpy(), state_of.reindex(tgt).to_numpy()
            for block, R in (("retained", np.outer(rs, rt)), ("all", None)):
                Q = P if R is None else P * R
                for x in STATES:
                    ix = ss == x
                    if not ix.any():
                        continue
                    row = Q[ix]
                    for y in STATES:
                        iy = st_ == y
                        m = float(row[:, iy].sum()) if iy.any() else 0.0
                        tr.append({"pair": unit, "block": block, "source_state": x, "target_state": y, "mass": m})
            Q = P[np.ix_(rs, rt)]
            if Q.size:
                j = Q.argmax(1)
                links.append(pd.DataFrame({"pair": unit, "source": src[rs], "target": tgt[rt][j],
                                           "top_mass": Q[np.arange(len(j)), j], "row_mass": Q.sum(1)}))
            print(key, unit, f"retained {int(rs.sum())}/{len(rs)} source, {int(rt.sum())}/{len(rt)} target", flush=True)
        pd.DataFrame(tr).to_csv(a.out / f"transitions_{key}.tsv", sep="\t", index=False)
        pd.concat(links).to_csv(a.out / f"links_{key}.tsv.gz", sep="\t", index=False, compression="gzip")


if __name__ == "__main__":
    main()
