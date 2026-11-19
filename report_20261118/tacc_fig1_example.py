"""Figure 1d source data, run on TACC (login node, read-only on the v3 grid): one valid N = 5,000 Splatter unit.

Unit: L2_depth_dropout, replicate 2, Case 2 population lost (Group 5 removed from the target), main configuration
ranknm256-midrank|none|scale|cos|pca50, M4-E. Reads the v3 worker outputs (cell_confidence.csv.gz, coupling_m4e.npz) and
the unit's input counts; writes small derived tables to --out (a new folder):
  cells.tsv        side, observation_id, group, should_reject, rejected, total_counts, umap1, umap2
  group_mass.tsv   block (all / retained), source_group, target_group, mass
  meta.json        shapes, sums, embedding recipe
The embedding is for display only (log1p CP10k, 2,000 top-variance genes, PCA 30, UMAP); it is not the OT input.

    python -I tacc_fig1_example.py --row ROWDIR --cond CONDDIR --out OUTDIR
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def group_col(obs):
    for c in obs.columns:
        v = obs[c].astype(str)
        if v.str.startswith("Group").mean() > 0.99:
            return c
    raise SystemExit(f"no Group column in {list(obs.columns)}")


def main():
    import anndata as ad
    import scipy.sparse as sp
    p = argparse.ArgumentParser()
    p.add_argument("--row", type=Path, required=True)
    p.add_argument("--cond", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    cc = pd.read_csv(a.row / "cell_confidence.csv.gz")
    cc = cc[cc.method == "M4-E"].copy()
    cc["rejected"] = cc.rejected.astype(str).str.lower().isin(["true", "1", "1.0"])
    z = np.load(a.row / "coupling_m4e.npz")
    P = z["coupling"].astype(np.float64)
    src = cc[cc.side == "source"]; tgt = cc[cc.side == "target"]
    assert P.shape == (len(src), len(tgt)), (P.shape, len(src), len(tgt))
    if "source_ids" in z.files:
        assert (z["source_ids"].astype(str) == src.observation_id.astype(str).to_numpy()).all()
        assert (z["target_ids"].astype(str) == tgt.observation_id.astype(str).to_numpy()).all()
    S = ad.read_h5ad(a.cond / "source.h5ad"); T = ad.read_h5ad(a.cond / "target.h5ad")
    gs, gt = group_col(S.obs), group_col(T.obs)
    grp = pd.concat([pd.Series(S.obs[gs].astype(str).to_numpy(), index="source|" + S.obs_names.astype(str)),
                     pd.Series(T.obs[gt].astype(str).to_numpy(), index="target|" + T.obs_names.astype(str))])
    cc["group"] = grp.reindex(cc.side + "|" + cc.observation_id.astype(str)).to_numpy()
    assert cc.group.notna().all()
    tgt_groups = set(cc.loc[cc.side == "target", "group"])
    src_groups = set(cc.loc[cc.side == "source", "group"])
    cc["should_reject"] = np.where(cc.side == "source", ~cc.group.isin(tgt_groups), ~cc.group.isin(src_groups))
    # display embedding
    X = sp.vstack([sp.csr_matrix(S.X), sp.csr_matrix(T.X)]).tocsr().astype(np.float64)
    tot = np.asarray(X.sum(1)).ravel()
    X = sp.diags(1e4 / np.maximum(tot, 1)) @ X
    X.data = np.log1p(X.data)
    mu = np.asarray(X.mean(0)).ravel(); var = np.asarray(X.multiply(X).mean(0)).ravel() - mu ** 2
    top = np.argsort(var)[::-1][:2000]
    Y = X[:, top].toarray(); Y -= Y.mean(0)
    from sklearn.decomposition import PCA
    Z = PCA(30, random_state=0).fit_transform(Y)
    try:
        import umap
        E = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0).fit_transform(Z); recipe = "UMAP (umap-learn) on PCA 30"
    except Exception as err:  # noqa: BLE001
        E = Z[:, :2]; recipe = f"PCA 1-2 (umap unavailable: {type(err).__name__})"
    order = pd.Index(np.r_[("source|" + S.obs_names.astype(str)).to_numpy(), ("target|" + T.obs_names.astype(str)).to_numpy()])
    key = cc.side + "|" + cc.observation_id.astype(str)
    pos = order.get_indexer(key); assert (pos >= 0).all()
    cc["umap1"], cc["umap2"] = E[pos, 0], E[pos, 1]
    cc["total_counts_display"] = tot[pos]
    cols = ["side", "observation_id", "group", "should_reject", "rejected", "total_counts_display", "umap1", "umap2"]
    cc[cols].to_csv(a.out / "cells.tsv", sep="\t", index=False)
    rows = []
    a_g = src.observation_id.astype(str).map(dict(zip(cc.loc[cc.side == "source", "observation_id"].astype(str), cc.loc[cc.side == "source", "group"]))).to_numpy()
    b_g = tgt.observation_id.astype(str).map(dict(zip(cc.loc[cc.side == "target", "observation_id"].astype(str), cc.loc[cc.side == "target", "group"]))).to_numpy()
    rs, rt = ~src.rejected.to_numpy(bool), ~tgt.rejected.to_numpy(bool)
    for block, ms, mt in (("all", np.ones_like(rs), np.ones_like(rt)), ("retained", rs, rt)):
        for x in sorted(set(a_g)):
            for y in sorted(set(b_g)):
                rows.append({"block": block, "source_group": x, "target_group": y,
                             "mass": float(P[np.ix_((a_g == x) & ms, (b_g == y) & mt)].sum())})
    pd.DataFrame(rows).to_csv(a.out / "group_mass.tsv", sep="\t", index=False)
    meta = {"row": str(a.row), "cond": str(a.cond), "coupling_shape": list(P.shape), "coupling_sum": float(P.sum()),
            "embedding": recipe, "source_cells": int(len(src)), "target_cells": int(len(tgt)),
            "source_rejected": int(src.rejected.sum()), "target_rejected": int(tgt.rejected.sum())}
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
