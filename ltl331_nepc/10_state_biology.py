"""LTL331 state-level readout of existing gates and couplings, for every fitted configuration of each branch (Non-Harmony:
the 64 configurations of the joint benchmark; author-provided Harmony: author_harmony|<key>|cos / sqeuc), no refit.

    python ltl331_nepc/10_state_biology.py --converted CONVERTED --run-root RUN_ROOT --out OUT \
        --gates nonharmony=PDX_CELL_GATES.tsv.gz [authorharmony=PDX_CELL_GATES_<...>.tsv.gz] \
        --coupling nonharmony=STATE_COUPLING.tsv.gz [...] --metrics nonharmony=PDX_GATE_COUPLING_METRICS.tsv [...]

States. Sar et al. (bioRxiv 10.1101/2025.04.25.650369; GSE297328) cluster the same 51,726 cells into 18 clusters
(seurat_clusters of the authors' annotated object, carried in our converted obs). Their text assigns: AR-high PRAD
11, 15; AR-low PRAD 0, 2, 4, 5, 6, 8, 12, 16 (cluster 6: RFX2 / RFX3 ciliogenesis regulons, "pre- or early EMT");
intermediate AR-low / NE-low EMT stem-like 17; NEPC 1, 3, 7, 9, 10, 13, 14, of which 10 early ASCL1+, 13 early ASCL1-,
7 the terminal ASCL1- state. STATE below follows that text; it is provisional until the authors' own annotation table
is supplied, and every cluster is also reported on its own. coarse_celltype (the multi-cohort Harmony object's
PRAD / NE / Basal / Other) is not used as a state: its Basal / Other are not the paper's intermediate (the paper finds
KRT5 / KRT14 absent).

Writes OUT/ (refuses if it exists):
  ANNOTATION_AUDIT.tsv      cells per time point and cluster (fraction of the time point), with state and coarse type
  PAPER_CLAIMS_CHECK.tsv    each composition statement of the paper's text against these cells
  STATE_RETENTION.tsv       branch, config, method, pair, side, level (state | cluster), group: cells, retained,
                            fraction with Wilson 95% interval, the same fraction standardised to the side's
                            pre-equalisation depth quintiles and to its nFeature_RNA quintiles (direct standardisation;
                            quintiles without cells of the group are left out and their weight renormalised), group
                            minus the rest of the side, and the fit's calibration flag (diagnostic only)
  STATE_COUPLING.tsv        branch, config, method, pair, block (all | retained), level, source group x target group:
                            mass, P(target group | source group), the target group's share of the block's mass,
                            enrichment = mass x total / (row mass x column mass) (1 = what the two marginals alone give)
  ROBUSTNESS.tsv            per branch, method, pair, side and group, over every fitted configuration: configurations,
                            median / quartiles / min / max of the retained fraction and of the depth-standardised one,
                            and the fraction of configurations in which the group is retained more than the rest
  COUPLING_ROBUSTNESS.tsv   per branch, method, pair, block and state pair: median / quartiles of the enrichment over
                            configurations
Nothing is filtered by calibration, convergence or depth; those stay columns. No ancestry or fate is read from a gate
or a coupling: a coupling links expression states of two time points, not cells to their descendants.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

STATE = {**{c: "AR-high PRAD" for c in (11, 15)}, **{c: "AR-low PRAD" for c in (0, 2, 4, 5, 8, 12, 16)}, 6: "pre-EMT (AR-low PRAD)",
         17: "intermediate EMT-like", 10: "early NEPC ASCL1+", 13: "early NEPC ASCL1-", 7: "terminal NEPC ASCL1-",
         **{c: "NEPC other" for c in (1, 3, 9, 14)}}
ORDER = ["AR-high PRAD", "AR-low PRAD", "pre-EMT (AR-low PRAD)", "intermediate EMT-like", "early NEPC ASCL1+",
         "early NEPC ASCL1-", "NEPC other", "terminal NEPC ASCL1-"]
NEPC = (1, 3, 7, 9, 10, 13, 14)
PRAD = (0, 2, 4, 5, 6, 8, 11, 12, 15, 16)


def wilson(k, n):
    z = 1.959964
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def pairs_of(spec: list[str]) -> dict[str, Path]:
    return {s.split("=", 1)[0]: Path(s.split("=", 1)[1]) for s in spec}


def standardised(g: pd.DataFrame, keys: list[str], group: str, stratum: str) -> pd.Series:
    """Direct standardisation of the retained fraction of each group to the side's stratum distribution."""
    w = g.groupby(keys + [stratum], observed=True).size(); w = w / w.groupby(level=keys, observed=True).transform("sum")
    r = g.groupby(keys + [group, stratum], observed=True).retained.mean().rename("r").dropna().reset_index()
    r = r.merge(w.rename("w").reset_index(), on=keys + [stratum])
    r["rw"] = r.r * r.w
    s = r.groupby(keys + [group], observed=True)[["rw", "w"]].sum()
    return s.rw / s.w


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--converted", type=Path, required=True)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gates", nargs="+", required=True)
    p.add_argument("--coupling", nargs="*", default=[])
    p.add_argument("--metrics", nargs="*", default=[])
    a = p.parse_args()
    if a.out.exists():
        sys.exit(f"{a.out} exists; not overwritten")
    a.out.mkdir(parents=True)
    obs = pd.concat([ad.read_h5ad(f, backed="r").obs[["seurat_clusters", "coarse_celltype", "timepoint", "nFeature_RNA", "nCount_RNA"]]
                     for f in sorted((a.converted / "h5ad").glob("*.h5ad"))])
    obs["cluster"] = obs.seurat_clusters.astype(int); obs["state"] = obs.cluster.map(STATE)
    obs["coarse_celltype"] = obs.coarse_celltype.astype(str); obs["timepoint"] = obs.timepoint.astype(str)
    depth = pd.read_csv(a.run_root / "inputs" / "ds" / "predownsample_depth.csv.gz").set_index("observation_id")["predownsample_total_counts"]

    # 1. annotation audit
    tp = ["preCx", "wk2", "wk4", "wk8", "wk12", "wk16", "wk20R", "wk22R"]
    au = obs.groupby(["timepoint", "cluster"]).size().rename("cells").reset_index()
    au["fraction_of_timepoint"] = au.cells / au.groupby("timepoint").cells.transform("sum")
    au["state"] = au.cluster.map(STATE)
    au = au.merge(obs.groupby("cluster").coarse_celltype.agg(lambda s: "; ".join(f"{k} {v}" for k, v in s.value_counts().items())).rename("coarse_celltype_counts"), on="cluster")
    au["timepoint"] = pd.Categorical(au.timepoint, tp); au.sort_values(["timepoint", "cluster"]).to_csv(a.out / "ANNOTATION_AUDIT.tsv", sep="\t", index=False)
    frac = au.pivot(index="cluster", columns="timepoint", values="fraction_of_timepoint").reindex(index=range(18), columns=tp).fillna(0)
    dom = frac.idxmax(); domf = frac.max()
    claims = [("cells", "51,726", len(obs)), ("clusters", "18", obs.cluster.nunique()),
              ("preCx cluster 11 fraction", "63%", f"{frac.loc[11, 'preCx']:.1%}"), ("preCx cluster 15 fraction", "15%", f"{frac.loc[15, 'preCx']:.1%}"),
              ("PRAD clusters present at preCx (>= 1 cell)", "8 of 10", f"{int((frac.loc[list(PRAD), 'preCx'] > 0).sum())} of 10"),
              ("NEPC-cluster cells at preCx", "absent", int(obs[(obs.timepoint == 'preCx') & obs.cluster.isin(NEPC)].shape[0])),
              ("AR-high (11, 15) cells at wk2", "undetectable", int(obs[(obs.timepoint == 'wk2') & obs.cluster.isin([11, 15])].shape[0]))]
    for t, c in zip(tp, (11, 2, 2, 0, 0, 4, 10, 1)):
        claims.append((f"dominant cluster at {t} (>= 50%)", str(c), f"{dom[t]} ({domf[t]:.0%})"))
    claims.append(("cluster 17 by time point (emerges 8-12 wk)", "8-12 wk", "; ".join(f"{t} {frac.loc[17, t]:.2%}" for t in tp)))
    claims.append(("cluster 10 by time point (appears 8-12 wk)", "8-12 wk", "; ".join(f"{t} {frac.loc[10, t]:.2%}" for t in tp)))
    claims.append(("NEPC clusters by time point (mainly from 20 wk)", "20 wk", "; ".join(f"{t} {frac.loc[list(NEPC), t].sum():.1%}" for t in tp)))
    pd.DataFrame(claims, columns=["statement", "paper_text", "these_cells"]).to_csv(a.out / "PAPER_CLAIMS_CHECK.tsv", sep="\t", index=False)

    # 2. retention per state / cluster
    valid = {}
    for b, f in pairs_of(a.metrics).items():
        m = pd.read_csv(f, sep="\t")
        valid[b] = m.assign(branch=b)[["branch", "config_id", "pair", "method", "m4e_inference_valid", "m4r_calibration_valid", "outer_converged"]]
    valid = pd.concat(valid.values()) if valid else None
    ret, rob = [], []
    for b, f in pairs_of(a.gates).items():
        g = pd.read_csv(f, sep="\t", usecols=["config_id", "pair_id", "method", "side", "observation_id", "rejected"],
                        dtype={"config_id": "category", "pair_id": "category", "method": "category", "side": "category"})
        g["pair"] = g.pair_id.astype(str).str.replace("LTL331__", "", regex=False).astype("category")
        g["retained"] = ~g.rejected.astype(bool)
        cells = obs.reindex(g.observation_id)
        g["cluster"] = cells.cluster.to_numpy(); g["state"] = cells.state.to_numpy()
        g["depth"] = g.observation_id.map(depth).to_numpy(); g["nfeature"] = cells.nFeature_RNA.to_numpy()
        side_cells = g.drop_duplicates(["pair", "side", "observation_id"])[["pair", "side", "observation_id", "depth", "nfeature"]]
        for col in ("depth", "nfeature"):
            side_cells[f"q_{col}"] = side_cells.groupby(["pair", "side"], observed=True)[col].transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False))
        g = g.merge(side_cells[["pair", "side", "observation_id", "q_depth", "q_nfeature"]], on=["pair", "side", "observation_id"])
        keys = ["config_id", "method", "pair", "side"]
        for level in ("state", "cluster"):
            t = g.groupby(keys + [level], observed=True).retained.agg(cells="size", retained_n="sum").reset_index()
            t["retained_fraction"] = t.retained_n / t.cells
            t["lo"], t["hi"] = wilson(t.retained_n.to_numpy(float), t.cells.to_numpy(float))
            tot = g.groupby(keys, observed=True).retained.agg(side_cells="size", side_retained="sum").reset_index()
            t = t.merge(tot, on=keys)
            t["rest_fraction"] = (t.side_retained - t.retained_n) / (t.side_cells - t.cells).replace(0, np.nan)
            t["group_minus_rest"] = t.retained_fraction - t.rest_fraction
            for strat in ("q_depth", "q_nfeature"):
                s = standardised(g, keys, level, strat).rename(f"retained_fraction_std_{strat[2:]}").reset_index()
                t = t.merge(s, on=keys + [level], how="left")
            t = t.rename(columns={level: "group"}).assign(level=level, branch=b)
            ret.append(t)
        print(b, "retention done", flush=True)
    ret = pd.concat(ret, ignore_index=True)
    if valid is not None:
        ret = ret.merge(valid, on=["branch", "config_id", "pair", "method"], how="left")
    ret["group"] = ret.group.astype(str)
    ret.to_csv(a.out / "STATE_RETENTION.tsv", sep="\t", index=False)
    rk = ["branch", "method", "pair", "side", "level", "group"]
    q = ret.groupby(rk, observed=True).agg(configs=("config_id", "nunique"), cells=("cells", "median"),
                            median=("retained_fraction", "median"), q25=("retained_fraction", lambda x: x.quantile(.25)),
                            q75=("retained_fraction", lambda x: x.quantile(.75)), min=("retained_fraction", "min"), max=("retained_fraction", "max"),
                            median_std_depth=("retained_fraction_std_depth", "median"), median_std_nfeature=("retained_fraction_std_nfeature", "median"),
                            fraction_configs_group_above_rest=("group_minus_rest", lambda x: float((x > 0).mean()) if x.notna().any() else np.nan),
                            median_group_minus_rest=("group_minus_rest", "median")).reset_index()
    q.to_csv(a.out / "ROBUSTNESS.tsv", sep="\t", index=False)

    # 3. coupling between states
    cp = []
    for b, f in pairs_of(a.coupling).items():
        c = pd.read_csv(f, sep="\t")
        c = c[c.block.isin(["all", "retained"])].copy()
        c["source_cluster"] = c.source_cluster.astype(int); c["target_cluster"] = c.target_cluster.astype(int)
        c["pair"] = c.pair_id.str.replace("LTL331__", "", regex=False)
        for level in ("state", "cluster"):
            x = c.assign(source=c.source_cluster.map(STATE) if level == "state" else c.source_cluster.astype(str),
                         target=c.target_cluster.map(STATE) if level == "state" else c.target_cluster.astype(str))
            k = ["config_id", "pair", "method", "block"]
            x = x.groupby(k + ["source", "target"]).mass.sum().reset_index()
            x["row"] = x.groupby(k + ["source"]).mass.transform("sum"); x["col"] = x.groupby(k + ["target"]).mass.transform("sum")
            x["total"] = x.groupby(k).mass.transform("sum")
            x["p_target_given_source"] = x.mass / x.row.replace(0, np.nan)
            x["target_share"] = x.col / x.total.replace(0, np.nan)
            x["enrichment"] = x.mass * x.total / (x.row * x.col).replace(0, np.nan)
            cp.append(x.assign(level=level, branch=b))
    if cp:
        cp = pd.concat(cp, ignore_index=True); cp.to_csv(a.out / "STATE_COUPLING.tsv", sep="\t", index=False)
        cr = cp[cp.level == "state"].groupby(["branch", "method", "pair", "block", "source", "target"]).agg(
            configs=("config_id", "nunique"), median_enrichment=("enrichment", "median"),
            q25=("enrichment", lambda x: x.quantile(.25)), q75=("enrichment", lambda x: x.quantile(.75)),
            median_p_target_given_source=("p_target_given_source", "median"), median_target_share=("target_share", "median")).reset_index()
        cr.to_csv(a.out / "COUPLING_ROBUSTNESS.tsv", sep="\t", index=False)
    print(pd.read_csv(a.out / "PAPER_CLAIMS_CHECK.tsv", sep="\t").to_string(index=False))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
