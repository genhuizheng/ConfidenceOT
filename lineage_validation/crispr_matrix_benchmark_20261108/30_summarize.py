"""Summary of the matrix-based and tree-based CRISPR scores over the 296 configurations (MATRIX_METRICS.md section 4).

    python lineage_validation/crispr_matrix_benchmark_20261108/30_summarize.py --scores SCORES --l1 L1_DIR \
        --splatter-ranking results/v3_grid_20261104/figures/splatter/splatter_ranking.tsv --out OUT

SCORES/<unit>/ are the outputs of 10_score_unit.py; L1_DIR holds 20_l1_family.py's tables.
Per level, dataset (KP, Quinn) and method: unit value -> lineage / family -> mouse mean -> dataset mean (the primary
number), with the coverage beside it. Mouse = KP mouse; a Quinn experiment counts as one mouse.
Levels:
  L1_family   KP d, mean of mouse means (20_l1_family.py)
  L1_clone    Quinn own-clone mass gain
  L2_intbc    matrix L2, intBC-allele partitions: BA - chance (and BA - size rule beside)
  L2_upgma    matrix L2, UPGMA partition
  L3_matrix   median dS_norm per lineage (L2/L3 cases; KP L1 same-family pairs as a separate stratum)
  L2_tree     subclone BA - chance (tree-dependent)
  L3_tree     median dD_norm per family (tree-dependent)
Best set: configurations whose paired mouse-bootstrap 95% interval (10,000 resamples, seed 20261108) of the difference
from the top configuration contains 0; leave-one-mouse-out membership; KP also without 3724. Every configuration is
listed with its Splatter v3 rank. Writes OUT/mouse_values.tsv.gz, OUT/levels.tsv, OUT/best_sets.tsv, OUT/lomo.tsv,
OUT/depth.tsv, OUT/agreement.tsv. No composite score.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SEED, B = 20261108, 10000


def unit_meta(uid: str) -> tuple[str, str, str]:
    """(dataset, mouse, stratum) from a unit id."""
    if uid.startswith("L23__Quinn2021_"):
        return "Quinn", uid.replace("L23__Quinn2021_", "").split("__")[0], "case"
    if uid.startswith("L23__KPTracer_"):
        return "KP", uid.replace("L23__KPTracer_", "").split("__")[0], "case"
    return "KP", uid.split("__")[1], "pair"


def read(scores: Path, name: str) -> pd.DataFrame:
    fr = [pd.read_csv(f, sep="\t") for f in sorted(scores.glob(f"*/{name}.tsv.gz"))]
    if not fr:
        return pd.DataFrame()
    d = pd.concat(fr, ignore_index=True)
    meta = d.unit_id.map(unit_meta)
    d["dataset"] = [m[0] for m in meta]; d["mouse"] = [m[1] for m in meta]; d["stratum"] = [m[2] for m in meta]
    return d


def mouse_values(scores: Path, l1: Path) -> pd.DataFrame:
    """One row per level x dataset x stratum x method x configuration x mouse: value and coverage."""
    out = []
    keys = ["dataset", "stratum", "method", "config_id", "mouse"]
    m1 = pd.read_csv(l1 / "l1_metastasis.tsv.gz", sep="\t", dtype={"mouse": str})
    g = m1.groupby(["method", "config_id", "mouse"]).d.mean().reset_index().rename(columns={"d": "value"})
    out.append(g.assign(level="L1_family", dataset="KP", stratum="pair", coverage=1.0))
    c = read(scores, "l1_clone")
    if len(c):
        g = c.groupby(keys).agg(value=("mean_gain", "mean"), coverage=("coverage", "mean")).reset_index()
        out.append(g.assign(level="L1_clone"))
    p = read(scores, "l2_partition")
    if len(p):
        p = p[p.status == "evaluable"]
        for kind, lev in (("intbc", "L2_intbc"), ("upgma", "L2_upgma")):
            q = p[p.kind == kind].copy()
            q["cov"] = q.scored / q.target_in_classes.where(q.target_in_classes > 0)
            lin = q.groupby(keys + ["unit_id", "lineage"]).agg(v=("ba_minus_chance", "mean"), vs=("ba_minus_size", "mean"),
                                                                cov=("cov", "mean")).reset_index()
            g = lin.groupby(keys).agg(value=("v", "mean"), value_vs_size=("vs", "mean"), coverage=("cov", "mean")).reset_index()
            out.append(g.assign(level=lev))
    l3 = read(scores, "l3_lineage")
    if len(l3):
        l3["cov"] = l3.scored / l3.targets
        lin = l3.groupby(keys + ["unit_id", "lineage"]).agg(v=("median_dS_norm", "mean"), dS=("mean_dS", "mean"), cov=("cov", "mean"),
                                                            t=("targets", "sum")).reset_index()
        g = lin.groupby(keys).agg(value=("v", "mean"), mean_dS=("dS", "mean"), coverage=("cov", "mean")).reset_index()
        out.append(g.assign(level="L3_matrix"))
    t2 = read(scores, "tree_l2")
    if len(t2):
        t2 = t2[t2.evaluation_status == "evaluable"].copy()
        t2["v"] = t2.balanced_accuracy_enrichment - t2.chance
        fam = t2.groupby(keys + ["family"]).agg(v=("v", "mean"), cov=("coverage_scored", "mean")).reset_index()
        out.append(fam.groupby(keys).agg(value=("v", "mean"), coverage=("cov", "mean")).reset_index().assign(level="L2_tree"))
    t3 = read(scores, "tree_l3")
    if len(t3):
        out.append(t3.groupby(keys).agg(value=("median_dD_norm", "mean"), coverage=("coverage", "mean")).reset_index().assign(level="L3_tree"))
    return pd.concat(out, ignore_index=True)


def best_set(mv: pd.DataFrame, rng_seed=SEED) -> pd.DataFrame:
    """Per configuration: mean of mouse means, rank, and best-set membership (paired mouse bootstrap)."""
    W = mv.pivot_table(index="config_id", columns="mouse", values="value")
    W = W.dropna(axis=1, how="all")
    mice = W.columns.to_numpy()
    if len(mice) == 0:
        return pd.DataFrame()
    obs = W.mean(1, skipna=True)
    top = obs.idxmax()
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, len(mice), size=(B, len(mice)))
    X = W.to_numpy(float)
    boot = np.stack([np.nanmean(X[:, r], 1) for r in idx], 1)          # configs x B
    diff = boot - boot[W.index.get_loc(top)][None, :]
    lo, hi = np.nanpercentile(diff, 2.5, 1), np.nanpercentile(diff, 97.5, 1)
    return pd.DataFrame({"config_id": W.index, "value": obs.to_numpy(), "rank": obs.rank(ascending=False, method="min").to_numpy(),
                         "diff_lo": lo, "diff_hi": hi, "best_set": (lo <= 0) & (hi >= 0), "mice": W.notna().sum(1).to_numpy(),
                         "top": W.index == top})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--l1", type=Path, required=True)
    p.add_argument("--splatter-ranking", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    mv = mouse_values(a.scores, a.l1)
    mv.to_csv(a.out / "mouse_values.tsv.gz", sep="\t", index=False, compression="gzip")
    spl = pd.read_csv(a.splatter_ranking, sep="\t").set_index("config_id")["rank"]
    levels, best, lomo = [], [], []
    for (lev, ds, stratum, method), g in mv.groupby(["level", "dataset", "stratum", "method"]):
        variants = [("all", g)]
        if ds == "KP":
            variants.append(("without_3724", g[g.mouse != "3724"]))
        for var, h in variants:
            bs = best_set(h)
            if bs.empty:
                continue
            cov = h.groupby("config_id").coverage.mean()
            bs = bs.assign(level=lev, dataset=ds, stratum=stratum, method=method, variant=var,
                           coverage=bs.config_id.map(cov), splatter_rank=bs.config_id.map(spl))
            best.append(bs)
            if var == "all":
                for mouse in h.mouse.unique():
                    lb = best_set(h[h.mouse != mouse])
                    if len(lb):
                        lomo.append(lb[["config_id", "best_set", "rank"]].assign(level=lev, dataset=ds, stratum=stratum, method=method,
                                                                                 left_out=mouse))
    B_ = pd.concat(best, ignore_index=True)
    B_.to_csv(a.out / "best_sets.tsv", sep="\t", index=False)
    if lomo:
        pd.concat(lomo, ignore_index=True).to_csv(a.out / "lomo.tsv", sep="\t", index=False)
    lv = B_.groupby(["level", "dataset", "stratum", "method", "variant"]).agg(
        configs=("config_id", "size"), mice=("mice", "max"), top=("config_id", lambda s: s[B_.loc[s.index, "top"]].iloc[0]),
        top_value=("value", "max"), median_value=("value", "median"), best_set_size=("best_set", "sum"),
        median_coverage=("coverage", "median")).reset_index()
    lv.to_csv(a.out / "levels.tsv", sep="\t", index=False)
    # matrix vs tree: rank agreement across configurations
    agr = []
    for (ds, stratum, method), g in B_[B_.variant == "all"].groupby(["dataset", "stratum", "method"]):
        w = g.pivot_table(index="config_id", columns="level", values="value")
        for x, y in (("L3_matrix", "L3_tree"), ("L2_intbc", "L2_tree"), ("L2_upgma", "L2_tree"), ("L3_matrix", "L2_intbc"), ("L1_clone", "L3_matrix")):
            if x in w and y in w:
                ok = w[[x, y]].dropna()
                if len(ok) > 10:
                    agr.append({"dataset": ds, "stratum": stratum, "method": method, "x": x, "y": y, "configs": len(ok),
                                "spearman": float(spearmanr(ok[x], ok[y])[0])})
    pd.DataFrame(agr).to_csv(a.out / "agreement.tsv", sep="\t", index=False)
    # depth
    l3 = read(a.scores, "l3_lineage"); dep = read(a.scores, "depth")
    rows = []
    if len(l3):
        rows.append(l3.groupby(["dataset", "stratum", "method", "config_id"]).rho_dS_depth.median().rename("median_rho_dS_depth"))
    if len(dep):
        rows.append(dep.groupby(["dataset", "stratum", "method", "config_id"])[["rho_margin_depth_source", "rho_margin_depth_target"]].median())
    if rows:
        pd.concat(rows, axis=1).reset_index().to_csv(a.out / "depth.tsv", sep="\t", index=False)
    print(lv.to_string(index=False))


if __name__ == "__main__":
    main()
