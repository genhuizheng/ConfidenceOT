"""Biological-structure check of the L−1 corrections (SELECTION_RULES.md): what each transform of the joint PCA does to
the populations, without any OT. Read only; uses the generated conditions of the local root.

    python realdata_benchmark/splatter_local/structure_check.py --root ROOT [--replicates 1 2]

Per pair and correction (none, harmony, center, quantile, mnn; representation ranknm256-production|P10s1|noscale|cos):
  within_knn15         share of a cell's 15 nearest same-side neighbours of its own population (within-side structure)
  shared_cross_1nn     for cells of populations present on both sides: the nearest other-side cell is of the same
                       population (cross-side alignment of what should match)
  unmatched_false_near for cells of a population absent from the other side: the nearest other-side cell is closer
                       than the cell's median distance to its 15 same-side neighbours (a false match being created)
Writes ROOT/STRUCTURE_CHECK.tsv and prints the means by level and correction.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_local_benchmark as lb  # noqa: E402

CORRECTIONS = ["", "harmony", "center", "quantile", "mnn"]


def one(task):
    import anndata as ad
    import rb_common as rb
    from sklearn.neighbors import NearestNeighbors
    man = pd.read_csv(task["manifest"]); i = task["pair_index"]; r = man.iloc[i]
    S, T = ad.read_h5ad(r.source_h5ad), ad.read_h5ad(r.target_h5ad)
    spec = task["spec"]
    tg = lb.depth_target(task["manifest"], spec["ds"]); s = int(float(spec["ds_seed"]))
    S, T = rb.equalised(S, tg, 1000 * s + 1), rb.equalised(T, tg, 1000 * s + 2)
    X0, Y0, _ = rb.build_config(dict(spec), S, T, 20260925 + i)
    truth = pd.read_csv(task["truth"], sep="\t").set_index(["side", "observation_id"])
    gs = truth.reindex(pd.MultiIndex.from_arrays([["source"] * S.n_obs, S.obs_names])).group.to_numpy()
    gt = truth.reindex(pd.MultiIndex.from_arrays([["target"] * T.n_obs, T.obs_names])).group.to_numpy()
    shared = set(gs) & set(gt)
    rows = []
    for corr in CORRECTIONS:
        X, Y = lb.correct(X0, Y0, corr)
        rec = {k: task[k] for k in ("level", "replicate", "case")} | {"correction": corr or "none"}
        w = []
        for V, g in ((X, gs), (Y, gt)):
            idx = NearestNeighbors(n_neighbors=16).fit(V).kneighbors(V, return_distance=False)[:, 1:]
            w.append(float((g[idx] == g[:, None]).mean()))
        rec["within_knn15"] = float(np.mean(w))
        cross_ok, false_near = [], []
        for A, ga, B, gb in ((X, gs, Y, gt), (Y, gt, X, gs)):
            d_cross, j = NearestNeighbors(n_neighbors=1).fit(B).kneighbors(A)
            d_in = NearestNeighbors(n_neighbors=16).fit(A).kneighbors(A)[0][:, 1:]
            sh = np.isin(ga, list(shared))
            if sh.any():
                cross_ok.append(float((gb[j[:, 0]][sh] == ga[sh]).mean()))
            if (~sh).any():
                false_near.append(float((d_cross[~sh, 0] < np.median(d_in[~sh], axis=1)).mean()))
        rec["shared_cross_1nn"] = float(np.mean(cross_ok)) if cross_ok else np.nan
        rec["unmatched_false_near"] = float(np.mean(false_near)) if false_near else np.nan
        rows.append(rec)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--replicates", type=int, nargs="+", default=[1, 2])
    p.add_argument("--jobs", type=int, default=24)
    a = p.parse_args()
    reg = pd.read_csv(lb.REPO / "results" / "realdata_benchmark_20261031" / "PREPROCESSING_CONFIG_REGISTRY.tsv", sep="\t").set_index("config_id")
    spec = reg.loc[lb.BASELINE].to_dict()
    tasks = []
    for rep in a.replicates:
        for lv in lb.ALL_LEVELS:
            cond = a.root / "conditions" / f"rep{rep}" / lv
            man = pd.read_csv(cond / "manifest.csv")
            for i, r in man.iterrows():
                tasks.append({"level": lv, "replicate": rep, "case": r.case, "manifest": str(cond / "manifest.csv"), "pair_index": i, "spec": spec,
                              "truth": str(cond / str(r.pair_id) / "truth" / "population_truth.tsv")})
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        rows = [x for rs in ex.map(one, tasks) for x in rs]
    t = pd.DataFrame(rows); t.to_csv(a.root / "STRUCTURE_CHECK.tsv", sep="\t", index=False)
    t["batch"] = np.where(t.level.isin(["L3_batch", "L4_batch_perturb", "L5_batch_perturb_imbalance"]), "batch L3-L5", "no batch")
    print(t.groupby(["batch", "correction"])[["within_knn15", "shared_cross_1nn", "unmatched_false_near"]].mean().round(3).to_string())
    print(t[t.case.str.startswith("Case3")].groupby(["batch", "correction"])[["unmatched_false_near"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
