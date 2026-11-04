"""Shared-evidence statistics v2 (development seeds only): label-free statistics computed after the credible-anchor MNN
translation, each normalised by two references built from the data itself, so a decision rule needs no threshold
fitted to labels:
  shared reference    the same statistic between two random halves of the source (shares everything, no batch)
  disjoint reference  the same statistic between the source's two k-means halves (two different population sets)
  score               (observed - disjoint) / (shared - disjoint), clipped to [0, 1]
The truth (case) is joined afterwards only to look at the separation.

    python realdata_benchmark/splatter_local/shared_evidence_v2.py --root ROOT [--replicates 1 2]

Statistics (A = source, B = target; anchors = cross-side mutual nearest neighbours, k = 20; credible = after the
iterative translation, closer than both cells' own neighbourhood radius):
  share20      credible share of the anchors, radius = 20th same-side neighbour (the mnncred statistic)
  share5pct    the same with radius = ceil(5% of the side)-th neighbour (size-adaptive)
  cover_min    share of cells covered by a credible anchor, the smaller of the two sides
  regions      share of the 8 k-means regions of each side holding >= 3 credible anchors, the smaller of the two sides
Writes ROOT/SHARED_EVIDENCE_V2.tsv and prints, per statistic, the score's range for shared (Case 1-2) and none-shared
(Case 3) pairs at every level.
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

STATS = ["share20", "share5pct", "cover_min", "regions"]


def statistics(A: np.ndarray, B: np.ndarray) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors
    k = min(20, len(A) - 1, len(B) - 1)
    na = NearestNeighbors(n_neighbors=k).fit(A).kneighbors(B, return_distance=False)
    nb = NearestNeighbors(n_neighbors=k).fit(B).kneighbors(A, return_distance=False)
    pairs = [(i, j) for j in range(len(B)) for i in na[j] if j in set(nb[i])]
    if not pairs:
        return {s: 0.0 for s in STATS}
    i, j = map(np.asarray, zip(*pairs))
    D = B[j] - A[i]
    def radius(V, kk):
        kk = int(min(max(kk, 2), len(V) - 1))
        return NearestNeighbors(n_neighbors=kk + 1).fit(V).kneighbors(V)[0][:, kk]
    out = {}
    for name, ka, kb in (("share20", k, k), ("share5pct", int(np.ceil(0.05 * len(A))), int(np.ceil(0.05 * len(B))))):
        ra, rb = radius(A, ka), radius(B, kb)
        shift = np.median(D, axis=0)
        for _ in range(3):
            ok = np.linalg.norm(D - shift, axis=1) < np.minimum(ra[i], rb[j])
            if ok.sum() < 5:
                break
            shift = np.median(D[ok], axis=0)
        out[name] = float(ok.mean())
        if name == "share20":
            okc, ii, jj = ok, i, j
    out["cover_min"] = float(min(len(set(ii[okc])) / len(A), len(set(jj[okc])) / len(B)))
    la = KMeans(n_clusters=min(8, len(A)), n_init=10, random_state=0).fit(A).labels_
    lbb = KMeans(n_clusters=min(8, len(B)), n_init=10, random_state=0).fit(B).labels_
    fa = np.mean([(okc & (la[ii] == r)).sum() >= 3 for r in np.unique(la)])
    fb = np.mean([(okc & (lbb[jj] == r)).sum() >= 3 for r in np.unique(lbb)])
    out["regions"] = float(min(fa, fb))
    return out


def one(task):
    import anndata as ad
    import rb_common as rb
    from sklearn.cluster import KMeans
    man = pd.read_csv(task["manifest"]); i = task["pair_index"]; r = man.iloc[i]
    S, T = ad.read_h5ad(r.source_h5ad), ad.read_h5ad(r.target_h5ad)
    spec = task["spec"]
    tg = lb.depth_target(task["manifest"], spec["ds"]); s = int(float(spec["ds_seed"]))
    S, T = rb.equalised(S, tg, 1000 * s + 1), rb.equalised(T, tg, 1000 * s + 2)
    X, Y, _ = rb.build_config(dict(spec), S, T, 20260925 + i)
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    obs = statistics(X, Y)
    rng = np.random.default_rng(0); o = rng.permutation(len(X)); h = len(X) // 2
    ref = statistics(X[o[:h]], X[o[h:2 * h]])
    lab = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X).labels_
    dis = statistics(X[lab == 0], X[lab == 1]) if min((lab == 0).sum(), (lab == 1).sum()) > 21 else {s_: 0.0 for s_ in STATS}
    rec = {k_: task[k_] for k_ in ("level", "replicate", "case")}
    for s_ in STATS:
        rec[s_] = obs[s_]; rec[f"{s_}_ref"] = ref[s_]; rec[f"{s_}_dis"] = dis[s_]
        rec[f"{s_}_score"] = float(np.clip((obs[s_] - dis[s_]) / (ref[s_] - dis[s_]), 0, 1)) if ref[s_] > dis[s_] else 0.0
    return rec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--replicates", type=int, nargs="+", default=[1, 2])
    p.add_argument("--jobs", type=int, default=30)
    a = p.parse_args()
    reg = pd.read_csv(lb.REPO / "results" / "realdata_benchmark_20261031" / "PREPROCESSING_CONFIG_REGISTRY.tsv", sep="\t").set_index("config_id")
    spec = reg.loc[lb.BASELINE].to_dict()
    tasks = []
    for rep in a.replicates:
        for lv in lb.ALL_LEVELS:
            cond = a.root / "conditions" / f"rep{rep}" / lv
            man = pd.read_csv(cond / "manifest.csv")
            for i, r in man.iterrows():
                tasks.append({"level": lv, "replicate": rep, "case": r.case, "manifest": str(cond / "manifest.csv"), "pair_index": i, "spec": spec})
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        t = pd.DataFrame(list(ex.map(one, tasks)))
    t.to_csv(a.root / "SHARED_EVIDENCE_V2.tsv", sep="\t", index=False)
    t["kind"] = np.where(t.case.str.startswith("Case3"), "none shared", "shared")
    pd.set_option("display.width", 250)
    for s_ in STATS:
        v = t.groupby(["kind", "level"])[f"{s_}_score"].agg(["min", "max"]).unstack("kind").round(2)
        sep = t[t.kind == "shared"].groupby("level")[f"{s_}_score"].min() - t[t.kind == "none shared"].groupby("level")[f"{s_}_score"].max()
        print(f"== {s_}: min(shared) - max(none shared) by level: {sep.round(2).to_dict()}")
        print(v.to_string())


if __name__ == "__main__":
    main()
