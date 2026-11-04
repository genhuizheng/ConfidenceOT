"""Label-free shared-evidence diagnostics (development seeds only): what separates a pair whose sides share populations
from one that shares none, before deciding to apply a batch translation. No OT, no truth used in the statistics; the
truth (case) is joined afterwards only to look at the separation.

    python realdata_benchmark/splatter_local/shared_evidence_diagnostics.py --root ROOT [--replicates 1 2]

Per pair (representation ranknm256-production|P10s1|noscale|cos, PCA 30), with the MNN translation (k = 20) applied:
  anchors_share_{source,target}  share of the side's cells in a mutual-nearest-neighbour pair
  consistency                    regional consistency of the anchor displacements (as the mnnc correction)
  coverage_{source,target}       after the translation, share of the side's cells whose nearest other-side cell is closer
                                 than their own 20th same-side neighbour (they would sit inside their own neighbourhood)
  coverage_raw_{source,target}   the same without the translation
  credible_anchor_share          after the translation, share of anchor pairs whose distance is below the 20th
                                 same-side neighbour distance of both cells
  displacement_dispersion        median |anchor displacement - median displacement| / median 20th same-side NN distance
Writes ROOT/SHARED_EVIDENCE_DIAGNOSTICS.tsv and prints the diagnostics by level x case.
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

K = 20


def diag(X: np.ndarray, Y: np.ndarray) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors
    k = min(K, len(X) - 1, len(Y) - 1)
    rx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)[0][:, k]  # 20th same-side neighbour distance
    ry = NearestNeighbors(n_neighbors=k + 1).fit(Y).kneighbors(Y)[0][:, k]
    nx = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(Y, return_distance=False)
    ny = NearestNeighbors(n_neighbors=k).fit(Y).kneighbors(X, return_distance=False)
    pairs = [(i, j) for j in range(len(Y)) for i in nx[j] if j in set(ny[i])]
    out = {"anchors": len(pairs)}
    def coverage(A, B, rA):
        d = NearestNeighbors(n_neighbors=1).fit(B).kneighbors(A)[0][:, 0]
        return float((d < rA).mean())
    out["coverage_raw_source"], out["coverage_raw_target"] = coverage(X, Y, rx), coverage(Y, X, ry)
    if not pairs:
        return out
    i, j = map(np.asarray, zip(*pairs))
    out["anchors_share_source"] = len(set(i)) / len(X); out["anchors_share_target"] = len(set(j)) / len(Y)
    D = Y[j] - X[i]; shift = np.median(D, axis=0)
    region = KMeans(n_clusters=min(8, len(X)), n_init=10, random_state=0).fit(X).labels_[i]
    V = np.array([np.median(D[region == r], axis=0) for r in np.unique(region) if (region == r).sum() >= 5])
    out["consistency"] = float(np.sum(V.mean(0) ** 2) / max(np.mean(np.sum(V ** 2, axis=1)), 1e-12)) if len(V) else np.nan
    Ys = Y - shift
    out["coverage_source"], out["coverage_target"] = coverage(X, Ys, rx), coverage(Ys, X, ry)
    dij = np.linalg.norm(Ys[j] - X[i], axis=1)
    out["credible_anchor_share"] = float(((dij < rx[i]) & (dij < ry[j])).mean())
    out["displacement_dispersion"] = float(np.median(np.linalg.norm(D - shift, axis=1)) / np.median(np.concatenate([rx, ry])))
    return out


def one(task):
    import anndata as ad
    import rb_common as rb
    man = pd.read_csv(task["manifest"]); i = task["pair_index"]; r = man.iloc[i]
    S, T = ad.read_h5ad(r.source_h5ad), ad.read_h5ad(r.target_h5ad)
    spec = task["spec"]
    tg = lb.depth_target(task["manifest"], spec["ds"]); s = int(float(spec["ds_seed"]))
    S, T = rb.equalised(S, tg, 1000 * s + 1), rb.equalised(T, tg, 1000 * s + 2)
    X, Y, _ = rb.build_config(dict(spec), S, T, 20260925 + i)
    return {k: task[k] for k in ("level", "replicate", "case")} | diag(np.asarray(X, float), np.asarray(Y, float))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--replicates", type=int, nargs="+", default=[1, 2])
    p.add_argument("--jobs", type=int, default=8)
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
    t.to_csv(a.root / "SHARED_EVIDENCE_DIAGNOSTICS.tsv", sep="\t", index=False)
    cols = ["anchors_share_source", "anchors_share_target", "consistency", "coverage_raw_source", "coverage_raw_target",
            "coverage_source", "coverage_target", "credible_anchor_share", "displacement_dispersion"]
    pd.set_option("display.width", 250)
    print(t.groupby(["case"])[cols].agg(["min", "median", "max"]).round(2).T.to_string())


if __name__ == "__main__":
    main()
