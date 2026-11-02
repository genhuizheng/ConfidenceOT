"""Representation, downsampling and rank-stability diagnostics of one N = 5,000 condition pair, before / beside OT.

    python realdata_benchmark/splatter_n5000/26_representation_diagnostics.py --manifest CONDITION_MANIFEST \
        --pair-index I --truth TRUTH_DIR --out OUT

Configurations (fixed): the eight run configurations (logcpm, ranknm256 production, ranknm1024 expected, rankhvg2000
midrank; DS none and P10 seed 1; noscale, cosine) plus the representation-only Rank-K sweep at DS none (ranknm128 /
256 / 512 / 2048 expected) and rankhvg3000 midrank. Representations come from rb_common.build_config (production code
by label where one exists; the benchmark encoders otherwise), source and target transformed jointly, seed 20260925 +
pair index; no ground-truth label is used to build or tune any of them. DS targets: the pooled 10th / 25th / 50th
percentile of every file of the condition manifest (as the worker), draws seeded 1000 x s + 1 / + 2.

representation.tsv  per configuration: realised nCount / nFeature per side, the diagnostics of
                    realdata_benchmark/04_representation_diagnostics.py (cost scale and spread relative to epsilon,
                    kernel row entropy, kNN population retrieval, cross-side 1-NN agreement and its chance level,
                    cross vs within separation, OT-free kNN-gate retention, depth association of the kNN cost and
                    gate), and, with the simulator's truth: the OT-free kNN gate's false rejection of should-retain
                    cells and recall of should-reject cells per side, and the AUC of the kNN cost for should-reject
                    against should-retain cells per side (how separable the unmatched population is in that geometry)
downsampling.tsv    per side and DS level (P10 / P25 / P50) and seed (1-3): target, untouched fraction, median nCount and
                    nFeature before / after, fraction of detected genes lost; top-256 membership Jaccard of the
                    production encoding seed 1 vs seeds 2 / 3 and vs DS none; share of the 256 slots decided by ties
rank_stability.tsv  per Rank-K / Rank-HVG configuration at DS none and P10: tie statistics at the boundary, and the PCA
                    subspace similarity / cost Spearman of the production vs the expected tie rule (Rank-K) or the
                    midrank vs column-order rule (Rank-HVG)
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import rb_common as rb  # noqa: E402
from confidenceot.preprocessing import median_pair_scale  # noqa: E402

_spec = importlib.util.spec_from_file_location("diag04", HERE.parent / "04_representation_diagnostics.py")
diag04 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(diag04)
SEED = 20260925
RUN = [f"{e}|{d}|noscale|cos" for e in ("logcpm", "ranknm256-production", "ranknm1024-expected", "rankhvg2000-midrank") for d in ("none", "P10s1")]
EXTRA = [f"ranknm{k}-expected|none|noscale|cos" for k in (128, 256, 512, 2048)] + ["rankhvg3000-midrank|none|noscale|cos"]


def spec_of(cid: str) -> dict:
    enc_part, ds, scale, cost = cid.split("|")
    enc, tie = (enc_part.split("-") + [None])[:2]
    s = {"encoding": "logcpm" if enc == "logcpm" else "ranknm" if enc.startswith("ranknm") else "rankhvg", "tie": tie,
         "scale": scale == "scale", "cost": cost, "ds": "none" if ds == "none" else ds[:3]}
    if enc.startswith("ranknm"):
        s["K"] = int(enc[6:])
    if enc.startswith("rankhvg"):
        s["H"] = int(enc[7:])
    return s


def jaccard(A, B):
    inter = np.asarray(A.multiply(B).sum(1)).ravel(); union = np.asarray(A.sum(1)).ravel() + np.asarray(B.sum(1)).ravel() - inter
    return inter / np.maximum(union, 1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--pair-index", type=int, required=True)
    p.add_argument("--truth", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if (a.out / "representation.tsv").exists():
        sys.exit(f"{a.out}/representation.tsv exists; not overwritten")
    a.out.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(a.manifest); row = man.iloc[a.pair_index]
    files = sorted(set(man.source_h5ad) | set(man.target_h5ad))
    pooled = np.concatenate([np.asarray(ad.read_h5ad(f, backed="r").X[:].sum(1)).ravel() for f in files])
    targets = {k: int(np.quantile(pooled, q)) for k, q in rb.DS_QUANTILES.items()}
    s0, t0 = ad.read_h5ad(row.source_h5ad), ad.read_h5ad(row.target_h5ad)
    truth = pd.read_csv(a.truth / "population_truth.tsv", sep="\t").set_index(["side", "observation_id"])
    grp = {s: truth.loc[[(s, o) for o in d.obs_names.astype(str)], "group"].to_numpy() for s, d in (("source", s0), ("target", t0))}
    rej = {s: truth.loc[[(s, o) for o in d.obs_names.astype(str)], "should_reject"].to_numpy(bool) for s, d in (("source", s0), ("target", t0))}
    depth = {s: np.asarray(d.X.sum(1)).ravel() for s, d in (("source", s0), ("target", t0))}
    cache = {}

    def inputs(ds, seed=1):
        key = (ds, seed)
        if key not in cache:
            cache[key] = (s0, t0) if ds == "none" else (rb.equalised(s0, targets[ds], 1000 * seed + 1), rb.equalised(t0, targets[ds], 1000 * seed + 2))
        return cache[key]

    rows = []
    for cid in RUN + EXTRA:
        spec = spec_of(cid); s, t = inputs(spec["ds"])
        X, Y, prov = rb.build_config(spec, s, t, SEED + a.pair_index)
        scale = median_pair_scale(X, Y, rng=np.random.default_rng(SEED + a.pair_index * 104729))
        d, sg, tg, _ = diag04.diagnose(X, Y, scale, grp["source"], grp["target"], depth["source"], depth["target"], np.random.default_rng(a.pair_index))
        r = {"config_id": cid, "path": prov["path"], **d}
        for side, g, A, B in (("source", sg, X, Y), ("target", tg, Y, X)):
            y = rej[side]
            r[f"{side}_knn_false_rejection"] = float((~g[~y]).mean()) if (~y).any() else np.nan
            r[f"{side}_knn_recall"] = float((~g[y]).mean()) if y.any() else np.nan
            from confidenceot.preprocessing import squared_euclidean
            kc = rb.knn_mean(squared_euclidean(A, B).astype(np.float32) / scale, 10)
            r[f"{side}_knn_cost_auc_should_reject"] = float(roc_auc_score(y, kc)) if 0 < y.mean() < 1 else np.nan
        r.update({f"{k}_median_nCount": float(np.median(np.asarray(v.X.sum(1)).ravel())) for k, v in (("source", s), ("target", t))})
        r.update({f"{k}_median_nFeature": float(np.median(np.asarray((v.X > 0).sum(1)).ravel())) for k, v in (("source", s), ("target", t))})
        rows.append(r)
    pd.DataFrame(rows).to_csv(a.out / "representation.tsv", sep="\t", index=False)

    ds_rows = []
    for side, base in (("source", s0), ("target", t0)):
        X0 = sp.csr_matrix(base.X).astype(np.float64)
        top0 = (rb.rank_encode(X0, 256, False, "production")[0] > 0).astype(np.float32)
        for ds in ("P10", "P25", "P50"):
            tops = {}
            for seed in (1, 2, 3):
                E = sp.csr_matrix(inputs(ds, seed)[0 if side == "source" else 1].X).astype(np.float64)
                enc, stt = rb.rank_encode(E, 256, False, "production")
                tops[seed] = (enc > 0).astype(np.float32)
                det0, det = np.diff(X0.indptr), np.diff(sp.csr_matrix(E).indptr)
                ds_rows.append({"side": side, "ds": ds, "seed": seed, "target": targets[ds], "untouched_fraction": float(np.mean(depth[side] <= targets[ds])),
                                "median_nCount_before": float(np.median(depth[side])), "median_nCount_after": float(np.median(np.asarray(E.sum(1)).ravel())),
                                "median_nFeature_before": float(np.median(det0)), "median_nFeature_after": float(np.median(det)),
                                "median_fraction_detected_lost": float(np.median(1 - det / np.maximum(det0, 1))),
                                "median_top256_slots_from_ties": float(np.median(stt.tied_slots_at_cut / 256)),
                                "median_top256_jaccard_vs_none": float(np.median(jaccard(tops[seed], top0)))})
            for seed in (2, 3):
                ds_rows[-4 + seed]["median_top256_jaccard_seed1"] = float(np.median(jaccard(tops[1], tops[seed])))
    pd.DataFrame(ds_rows).to_csv(a.out / "downsampling.tsv", sep="\t", index=False)

    rs = []
    for ds in ("none", "P10"):
        s, t = inputs(ds)
        for enc, alt in (({"encoding": "ranknm", "K": 256, "tie": "production"}, "expected"), ({"encoding": "ranknm", "K": 1024, "tie": "production"}, "expected"),
                         ({"encoding": "rankhvg", "H": 2000, "tie": "midrank"}, "order")):
            base = {**enc, "scale": False, "cost": "cos", "ds": ds}
            X1, Y1, _ = rb.build_config(base, s, t, SEED + a.pair_index); X2, Y2, _ = rb.build_config({**base, "tie": alt}, s, t, SEED + a.pair_index)
            J1, J2 = np.vstack([X1, Y1]), np.vstack([X2, Y2])
            g = np.random.default_rng(0); i, j = g.integers(len(X1), size=100_000), g.integers(len(Y1), size=100_000)
            c1, c2 = np.sum((X1[i] - Y1[j]) ** 2, 1), np.sum((X2[i] - Y2[j]) ** 2, 1)
            st_ = rb.rank_encode(rb.joint_counts(s, t), enc.get("K", 256), False, "production")[1] if enc["encoding"] == "ranknm" else None
            rs.append({"ds": ds, "encoding": enc["encoding"], "K_or_H": enc.get("K", enc.get("H")), "rule_a": enc["tie"], "rule_b": alt,
                       "median_boundary_slots_from_ties": float(np.median(st_.tied_slots_at_cut / enc["K"])) if st_ is not None else np.nan,
                       "subspace_similarity": rb.subspace_similarity(J1, J2), "cost_spearman": float(spearmanr(c1, c2)[0])})
    pd.DataFrame(rs).to_csv(a.out / "rank_stability.tsv", sep="\t", index=False)
    print(f"diagnostics -> {a.out}")


if __name__ == "__main__":
    main()
