"""REPRESENTATION_DIAGNOSTICS.tsv and COST_CORRELATION.tsv: every registered configuration on every LTL331 pair, before
any OT. Each representation is built by the production code from its label (Preprocessing.from_label ->
cancer_metastasis/common.prepare_joint_representation, the run seed 20260831 + pair index, the median scale with the
02_run_pair.py generator), the same path ltl331_nepc/07_validity_audit.py reproduced against run.json.

    python realdata_benchmark/04_representation_diagnostics.py --run-root RUN_ROOT --converted CONVERTED --registry REGISTRY --out OUT_DIR

Per pair x configuration:
  input        median depth and detected genes per side of the matrix the label reads (ds or original counts);
  scale        cost scale, and the 5% / 50% / 95% / max of 200,000 sampled scaled cross-side costs;
  epsilon      (q95 - q05) / epsilon and the median per-row (q95 - q05) / epsilon over 500 source rows (epsilon 0.1,
               02_run_pair.py default); the normalised entropy of exp(-C / epsilon) per row (1 = uniform), the
               kernel's own spread before any marginal constraint;
  stability    PCA subspace similarity between the run seed and seed + 1 (sklearn's randomised solver);
               not computed when there is no PCA;
  retrieval    kNN (k = 15) coarse_celltype agreement in the joint representation; cross-side 1-NN type agreement for
               source cells against its chance level (the two sides' type frequencies);
  separation   AUC of cross-side vs within-side k = 10 mean cost (matched reference sizes; 0.5 = the sides
               interleave); the OT-free kNN gate's retained fractions;
  depth        Spearman correlation of each side's cross-side kNN cost with the cell's pre-equalisation depth;
               16 -> 20 source NE minus non-NE kNN-gate retention.
Sampled cost vectors are kept per pair (OUT/source/costs_<pair>.npz) and COST_CORRELATION.tsv holds the Spearman
correlation of every configuration pair on the same sampled cell pairs. Configurations whose representation fails to
build are listed with the error, never dropped.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

import rb_common as rb
from common import prepare_joint_representation
from confidenceot import Preprocessing
from confidenceot.preprocessing import median_pair_scale, squared_euclidean

EPSILON = 0.1


def build(label: str, s, t, seed: int):
    c = Preprocessing.from_label(label)
    return prepare_joint_representation(
        s, t, n_hvg=c.n_hvg, n_pcs=c.n_pcs, seed=seed, representation=c.normalisation, rank_top_n=c.rank_top_n,
        cost=c.cost, equalise_depth=c.equalise_depth, regress_out=c.regress_out, regress_on_ranks=c.regress_on_ranks,
        scale_genes=c.scale_genes, label_stem=c.label_stem, allow_precomputed=c.normalisation == "precomputed")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--converted", type=Path, required=True)
    p.add_argument("--registry", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--labels", nargs="*", default=None)
    a = p.parse_args()
    target = a.out / "REPRESENTATION_DIAGNOSTICS.tsv"
    if target.exists():
        sys.exit(f"{target} exists; not overwritten")
    (a.out / "source").mkdir(parents=True, exist_ok=True)
    labels = a.labels or pd.read_csv(a.registry, sep="\t")["label"].tolist()
    ctype = rb.cell_types(a.converted)
    depth = pd.read_csv(a.run_root / "inputs" / "ds" / "predownsample_depth.csv.gz").set_index("observation_id")["predownsample_total_counts"]
    rows, corr = [], []
    n_pairs = len(rb.ltl_pairs(a.run_root, True))
    for index in range(n_pairs):
        cache, costs = {}, {}
        for label in labels:
            eq = Preprocessing.from_label(label).equalise_depth
            if eq not in cache:
                row = rb.ltl_pairs(a.run_root, eq).iloc[index]
                cache[eq] = (row.pair_id.replace("LTL331__", ""), ad.read_h5ad(rb.local(row.source_h5ad, a.run_root)),
                             ad.read_h5ad(rb.local(row.target_h5ad, a.run_root)))
            pair, s, t = cache[eq]
            r = {"pair": pair, "label": label, "input": "ds" if eq else "original"}
            t0 = time.time()
            try:
                X, Y, _, prov = build(label, s, t, rb.SEED + index)
            except Exception as error:  # reported, never dropped
                rows.append({**r, "status": f"failed: {type(error).__name__}: {error}"}); print(pair, label, "FAILED", error, flush=True)
                continue
            X, Y = np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)
            scale = median_pair_scale(X, Y, rng=np.random.default_rng(rb.SEED + index * 104729))
            rng = np.random.default_rng(index)
            li, lj = rng.integers(len(X), size=200_000), rng.integers(len(Y), size=200_000)
            cs = np.sum((X[li] - Y[lj]) ** 2, 1) / scale; costs[label] = cs.astype(np.float32)
            st = ctype.loc[s.obs_names.astype(str)].to_numpy(); tt = ctype.loc[t.obs_names.astype(str)].to_numpy()
            sd, td = depth.loc[s.obs_names.astype(str)].to_numpy(), depth.loc[t.obs_names.astype(str)].to_numpy()
            rows_idx = rng.choice(len(X), min(500, len(X)), replace=False)
            Crow = squared_euclidean(X[rows_idx], Y) / scale
            q = np.quantile(Crow, [0.05, 0.95], axis=1)
            K = np.exp(-(Crow - Crow.min(1, keepdims=True)) / EPSILON); K /= K.sum(1, keepdims=True)
            H = -(K * np.log(np.where(K > 0, K, 1))).sum(1) / np.log(Crow.shape[1])
            C = squared_euclidean(X, Y).astype(np.float32) / scale
            src_knn, tgt_knn = rb.knn_mean(C, 10), rb.knn_mean(C.T, 10)

            def gate(A, B, Cab, seed):
                g = np.random.default_rng(seed); n = len(A); perm = g.permutation(n); u, v = perm[: n // 2], perm[n // 2:]
                m = min(n // 2, len(B)); ref = g.choice(len(B), m, replace=False); cross = rb.knn_mean(Cab[:, ref], 10)
                within = np.concatenate([rb.knn_mean(squared_euclidean(A[u], A[v[:m]]).astype(np.float32) / scale, 10),
                                         rb.knn_mean(squared_euclidean(A[v], A[u[:m]]).astype(np.float32) / scale, 10)])
                return cross, within, cross <= np.quantile(within, 0.99)
            sc, sw, sg = gate(X, Y, C, 7); tc, tw, tg = gate(Y, X, C.T, 8)
            del C
            J = np.vstack([X, Y]); types = np.concatenate([st, tt])
            idx = NearestNeighbors(n_neighbors=16).fit(J).kneighbors(J, return_distance=False)[:, 1:]
            nn_t = NearestNeighbors(n_neighbors=1).fit(Y).kneighbors(X, return_distance=False)[:, 0]
            ps = pd.Series(st).value_counts(normalize=True); pt = pd.Series(tt).value_counts(normalize=True)
            r.update(status="ok", seconds=round(time.time() - t0, 1), dimension=X.shape[1], common_genes=prov.get("common_gene_n"),
                     hvg=prov.get("hvg_n"), pca_components=prov.get("pca_components"),
                     source_median_depth=float(np.median(np.asarray(s.X.sum(1)).ravel())), target_median_depth=float(np.median(np.asarray(t.X.sum(1)).ravel())),
                     source_median_detected=float(np.median(np.diff(s.X.tocsr().indptr))) if hasattr(s.X, "tocsr") else np.nan,
                     target_median_detected=float(np.median(np.diff(t.X.tocsr().indptr))) if hasattr(t.X, "tocsr") else np.nan,
                     cost_scale=scale, cost_q05=float(np.quantile(cs, 0.05)), cost_median=float(np.median(cs)), cost_q95=float(np.quantile(cs, 0.95)), cost_max=float(cs.max()),
                     spread_over_epsilon=float((np.quantile(cs, 0.95) - np.quantile(cs, 0.05)) / EPSILON),
                     row_spread_over_epsilon_median=float(np.median((q[1] - q[0]) / EPSILON)),
                     kernel_row_entropy_median=float(np.median(H)),
                     knn15_celltype_retrieval=float(np.mean(types[idx] == types[:, None])),
                     cross_1nn_type_agreement=float(np.mean(tt[nn_t] == st)), cross_1nn_type_chance=float(sum(ps.get(k, 0) * pt.get(k, 0) for k in set(ps.index) | set(pt.index))),
                     source_cross_vs_within_auc=float(roc_auc_score(np.r_[np.ones(len(sc)), np.zeros(len(sw))], np.r_[sc, sw])),
                     target_cross_vs_within_auc=float(roc_auc_score(np.r_[np.ones(len(tc)), np.zeros(len(tw))], np.r_[tc, tw])),
                     source_knn_retained=float(sg.mean()), target_knn_retained=float(tg.mean()),
                     source_knn_cost_depth_spearman=float(spearmanr(src_knn, sd)[0]), target_knn_cost_depth_spearman=float(spearmanr(tgt_knn, td)[0]),
                     source_knn_gate_ne_minus_nonne=float(sg[st == "NE"].mean() - sg[st != "NE"].mean()) if (st == "NE").sum() >= 20 else np.nan)
            if prov.get("pca_components"):
                try:
                    X2, Y2, _, _ = build(label, s, t, rb.SEED + index + 1)
                    r["pca_seed_subspace_similarity"] = rb.subspace_similarity(np.vstack([X, Y]), np.vstack([X2, Y2]).astype(np.float32))
                except Exception as error:
                    r["pca_seed_subspace_similarity"] = np.nan; r["status"] = f"ok; seed rebuild failed: {error}"
            rows.append(r)
            print(pair, label, f"{r['seconds']}s", flush=True)
            pd.DataFrame(rows).to_csv(a.out / "REPRESENTATION_DIAGNOSTICS.partial.tsv", sep="\t", index=False)
        np.savez_compressed(a.out / "source" / f"costs_{pair}.npz", **costs)
        names = list(costs)
        for i, x in enumerate(names):
            for y in names[i + 1:]:
                corr.append({"pair": pair, "label_a": x, "label_b": y, "cost_spearman": float(spearmanr(costs[x], costs[y])[0])})
    pd.DataFrame(rows).to_csv(target, sep="\t", index=False)
    pd.DataFrame(corr).to_csv(a.out / "COST_CORRELATION.tsv", sep="\t", index=False)
    (a.out / "REPRESENTATION_DIAGNOSTICS.partial.tsv").unlink(missing_ok=True)
    print(f"-> {target}")


if __name__ == "__main__":
    main()
