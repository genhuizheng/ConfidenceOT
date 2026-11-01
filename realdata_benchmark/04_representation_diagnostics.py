"""REPRESENTATION_DIAGNOSTICS.tsv (+ COST_CORRELATION.tsv): every configuration of the registry with in_new_grid on
every LTL331 pair, before any OT (JOINT_BENCHMARK_PROTOCOL.md section 5, the no-OT part).

    python realdata_benchmark/04_representation_diagnostics.py --run-root RUN_ROOT --converted CONVERTED \
        --registry REGISTRY --out OUT_DIR [--pairs 5 0 1 2 3 4 6]

Inputs per configuration: DS none = the original counts of the analysed cells (inputs/rawqc); P10 frozen = the 27_
equalised files (inputs/ds); P10 / P25 / P50 with a seed = the original counts equalised here (rb_common.equalised;
targets 2,817 / 5,023 / 9,360). Coordinates come from rb_common.build_config: production-equivalent configurations
through the production code by label, the others through the benchmark encoders. Seed 20260831 + pair index, median
scale with the 02_run_pair.py generator, as the frozen run.

Per pair x configuration: input depth / detected genes; cost scale and quantiles of 200,000 sampled scaled costs;
spread relative to epsilon (0.1) and the kernel row entropy of exp(-C / epsilon) over 500 source rows; kNN (k = 15)
coarse_celltype retrieval and cross-side 1-NN type agreement with its chance level; AUC of cross-side vs within-side
k = 10 mean cost; OT-free kNN-gate retained fractions; Spearman of each side's kNN cost with pre-equalisation depth
and the AUC of depth for kNN-rejected cells; 16 -> 20 source NE minus non-NE kNN retention (and for every pair where
NE >= 20). Per pair the kNN gates, the sampled costs and the coordinates (float16) are kept in
OUT/source/diag_pair<i>.npz, from which 09_ computes DS-seed / tie / cross-configuration stability. Failures are
listed with the error, never dropped.
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
from confidenceot.preprocessing import median_pair_scale, squared_euclidean

EPSILON = 0.1


def gate(A, B, Cab, scale, seed):
    g = np.random.default_rng(seed); n = len(A); perm = g.permutation(n); u, v = perm[: n // 2], perm[n // 2:]
    m = min(n // 2, len(B)); ref = g.choice(len(B), m, replace=False); cross = rb.knn_mean(Cab[:, ref], 10)
    within = np.concatenate([rb.knn_mean(squared_euclidean(A[u], A[v[:m]]).astype(np.float32) / scale, 10),
                             rb.knn_mean(squared_euclidean(A[v], A[u[:m]]).astype(np.float32) / scale, 10)])
    return cross, within, cross <= np.quantile(within, 0.99)


def diagnose(X, Y, scale, st, tt, sd, td, rng):
    li, lj = rng.integers(len(X), size=200_000), rng.integers(len(Y), size=200_000)
    cs = np.sum((X[li] - Y[lj]) ** 2, 1) / scale
    rows_idx = rng.choice(len(X), min(500, len(X)), replace=False)
    Crow = squared_euclidean(X[rows_idx], Y) / scale
    q = np.quantile(Crow, [0.05, 0.95], axis=1)
    K = np.exp(-(Crow - Crow.min(1, keepdims=True)) / EPSILON); K /= K.sum(1, keepdims=True)
    H = -(K * np.log(np.where(K > 0, K, 1))).sum(1) / np.log(Crow.shape[1])
    C = squared_euclidean(X, Y).astype(np.float32) / scale
    src_knn, tgt_knn = rb.knn_mean(C, 10), rb.knn_mean(C.T, 10)
    sc, sw, sg = gate(X, Y, C, scale, 7); tc, tw, tg = gate(Y, X, C.T, scale, 8)
    del C
    J = np.vstack([X, Y]).astype(np.float32); types = np.concatenate([st, tt])
    idx = NearestNeighbors(n_neighbors=16).fit(J).kneighbors(J, return_distance=False)[:, 1:]
    nn_t = NearestNeighbors(n_neighbors=1).fit(Y).kneighbors(X, return_distance=False)[:, 0]
    ps = pd.Series(st).value_counts(normalize=True); pt = pd.Series(tt).value_counts(normalize=True)
    auc = lambda y, x: float(roc_auc_score(y, x)) if 0 < np.mean(y) < 1 else np.nan
    ne = (st == "NE").sum() >= 20
    r = dict(cost_scale=scale, cost_q05=float(np.quantile(cs, 0.05)), cost_median=float(np.median(cs)), cost_q95=float(np.quantile(cs, 0.95)),
             cost_max=float(cs.max()), spread_over_epsilon=float((np.quantile(cs, 0.95) - np.quantile(cs, 0.05)) / EPSILON),
             row_spread_over_epsilon_median=float(np.median((q[1] - q[0]) / EPSILON)), kernel_row_entropy_median=float(np.median(H)),
             knn15_celltype_retrieval=float(np.mean(types[idx] == types[:, None])),
             cross_1nn_type_agreement=float(np.mean(tt[nn_t] == st)),
             cross_1nn_type_chance=float(sum(ps.get(k, 0) * pt.get(k, 0) for k in set(ps.index) | set(pt.index))),
             source_cross_vs_within_auc=auc(np.r_[np.ones(len(sc)), np.zeros(len(sw))], np.r_[sc, sw]),
             target_cross_vs_within_auc=auc(np.r_[np.ones(len(tc)), np.zeros(len(tw))], np.r_[tc, tw]),
             source_knn_retained=float(sg.mean()), target_knn_retained=float(tg.mean()),
             source_knn_cost_depth_spearman=float(spearmanr(src_knn, sd)[0]), target_knn_cost_depth_spearman=float(spearmanr(tgt_knn, td)[0]),
             source_knn_rejected_depth_auc=auc(~sg, sd), target_knn_rejected_depth_auc=auc(~tg, td),
             source_knn_ne_minus_nonne=float(sg[st == "NE"].mean() - sg[st != "NE"].mean()) if ne else np.nan)
    return r, sg, tg, cs.astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--converted", type=Path, required=True)
    p.add_argument("--registry", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pairs", type=int, nargs="*", default=[5, 0, 1, 2, 3, 4, 6])
    a = p.parse_args()
    target = a.out / "REPRESENTATION_DIAGNOSTICS.tsv"
    if target.exists():
        sys.exit(f"{target} exists; not overwritten")
    (a.out / "source").mkdir(parents=True, exist_ok=True)
    reg = pd.read_csv(a.registry, sep="\t")
    reg = reg[reg.in_new_grid.astype(bool)].reset_index(drop=True)
    ctype = rb.cell_types(a.converted)
    depth = pd.read_csv(a.run_root / "inputs" / "ds" / "predownsample_depth.csv.gz").set_index("observation_id")["predownsample_total_counts"]
    targets = rb.ds_targets(depth)
    rows, corr = [], []
    for index in a.pairs:
        orig = rb.ltl_pairs(a.run_root, False).iloc[index]; froz = rb.ltl_pairs(a.run_root, True).iloc[index]
        pair = orig.pair_id.replace("LTL331__", "")
        base = {"none": (ad.read_h5ad(rb.local(orig.source_h5ad, a.run_root)), ad.read_h5ad(rb.local(orig.target_h5ad, a.run_root))),
                "P10sfrozen": (ad.read_h5ad(rb.local(froz.source_h5ad, a.run_root)), ad.read_h5ad(rb.local(froz.target_h5ad, a.run_root)))}
        if list(base["none"][0].obs_names) != list(base["P10sfrozen"][0].obs_names):
            sys.exit(f"{pair}: original and equalised files hold different cells")
        inputs = {}

        def get_input(ds, seed):
            key = "none" if ds == "none" else f"{ds}s{seed}"
            if key not in inputs:
                if key in base:
                    inputs[key] = base[key]
                else:
                    s0, t0 = base["none"]
                    inputs[key] = (rb.equalised(s0, targets[ds], 1000 * int(seed) + 1), rb.equalised(t0, targets[ds], 1000 * int(seed) + 2))
            return inputs[key]
        store = {}
        for _, spec in reg.iterrows():
            spec = spec.to_dict()
            seed_tag = str(spec["ds_seed"]).replace(" (27_)", "") if spec["ds"] != "none" else None
            s, t = get_input(spec["ds"], seed_tag if seed_tag != "frozen" else "frozen")
            st = ctype.loc[s.obs_names.astype(str)].to_numpy(); tt = ctype.loc[t.obs_names.astype(str)].to_numpy()
            sd, td = depth.loc[s.obs_names.astype(str)].to_numpy(), depth.loc[t.obs_names.astype(str)].to_numpy()
            r = {"pair": pair, "pair_index": index, "config_id": spec["config_id"], "kind": spec["kind"], "role": spec["role"],
                 "contrasts": spec["contrasts"], "ds": spec["ds"], "ds_seed": seed_tag}
            t0 = time.time()
            try:
                if spec["kind"] == "historical":
                    spec = {**spec, "encoding": "label", "label": spec["label"]}
                X, Y, prov = rb.build_config(spec, s, t, rb.SEED + index)
            except Exception as error:
                rows.append({**r, "status": f"failed: {type(error).__name__}: {error}"}); print(pair, r["config_id"], "FAILED", error, flush=True)
                continue
            scale = median_pair_scale(X, Y, rng=np.random.default_rng(rb.SEED + index * 104729))
            d, sg, tg, cs = diagnose(X, Y, scale, st, tt, sd, td, np.random.default_rng(index))
            r.update(status="ok", path=prov["path"], dimension=X.shape[1], hvg_n=prov.get("hvg_n"),
                     source_median_depth=float(np.median(np.asarray(s.X.sum(1)).ravel())), target_median_depth=float(np.median(np.asarray(t.X.sum(1)).ravel())),
                     source_median_detected=float(np.median(np.asarray((s.X > 0).sum(1)).ravel())), target_median_detected=float(np.median(np.asarray((t.X > 0).sum(1)).ravel())),
                     **d, seconds=round(time.time() - t0, 1))
            rows.append(r)
            store[f"{spec['config_id']}::source_gate"] = sg; store[f"{spec['config_id']}::target_gate"] = tg
            store[f"{spec['config_id']}::costs"] = cs; store[f"{spec['config_id']}::coords"] = np.vstack([X, Y]).astype(np.float16)
            print(pair, r["config_id"], f"{r['seconds']}s", flush=True)
            pd.DataFrame(rows).to_csv(a.out / "REPRESENTATION_DIAGNOSTICS.partial.tsv", sep="\t", index=False)
        np.savez_compressed(a.out / "source" / f"diag_pair{index}.npz", **store)
        names = [k[:-7] for k in store if k.endswith("::costs")]
        for i, x in enumerate(names):
            for y in names[i + 1:]:
                corr.append({"pair": pair, "config_a": x, "config_b": y, "cost_spearman": float(spearmanr(store[x + "::costs"], store[y + "::costs"])[0])})
        pd.DataFrame(corr).to_csv(a.out / "COST_CORRELATION.partial.tsv", sep="\t", index=False)
    pd.DataFrame(rows).to_csv(target, sep="\t", index=False)
    pd.DataFrame(corr).to_csv(a.out / "COST_CORRELATION.tsv", sep="\t", index=False)
    for f in ("REPRESENTATION_DIAGNOSTICS.partial.tsv", "COST_CORRELATION.partial.tsv"):
        (a.out / f).unlink(missing_ok=True)
    print(f"-> {target}")


if __name__ == "__main__":
    main()
