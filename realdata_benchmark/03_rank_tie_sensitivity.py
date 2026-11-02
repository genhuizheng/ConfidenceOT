"""RANK_TIE_SENSITIVITY.tsv: how much of the rank representation the tie rule decides, on every LTL331 pair.

    python realdata_benchmark/03_rank_tie_sensitivity.py --run-root RUN_ROOT --converted CONVERTED --out OUT_DIR

For rank_value (rank256) and rank_no_median (ranknm256), on the equalised (ds) and the original counts, each tie rule
of rb_common.rank_encode (production, alphabetical, random seeds 1-3, expected) is pushed through the same downstream
as the production arm (2,000 top-variance genes, no gene scaling, PCA 30 with the run seed, cosine) and compared with
the production rule on: tie statistics at the top-256 cut; top-256 membership (Jaccard); PCA subspace similarity of
the joint coordinates; Spearman correlation of 200,000 sampled cross-side costs; the OT-free kNN gate of
ltl331_nepc/07_validity_audit.py (agreement kappa with the production rule's kNN gate, source / target kNN-retained
fractions, NE minus non-NE kNN retention on the source side); kNN (k = 15) coarse_celltype retrieval within the joint
representation. No OT is fitted here (the OT effect of the tie rule is in the 16 -> 20 pilot).

Options (added 2026-11-01 for criterion 3 of the eligibility section, which had been measured with cosine and K = 256
only): --cost sqeuc (no unit rows; costs and kNN gates on the PCA coordinates), --k K (top-K; the 'top256' columns are
then named top<K>), --encodings ranknm (rank_no_median only), --ds-only (equalised counts only). With any non-default
option the table is RANK_TIE_SENSITIVITY_<cost>_K<k>[_<encodings>][_ds].tsv, so the default table is never touched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

import rb_common as rb
from confidenceot.preprocessing import median_pair_scale, squared_euclidean

RULES = [("production", 0), ("alphabetical", 0), ("random", 1), ("random", 2), ("random", 3), ("expected", 0)]


def knn_gate(X, Y, scale, rng, k=10):
    C = squared_euclidean(X, Y).astype(np.float32) / scale
    n = len(X); perm = rng.permutation(n); a, b = perm[: n // 2], perm[n // 2:]; m = min(n // 2, len(Y))
    ref = rng.choice(len(Y), m, replace=False)
    cross = rb.knn_mean(C[:, ref], k)
    w = np.concatenate([rb.knn_mean(squared_euclidean(X[a], X[b[:m]]).astype(np.float32) / scale, k),
                        rb.knn_mean(squared_euclidean(X[b], X[a[:m]]).astype(np.float32) / scale, k)])
    return cross <= np.quantile(w, 0.99)


def retrieval(coords, labels, k=15):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    idx = nn.kneighbors(coords, return_distance=False)[:, 1:]
    return float(np.mean(labels[idx] == labels[:, None]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--converted", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--resume", action="store_true",
                   help="keep every (configuration, pair) of the partial table that has all tie rules, compute the rest")
    p.add_argument("--cost", choices=("cos", "sqeuc"), default="cos")
    p.add_argument("--k", type=int, default=256)
    p.add_argument("--encodings", choices=("both", "ranknm"), default="both")
    p.add_argument("--ds-only", action="store_true")
    a = p.parse_args()
    cosine, K = a.cost == "cos", a.k
    gms = (False, True) if a.encodings == "both" else (False,)
    default = cosine and K == 256 and a.encodings == "both" and not a.ds_only
    target = a.out / ("RANK_TIE_SENSITIVITY.tsv" if default else
                      f"RANK_TIE_SENSITIVITY_{a.cost}_K{K}" + ("" if a.encodings == "both" else f"_{a.encodings}") + ("_ds" if a.ds_only else "") + ".tsv")
    name = lambda gm, equalised: ("rank" if gm else "ranknm") + str(K) + "_noscale" + ("_ds" if equalised else "") + ("_cos" if cosine else "")
    if target.exists():
        sys.exit(f"{target} exists; not overwritten")
    a.out.mkdir(parents=True, exist_ok=True)
    ctype = rb.cell_types(a.converted)
    rows, done = [], set()
    partial = target.with_suffix(".partial.tsv")
    if a.resume and partial.exists():
        old = pd.read_csv(partial, sep="\t")
        complete = old.groupby(["config", "pair"]).size()
        done = set(complete[complete == len(RULES)].index)
        rows = old[[(c, q) in done for c, q in zip(old.config, old.pair)]].to_dict("records")
        print(f"resume: {len(done)} configuration-pairs kept", flush=True)
    for equalised in ((True,) if a.ds_only else (True, False)):
        pairs = rb.ltl_pairs(a.run_root, equalised)
        for index, row in pairs.iterrows():
            pair = row.pair_id.replace("LTL331__", "")
            names = [name(gm, equalised) for gm in gms]
            if all((n, pair) in done for n in names):
                continue
            s = ad.read_h5ad(rb.local(row.source_h5ad, a.run_root)); t = ad.read_h5ad(rb.local(row.target_h5ad, a.run_root))
            joint = rb.joint_counts(s, t)
            ns = s.n_obs
            types = np.concatenate([ctype.loc[s.obs_names.astype(str)].to_numpy(), ctype.loc[t.obs_names.astype(str)].to_numpy()])
            sample_rng = np.random.default_rng(index)
            li, lj = sample_rng.integers(ns, size=200_000), sample_rng.integers(t.n_obs, size=200_000)
            for gm in gms:
                config = name(gm, equalised)
                if (config, pair) in done:
                    continue
                ref = None
                for rule, seed in RULES:
                    enc, st = rb.rank_encode(joint, K, gm, rule, seed)
                    coords, _ = rb.pca_coordinates(enc, 2000, 30, False, cosine, rb.SEED + index)
                    X, Y = coords[:ns], coords[ns:]
                    scale = median_pair_scale(X, Y, rng=np.random.default_rng(rb.SEED + index * 104729))
                    costs = np.sum((X[li] - Y[lj]) ** 2, 1) / scale
                    gs = knn_gate(X, Y, scale, np.random.default_rng(7)); gt = knn_gate(Y, X, scale, np.random.default_rng(8))
                    member = (enc > 0).astype(np.float32)
                    r = {"config": config, "pair": pair, "tie_rule": rule, "seed": seed,
                         "median_tied_slots_at_cut": float(np.median(st.tied_slots_at_cut)),
                         f"median_fraction_of_top{K}_from_cut_ties": float(np.median(st.tied_slots_at_cut / K)),
                         "median_cut_tie_group_size": float(np.median(st.cut_tie_group_size)),
                         "median_cut_value": float(np.nanmedian(st.cut_value)),
                         "cells_with_cut_ties": float(np.mean(st.tied_slots_at_cut > 0)),
                         "cost_scale": scale, "source_knn_retained": float(gs.mean()), "target_knn_retained": float(gt.mean()),
                         "source_knn_ne_minus_nonne": float(gs[types[:ns] == "NE"].mean() - gs[types[:ns] != "NE"].mean()) if (types[:ns] == "NE").sum() >= 20 else np.nan,
                         "knn15_celltype_retrieval": retrieval(coords, types)}
                    if ref is None:
                        ref = {"member": member, "coords": coords, "costs": costs, "gs": gs, "gt": gt}
                    inter = np.asarray(member.multiply(ref["member"]).sum(1)).ravel()
                    union = np.asarray(member.sum(1)).ravel() + np.asarray(ref["member"].sum(1)).ravel() - inter
                    r.update(**{f"top{K}_jaccard_vs_production_median": float(np.median(inter / np.maximum(union, 1)))},
                             pca_subspace_similarity_vs_production=rb.subspace_similarity(coords, ref["coords"]),
                             cost_spearman_vs_production=float(spearmanr(costs, ref["costs"])[0]),
                             source_knn_gate_kappa_vs_production=rb.kappa(gs, ref["gs"]),
                             target_knn_gate_kappa_vs_production=rb.kappa(gt, ref["gt"]))
                    rows.append(r)
                print(config, pair, "done", flush=True)
                pd.DataFrame(rows).to_csv(target.with_suffix(".partial.tsv"), sep="\t", index=False)
    pd.DataFrame(rows).to_csv(target, sep="\t", index=False)
    target.with_suffix(".partial.tsv").unlink(missing_ok=True)
    print(f"-> {target}")


if __name__ == "__main__":
    main()
