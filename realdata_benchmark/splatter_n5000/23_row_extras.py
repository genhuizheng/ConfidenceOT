"""Per-row evaluation beyond 40_score.py for the N = 5,000 benchmark: coupling metrics of M4-E, M4-R and entropic
balanced OT on the same cost, and the DEG-B contrast (predicted rejected vs retained) against the simulator's truth.

    python realdata_benchmark/splatter_n5000/23_row_extras.py --fit-out FIT --manifest CONDITION_MANIFEST \
        --pair-index I --truth TRUTH_DIR --out OUT

FIT is a row directory of 10_tacc_worker.py (runs/<pair>/scope_all/budget_*/: cell_confidence.csv, coupling_m4e.npz,
coupling_m4r.npz, joint_pca_coordinates.csv.gz, run.json). The cost is rebuilt from the saved joint coordinates and
the run's cost_scale, so it is the run's own matrix. Truth: 21_build_conditions_n5000.py's population_truth.tsv and
deg_truth.tsv (observation ids are the h5ad cell names the gates carry).

coupling.tsv, one row per plan (M4-E, M4-R, balanced full, balanced on the M4-E retained x retained block):
  transported_mass; same_population_mass (mass on pairs of one group / transported: the conditional accuracy);
  supported_mass (absolute mass on same-group pairs of cells that should not be rejected, over the whole problem's
  unit mass); unsupported_mass (mass touching a should-reject cell); coverage_shared_source / _target (fraction of
  should-retain cells sending / receiving at least half their nominal mass); median normalised row entropy. Balanced
  OT has no gate: its full-mass accuracy is not comparable to a gated plan's conditional accuracy without the matched
  block, which is why both are given.
deg_b.tsv, per side, M4-E gate: rejected vs retained cells of that side only (never pooled across sides), original
  counts, library size 1e4 + log1p, scanpy Wilcoxon. Against the side's deg_truth (Case 2 sides that hold an
  unmatched population): AUROC and AUPRC of |score| for true_de, direction agreement on true-DE genes, Spearman of
  the log fold change with the true one on true-DE genes, overlap of the top 100 genes by |score| with the top 100 by
  |true log2FC|; always: rejected / retained counts, median nCount / nFeature per group and their AUC for rejected.
  Fewer than 20 cells in a group: not_estimable. A DEG agreement is descriptive and never by itself evidence that the
  gate is right.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from traditional_ot.balanced import balanced_ot

EPSILON = 0.1


def plan_metrics(P, src, tgt, name):
    n, m = P.shape
    same = src.group.to_numpy()[:, None] == tgt.group.to_numpy()[None, :]
    keep_s, keep_t = ~src.should_reject.to_numpy(bool), ~tgt.should_reject.to_numpy(bool)
    total = P.sum()
    supported = P[np.ix_(keep_s, keep_t)][same[np.ix_(keep_s, keep_t)]].sum()
    unsupported = total - P[np.ix_(keep_s, keep_t)].sum()
    rows = P.sum(1) * n; cols = P.sum(0) * m
    R = P / np.maximum(P.sum(1, keepdims=True), 1e-300)
    H = -(R * np.log(np.where(R > 0, R, 1))).sum(1) / np.log(m)
    return {"plan": name, "transported_mass": float(total),
            "same_population_mass": float(P[same].sum() / total) if total > 0 else np.nan,
            "supported_mass": float(supported), "unsupported_mass": float(unsupported),
            "coverage_shared_source": float(np.mean(rows[keep_s] >= 0.5)) if keep_s.any() else np.nan,
            "coverage_shared_target": float(np.mean(cols[keep_t] >= 0.5)) if keep_t.any() else np.nan,
            "row_entropy_median": float(np.median(H[P.sum(1) > 0])) if (P.sum(1) > 0).any() else np.nan}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fit-out", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--pair-index", type=int, required=True)
    p.add_argument("--truth", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if (a.out / "coupling.tsv").exists():
        sys.exit(f"{a.out}/coupling.tsv exists; not overwritten")
    a.out.mkdir(parents=True, exist_ok=True)
    row = pd.read_csv(a.manifest).iloc[a.pair_index]
    run = next((a.fit_out / "runs").glob(f"{row.pair_id}/scope_all/budget_*"))
    info = json.loads((run / "run.json").read_text())
    cc = pd.read_csv(run / "cell_confidence.csv")
    truth = pd.read_csv(a.truth / "population_truth.tsv", sep="\t")
    coords = pd.read_csv(run / "joint_pca_coordinates.csv.gz")
    pcs = [c for c in coords if c.startswith("PC")]
    side_name = {"primary": "source", "metastasis": "target"}
    coords["side"] = coords.side.map(side_name).fillna(coords.side)
    t_by = truth.set_index(["side", "observation_id"])
    X = {s: coords[coords.side == s] for s in ("source", "target")}
    meta = {s: t_by.loc[[(s, o) for o in X[s].observation_id]].reset_index() for s in ("source", "target")}
    A, B = X["source"][pcs].to_numpy(np.float64), X["target"][pcs].to_numpy(np.float64)
    C = (np.sum(A * A, 1)[:, None] + np.sum(B * B, 1)[None, :] - 2 * A @ B.T).clip(min=0) / float(info["cost_scale"])
    rows = []
    gates = {}
    for method in ("M4-E", "M4-R"):
        f = run / f"coupling_{method.lower().replace('-', '')}.npz"
        g = cc[cc.method == method].set_index(["side", "observation_id"])["rejected"]
        gates[method] = {s: ~g.loc[[(s, o) for o in X[s].observation_id]].to_numpy(bool) for s in ("source", "target")}
        if f.exists():
            rows.append({**plan_metrics(np.load(f)["coupling"], meta["source"], meta["target"], method),
                         "source_retained": float(gates[method]["source"].mean()), "target_retained": float(gates[method]["target"].mean())})
    bot = balanced_ot(C, epsilon=EPSILON)
    rows.append({**plan_metrics(bot.coupling, meta["source"], meta["target"], "balanced_full"), "converged": bool(bot.converged)})
    rs, rt = gates["M4-E"]["source"], gates["M4-E"]["target"]
    if rs.sum() >= 4 and rt.sum() >= 4:
        sub = balanced_ot(C[np.ix_(rs, rt)], epsilon=EPSILON)
        P = np.zeros_like(C); P[np.ix_(rs, rt)] = sub.coupling * (rs.sum() / len(rs))  # the block's share of the source mass
        rows.append({**plan_metrics(P, meta["source"], meta["target"], "balanced_on_M4E_retained_block"), "converged": bool(sub.converged)})
    else:
        rows.append({"plan": "balanced_on_M4E_retained_block", "status": "not_estimable: < 4 retained cells on a side"})
    pd.DataFrame(rows).to_csv(a.out / "coupling.tsv", sep="\t", index=False)

    deg = pd.read_csv(a.truth / "deg_truth.tsv", sep="\t")
    out_rows = []
    for s in ("source", "target"):
        data = ad.read_h5ad(row[f"{s}_h5ad"])
        data = data[X[s].observation_id.to_numpy()].copy()
        counts = np.asarray(data.X.sum(1)).ravel(); feats = np.asarray((data.X > 0).sum(1)).ravel()
        retained = gates["M4-E"][s]
        rec = {"side": s, "rejected_n": int((~retained).sum()), "retained_n": int(retained.sum()),
               "median_nCount_rejected": float(np.median(counts[~retained])) if (~retained).any() else np.nan,
               "median_nCount_retained": float(np.median(counts[retained])) if retained.any() else np.nan,
               "median_nFeature_rejected": float(np.median(feats[~retained])) if (~retained).any() else np.nan,
               "median_nFeature_retained": float(np.median(feats[retained])) if retained.any() else np.nan}
        if 0 < retained.mean() < 1:
            rec["auc_nCount_rejected"] = float(roc_auc_score(~retained, counts)); rec["auc_nFeature_rejected"] = float(roc_auc_score(~retained, feats))
        if min(rec["rejected_n"], rec["retained_n"]) < 20:
            out_rows.append({**rec, "status": "not_estimable: < 20 cells in a group"}); continue
        sc.pp.normalize_total(data, target_sum=1e4); sc.pp.log1p(data)
        data.obs["g"] = np.where(retained, "retained", "rejected")
        sc.tl.rank_genes_groups(data, "g", groups=["rejected"], reference="retained", method="wilcoxon")
        res = sc.get.rank_genes_groups_df(data, "rejected").set_index("names")
        rec.update(status="computed", median_score=float(res.scores.median()), positive_share=float((res.scores > 0).mean()))
        tr = deg[deg.side == s]
        if len(tr):
            tr = tr.set_index("gene_id").reindex(res.index)
            y = tr.true_de.fillna(False).to_numpy(bool); sabs = res.scores.abs().to_numpy()
            if 0 < y.mean() < 1:
                rec.update(true_de_genes=int(y.sum()), deg_auroc=float(roc_auc_score(y, sabs)), deg_auprc=float(average_precision_score(y, sabs)),
                           deg_auprc_baseline=float(y.mean()),
                           direction_agreement=float(np.mean(np.sign(res.logfoldchanges.to_numpy()[y]) == np.sign(tr.true_log2fc.to_numpy()[y]))),
                           effect_spearman=float(spearmanr(res.logfoldchanges.to_numpy()[y], tr.true_log2fc.to_numpy()[y])[0]),
                           top100_overlap=len(set(res.index[np.argsort(-sabs)[:100]]) & set(tr.true_log2fc.abs().sort_values(ascending=False).index[:100])) / 100)
        out_rows.append(rec)
    pd.DataFrame(out_rows).to_csv(a.out / "deg_b.tsv", sep="\t", index=False)
    print(f"extras written to {a.out}")


if __name__ == "__main__":
    main()
