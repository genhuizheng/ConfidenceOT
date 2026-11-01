"""Biological validity audit of the LTL331 run (local, after the TACC download). Reads the frozen OT outputs; writes
only under OUT. Nothing is refitted here (the composition check that refits is 08_composition_sensitivity.py).

    python ltl331_nepc/07_validity_audit.py --run-root RUN_ROOT --converted CONVERTED --out OUT

Per label (ranknm256_noscale_ds_cos, logcpm), every pair, both sides, M4-E unless stated:

calibration_validity.tsv  every pair_metrics row's calibration flags; M4-E inference valid is the condition for a
                          pair to enter a formal statement, invalid pairs are marked, never pooled silently.
m4r_failure_mode.tsv      M4-R convergence: outer converged, cycle detected / length, outer iterations, the M4-R
                          calibration flags, and its rejected fractions beside M4-E's.
retention_by_type.tsv     retained / rejected counts and fractions per coarse_celltype with Wilson 95% intervals.
ne_effect.tsv             NE against the other three types pooled and against each type: difference in retained
                          fraction with a cell bootstrap 95% interval (1,000 resamples, stratified by group); the
                          Mantel-Haenszel odds ratio of retention over depth quintiles; logistic regressions of
                          retention on NE + log depth, and on NE + log depth + the OT-free kNN cost (below), with
                          Wald 95% intervals. Cells are the units (one tumour per time point): intervals describe
                          cell-sampling uncertainty in this tumour, not biological replication. Requires >= 20 NE and
                          >= 20 comparison cells.
knn_baseline_cells.tsv.gz per cell: the OT gate, the OT-free costs and the kNN-baseline gate.
knn_baseline_by_type.tsv  kNN-baseline retention by type and its agreement (kappa) with the OT gate.
                          OT-free RNA similarity: the cost OT used (reproduced from the run's own representation,
                          checked against run.json's cost scale / median / max) between each cell and the other side:
                          the nearest-neighbour cost and the mean of the k = 10 smallest. kNN baseline gate: a cell is
                          retained if its k-mean cost to the other side is <= the 0.99 quantile of the same quantity
                          inside its own side (a random half against the other half, the within-side split idea
                          without OT), the other side subsampled to the same reference size m = min(half own side,
                          other side) so the two distances are taken against equally many cells.
representation_check.tsv  reproduced scale / median / max against run.json.
coupling_enrichment.tsv   coupling mass between source and target types (full P and the retained block) against the
                          target-abundance-preserving permutation null (target type labels shuffled, 1,000 times):
                          observed / null mean, z, two-sided empirical p.
arm_robustness.tsv        cell-type level (type rejected fractions and NE minus non-NE retention per pair-side in the
                          two labels: sign agreement) against cell level (kappa on the same cells).
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.stats.contingency_tables import StratifiedTable

from common import prepare_joint_representation
from confidenceot import Preprocessing
from confidenceot.preprocessing import median_pair_scale, squared_euclidean

warnings.filterwarnings("ignore")
TYPES = ["PRAD", "Basal", "Other", "NE"]
LABELS = ["ranknm256_noscale_ds_cos", "logcpm"]
MANIFESTS = {"ranknm256_noscale_ds_cos": "ds/pair_manifest_downsampled.csv", "logcpm": "rawqc/pair_manifest_rawqc.csv"}
SEED = 20260831  # 02_run_pair.py default


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    z = 1.959964; p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def kappa(x, y) -> float:
    x, y = np.asarray(x, int), np.asarray(y, int)
    po = np.mean(x == y); pe = x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else np.nan


def local(path: str, run_root: Path) -> Path:
    return run_root / "inputs" / path.replace("\\", "/").split("/inputs/")[1]


def knn_mean(C: np.ndarray, k: int) -> np.ndarray:
    k = min(k, C.shape[1])
    return np.partition(C, k - 1, axis=1)[:, :k].mean(1)


def side_costs(X: np.ndarray, Y: np.ndarray, scale: float, k: int, rng) -> dict:
    """OT-free costs of the cells of X against Y, and the within-X split reference for the kNN gate."""
    C = squared_euclidean(X, Y).astype(np.float32) / scale
    out = {"nn_cost": C.min(1), "knn_cost": knn_mean(C, k)}
    n = len(X); half = rng.permutation(n); a, b = half[: n // 2], half[n // 2:]
    m = min(n // 2, len(Y))
    ref = rng.choice(len(Y), m, replace=False)
    cross = knn_mean(C[:, ref], k)
    wa = knn_mean(squared_euclidean(X[a], X[b[:m]]).astype(np.float32) / scale, k)
    wb = knn_mean(squared_euclidean(X[b], X[a[:m]]).astype(np.float32) / scale, k)
    threshold = np.quantile(np.concatenate([wa, wb]), 0.99)
    out.update(knn_cost_matched=cross, knn_threshold=threshold, knn_reference_m=m, knn_retained=cross <= threshold)
    return out


def logit(df: pd.DataFrame, terms: list[str]) -> dict:
    X = sm.add_constant(df[terms].astype(float)); y = df["retained"].astype(float)
    try:
        f = sm.Logit(y, X).fit(disp=0)
        b, se = f.params["is_ne"], f.bse["is_ne"]
        return {"or": np.exp(b), "or_lo": np.exp(b - 1.96 * se), "or_hi": np.exp(b + 1.96 * se), "ok": True}
    except Exception as error:  # perfect separation and the like: reported, not hidden
        return {"or": np.nan, "or_lo": np.nan, "or_hi": np.nan, "ok": f"failed: {type(error).__name__}"}


def ne_effect(u: pd.DataFrame, other: list[str], rng, n_boot: int = 1000) -> dict | None:
    ne = u[u.coarse_celltype == "NE"]; ot = u[u.coarse_celltype.isin(other)]
    if len(ne) < 20 or len(ot) < 20:
        return None
    rn, ro = ne.retained.to_numpy(float), ot.retained.to_numpy(float)
    boot = np.array([rng.choice(rn, len(rn)).mean() - rng.choice(ro, len(ro)).mean() for _ in range(n_boot)])
    v = pd.concat([ne, ot]).assign(is_ne=lambda d: (d.coarse_celltype == "NE").astype(int))
    v["log_depth"] = np.log(v.depth); v["q"] = pd.qcut(v.depth, 5, labels=False, duplicates="drop")
    tables = []
    for _, s in v.groupby("q"):
        t = np.array([[((s.is_ne == 1) & s.retained).sum(), ((s.is_ne == 1) & ~s.retained).sum()],
                      [((s.is_ne == 0) & s.retained).sum(), ((s.is_ne == 0) & ~s.retained).sum()]], float)
        if (t.sum(1) > 0).all():
            tables.append(t + 0.5)  # Haldane correction so an all-retained or all-rejected stratum still counts
    mh = StratifiedTable(tables) if tables else None
    l1 = logit(v, ["is_ne", "log_depth"]); l2 = logit(v, ["is_ne", "log_depth", "knn_cost"])
    return {"ne_n": len(ne), "comparison_n": len(ot), "ne_retained": rn.mean(), "comparison_retained": ro.mean(),
            "difference": rn.mean() - ro.mean(), "difference_lo": np.quantile(boot, 0.025), "difference_hi": np.quantile(boot, 0.975),
            "mh_or_depth_quintiles": mh.oddsratio_pooled if mh else np.nan,
            "mh_or_lo": mh.oddsratio_pooled_confint()[0] if mh else np.nan, "mh_or_hi": mh.oddsratio_pooled_confint()[1] if mh else np.nan,
            "logit_or_ne_given_depth": l1["or"], "logit_or_lo": l1["or_lo"], "logit_or_hi": l1["or_hi"], "logit_status": l1["ok"],
            "logit_or_ne_given_depth_knn": l2["or"], "logit_knn_or_lo": l2["or_lo"], "logit_knn_or_hi": l2["or_hi"], "logit_knn_status": l2["ok"]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--converted", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--permutations", type=int, default=1000)
    a = p.parse_args()
    if a.out.exists():
        sys.exit(f"{a.out} exists; not overwritten")
    a.out.mkdir(parents=True)
    rng = np.random.default_rng(20261031)

    obs = []
    for f in sorted((a.converted / "h5ad").glob("*.h5ad")):
        x = ad.read_h5ad(f, backed="r"); obs.append(x.obs[["coarse_celltype"]].astype(str)); x.file.close()
    ctype = pd.concat(obs)["coarse_celltype"]
    depth = pd.read_csv(a.run_root / "inputs" / "ds" / "predownsample_depth.csv.gz").set_index("observation_id")["predownsample_total_counts"]

    metrics, rep_check, cells, coup = [], [], [], []
    for label in LABELS:
        cfg = Preprocessing.from_label(label)
        man = pd.read_csv(a.run_root / "inputs" / MANIFESTS[label]); man = man[man.eligible.astype(bool)].reset_index(drop=True)
        for index, row in man.iterrows():
            pair = row.pair_id.replace("LTL331__", "")
            d = next((a.run_root / "runs" / "acc0.99" / "ltl331" / label / row.pair_id / "scope_malignant").glob("budget_*"))
            m = pd.read_csv(d / "pair_metrics.csv"); m.insert(0, "pair", pair); m.insert(0, "label", label); metrics.append(m)
            run = json.loads((d / "run.json").read_text())
            s = ad.read_h5ad(local(row.source_h5ad, a.run_root)); t = ad.read_h5ad(local(row.target_h5ad, a.run_root))
            sp, tp, _, prep = prepare_joint_representation(
                s, t, n_hvg=cfg.n_hvg or 2000, n_pcs=cfg.n_pcs or 30, seed=SEED + index, representation=cfg.normalisation,
                rank_top_n=cfg.rank_top_n, cost=cfg.cost, equalise_depth=cfg.equalise_depth, regress_out=cfg.regress_out,
                regress_on_ranks=cfg.regress_on_ranks, scale_genes=cfg.scale_genes, label_stem=cfg.label_stem)
            scale = median_pair_scale(sp, tp, rng=np.random.default_rng(SEED + index * 104729))
            C = squared_euclidean(sp, tp) / scale
            chk = {"label": label, "pair": pair, "scale": scale, "run_scale": run["cost_scale"], "median": float(np.median(C)),
                   "run_median": run["cost_median"], "max": float(C.max()), "run_max": run["cost_max"]}
            chk["reproduced"] = all(abs(chk[k] - chk["run_" + k]) <= 1e-4 * max(1.0, abs(chk["run_" + k])) for k in ("scale", "median", "max"))
            rep_check.append(chk)
            if not chk["reproduced"]:
                sys.exit(f"{label} {pair}: representation not reproduced {chk}")
            del C
            cc = pd.read_csv(d / "cell_confidence.csv", usecols=["method", "side", "observation_id", "rejected"])
            cc = cc[cc.method == "M4-E"]
            for side, X, Y, ids in (("source", sp, tp, s.obs_names), ("target", tp, sp, t.obs_names)):
                g = cc[cc.side == side].set_index("observation_id").loc[ids.astype(str)]
                r = side_costs(X, Y, scale, a.k, rng)
                cells.append(pd.DataFrame({"label": label, "pair": pair, "pair_index": index, "side": side, "observation_id": ids.astype(str),
                                           "coarse_celltype": ctype.loc[ids.astype(str)].to_numpy(), "depth": depth.loc[ids.astype(str)].to_numpy(),
                                           "retained": ~g.rejected.to_numpy(bool), "nn_cost": r["nn_cost"], "knn_cost": r["knn_cost"],
                                           "knn_cost_matched": r["knn_cost_matched"], "knn_threshold": r["knn_threshold"],
                                           "knn_reference_m": r["knn_reference_m"], "knn_retained": r["knn_retained"]}))
            # annotation-level transport against the abundance-preserving permutation null
            P = np.load(d / "coupling_m4e.npz")["coupling"]
            src = cc[cc.side == "source"].set_index("observation_id").loc[s.obs_names.astype(str)]
            tgt = cc[cc.side == "target"].set_index("observation_id").loc[t.obs_names.astype(str)]
            st = ctype.loc[s.obs_names.astype(str)].to_numpy(); tt = ctype.loc[t.obs_names.astype(str)].to_numpy()
            for block, ms, mt in (("full", np.ones(len(st), bool), np.ones(len(tt), bool)),
                                  ("retained", ~src.rejected.to_numpy(bool), ~tgt.rejected.to_numpy(bool))):
                Q = P[np.ix_(ms, mt)]; qs, qt = st[ms], tt[mt]
                if Q.sum() <= 0:
                    continue
                A = np.stack([(qs == x).astype(float) for x in TYPES]); B = np.stack([(qt == y).astype(float) for y in TYPES], 1)
                SQ = A @ Q  # 4 x n_target
                obs_m = SQ @ B
                null = np.empty((a.permutations, 4, 4))
                for i in range(a.permutations):
                    null[i] = SQ @ B[rng.permutation(len(qt))]
                mu, sd = null.mean(0), null.std(0)
                for i, x in enumerate(TYPES):
                    for j, y in enumerate(TYPES):
                        if A[i].sum() == 0 or B[:, j].sum() == 0:
                            continue
                        pe = (np.sum(np.abs(null[:, i, j] - mu[i, j]) >= abs(obs_m[i, j] - mu[i, j])) + 1) / (a.permutations + 1)
                        coup.append({"label": label, "pair": pair, "block": block, "source_type": x, "target_type": y,
                                     "source_n": int(A[i].sum()), "target_n": int(B[:, j].sum()), "mass": obs_m[i, j],
                                     "null_mean": mu[i, j], "observed_over_null": obs_m[i, j] / mu[i, j] if mu[i, j] > 0 else np.nan,
                                     "z": (obs_m[i, j] - mu[i, j]) / sd[i, j] if sd[i, j] > 0 else np.nan, "empirical_p": pe})
            print(f"{label} {pair}: representation reproduced, {len(s.obs_names)} x {len(t.obs_names)}", flush=True)

    metrics = pd.concat(metrics)
    cal = metrics[["label", "pair", "method", "calibration_feasible_cost_found", "calibration_m4e_clean", "calibration_m4e_inference_valid",
                   "calibration_m4r_clean", "calibration_valid_for_m4r", "rejection_cost"]]
    cal.to_csv(a.out / "calibration_validity.tsv", sep="\t", index=False)
    wide = metrics.pivot_table(index=["label", "pair"], columns="method",
                               values=["source_final_rejection_rate", "target_final_rejection_rate"]).round(4)
    wide.columns = [f"{v}_{m}" for v, m in wide.columns]
    m4r = metrics[metrics.method == "M4-R"][["label", "pair", "inner_converged", "outer_converged", "cycle_detected", "cycle_length",
                                              "outer_iterations", "total_inner_iterations", "calibration_m4r_clean", "calibration_valid_for_m4r"]]
    m4r.merge(wide.reset_index(), on=["label", "pair"]).to_csv(a.out / "m4r_failure_mode.tsv", sep="\t", index=False)
    pd.DataFrame(rep_check).to_csv(a.out / "representation_check.tsv", sep="\t", index=False)

    cells = pd.concat(cells, ignore_index=True)
    cells.to_csv(a.out / "knn_baseline_cells.tsv.gz", sep="\t", index=False, compression="gzip")
    valid = cal[cal.method == "M4-E"].set_index(["label", "pair"])["calibration_m4e_inference_valid"]
    rows, eff, base = [], [], []
    for (label, pair, side), u in cells.groupby(["label", "pair", "side"]):
        ok = bool(valid.loc[(label, pair)])
        for t_, w in u.groupby("coarse_celltype"):
            k_ = int(w.retained.sum()); lo, hi = wilson(k_, len(w))
            rows.append({"label": label, "pair": pair, "side": side, "m4e_inference_valid": ok, "coarse_celltype": t_, "n": len(w),
                         "retained_n": k_, "rejected_n": len(w) - k_, "retained_fraction": k_ / len(w), "retained_lo": lo, "retained_hi": hi})
            kb = int(w.knn_retained.sum()); blo, bhi = wilson(kb, len(w))
            base.append({"label": label, "pair": pair, "side": side, "coarse_celltype": t_, "n": len(w), "ot_retained_fraction": k_ / len(w),
                         "knn_retained_fraction": kb / len(w), "knn_retained_lo": blo, "knn_retained_hi": bhi,
                         "median_knn_cost": w.knn_cost.median(), "kappa_ot_vs_knn": kappa(w.retained, w.knn_retained)})
        for name, other in [("NE_vs_nonNE", ["PRAD", "Basal", "Other"])] + [(f"NE_vs_{x}", [x]) for x in ["PRAD", "Basal", "Other"]]:
            e = ne_effect(u, other, rng)
            if e is not None:
                eff.append({"label": label, "pair": pair, "side": side, "m4e_inference_valid": ok, "contrast": name, **e})
        base.append({"label": label, "pair": pair, "side": side, "coarse_celltype": "all", "n": len(u), "ot_retained_fraction": u.retained.mean(),
                     "knn_retained_fraction": u.knn_retained.mean(), "kappa_ot_vs_knn": kappa(u.retained, u.knn_retained),
                     "median_knn_cost": u.knn_cost.median()})
    pd.DataFrame(rows).to_csv(a.out / "retention_by_type.tsv", sep="\t", index=False)
    pd.DataFrame(eff).to_csv(a.out / "ne_effect.tsv", sep="\t", index=False)
    pd.DataFrame(base).to_csv(a.out / "knn_baseline_by_type.tsv", sep="\t", index=False)
    pd.DataFrame(coup).to_csv(a.out / "coupling_enrichment.tsv", sep="\t", index=False)

    rob = []
    r = pd.DataFrame(rows).pivot_table(index=["pair", "side", "coarse_celltype"], columns="label", values="retained_fraction")
    e = pd.DataFrame(eff); e = e[e.contrast == "NE_vs_nonNE"].pivot_table(index=["pair", "side"], columns="label", values="difference")
    w = cells.pivot_table(index=["pair", "side", "observation_id"], columns="label", values="retained")
    for (pair, side), u in w.groupby(level=[0, 1]):
        tr = r.loc[(pair, side)].dropna()
        rob.append({"pair": pair, "side": side,
                    "type_level_spearman": tr[LABELS[0]].corr(tr[LABELS[1]], method="spearman") if len(tr) >= 3 else np.nan,
                    "ne_minus_nonne_" + LABELS[0]: e.loc[(pair, side), LABELS[0]] if (pair, side) in e.index else np.nan,
                    "ne_minus_nonne_" + LABELS[1]: e.loc[(pair, side), LABELS[1]] if (pair, side) in e.index else np.nan,
                    "cell_level_kappa": kappa(u[LABELS[0]], u[LABELS[1]]), "cells": len(u)})
    rob = pd.DataFrame(rob)
    rob["ne_direction_agrees"] = np.sign(rob["ne_minus_nonne_" + LABELS[0]]) == np.sign(rob["ne_minus_nonne_" + LABELS[1]])
    rob.to_csv(a.out / "arm_robustness.tsv", sep="\t", index=False)
    print(f"done: {len(cells)} cell rows, {len(eff)} NE contrasts, {len(coup)} coupling cells; {a.out}")


if __name__ == "__main__":
    main()
