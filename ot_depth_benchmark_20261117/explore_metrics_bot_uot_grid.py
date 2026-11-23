"""EXPLORATORY metrics on the BOT vs UOT v3 grid (user 2026-11-23: "再探索多个指标进行尝试").

    python ot_depth_benchmark_20261117/explore_metrics_bot_uot_grid.py

Not the benchmark readout. The readout and selection stay the protocol's (Case 1 FRR, Case 2 Rejection F1 primary,
Case 3 unmatched recall; Pareto rule; RESULTS_BOT_UOT_GRID.md). Everything here is computed from columns the grid already
writes (GRID_SUMMARY_v3tacc.tsv, RESULTS_grid_v3tacc.tsv, FIT_METRICS_v3tacc.tsv.gz); the one threshold used (a side
counts as whole-side rejected at rejected_fraction >= 0.99) is summarize_local.py's own one_side_all rule.

Explored, per backbone x level x distance (median over configurations) and for the main configuration C1:
  E1 Case 2 precision and recall          the two parts of F1 (summarize_local columns)
  E2 Case 2 shared-population FRR         false rejection of the shared populations inside a Case 2 unit
  E3 Case 2 F1 above reject-all           case2_f1_eff - reject_all_f1 (reject_all_f1 is the protocol's reference value)
  E4 Case 3 false asserted mass           mass asserted between non-corresponding populations
  E5 Case 1/2 whole-side rejection        share of Case 1/2 fits where one side is >= 99% rejected (GRID_V3_REPORT's
                                          failure indicator)
  E6 replicate SD of Case 2 F1            stability over the three Splatter replicates
  E7 configuration-order agreement        Spearman of Case 2 F1 between backbones over the 296 configurations, per level
  E8 M4-E / M4-R agreement                share of fits where the two estimators reject the same number of cells
  E9 fit cost                             M4-E / M4-R fit seconds, outer convergence, cycle rate, transported mass
Writes bot_uot_grid/tables/exploratory/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R = HERE.parent / "results" / "ot_depth_benchmark_20261117" / "bot_uot_grid"
OUT = R / "tables" / "exploratory"
LEVELS = ["L1_depth", "L2_depth_dropout", "L3_batch", "L4_batch_perturb", "L5_batch_perturb_imbalance"]
C1 = "ranknm256-midrank|none|scale|cos|pca50"
B = {"uot": "UOT", "bot": "balanced"}


def summary() -> pd.DataFrame:
    g = []
    for b, name in B.items():
        x = pd.read_csv(R / b / "GRID_SUMMARY_v3tacc.tsv", sep="\t")
        x.insert(0, "backbone", name); g.append(x)
    g = pd.concat(g, ignore_index=True)
    g["distance"] = g.config_id.str.split("|").str[3]
    g["E3_f1_above_reject_all"] = g.case2_f1_eff - g.reject_all_f1
    return g


def per_fit() -> pd.DataFrame:
    cols = ["level", "replicate", "case", "pair_id", "config_id", "method", "side", "rejected", "rejected_fraction"]
    f = []
    for b, name in B.items():
        x = pd.read_csv(R / b / "RESULTS_grid_v3tacc.tsv", sep="\t", usecols=cols)
        x.insert(0, "backbone", name); f.append(x)
    return pd.concat(f, ignore_index=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    g = summary()
    E = {"case2_precision": "E1_precision", "case2_recall": "E1_recall", "case2_other_frr": "E2_shared_frr",
         "E3_f1_above_reject_all": "E3_f1_above_reject_all", "case3_false_asserted_mass": "E4_false_asserted_mass",
         "case2_f1_eff_rep_sd": "E6_f1_rep_sd"}
    med = g.groupby(["level", "distance", "backbone"])[list(E)].median().rename(columns=E)
    # E5 whole-side rejection in Case 1/2, M4-E
    f = per_fit()
    e = f[(f.method == "M4-E") & f.case.str.startswith(("Case1", "Case2"))]
    w = e.groupby(["backbone", "level", "replicate", "case", "pair_id", "config_id"]).rejected_fraction.max().ge(0.99).rename("whole_side").reset_index()
    w["distance"] = w.config_id.str.split("|").str[3]
    med = med.join(w.groupby(["level", "distance", "backbone"]).whole_side.mean().rename("E5_case12_whole_side_share"))
    med.round(3).to_csv(OUT / "E_median_by_level_distance_backbone.tsv", sep="\t")
    c1 = g[g.config_id == C1].set_index(["level", "backbone"])[list(E)].rename(columns=E)
    c1 = c1.join(w[w.config_id == C1].groupby(["level", "backbone"]).whole_side.mean().rename("E5_case12_whole_side_share"))
    c1.round(3).to_csv(OUT / "E_C1_by_level_backbone.tsv", sep="\t")
    # E7 order agreement between backbones
    p = g.pivot_table(index=["level", "config_id"], columns="backbone", values="case2_f1_eff").reset_index()
    e7 = p.groupby("level").apply(lambda d: d[["UOT", "balanced"]].corr(method="spearman").iloc[0, 1]).rename("E7_spearman_f1_order")
    e7.round(3).to_csv(OUT / "E7_config_order_agreement.tsv", sep="\t")
    # E8 M4-E vs M4-R agreement on rejected counts
    k = ["backbone", "level", "replicate", "case", "pair_id", "config_id", "side"]
    q = f.pivot_table(index=k, columns="method", values="rejected").reset_index()
    q["same"] = q["M4-E"] == q["M4-R"]
    e8 = q.groupby(["backbone", "level"]).same.mean().rename("E8_m4e_m4r_same_rejected_count")
    e8.round(3).to_csv(OUT / "E8_m4e_m4r_agreement.tsv", sep="\t")
    # E9 fit cost
    fm = []
    for b, name in B.items():
        x = pd.read_csv(R / b / "FIT_METRICS_v3tacc.tsv.gz", sep="\t", low_memory=False,
                        usecols=["method", "fit_seconds", "outer_converged", "cycle_detected", "transported_mass"])
        x.insert(0, "backbone", name); fm.append(x)
    fm = pd.concat(fm)
    e9 = fm.groupby(["backbone", "method"]).agg(fits=("fit_seconds", "size"), fit_seconds_median=("fit_seconds", "median"),
                                                outer_converged=("outer_converged", "mean"), cycle_rate=("cycle_detected", "mean"),
                                                transported_mass_median=("transported_mass", "median"))
    e9.round(3).to_csv(OUT / "E9_fit_cost.tsv", sep="\t")
    pd.set_option("display.width", 250)
    print(med.round(3).to_string()); print(c1.round(3).to_string()); print(e7.round(3).to_string()); print(e8.round(3).to_string()); print(e9.round(3).to_string())


if __name__ == "__main__":
    main()
