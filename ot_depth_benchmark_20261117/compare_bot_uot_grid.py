"""Balanced (BOT) vs unbalanced (UOT) backbone of ConfidenceOT over the full v3 grid on Splatter (PLAN.md post-hoc
section of 2026-11-18, "Balanced-backbone ConfidenceOT"; grid rerun requested by the user 2026-11-18).

    python ot_depth_benchmark_20261117/compare_bot_uot_grid.py

Inputs (copied from TACC, gitignored): results/ot_depth_benchmark_20261117/bot_uot_grid/{uot,bot}/GRID_SUMMARY_v3tacc.tsv
(UOT: v3_grid_20261104, BOT: v3_grid_bot_20261118; same worklist, same frozen calibration v3, M4-E, acceptance 0.99,
effective readout). The comparison is paired by configuration x level. Nothing is selected: every configuration is
shown, the metrics are the grid's own (Case 1 FRR, Case 2 Rejection F1 = the primary metric, Case 3 unmatched recall,
Case 3 false asserted mass), and no composite is formed.
Writes bot_uot_grid/{tables,figures}.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "realdata_benchmark" / "v3_grid" / "figures"))
import nbt_style as ns  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

R = HERE.parent / "results" / "ot_depth_benchmark_20261117" / "bot_uot_grid"
TAB, FIG = R / "tables", R / "figures"
LEVELS = ["L1_depth", "L2_depth_dropout", "L3_batch", "L4_batch_perturb", "L5_batch_perturb_imbalance"]
METRICS = ["case1_frr_eff", "case2_f1_eff", "case3_recall_eff", "case3_false_asserted_mass"]
NAMED = {"C1 main": "ranknm256-midrank|none|scale|cos|pca50", "C3 grid best (local)": "ranknm512-expected|none|noscale|cos|pca30",
         "C4 logCPM cos": "logcpm|none|noscale|cos|pca30", "baseline ranknm256": "ranknm256-production|none|noscale|cos|pca30",
         "logCPM scale sqeuc pca50": "logcpm|none|scale|sqeuc|pca50"}
COL = {"cos": "#C0392B", "sqeuc": "#37536B"}


def load() -> pd.DataFrame:
    k = ["config_id", "level"]
    u = pd.read_csv(R / "uot" / "GRID_SUMMARY_v3tacc.tsv", sep="\t")[k + METRICS]
    b = pd.read_csv(R / "bot" / "GRID_SUMMARY_v3tacc.tsv", sep="\t")[k + METRICS]
    m = u.merge(b, on=k, suffixes=("_uot", "_bot"), how="outer", indicator=True)
    if (m._merge != "both").any():
        print("unpaired rows:", m._merge.value_counts().to_dict())
    m = m[m._merge == "both"].drop(columns="_merge")
    m["distance"] = m.config_id.str.split("|").str[3]
    for x in METRICS:
        m[f"{x}_diff"] = m[f"{x}_bot"] - m[f"{x}_uot"]
    return m


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def scatter(m, metric, level, label):
    d = m[m.level == level]
    fig, ax = plt.subplots(figsize=(1.9, 1.9))
    lo = np.nanmin([d[f"{metric}_uot"].min(), d[f"{metric}_bot"].min(), 0]); hi = max(1.0, np.nanmax(d[f"{metric}_bot"]))
    ax.plot([lo, hi], [lo, hi], color=ns.RULE, lw=ns.LW_REF, zorder=1)
    for dist, c in COL.items():
        e = d[d.distance == dist]
        ax.scatter(e[f"{metric}_uot"], e[f"{metric}_bot"], s=5, color=c, alpha=0.7, lw=0, label=dist, zorder=2)
    ax.set_xlabel(f"{label}, UOT backbone"); ax.set_ylabel(f"{label}, balanced backbone")
    ax.set_title(level.replace("_", " "))
    ax.set_xlim(lo - 0.02, hi + 0.02); ax.set_ylim(lo - 0.02, hi + 0.02)
    ax.legend(frameon=False, loc="upper left", handletextpad=0.2)
    save(fig, f"scatter_{metric}_{level}")


def main():
    ns.apply()
    TAB.mkdir(parents=True, exist_ok=True)
    m = load()
    m.to_csv(TAB / "PAIRED_bot_vs_uot_by_config_level.tsv", sep="\t", index=False)
    # distribution of the paired difference, by level and distance
    rows = []
    for (lv, dist), d in m.groupby(["level", "distance"]):
        r = {"level": lv, "distance": dist, "n_configs": len(d)}
        for x in METRICS:
            v = d[f"{x}_diff"].dropna()
            r |= {f"{x}_uot_median": d[f"{x}_uot"].median(), f"{x}_bot_median": d[f"{x}_bot"].median(),
                  f"{x}_diff_median": v.median(), f"{x}_diff_q10": v.quantile(0.1), f"{x}_diff_q90": v.quantile(0.9)}
        rows.append(r)
    s = pd.DataFrame(rows)
    s["level"] = pd.Categorical(s.level, LEVELS); s = s.sort_values(["level", "distance"])
    s.round(3).to_csv(TAB / "DIFF_by_level_distance.tsv", sep="\t", index=False)
    # named configurations, both backbones, every level
    n = m[m.config_id.isin(NAMED.values())].copy()
    n.insert(0, "name", n.config_id.map({v: k for k, v in NAMED.items()}))
    n["level"] = pd.Categorical(n.level, LEVELS)
    cols = ["name", "config_id", "level"] + [f"{x}_{b}" for x in METRICS for b in ("uot", "bot")]
    n.sort_values(["name", "level"])[cols].round(3).to_csv(TAB / "NAMED_configs_bot_vs_uot.tsv", sep="\t", index=False)
    # over L1-L5 means per configuration, both backbones (no ranking beyond sorting by the UOT F1 for reading)
    o = m.groupby("config_id")[[f"{x}_{b}" for x in METRICS for b in ("uot", "bot")]].mean()
    o["worst_case2_f1_uot"] = m.groupby("config_id").case2_f1_eff_uot.min()
    o["worst_case2_f1_bot"] = m.groupby("config_id").case2_f1_eff_bot.min()
    o.sort_values("case2_f1_eff_uot", ascending=False).round(3).to_csv(TAB / "OVERALL_L1L5_mean_by_config.tsv", sep="\t")
    for lv in LEVELS:
        scatter(m, "case2_f1_eff", lv, "Case 2 F1")
        scatter(m, "case1_frr_eff", lv, "Case 1 FRR")
        scatter(m, "case3_false_asserted_mass", lv, "Case 3 false asserted")
    print(s[["level", "distance", "n_configs", "case2_f1_eff_uot_median", "case2_f1_eff_bot_median", "case2_f1_eff_diff_median", "case1_frr_eff_diff_median",
             "case3_false_asserted_mass_diff_median"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
