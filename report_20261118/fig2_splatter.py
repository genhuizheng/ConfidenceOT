"""Figure 2 - Splatter: does ConfidenceOT work under technical variation? (read-only on every result table)

Sources (audited 2026-11-18, scratchpad audit/splatter.md):
  N = 5,000 grid  results/v3_grid_20261104/splatter/{GRID_SUMMARY_v3tacc.tsv, RESULTS_grid_v3tacc.tsv}
  N = 200 fixed-cost benchmark  results/ot_depth_benchmark_20261117/splatter/METRICS_s1.tsv (S1 native),
      splatter/tables/S2_group12_mass_shares.tsv and splatter/METRICS_s2.tsv (S2, L0), splatter_s2b/METRICS_s2.tsv (S2b,
      L1-L5; rerun after the pair_id fix e5bc48a). Discarded folders (discarded_run1_pot_uot, discarded_pairid_collision,
      couplings_s1_INVALID_pairid_collision) are never read.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402
import glyphs as g  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
GRID = REPO / "results/v3_grid_20261104/splatter"
OTDB = REPO / "results/ot_depth_benchmark_20261117"
OUT = fs.OUT / "Figure2"
DATA = OUT / "data"
MAIN = "ranknm256-midrank|none|scale|cos|pca50"
LEVELS5 = ["L1_depth", "L2_depth_dropout", "L3_batch", "L4_batch_perturb", "L5_batch_perturb_imbalance"]
LEVELS6 = ["L0_matched"] + LEVELS5
LV = {l: l.split("_")[0] for l in LEVELS6}

ARMS = [  # (config id, legend label, colour)
    (MAIN, "Main configuration", fs.METHOD["ConfidenceOT"]),
    ("ranknm256-midrank|none|noscale|cos|pca50", "Main, no gene scaling", "#37536B"),
    ("ranknm256-production|none|noscale|cos|pca30", "Earlier default (PCA 30)", "#6E8E7D"),
    ("logcpm|none|noscale|cos|pca30", "log CPM, cosine", "#BCA177"),
    ("logcpm|none|scale|sqeuc|pca50", "log CPM, Euclidean *", "#8FB0C6"),
    ("ranknm256-midrank|none|scale|sqeuc|pca50", "Main, Euclidean cost", "#444444"),
]


def load():
    DATA.mkdir(parents=True, exist_ok=True)
    g = pd.read_csv(GRID / "GRID_SUMMARY_v3tacc.tsv", sep="\t")
    r = pd.read_csv(GRID / "RESULTS_grid_v3tacc.tsv", sep="\t",
                    usecols=["level", "replicate", "case", "config_id", "arm", "method", "side", "cells", "to_reject",
                             "rejected", "f1", "false_rejection_rate", "m4e_inference_valid", "status"])
    r = r[(r.config_id == MAIN) & (r.method == "M4-E") & (r.arm == "within_side_split+mnncovlocal")]
    assert (r.status == "ok").all()
    # b: Case 2 F1 (effective readout, mean of 6 fits) +- replicate SD; per-direction F1; Case 1 FRR per side
    b = g[g.config_id == MAIN].set_index("level").loc[LEVELS5, ["case2_f1_eff", "case2_f1_eff_rep_sd", "case1_frr_eff", "reject_all_f1"]]
    lost = r[(r.case == "Case2_source_unmatched") & (r.side == "source")].groupby("level").f1.agg(["mean", "std", "count"])
    emer = r[(r.case == "Case2_target_unmatched") & (r.side == "target")].groupby("level").f1.agg(["mean", "std", "count"])
    c1 = r[r.case == "Case1_all_shared"].groupby(["level", "side"]).false_rejection_rate.agg(["mean", "std", "count"]).unstack("side")
    b = b.join(lost.add_prefix("lost_src_f1_")).join(emer.add_prefix("emerged_tgt_f1_"))
    b["case1_frr_source"] = c1[("mean", "source")]; b["case1_frr_target"] = c1[("mean", "target")]
    b["case1_n_fits_per_side"] = c1[("count", "source")]
    b.to_csv(DATA / "b_main_config_by_level.tsv", sep="\t")
    flags = r.groupby(["level", "case"]).m4e_inference_valid.apply(lambda s: int((~s.astype(bool)).sum())).unstack()
    flags.to_csv(DATA / "b_calibration_flag_false_counts.tsv", sep="\t")
    # c: all 296 Case 2 F1 by level + named arms
    c = g.pivot(index="config_id", columns="level", values="case2_f1_eff")[LEVELS5]
    assert len(c) == 296
    c.to_csv(DATA / "c_case2_f1_all296.tsv", sep="\t")
    # d: N = 200 S1, cost C1, Case 2 population lost, L0-L5, 8 replicates
    m = pd.read_csv(OTDB / "splatter/METRICS_s1.tsv", sep="\t")
    m = m[(m.cost == "C1") & (m.case == "population_lost") & m.level.isin(LEVELS6) & m.solver.isin(["sinkhorn", "uot", "confidenceot"])].copy()
    assert (m.status == "ok").all()
    co = m.solver == "confidenceot"
    m["total_mass"] = m.transported_mass
    m["accuracy"] = np.where(co, 1 - m.false_asserted_share, m.correct_mass)
    m["correct_mass_abs"] = np.where(co, m.asserted_mass * (1 - m.false_asserted_share), m.transported_mass * m.correct_mass)
    m["retained_mass"] = np.where(co, m.asserted_mass, np.nan)
    keep = ["total_mass", "retained_mass", "accuracy", "correct_mass_abs", "rejected_fraction_unmatched_source",
            "rejected_fraction_shared_source", "rejected_fraction_shared_target"]
    d = m.groupby(["solver", "level"])[keep].agg(["mean", "sem", "count"])
    d.columns = ["_".join(x) for x in d.columns]
    d.to_csv(DATA / "d_s1_C1_population_lost.tsv", sep="\t")
    # e: S2 conflict arm at L0 (group-1/2 source mass shares), mean / sem over 8 replicates
    s2 = pd.read_csv(OTDB / "splatter/tables/S2_group12_mass_shares.tsv", sep="\t", header=[0, 1], index_col=[0, 1, 2, 3])
    s2.columns = ["_".join(x) for x in s2.columns]
    s2 = s2.reset_index()
    e = s2[s2.arm == "conflict"].copy()
    e.to_csv(DATA / "e_s2_conflict_L0_group12_shares.tsv", sep="\t", index=False)
    # f: global arm (every population shared), ConfidenceOT on C1, S2 (L0) + S2b (L1-L5)
    gl = []
    for f_ in (OTDB / "splatter/METRICS_s2.tsv", OTDB / "splatter_s2b/METRICS_s2.tsv"):
        t = pd.read_csv(f_, sep="\t")
        t = t[(t.arm == "global") & (t.solver == "confidenceot") & (t.cost == "C1") & t.level.isin(LEVELS6)]
        assert (t.status == "ok").all()
        gl.append(t)
    gl = pd.concat(gl)
    gl["depth_ratio"] = gl.median_ncount_source / gl.median_ncount_target
    fdat = gl.groupby(["level", "f"]).agg(depth_ratio=("depth_ratio", "mean"), rej_src=("rejected_fraction_shared_source", "mean"),
                                          rej_src_sem=("rejected_fraction_shared_source", "sem"), rej_tgt=("rejected_fraction_shared_target", "mean"),
                                          asserted=("asserted_mass", "mean"), n=("replicate", "nunique")).reset_index()
    fdat.to_csv(DATA / "f_global_arm_confidenceot_C1.tsv", sep="\t", index=False)
    return b, c, d, e, fdat, flags


# ------------------------------------------------------------------ panels
def panel_a(sh):
    x0, y0, x1, y1 = 0.0, 3.62, 3.05, 5.40
    sh.panel("a", x0, y0, x1, y1, "Benchmark design (Splatter, known truth)")
    ax = sh.ax(0.05, 3.64, 2.98, 1.46); ax.set_xlim(0, 2.98); ax.set_ylim(0, 1.46); fs.blank(ax)
    fs.schematic_tag(ax, 1.0, 1.04)
    # ---- cases: which of the 5 groups each side holds
    xs0, xt0, dx = 0.84, 1.24, 0.066
    g.label(ax, xs0 + 2 * dx, 1.36, "Source", color=fs.SRC, size=fs.FS_ANNOT)
    g.label(ax, xt0 + 2 * dx, 1.36, "Target", color=fs.TGT, size=fs.FS_ANNOT)
    cases = [("Case 1", "all shared", [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
             ("Case 2", "population lost", [1, 2, 3, 4, 5], [1, 2, 3, 4]),
             ("Case 2", "population emerged", [1, 2, 3, 4], [1, 2, 3, 4, 5]),
             ("Case 3", "none shared", [1, 2], [3, 4, 5])]
    for i, (c, sub, s_, t_) in enumerate(cases):
        y = 1.13 - i * 0.29
        g.label(ax, 0.0, y + 0.05, c, ha="left", size=fs.FS_ANNOT)
        g.label(ax, 0.0, y - 0.07, sub, ha="left", color=fs.SUBTLE)
        for x_, groups in ((xs0, s_), (xt0, t_)):
            for k in range(1, 6):
                if k in groups:
                    g.cell(ax, x_ + (k - 1) * dx, y, 0.027, fs.POP[k - 1], nucleus=False)
                else:
                    ax.add_patch(__import__("matplotlib.patches", fromlist=["Circle"]).Circle(
                        (x_ + (k - 1) * dx, y), 0.027, fc="white", ec=fs.NEUTRAL, lw=0.5, ls=(0, (1.2, 1.0)), zorder=4))
        if i < 3:
            g.rule(ax, 0.0, 1.55, y - 0.145)
    for k in range(5):
        g.cell(ax, 0.50 + k * 0.21, 0.03, 0.026, fs.POP[k], nucleus=False)
        g.label(ax, 0.545 + k * 0.21, 0.03, f"G{k + 1}", ha="left", color=fs.POP[k])
    g.label(ax, 0.0, 0.03, "Groups:", ha="left", color=fs.SUBTLE)
    # ---- technical levels, cumulative, each with an icon
    xl, xi, xt = 1.64, 1.90, 2.06
    g.label(ax, xl, 1.38, "Technical levels (cumulative)", ha="left", size=fs.FS_ANNOT)
    lv = [("L1", "target depth x 0.5"), ("L2", "+ dropout"), ("L3", "+ batch effect"),
          ("L4", "+ Group 1 shifted\non the target"), ("L5", "+ unequal target\nproportions")]
    for i, (l, t) in enumerate(lv):
        y = 1.06 - i * 0.235
        g.label(ax, xl, y, l, ha="left", size=fs.FS_ANNOT)
        s = 0.075
        if l == "L1":
            from matplotlib.patches import Rectangle as R
            ax.add_patch(R((xi - 0.06, y - s), 0.05, 2 * s, fc=g.tint(fs.SRC, 0.8), ec="none"))
            ax.add_patch(R((xi + 0.01, y - s), 0.05, s, fc=g.tint(fs.TGT, 0.8), ec="none"))
            ax.plot([xi - 0.08, xi + 0.08], [y - s, y - s], color=fs.INK, lw=0.45)
        elif l == "L2":
            g.matrix_icon(ax, xi, y, s, fs.TGT, rows=4, cols=5, seed=4, holes=0.4)
        elif l == "L3":
            rng = np.random.default_rng(3)
            ax.scatter(xi - 0.04 + rng.normal(0, 0.018, 10), y + 0.03 + rng.normal(0, 0.018, 10), s=2.4, color=fs.SRC, lw=0)
            ax.scatter(xi + 0.04 + rng.normal(0, 0.018, 10), y - 0.03 + rng.normal(0, 0.018, 10), s=2.4, color=fs.TGT, lw=0)
        elif l == "L4":
            g.cell(ax, xi - 0.04, y, 0.028, fs.POP[0], nucleus=False)
            g.arrow(ax, xi - 0.005, y, xi + 0.025, y, color=fs.POP[0], lw=0.6, ms=4)
            g.cell(ax, xi + 0.06, y, 0.028, fs.POP[0], nucleus=False)
        else:
            from matplotlib.patches import Rectangle as R
            xx = xi - 0.08
            for k, f in enumerate([0.40, 0.25, 0.15, 0.12, 0.08]):
                ax.add_patch(R((xx, y - 0.03), 0.16 * f, 0.06, fc=fs.POP[k], ec="white", lw=0.3)); xx += 0.16 * f
        g.label(ax, xt, y, t, ha="left")
    g.label(ax, 2.98, 1.25, "N = 5,000 per side, 3 replicates", ha="right", color=fs.SUBTLE)


def panel_b(sh, b):
    x0, y0, x1, y1 = 3.10, 3.62, 7.20, 5.40
    sh.panel("b", x0, y0, x1, y1, "Matching and rejection, main configuration (N = 5,000)")
    xs = np.arange(5)
    ax = sh.ax(3.55, 4.02, 1.75, 0.92); fs.tidy(ax)
    ax.axhline(b.reject_all_f1.mean(), **fs.REF_KW)
    ax.text(4.25, b.reject_all_f1.mean() + 0.03, "reject all", fontsize=fs.FS_SMALL, color=fs.SUBTLE, ha="right", va="bottom")
    ax.errorbar(xs, b.case2_f1_eff, yerr=b.case2_f1_eff_rep_sd, color=fs.METHOD["ConfidenceOT"], lw=1.2, marker="o", ms=3.2,
                capsize=1.6, elinewidth=0.6, label="Case 2 (mean of both directions)", zorder=4)
    ax.plot(xs, b.lost_src_f1_mean, color=fs.SRC, lw=0.8, marker="s", ms=2.4, label="lost population (source side)")
    ax.plot(xs, b.emerged_tgt_f1_mean, color=fs.TGT, lw=0.8, marker="^", ms=2.6, label="emerged population (target side)")
    ax.set_xticks(xs, [LV[l] for l in LEVELS5]); ax.set_ylim(0, 1.05); ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("Rejection F1"); ax.set_title("Unmatched cells found", fontsize=fs.FS_TITLE, pad=11)
    ax.legend(loc="center left", fontsize=fs.FS_SMALL, handlelength=1.4, borderaxespad=0.1, bbox_to_anchor=(0.0, 0.58))
    for i, l in enumerate(LEVELS5):
        ax.text(i, 1.09, f"{b.case2_f1_eff.iloc[i]:.2f}", ha="center", va="bottom", fontsize=fs.FS_SMALL,
                color=fs.METHOD["ConfidenceOT"], clip_on=False)
    ax.text(-0.45, 1.09, "Case 2:", ha="right", va="bottom", fontsize=fs.FS_SMALL, color=fs.METHOD["ConfidenceOT"], clip_on=False)
    ax2 = sh.ax(5.75, 4.02, 1.30, 0.92); fs.tidy(ax2)
    w = 0.36
    ax2.bar(xs - w / 2, b.case1_frr_source, w, color=fs.SRC, label="source")
    ax2.bar(xs + w / 2, b.case1_frr_target, w, color=fs.TGT, label="target")
    ax2.set_xticks(xs, [LV[l] for l in LEVELS5]); ax2.set_ylim(0, 0.3); ax2.set_yticks([0, 0.1, 0.2, 0.3])
    ax2.set_ylabel("False rejection rate"); ax2.set_title("Case 1: nothing to reject", fontsize=fs.FS_TITLE, pad=11)
    ax2.legend(loc="upper left", fontsize=fs.FS_SMALL, handlelength=0.9, borderaxespad=0.1)
    sh.text(3.55, 3.68, "Each point: mean of 3 replicates (Case 2 line: 2 directions x 3 replicates, bar = SD of replicate means).",
            fontsize=fs.FS_SMALL, color=fs.SUBTLE, va="bottom")


def panel_c(sh, c):
    x0, y0, x1, y1 = 0.0, 1.86, 3.05, 3.58
    sh.panel("c", x0, y0, x1, y1, "Preprocessing sensitivity (296 configurations)")
    ax = sh.ax(0.42, 2.26, 1.40, 1.02); fs.tidy(ax)
    xs = np.arange(5)
    rng = np.random.default_rng(0)
    for i, l in enumerate(LEVELS5):
        v = c[l].dropna().to_numpy()
        ax.scatter(i + rng.uniform(-0.22, 0.22, len(v)), v, s=1.2, color=fs.NEUTRAL, lw=0, alpha=0.8, zorder=1)
    for cid, lab, col in ARMS[::-1]:
        lw, ms = (1.3, 3.0) if cid == MAIN else (0.75, 2.0)
        ax.plot(xs, c.loc[cid, LEVELS5], color=col, lw=lw, marker="o", ms=ms, zorder=5 if cid == MAIN else 3)
    ax.set_xticks(xs, [LV[l] for l in LEVELS5]); ax.set_ylim(0, 1.05); ax.set_yticks([0, 0.5, 1]); ax.set_ylabel("Case 2 rejection F1")
    lax = sh.ax(1.92, 2.26, 1.10, 1.02); fs.blank(lax); lax.set_xlim(0, 1); lax.set_ylim(0, 1)
    for i, (cid, lab, col) in enumerate(ARMS):
        yy = 0.97 - i * 0.13
        lax.plot([0, 0.13], [yy, yy], color=col, lw=1.3 if cid == MAIN else 0.8)
        lax.text(0.17, yy, lab, fontsize=fs.FS_SMALL, va="center")
    lax.scatter([0.065], [0.97 - 6 * 0.13], s=4, color=fs.NEUTRAL); lax.text(0.17, 0.97 - 6 * 0.13, "All 296", fontsize=fs.FS_SMALL, va="center")
    lax.text(0.0, 0.97 - 7.2 * 0.13, "* closest grid arm to log1p-PCA", fontsize=fs.FS_SMALL, va="center", color=fs.SUBTLE)
    sh.text(0.42, 1.94, f"L5 median over 296 = {c['L5_batch_perturb_imbalance'].median():.2f}; main = {c.loc[MAIN, 'L5_batch_perturb_imbalance']:.2f}"
            " (selected on L5)", fontsize=fs.FS_SMALL, color=fs.SUBTLE, va="center")


def panel_d(sh, d):
    x0, y0, x1, y1 = 3.10, 1.86, 7.20, 3.58
    sh.panel("d", x0, y0, x1, y1, "Balanced OT vs UOT vs ConfidenceOT (same cost, same epsilon; N = 200)")
    sol = [("sinkhorn", "Balanced OT", fs.METHOD["Balanced OT"]), ("uot", "UOT", fs.METHOD["UOT"]),
           ("confidenceot", "ConfidenceOT", fs.METHOD["ConfidenceOT"])]
    mets = [("total_mass", "Transported mass"), ("accuracy", "Accuracy"),
            ("correct_mass_abs", "Correct mass"), ("rej", "Rejected cells")]
    xs = np.arange(6)
    w, gap, left = 0.78, 0.20, 3.42
    for j, (k, title) in enumerate(mets):
        ax = sh.ax(left + j * (w + gap), 2.26, w, 0.80); fs.tidy(ax)
        if k == "rej":
            t = d.loc["confidenceot"].reindex(LEVELS6)
            ax.plot(xs, t.rejected_fraction_unmatched_source_mean, color=fs.REJECT, lw=1.0, marker="o", ms=2.4)
            ax.plot(xs, t.rejected_fraction_shared_source_mean, color=fs.REJECT, lw=0.8, ls=fs.DASH, marker="o", ms=2.0, mfc="white")
            ax.text(2.6, 0.80, "lost Group 5", fontsize=fs.FS_SMALL, color=fs.REJECT, ha="center", va="bottom")
            ax.text(2.5, 0.10, "shared cells", fontsize=fs.FS_SMALL, color=fs.REJECT, ha="center", va="bottom")
        else:
            for s, lab, col in sol:
                t = d.loc[s].reindex(LEVELS6)
                ax.errorbar(xs, t[f"{k}_mean"], yerr=t[f"{k}_sem"], color=col, lw=1.0 if s == "confidenceot" else 0.8, marker="o",
                            ms=2.2, capsize=0, elinewidth=0.5, label=lab)
            if k == "total_mass":
                t = d.loc["confidenceot"].reindex(LEVELS6)
                ax.plot(xs, t.retained_mass_mean, color=fs.METHOD["ConfidenceOT"], lw=0.7, ls=fs.DASH)
        ax.set_ylim(0, 1.05); ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["0", "0.5", "1"] if j == 0 else [])
        ax.set_xticks(xs, [str(i) for i in range(6)]); ax.set_xlabel("Level L", labelpad=1)
        ax.set_title(title, fontsize=fs.FS_SMALL + 0.4, pad=2)
        if j == 0:
            ax.set_ylabel("Mean of 8 replicates")
    hx = sh.ax(3.42, 3.24, 3.70, 0.10); fs.blank(hx); hx.set_xlim(0, 1); hx.set_ylim(0, 1)
    for i, (s, lab, col) in enumerate(sol):
        hx.plot([i * 0.27, i * 0.27 + 0.045], [0.5, 0.5], color=col, lw=1.2); hx.text(i * 0.27 + 0.06, 0.5, lab, fontsize=fs.FS_SMALL, va="center")
    hx.plot([0.81, 0.855], [0.5, 0.5], color=fs.METHOD["ConfidenceOT"], lw=0.7, ls=fs.DASH)
    hx.text(0.87, 0.5, "retained block", fontsize=fs.FS_SMALL, va="center")
    sh.text(3.42, 1.94, "Case 2 (Group 5 lost), main cost, epsilon 0.1; accuracy = correct / transported mass"
            , fontsize=fs.FS_SMALL, color=fs.SUBTLE, va="center")


def panel_e(sh, e):
    x0, y0, x1, y1 = 0.0, 0.0, 3.55, 1.82
    sh.panel("e", x0, y0, x1, y1, "Failure 1: depth can win over biology (constructed conflict, N = 200)")
    ax = sh.ax(0.45, 0.44, 1.55, 0.98); fs.tidy(ax)
    fvals = [1.0, 0.5, 0.25, 0.1]; xs = np.arange(4)

    def row(cost, solver):
        t = e[(e.cost == cost) & (e.solver == solver)].set_index("f").reindex(fvals)
        return t
    lines = [("C2", "uot", "UOT, log1p-PCA cost", fs.METHOD["UOT"], "-"), ("C2", "sinkhorn", "Balanced OT, log1p-PCA cost", fs.METHOD["Balanced OT"], "-"),
             ("C1", "uot", "UOT, main cost", fs.METHOD["UOT"], fs.DASH)]
    for cost, s, lab, col, ls in lines:
        t = row(cost, s)
        ax.errorbar(xs, t.correct_mean, yerr=t.correct_sem, color=col, ls=ls, lw=1.0, marker="o", ms=2.4, capsize=0, elinewidth=0.5, label=lab)
    t = row("C2", "uot")
    ax.plot(xs, t.wrong_similar_depth_mean, color=fs.REJECT, lw=0.9, marker="x", ms=3, label="UOT, log1p-PCA: mass on wrong,\ndepth-similar groups")
    ax.set_xticks(xs, ["1", "0.5", "0.25", "0.1"]); ax.set_xlabel("Target Groups 1-2: fraction of counts kept")
    ax.set_ylim(0, 1.05); ax.set_yticks([0, 0.5, 1]); ax.set_ylabel("Share of mass\n(source Groups 1-2)")
    ax.text(3.05, t.correct_mean.iloc[-1] + 0.03, f"{t.correct_mean.iloc[-1]:.2f}", fontsize=fs.FS_SMALL, color=fs.METHOD["UOT"], va="bottom")
    ax.text(3.05, t.wrong_similar_depth_mean.iloc[-1] - 0.02, f"{t.wrong_similar_depth_mean.iloc[-1]:.2f}", fontsize=fs.FS_SMALL, color=fs.REJECT, va="top")
    lax = sh.ax(2.08, 0.44, 1.45, 0.98); fs.blank(lax); lax.set_xlim(0, 1); lax.set_ylim(0, 1)
    h, l = ax.get_legend_handles_labels()
    order = [1, 2, 3, 0]
    lax.legend([h[i] for i in order], [l[i].replace("UOT, log1p-PCA: mass", "UOT, log1p-PCA: mass") for i in order], loc="upper left",
               fontsize=fs.FS_SMALL, handlelength=1.5, borderaxespad=0, labelspacing=0.6, title="Share on the correct group:",
               title_fontsize=fs.FS_SMALL, alignment="left")
    sh.text(0.45, 0.06, "Groups 3-5 keep source-like depth (the technical lure); mean +- s.e.m. of 8 replicates.",
            fontsize=fs.FS_SMALL, color=fs.SUBTLE, va="center")


def panel_f(sh, fdat):
    x0, y0, x1, y1 = 3.60, 0.0, 7.20, 1.82
    sh.panel("f", x0, y0, x1, y1, "Failure 2: a very large depth gap (all populations shared, N = 200)")
    ax = sh.ax(4.05, 0.44, 1.75, 0.98); fs.tidy(ax)
    cols = {"L0_matched": "#9A9A9A", "L1_depth": "#5B8DB8", "L2_depth_dropout": "#7FB07F", "L3_batch": "#C9A54B",
            "L4_batch_perturb": "#B07AA1", "L5_batch_perturb_imbalance": "#7A9E9F"}
    for l in LEVELS6:
        t = fdat[fdat.level == l].sort_values("depth_ratio")
        ax.plot(t.depth_ratio, t.rej_src, color=cols[l], lw=0.9, marker="o", ms=2.4, label=LV[l])
        ax.plot(t.depth_ratio, t.rej_tgt, color=cols[l], lw=0.6, ls=fs.DASH, marker="o", ms=1.8, mfc="white")
    ax.set_xscale("log"); ax.set_xticks([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"]); ax.minorticks_off()
    ax.set_xlabel("Source / target median depth"); ax.set_ylim(-0.02, 1.05); ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("Shared cells rejected\n(all are false)")
    lax = sh.ax(5.95, 0.44, 1.20, 0.98); fs.blank(lax); lax.set_xlim(0, 1); lax.set_ylim(0, 1)
    for i, l in enumerate(LEVELS6):
        lax.plot([0, 0.12], [0.95 - i * 0.12] * 2, color=cols[l], lw=1.0); lax.text(0.16, 0.95 - i * 0.12, LV[l], fontsize=fs.FS_SMALL, va="center")
    lax.plot([0, 0.12], [0.95 - 6.3 * 0.12] * 2, color=fs.INK, lw=0.9); lax.text(0.16, 0.95 - 6.3 * 0.12, "source side", fontsize=fs.FS_SMALL, va="center")
    lax.plot([0, 0.12], [0.95 - 7.3 * 0.12] * 2, color=fs.INK, lw=0.6, ls=fs.DASH); lax.text(0.16, 0.95 - 7.3 * 0.12, "target side", fontsize=fs.FS_SMALL, va="center")
    sh.text(4.05, 0.06, "Whole target thinned (f = 1, 0.5, 0.25, 0.1); ConfidenceOT, main cost; mean of 8 replicates.",
            fontsize=fs.FS_SMALL, color=fs.SUBTLE, va="center")


def write_text(b, c, d, e, fdat, flags):
    L5 = "L5_batch_perturb_imbalance"
    dl = d.loc["confidenceot"]
    t_e = e[(e.cost == "C2") & (e.solver == "uot")].set_index("f")
    t_eb = e[(e.cost == "C2") & (e.solver == "sinkhorn")].set_index("f")
    big = fdat[(fdat.f == 0.1) & (fdat.level != "L0_matched")]
    cap = f"""# Figure 2 | Splatter: what works, what fails, and under which conditions

**a**, Benchmark design. Splatter 1.30.0 simulations with 5 populations (Groups 1-5) and 10,000 genes. Case 1: all
populations on both sides. Case 2: Group 5 removed from the target (population lost, source cells should be rejected) or
from the source (population emerged, target cells should be rejected). Case 3: no population shared. Technical levels
are cumulative: L1 target library size x 0.5; L2 + dropout on both sides; L3 + batch effect; L4 + target-only shift of
Group 1 (keeps its label); L5 + target proportions 40/25/15/12/8%. The depth ratio, the Group 1 shift, the L5 proportions
and the paired count sampler are custom; all other parameters are Splatter defaults. N = 5,000 cells per side, 3
replicates. Schematic.

**b**, Main configuration `{MAIN}` (ConfidenceOT M4-E, calibration v3). Left: rejection F1 for Case 2 (red; mean of 2
directions x 3 replicates, error bar = SD of the 3 replicate means) and for each direction (mean of 3 replicates). Dashed
line: F1 of rejecting every cell. Case 2 F1 = {', '.join(f'{v:.2f}' for v in b.case2_f1_eff)} at L1-L5. The L5 drop comes
from the emerged direction (target F1 {b.emerged_tgt_f1_mean.iloc[-1]:.2f}): Group 5 is only 8% of the target and shared
target cells are also rejected. Right: Case 1 false rejection rate per side (mean of 3 replicates); L5 source
{b.case1_frr_source.iloc[-1]:.3f}, target {b.case1_frr_target.iloc[-1]:.3f}. The held-out calibration check
(`calibration_m4e_inference_valid`) is False for {int(flags.values.sum())} of the 60 fits; it is reported, not used to exclude fits.

**c**, Case 2 F1 for all 296 configurations of the N = 5,000 grid (grey; 37 encodings x PCA 30/50 x gene scaling x cosine /
squared Euclidean cost) with six configurations highlighted. The main configuration was selected as the best at L5 Case 2
({c.loc[MAIN, L5]:.3f}; the L5 median over 296 is {c[L5].median():.3f}); it was not selected on any real data set and is
0.01-0.03 below other configurations at L1-L3. Its L5 lead over the second configuration (0.009) is smaller than the
replicate SD (0.127). The cost (cosine vs squared Euclidean) changes the result most.

**d**, The same cost matrix (main configuration, with the v3 correction) and the same epsilon (0.1) for entropic balanced
OT, UOT (lambda 1) and ConfidenceOT (UOT backbone + rejection), Case 2 population lost, levels L0-L5, N = 200 per side,
mean +- s.e.m. of 8 replicates. Total transported mass: balanced 1 by construction, UOT and ConfidenceOT about
{dl.total_mass_mean.mean():.2f}; dashed: ConfidenceOT mass on retained x retained cells. Conditional accuracy: correct mass /
transported mass (ConfidenceOT: on the retained block). Correct matching mass: transported x accuracy. Rejected fraction:
ConfidenceOT only (lost Group 5 vs shared source cells); balanced OT and UOT make no decision. At N = 200 the gate is
weaker than at N = 5,000 (panel b), so these values are not comparable to panel b.

**e**, Constructed depth-identity conflict at L0 (N = 200, 8 replicates): target Groups 1-2 binomially thinned to a
fraction f of their counts, so their correct partners become shallow while wrong Groups 3-5 keep source-like depth. Share
of the mass of source Groups 1-2 that goes to the correct group (and, red, to wrong depth-similar groups) for UOT and
balanced OT on a scanpy log1p-PCA cost (normalize_total, log1p, 2,000 HVG, scale, PCA 50, squared Euclidean) and UOT on
the main cost. At f = 0.1, UOT on log1p-PCA puts {t_e.loc[0.1, 'correct_mean']:.2f} on the correct group and
{t_e.loc[0.1, 'wrong_similar_depth_mean']:.2f} on wrong, depth-similar groups; balanced OT on the same cost keeps
{t_eb.loc[0.1, 'correct_mean']:.2f}. ConfidenceOT on this cost equals UOT and does not reject (its gate stays open).

**f**, Global depth gap with every population shared (L0-L5, N = 200, 8 replicates; whole target thinned to f = 1, 0.5,
0.25, 0.1). Fraction of shared source (solid) and target (dashed) cells rejected by ConfidenceOT on the main cost; every
rejection here is false. At f = 0.1 on L1-L5 (source/target median depth about 20x) ConfidenceOT rejects
{big.rej_src.min():.2f}-{big.rej_src.max():.2f} of the source and asserts almost no mass. The N = 5,000 grid has no arm
this extreme (its depth ratio is 2x).
"""
    (OUT / "caption.md").write_text(cap, encoding="utf-8")
    src = f"""# Figure 2 sources

| Panel | Source table | Filter / derivation | Derived table |
|---|---|---|---|
| a | design constants from `realdata_benchmark/splatter_n5000/20_generate_n5000.R`, `21_build_conditions_n5000.py`, `benchmark/20_build_conditions.py` | — | — |
| b | `results/v3_grid_20261104/splatter/GRID_SUMMARY_v3tacc.tsv`; `RESULTS_grid_v3tacc.tsv` | config `{MAIN}`, method M4-E, arm within_side_split+mnncovlocal; Case 2 F1 = `case2_f1_eff` (+ `_rep_sd`); per direction keyed by level + replicate + case (never pair_id) | `data/b_main_config_by_level.tsv`, `data/b_calibration_flag_false_counts.tsv` |
| c | `GRID_SUMMARY_v3tacc.tsv` | `case2_f1_eff`, all 296 configs x L1-L5 | `data/c_case2_f1_all296.tsv` |
| d | `results/ot_depth_benchmark_20261117/splatter/METRICS_s1.tsv` | cost C1, case population_lost, solvers sinkhorn / uot / confidenceot, L0-L5; ConfidenceOT accuracy = 1 - false_asserted_share on the retained block | `data/d_s1_C1_population_lost.tsv` |
| e | `results/ot_depth_benchmark_20261117/splatter/tables/S2_group12_mass_shares.tsv` | arm conflict, L0 | `data/e_s2_conflict_L0_group12_shares.tsv` |
| f | `results/ot_depth_benchmark_20261117/splatter/METRICS_s2.tsv` (L0) + `splatter_s2b/METRICS_s2.tsv` (L1-L5, after fix e5bc48a) | arm global, solver confidenceot, cost C1 | `data/f_global_arm_confidenceot_C1.tsv` |

Excluded: `splatter/discarded_run1_pot_uot/`, `splatter_s2b/discarded_pairid_collision/`,
`splatter/couplings_s1_INVALID_pairid_collision/`; superseded runs `results/splatter_n5000_20261101`,
`results/splatter_n5000_default_20261103`, `results/realdata_benchmark_20261031`, `results/BEST_PREPROCESSING_20261103.md`.
"""
    (OUT / "sources.md").write_text(src, encoding="utf-8")
    notes = {
        "a": {"title": "Benchmark design", "compared": "Simulated source and target samples with known populations: all shared, one lost, one emerged, none shared; five cumulative technical levels.",
              "observe": "Every cell has a known answer: keep it (it has a counterpart) or reject it (it does not).", "meaning": "We can score the decision exactly, and see which technical change breaks it."},
        "b": {"title": "How well does ConfidenceOT find unmatched cells?", "compared": "Rejection F1 for the unmatched population (Case 2) and false rejection when nothing should be rejected (Case 1), L1-L5, main configuration, N = 5,000.",
              "observe": f"F1 {b.case2_f1_eff.iloc[0]:.2f}-{b.case2_f1_eff.iloc[3]:.2f} at L1-L4 and {b.case2_f1_eff.iloc[-1]:.2f} at L5; the L5 drop is the rare emerged population (target F1 {b.emerged_tgt_f1_mean.iloc[-1]:.2f}).",
              "meaning": "Rejection is reliable under depth, dropout, batch and a shifted population; it weakens when the unmatched population is rare and the proportions differ."},
        "c": {"title": "Does preprocessing matter?", "compared": "Case 2 F1 for 296 preprocessing configurations; six highlighted.",
              "observe": "Most configurations agree at L1-L3; they separate at L4-L5; a squared Euclidean cost on the same representation fails everywhere.",
              "meaning": "The main configuration is the L5 best on this simulation, not a universal optimum; the cost metric is the most important choice."},
        "d": {"title": "Balanced OT vs UOT vs ConfidenceOT", "compared": "Same cost, same epsilon; four different quantities: transported mass, accuracy, correct mass, rejected fraction (N = 200).",
              "observe": "Balanced OT moves all mass, including the lost population; UOT moves less mass but decides nothing; ConfidenceOT adds an explicit rejection of the lost population.",
              "meaning": "Transported mass and accuracy are not the same thing; only ConfidenceOT says which cells have no counterpart."},
        "e": {"title": "Failure 1: depth can win over biology", "compared": "Mass from source Groups 1-2 when their true partners are made shallow and wrong groups keep similar depth.",
              "observe": f"On a log1p-PCA cost, UOT sends {t_e.loc[0.1, 'wrong_similar_depth_mean']:.0%} of this mass to wrong, depth-similar groups at f = 0.1; the main cost stays near its baseline.",
              "meaning": "With a depth-sensitive cost, unbalanced matching can follow sequencing depth instead of biology, and the gate does not catch it."},
        "f": {"title": "Failure 2: a very large depth gap", "compared": "All populations shared; the whole target thinned; any rejection is false.",
              "observe": "Up to about 4-8x depth difference the false rejection stays limited at L0; at about 20x ConfidenceOT rejects almost the whole source.",
              "meaning": "A large global depth gap can be read as 'no counterpart'; depth must be checked before reading rejection biologically."},
    }
    (OUT / "panel_notes.json").write_text(json.dumps(notes, indent=1), encoding="utf-8")


def main():
    fs.apply()
    b, c, d, e, fdat, flags = load()
    sh = fs.Sheet(7.2, 5.4, "Figure2")
    panel_a(sh); panel_b(sh, b); panel_c(sh, c); panel_d(sh, d); panel_e(sh, e); panel_f(sh, fdat)
    sh.save()
    write_text(b, c, d, e, fdat, flags)


if __name__ == "__main__":
    main()
