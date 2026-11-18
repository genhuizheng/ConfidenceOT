"""Splatter figures and result tables (PLAN.md): one PDF + PNG per panel, nbt style.

    python ot_depth_benchmark_20261117/fig_splatter.py

Reads results/ot_depth_benchmark_20261117/splatter/{METRICS,TRANSITIONS,GATES}_{s1,s2}; writes figures/ and tables/
there. Every number is a mean over the 8 replicates; error bars are the standard error over replicates.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "realdata_benchmark" / "v3_grid" / "figures"))
import nbt_style as ns  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

R = REPO / "results" / "ot_depth_benchmark_20261117" / "splatter"
FIG, TAB = R / "figures", R / "tables"
SOLVERS = ["independent", "emd", "sinkhorn", "uot", "partial", "confidenceot"]
LABEL = {"independent": "Independent a×b", "emd": "Exact balanced OT", "sinkhorn": "Entropic balanced OT",
         "uot": "Entropic UOT", "partial": "Partial OT (m 0.8)", "confidenceot": "ConfidenceOT M4-E"}
COL = {"independent": ns.NEUTRAL, "emd": "#37536B", "sinkhorn": "#6E8E7D", "uot": "#B4702A", "partial": "#7D3C98",
       "confidenceot": "#C0392B"}
COSTNAME = {"C1": "C1 rank-cosine (main)", "C2": "C2 log1p-PCA (standard)"}
THIN = ("Group1", "Group2")


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def s2_shares() -> pd.DataFrame:
    """Per replicate x arm x f x cost x solver: shares of the transported mass of Group1/2 source cells (correct, wrong
    thinned, wrong unthinned) and their transported fraction."""
    t = pd.read_csv(R / "TRANSITIONS_s2.tsv", sep="\t")
    g = pd.read_csv(R / "GATES_s2.tsv.gz", sep="\t")
    k = ["replicate", "arm", "f", "cost", "solver"]
    x = t[t.source_group.isin(THIN)].copy()
    x["cat"] = np.where(x.source_group == x.target_group, "correct",
                        np.where(x.target_group.isin(THIN), "wrong_dissimilar_depth", "wrong_similar_depth"))
    s = x.groupby(k + ["cat"]).mass.sum().unstack(fill_value=0.0)
    tot = s.sum(1)
    marg = g[(g.side == "source") & (g.cost == "C1")].groupby(["replicate", "arm", "f"]).group.apply(lambda v: v.isin(THIN).mean())
    s = s.div(tot, axis=0)
    s["transported_fraction"] = tot.values / marg.reindex(pd.MultiIndex.from_frame(tot.reset_index()[["replicate", "arm", "f"]])).values
    return s.reset_index()


def line_panel(d, y, title, name, ylab, ref=None, solvers=SOLVERS, ylim=(0, 1)):
    fig, ax = plt.subplots(figsize=(2.2, 1.9))
    for s in solvers:
        e = d[d.solver == s].groupby("f")[y].agg(["mean", "sem"]).reset_index()
        ax.errorbar(e.f, e["mean"], yerr=e["sem"], color=COL[s], lw=ns.LW_MAIN if s == "confidenceot" else 0.8,
                    marker="o", ms=2.4, capsize=1.2, elinewidth=ns.LW_ERR, label=LABEL[s],
                    ls="--" if s == "independent" else "-")
    if ref is not None:
        ax.axhline(ref, **ns.REF_KW)
    ax.set_xscale("log"); ax.set_xticks([0.1, 0.25, 0.5, 1.0]); ax.set_xticklabels(["0.1", "0.25", "0.5", "1"])
    ax.invert_xaxis()
    ax.set_xlabel("Thinning factor f of the thinned groups"); ax.set_ylabel(ylab); ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ns.tidy(ax)
    save(fig, name)


def legend_panel():
    fig, ax = plt.subplots(figsize=(1.6, 1.2))
    for s in SOLVERS:
        ax.plot([], [], color=COL[s], lw=1.0, marker="o", ms=2.4, ls="--" if s == "independent" else "-", label=LABEL[s])
    ax.legend(frameon=False, loc="center"); ax.axis("off")
    save(fig, "legend_solvers")


def heat(M, title, name, fmt="{:.2f}", vmin=None, vmax=None, cmap="viridis", clabel=""):
    fig, ax = plt.subplots(figsize=(0.42 * M.shape[1] + 1.4, 0.24 * M.shape[0] + 0.9))
    im = ax.imshow(M.values.astype(float), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(M.columns, rotation=45, ha="right")
    ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(M.index)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=ns.FS_ANNOT,
                        color="white" if (vmax is not None and v < (vmin + vmax) / 2) else ns.INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label(clabel); cb.outline.set_linewidth(ns.LW_AXIS)
    ax.set_title(title)
    save(fig, name)


def main():
    ns.apply()
    TAB.mkdir(parents=True, exist_ok=True)
    # ------------------------------------------------------------------------------------------- S2 key figure
    s = s2_shares()
    s.groupby(["arm", "f", "cost", "solver"]).agg(["mean", "sem"]).round(4).to_csv(TAB / "S2_group12_mass_shares.tsv", sep="\t")
    for cost in ("C1", "C2"):
        for arm in ("conflict", "aligned"):
            d = s[(s.arm == arm) & (s.cost == cost)]
            line_panel(d, "correct", f"{arm.capitalize()} arm, {COSTNAME[cost]}", f"S2_{arm}_{cost}_correct_share",
                       "Share of transported mass\non the correct group")
            if arm == "conflict":
                line_panel(d, "wrong_similar_depth", f"Conflict arm, {COSTNAME[cost]}", f"S2_conflict_{cost}_wrong_similar_depth_share",
                           "Share on wrong groups\nwith source-like depth")
                line_panel(d, "transported_fraction", f"Conflict arm, {COSTNAME[cost]}", f"S2_conflict_{cost}_transported_fraction",
                           "Transported fraction of\nGroup1/2 source mass", solvers=[x for x in SOLVERS if x in ("uot", "partial", "confidenceot")])
    m2 = pd.read_csv(R / "METRICS_s2.tsv", sep="\t")
    for cost in ("C1", "C2"):
        for arm in ("conflict", "aligned", "global"):
            d = m2[(m2.arm == arm) & (m2.cost == cost)]
            line_panel(d, "depth_corr_count", f"{arm.capitalize()} arm, {COSTNAME[cost]}", f"S2_{arm}_{cost}_depth_corr",
                       "Coupling-weighted corr.\nof log nCount", ref=0, ylim=(-0.1, 1))
        d = m2[(m2.arm == "global") & (m2.cost == cost)]
        line_panel(d, "correct_mass", f"Global arm, {COSTNAME[cost]}", f"S2_global_{cost}_correct_share",
                   "Share of transported mass\non the correct group")
    g2 = pd.read_csv(R / "GATES_s2.tsv.gz", sep="\t")
    rej = g2[(g2.side == "target") & ~g2.should_reject].groupby(["replicate", "arm", "f", "cost"]).rejected.mean().reset_index()
    rej["solver"] = "confidenceot"
    for cost in ("C1", "C2"):
        d = rej[(rej.arm == "global") & (rej.cost == cost)]
        line_panel(d, "rejected", f"Global arm, {COSTNAME[cost]}", f"S2_global_{cost}_target_rejected_fraction",
                   "Rejected fraction of\n(shared) target cells", solvers=["confidenceot"], ylim=(0, 0.5))
    legend_panel()
    # association vs harm: depth correlation against correct share, every S2 fit
    fig, ax = plt.subplots(figsize=(2.4, 2.1))
    mk = {"conflict": "o", "aligned": "s", "global": "^"}
    for arm in ("conflict", "aligned", "global"):
        for cost, fill in (("C1", True), ("C2", False)):
            d = m2[(m2.arm == arm) & (m2.cost == cost) & (m2.solver != "independent")].groupby(["f", "solver"])[["depth_corr_count", "correct_mass"]].mean().reset_index()
            ax.scatter(d.depth_corr_count, d.correct_mass, s=9, marker=mk[arm], facecolors=[COL[x] for x in d.solver] if fill else "none",
                       edgecolors=[COL[x] for x in d.solver], lw=0.6)
    for arm in mk:
        ax.scatter([], [], marker=mk[arm], s=9, facecolors="none", edgecolors=ns.INK, lw=0.6, label=f"{arm} (filled C1, open C2)")
    ax.legend(frameon=False, loc="lower left")
    ax.set_xlabel("Coupling-weighted corr. of log nCount"); ax.set_ylabel("Correct-group mass share (all cells)")
    ax.set_title("Depth association is not harm"); ns.tidy(ax)
    save(fig, "S2_association_vs_harm")
    # ------------------------------------------------------------------------------------------- S1 native
    m1 = pd.read_csv(R / "METRICS_s1.tsv", sep="\t")
    m1.groupby(["level", "case", "cost", "solver"]).mean(numeric_only=True).round(4).drop(columns=["replicate"]).to_csv(TAB / "S1_metrics_mean.tsv", sep="\t")
    lv_order = ["L0_matched", "L1_depth", "L2_depth_dropout", "A5_imbalance_only", "A4_perturb_only", "L3_batch", "L4_batch_perturb", "L5_batch_perturb_imbalance"]
    for cost in ("C1", "C2"):
        d = m1[(m1.case == "all_shared") & (m1.cost == cost)]
        for q, title, lim in (("depth_corr_count", "Coupling-weighted corr. of log nCount", (0, 0.3)),
                              ("depth_corr_count_within", "Within-group corr. of log nCount", (0, 0.3)),
                              ("correct_mass", "Correct-group mass share", (0, 1))):
            M = d.groupby(["level", "solver"])[q].mean().unstack().reindex(index=lv_order, columns=SOLVERS)
            M.columns = [LABEL[c] for c in M.columns]
            heat(M, f"All shared, {COSTNAME[cost]}", f"S1_allshared_{cost}_{q}", vmin=lim[0], vmax=lim[1], clabel=title)
        # unmatched populations: transported share of the unmatched population's mass (gate-free) and rejected fraction (gate)
        for case, side in (("population_lost", "source"), ("population_emerged", "target")):
            d2 = m1[(m1.case == case) & (m1.cost == cost)]
            M = d2.groupby(["level", "solver"])[f"transported_share_unmatched_{side}"].mean().unstack().reindex(index=lv_order, columns=SOLVERS)
            M.columns = [LABEL[c] for c in M.columns]
            heat(M, f"{case}, {COSTNAME[cost]}", f"S1_{case}_{cost}_transported_share_unmatched", vmin=0, vmax=1,
                 clabel=f"Transported share of the\nunmatched {side} population")
        R_ = []
        for case in ("all_shared", "population_lost", "population_emerged", "populations_disjoint"):
            d2 = m1[(m1.case == case) & (m1.cost == cost) & (m1.solver == "confidenceot")]
            for side in ("source", "target"):
                for kind in ("unmatched", "shared"):
                    col = f"rejected_fraction_{kind}_{side}"
                    if d2[col].notna().any():
                        R_.append(d2.groupby("level")[col].mean().rename(f"{case}: {kind} {side}"))
        M = pd.concat(R_, axis=1).reindex(lv_order)
        heat(M, f"ConfidenceOT gate, {COSTNAME[cost]}", f"S1_gate_{cost}_rejected_fraction", vmin=0, vmax=1, clabel="Rejected fraction")
        # depth-driven rejection: Spearman(log nCount, rejected) among shared cells
        M = m1[(m1.cost == cost) & (m1.solver == "confidenceot")].groupby(["level", "case"])[["rejection_depth_corr_source", "rejection_depth_corr_target"]].mean()
        M.round(3).to_csv(TAB / f"S1_gate_{cost}_rejection_depth_corr.tsv", sep="\t")
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
