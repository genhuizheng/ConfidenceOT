"""Figure 3 - PDX (LTL331, Sar et al. 2026, GSE297328): can ConfidenceOT capture changes in cell states?

Read-only on the repository's result tables (frozen calibration v3 grid, main configuration
ranknm256-midrank|none|scale|cos|pca50, M4-E). Writes report_20261118/out/Figure3/ only.

    python -I report_20261118/fig3_pdx.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figstyle as st  # noqa: E402

REPO = HERE.parent
PDX = REPO / "results/v3_grid_20261104/figures/pdx/data"
GRID = REPO / "results/v3_grid_20261104/ltl331"
INTER = REPO / "results/ltl331_nepc/v3_intermediate_check_20261117"
DEPTH = REPO / "results/ltl331_nepc/tacc_20261031/ltl331_confidenceot/inputs/ds/predownsample_depth.csv.gz"
CLAIMS = REPO / "results/ltl331_nepc/state_biology_20261101/nonharmony_v4/PAPER_CLAIMS_CHECK.tsv"
SAMPLES = REPO / "results/ltl331_nepc/tacc_20261031/data/ltl331_GSE297328/converted/samples.tsv"
MAIN = "ranknm256-midrank|none|scale|cos|pca50"
KEY = "splatter_l5_best"
OUT = st.OUT / "Figure3"
DATA = OUT / "data"

SAMPLE_ORDER = ["w00_preCX_1", "w02_2wkCX_2", "w04_4wkCX_1", "w08_8wkCX_1", "w12_12wkCX_2", "w16_16wkCX_1",
                "w20_20wkCX_1_R", "w22_22wkCX_2_R"]
WEEK = dict(zip(SAMPLE_ORDER, [0, 2, 4, 8, 12, 16, 20, 22]))
TLAB = dict(zip(SAMPLE_ORDER, ["pre", "2", "4", "8", "12", "16", "20R", "22R"]))
STATES = ["AR-high PRAD", "AR-low PRAD", "pre-EMT", "intermediate EMT-like", "NEPC ASCL1+", "early NEPC ASCL1-",
          "terminal NEPC ASCL1-", "NEPC other"]
SHORT = {"AR-high PRAD": "AR-high", "AR-low PRAD": "AR-low", "pre-EMT": "pre-EMT", "intermediate EMT-like": "Int. EMT-like",
         "NEPC ASCL1+": "ASCL1+", "early NEPC ASCL1-": "early ASCL1−", "terminal NEPC ASCL1-": "term. ASCL1−",
         "NEPC other": "NEPC other"}
CLUSTERS = {"AR-high PRAD": "11, 15", "AR-low PRAD": "0, 2, 4, 5, 8, 12, 16", "pre-EMT": "6", "intermediate EMT-like": "17",
            "NEPC ASCL1+": "10", "early NEPC ASCL1-": "13", "terminal NEPC ASCL1-": "1, 7, 9", "NEPC other": "3, 14"}
# PDX state colours (their own category; none equals a side / gate colour of figstyle)
SCOL = {"AR-high PRAD": "#1F3B5C", "AR-low PRAD": "#8DB3D3", "pre-EMT": "#D9B44A", "intermediate EMT-like": "#8C5E9E",
        "NEPC ASCL1+": "#2F8C86", "early NEPC ASCL1-": "#A8D5A2", "terminal NEPC ASCL1-": "#5E7D33", "NEPC other": "#A3A3A3"}
KEY_PAIRS = [("w12_12wkCX_2", "w16_16wkCX_1"), ("w16_16wkCX_1", "w20_20wkCX_1_R"), ("w20_20wkCX_1_R", "w22_22wkCX_2_R")]


def unit(s0, s1):
    return f"LTL331__{s0}__{s1}"


def plab(s0, s1):
    return f"{TLAB[s0]}→{TLAB[s1]}"


def auc(pos, neg):
    """P(pos > neg) + 0.5 P(tie) (Mann-Whitney)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    allv = np.r_[pos, neg]
    r = pd.Series(allv).rank().to_numpy()
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


# ------------------------------------------------------------------------------------------------------------ data
def load():
    DATA.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(PDX / "cells.tsv.gz", sep="\t")
    gates = pd.read_csv(PDX / f"gates_{KEY}.tsv.gz", sep="\t")
    pairs = [(SAMPLE_ORDER[i], SAMPLE_ORDER[i + 1]) for i in range(7)]
    rows = []
    for s0, s1 in pairs:
        g = gates[gates.pair == unit(s0, s1)].copy()
        g["sample_id"] = np.where(g.side == "source", s0, s1)
        rows.append(g)
    g = pd.concat(rows)
    g = g.merge(cells[["cell", "sample_id", "state"]], on=["cell", "sample_id"], how="left", validate="many_to_one")
    assert g.state.notna().all(), "gate cell without a state"
    dep = pd.read_csv(DEPTH).rename(columns={"observation_id": "cell"})
    g = g.merge(dep[["sample_id", "cell", "predownsample_total_counts"]], on=["sample_id", "cell"], how="left", validate="many_to_one")
    assert g.predownsample_total_counts.notna().all()
    fit = pd.read_csv(GRID / "FIT_METRICS_v3tacc.tsv.gz", sep="\t")
    fit = fit[(fit.config_id == MAIN) & (fit.method == "M4-E")].set_index("unit_id")
    trans = pd.read_csv(PDX / f"transitions_{KEY}.tsv", sep="\t")
    sc = pd.read_csv(INTER / "state_coupling_all_configs.tsv.gz", sep="\t")
    return cells, g, pairs, fit, trans, sc


def tables(cells, g, pairs, fit, trans, sc):
    # OT cells (each sample appears in one or two pairs; composition is per sample, de-duplicated)
    ot = g.drop_duplicates(["sample_id", "cell"])[["sample_id", "cell", "state"]]
    comp = pd.crosstab(ot.sample_id, ot.state).reindex(index=SAMPLE_ORDER, columns=STATES, fill_value=0)
    comp.to_csv(DATA / "b_composition_counts.tsv", sep="\t")
    samp = pd.read_csv(SAMPLES, sep="\t") if SAMPLES.exists() else None
    design = pd.DataFrame({"sample_id": SAMPLE_ORDER, "weeks_after_castration": [WEEK[s] for s in SAMPLE_ORDER],
                           "label": [TLAB[s] for s in SAMPLE_ORDER], "ot_cells": comp.sum(1).to_numpy()})
    conv = cells.groupby("sample_id").size().reindex(SAMPLE_ORDER)
    design["authors_qc_cells"] = conv.to_numpy()
    design["removed_by_27_qc"] = design.authors_qc_cells - design.ot_cells
    design.to_csv(DATA / "a_design.tsv", sep="\t", index=False)

    rej = []
    for s0, s1 in pairs:
        u = unit(s0, s1); f = fit.loc[u]
        for side in ("source", "target"):
            x = g[(g.pair == u) & (g.side == side)]
            rd = x.loc[~x.retained.astype(bool), "predownsample_total_counts"]
            kd = x.loc[x.retained.astype(bool), "predownsample_total_counts"]
            rej.append({"pair": plab(s0, s1), "unit_id": u, "side": side, "cells": len(x), "rejected": int((~x.retained.astype(bool)).sum()),
                        "rejected_fraction": float((~x.retained.astype(bool)).mean()),
                        "fit_metrics_rejection_rate": float(f[f"{side}_final_rejection_rate"]),
                        "m4e_inference_valid": bool(f.calibration_m4e_inference_valid), "transported_mass": float(f.transported_mass),
                        "depth_auc_rejected_deeper": auc(rd, kd)})
    rej = pd.DataFrame(rej)
    assert np.allclose(rej.rejected_fraction, rej.fit_metrics_rejection_rate, atol=1e-6), "gates disagree with FIT_METRICS"
    rej.to_csv(DATA / "c_rejected_fraction.tsv", sep="\t", index=False)

    # rejection by state, key pairs, both sides
    bys = (g.assign(rej=~g.retained.astype(bool)).groupby(["pair", "side", "state"])
           .agg(cells=("rej", "size"), rejected=("rej", "sum")).reset_index())
    bys["rejected_fraction"] = bys.rejected / bys.cells
    bys["pair_label"] = [plab(*p.split("__")[1:]) for p in bys.pair]
    bys.to_csv(DATA / "d_rejected_by_state.tsv", sep="\t", index=False)

    # retained-block P(target state | source state) for the key pairs, plus retained cell counts per state
    heat = {}
    for s0, s1 in KEY_PAIRS:
        u = unit(s0, s1)
        t = trans[(trans.pair == u) & (trans.block == "retained")]
        M = t.pivot_table(index="source_state", columns="target_state", values="mass", aggfunc="sum").reindex(index=STATES, columns=STATES).fillna(0.0)
        x = g[(g.pair == u) & g.retained.astype(bool)]
        ns = x[x.side == "source"].state.value_counts().reindex(STATES, fill_value=0)
        nt = x[x.side == "target"].state.value_counts().reindex(STATES, fill_value=0)
        P = M.div(M.sum(1).replace(0, np.nan), axis=0)
        long = P.stack(dropna=False).rename("p_target_given_source").reset_index()
        long.columns = ["source_state", "target_state", "p_target_given_source"]
        long["mass"] = M.stack().to_numpy()
        long["retained_source_cells"] = long.source_state.map(ns)
        long["retained_target_cells"] = long.target_state.map(nt)
        long.insert(0, "pair", plab(s0, s1))
        long.to_csv(DATA / f"c_transition_{TLAB[s0]}_{TLAB[s1]}.tsv", sep="\t", index=False)
        heat[(s0, s1)] = (P, ns, nt, float(M.to_numpy().sum()))

    # intermediate -> intermediate enrichment at 12->16 over all 296 configurations
    e = sc[(sc.pair == "12->16") & (sc.src == "intermediate EMT-like") & (sc.tgt == "intermediate EMT-like")][["config_id", "enrichment", "p_target_given_source", "mass"]]
    e.to_csv(DATA / "d_intermediate_enrichment_296.tsv", sep="\t", index=False)
    return comp, design, rej, bys, heat, e


# ------------------------------------------------------------------------------------------------------------ draw
def panel_a(sh, design):
    import matplotlib.patches as mpatches
    x0, y0, x1, y1 = 0.0, 3.62, 3.25, 5.4
    sh.panel("a", x0, y0, x1, y1, title="LTL331 PDX: sampling after host castration")
    ax = sh.ax(0.15, 4.50, 3.05, 0.58); st.blank(ax)
    ax.set_xlim(-2.4, 23.6); ax.set_ylim(-1.45, 1.35)
    ax.axvspan(18.9, 23.6, color=st.PALE, lw=0)
    ax.text(21.25, 1.3, "relapse\n(LTL331R)", ha="center", va="top", fontsize=st.FS_SMALL, color=st.SUBTLE, linespacing=1.1)
    ax.plot([-1.6, 23], [0, 0], color=st.INK, lw=0.7)
    ax.annotate("host castration", xy=(0, 0.08), xytext=(0.0, 1.0), ha="left", va="bottom", fontsize=st.FS_SMALL, color=st.INK,
                arrowprops=dict(arrowstyle="-|>", lw=0.6, color=st.INK, mutation_scale=5))
    ws = [r.weeks_after_castration - (1.3 if r.label == "pre" else 0) for _, r in design.iterrows()]
    for w, (_, r) in zip(ws, design.iterrows()):
        ax.plot([w], [0], "o", ms=3.2, color=st.INK, zorder=3)
        ax.text(w, -0.3, r.label, ha="center", va="top", fontsize=st.FS_TICK)
        ax.text(w, -0.86, f"{r.ot_cells:,}", ha="center", va="top", fontsize=st.FS_SMALL, color=st.SUBTLE)
    for i in range(7):
        ax.annotate("", xy=(ws[i + 1] - 0.35, 0.36), xytext=(ws[i] + 0.35, 0.36),
                    arrowprops=dict(arrowstyle="-|>", lw=0.5, color=st.RULE, mutation_scale=4))
    tot = int(design.ot_cells.sum()); rem = int(design.removed_by_27_qc.sum())
    sh.text(0.20, 4.43, "week after castration (pre = before); grey: cells after QC", fontsize=st.FS_SMALL, color=st.SUBTLE, va="top")
    sh.text(0.20, 4.31, f"{tot:,} cells; {rem:,} of {int(design.authors_qc_cells.sum()):,} authors' cells removed by QC before OT",
            fontsize=st.FS_SMALL, color=st.SUBTLE, va="top")
    sh.text(0.20, 4.19, "arrows: 7 adjacent pairs, Source = earlier, Target = next sample", fontsize=st.FS_SMALL, color=st.SUBTLE, va="top")
    sh.text(0.20, 4.04, "States (published map, Sar et al.; Seurat clusters)", fontsize=st.FS_ANNOT, va="top")
    for i, s in enumerate(STATES):
        cx = 0.22 + (i // 4) * 1.5; cy = 3.885 - (i % 4) * 0.073
        sh.fig.patches.append(mpatches.Rectangle((cx / sh.W, (cy - 0.026) / sh.H), 0.09 / sh.W, 0.052 / sh.H,
                              transform=sh.fig.transFigure, fc=SCOL[s], ec="none"))
        sh.text(cx + 0.12, cy, f"{SHORT[s]}  ({CLUSTERS[s]})", fontsize=st.FS_SMALL, va="center")


def panel_b(sh, comp):
    x0, y0, x1, y1 = 3.3, 3.62, 7.2, 5.4
    sh.panel("b", x0, y0, x1, y1, title="Cell-state composition per sample (data, no OT)")
    ax = sh.ax(3.85, 3.95, 3.25, 1.05); st.tidy(ax)
    frac = comp.div(comp.sum(1), axis=0)
    xs = np.arange(len(SAMPLE_ORDER)); bot = np.zeros(len(xs))
    for s in STATES:
        ax.bar(xs, frac[s].to_numpy(), bottom=bot, color=SCOL[s], width=0.72, lw=0, label=SHORT[s])
        bot += frac[s].to_numpy()
    for i, n in enumerate(comp.sum(1)):
        ax.text(i, 1.03, f"{n:,}", ha="center", va="bottom", fontsize=st.FS_SMALL, color=st.SUBTLE)
    ax.text(-0.6, 1.17, "n cells", ha="left", va="bottom", fontsize=st.FS_SMALL, color=st.SUBTLE, clip_on=False)
    ax.set_xticks(xs, [TLAB[s] for s in SAMPLE_ORDER]); ax.set_xlabel("week after castration (R = relapse)")
    ax.set_ylim(0, 1.0); ax.set_yticks([0, 0.5, 1.0]); ax.set_ylabel("fraction of cells")
    ax.set_xlim(-0.6, 7.5)


def panel_c(sh, rej, heat):
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap
    x0, y0, x1, y1 = 0.0, 1.80, 7.2, 3.58
    sh.panel("c", x0, y0, x1, y1, title="ConfidenceOT matching between adjacent samples (main configuration, M4-E)")
    ax = sh.ax(0.40, 2.25, 1.85, 0.92); st.tidy(ax)
    labs = list(rej.pair.unique()); xs = np.arange(len(labs))
    for side, col, dx in (("source", st.SRC, -0.14), ("target", st.TGT, 0.14)):
        r = rej[rej.side == side].set_index("pair").reindex(labs)
        ax.bar(xs + dx, r.rejected_fraction, width=0.27, color=col, lw=0, label=side.capitalize())
    for i, l in enumerate(labs):
        if not bool(rej[(rej.pair == l)].m4e_inference_valid.iloc[0]):
            ax.text(i, 1.0, "†", ha="center", va="bottom", fontsize=st.FS_TICK, color=st.INK)
    ax.set_xticks(xs, labs, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylim(0, 1.0); ax.set_yticks([0, 0.5, 1.0]); ax.set_ylabel("rejected fraction")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.07), ncol=2, handlelength=0.9, columnspacing=0.8, borderaxespad=0)
    sh.text(1.50, 3.245, "† M4-E calibration\n   flag (kept, not removed)", fontsize=st.FS_SMALL,
            color=st.SUBTLE, va="bottom", linespacing=1.1)
    cmap = LinearSegmentedColormap.from_list("ret", ["#FFFFFF", st.RETAIN]); cmap.set_bad(st.PALE)
    c = 0.115; top = 3.28; xs0 = [2.98, 4.30, 5.62]
    sh.text(3.65, 3.455, "Retained cells: P(Target state | Source state) in %, shown where ≥ 20; grey = retained Source cells",
            fontsize=st.FS_SMALL, color=st.SUBTLE, va="bottom")
    for k, (s0, s1) in enumerate(KEY_PAIRS):
        P, ns, nt, tot = heat[(s0, s1)]
        A = P.loc[STATES, STATES].to_numpy().copy()
        A[ns.reindex(STATES).to_numpy() == 0, :] = np.nan
        ax = sh.ax(xs0[k], top - 8 * c, 8 * c, 8 * c)
        ax.imshow(np.ma.masked_invalid(A), cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        for i in range(8):
            for j in range(8):
                if np.isfinite(A[i, j]) and A[i, j] >= 0.20:
                    ax.text(j, i, f"{100 * A[i, j]:.0f}", ha="center", va="center", fontsize=st.FS_SMALL,
                            color="white" if A[i, j] > 0.6 else st.INK)
            ax.text(8.0, i, f"{int(ns[STATES[i]]):,}", ha="left", va="center", fontsize=st.FS_SMALL, color=st.SUBTLE)
        ax.set_xticks(range(8), [SHORT[s] for s in STATES], rotation=90, fontsize=st.FS_SMALL)
        ax.set_yticks(range(8), [SHORT[s] for s in STATES] if k == 0 else [], fontsize=st.FS_SMALL)
        ax.tick_params(length=1.2, pad=1)
        if k:
            ax.tick_params(axis="y", length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(3.5, -0.75, plab(s0, s1), ha="center", va="bottom", fontsize=st.FS_ANNOT)
        if k == 0:
            ax.text(-0.6, 8.9, "Source ↓  Target →", ha="right", va="top", fontsize=st.FS_SMALL, color=st.SUBTLE)
    cax = sh.ax(7.0, 2.45, 0.05, 0.7)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=mpl.colors.Normalize(0, 1), ticks=[0, 0.5, 1])
    cb.set_ticklabels(["0", "50", "100"]); cax.set_title("%", fontsize=st.FS_SMALL, pad=2)
    cb.outline.set_visible(False); cax.tick_params(labelsize=st.FS_SMALL, length=1.5, pad=1)


def panel_d(sh, bys, e):
    from matplotlib.colors import LinearSegmentedColormap
    x0, y0, x1, y1 = 0.0, 0.0, 3.25, 1.76
    sh.panel("d", x0, y0, x1, y1, title="Which states keep a counterpart?")
    cmap = LinearSegmentedColormap.from_list("rej", ["#FFFFFF", st.REJECT])
    want = ["12→16", "16→20R", "20R→22R"]
    ax = sh.ax(0.78, 0.42, 1.25, 1.0); st.blank(ax)
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(7.6, -0.6)
    for k, pl in enumerate(want):
        x = bys[(bys.pair_label == pl) & (bys.side == "source")].set_index("state")
        for i, s in enumerate(STATES):
            if s in x.index and x.loc[s, "cells"] > 0:
                r = x.loc[s]
                ax.scatter(k - 0.16, i, s=3 + 5 * np.log10(r.cells) ** 2, c=[cmap(r.rejected_fraction)], edgecolor=st.INK, lw=0.3, zorder=3)
                ax.text(k + 0.04, i, f"{r.rejected_fraction:.2f}", ha="left", va="center", fontsize=st.FS_SMALL)
        ax.text(k, -0.95, pl, ha="center", va="bottom", fontsize=st.FS_TICK)
    for i, s in enumerate(STATES):
        ax.text(-0.5, i, SHORT[s], ha="right", va="center", fontsize=st.FS_SMALL)
        ax.plot([-0.45, 2.45], [i, i], color=st.GUIDE, lw=0.3, zorder=0)
    ax.text(0.85, 8.1, "colour: Source rejected fraction; size: cells", ha="center", va="top", fontsize=st.FS_SMALL, color=st.SUBTLE)
    for j, n in enumerate([10, 100, 1000, 10000]):
        xk = [-0.45, 0.2, 0.92, 1.78][j]
        ax.scatter(xk, 9.1, s=3 + 5 * np.log10(n) ** 2, c="white", edgecolor=st.INK, lw=0.3, clip_on=False)
        ax.text(xk + [0.12, 0.17, 0.22, 0.27][j], 9.1, f"{n:,}", ha="left", va="center", fontsize=st.FS_SMALL, color=st.SUBTLE)
    ax2 = sh.ax(2.42, 0.42, 0.5, 1.0); st.tidy(ax2)
    v = np.log2(e.enrichment.clip(lower=1e-3).to_numpy())
    ax2.hist(v, bins=np.linspace(-1, 7, 25), orientation="horizontal", color=st.NEUTRAL, lw=0)
    main = float(e[e.config_id == MAIN].enrichment.iloc[0]); med = float(np.median(e.enrichment))
    ax2.axhline(np.log2(main), color=st.REJECT, lw=0.9); ax2.axhline(np.log2(med), color=st.INK, lw=0.6)
    ax2.axhline(0, **st.REF_KW)
    ax2.text(1, np.log2(main) + 0.12, f"main {main:.1f}×", ha="left", va="bottom", fontsize=st.FS_SMALL, color=st.REJECT)
    ax2.text(1, np.log2(med) - 0.12, f"median {med:.1f}×", ha="left", va="top", fontsize=st.FS_SMALL, color=st.INK)
    ax2.yaxis.tick_right(); ax2.yaxis.set_label_position("right"); ax2.spines["left"].set_visible(False); ax2.spines["right"].set_visible(True)
    ax2.set_yticks([0, 2, 4, 6], ["1", "4", "16", "64"]); ax2.set_ylim(-1, 7)
    ax2.set_ylabel("Int.→Int. enrichment (12→16)", fontsize=st.FS_SMALL, labelpad=1)
    ax2.set_xlabel(f"configurations\n(n = {len(e)})", fontsize=st.FS_SMALL)
    ax2.set_title(f"> 1 in {float((e.enrichment > 1).mean()):.0%}", fontsize=st.FS_SMALL, pad=2)


def panel_e(sh, comp, rej, heat, e):
    x0, y0, x1, y1 = 3.3, 0.0, 7.2, 1.76
    sh.panel("e", x0, y0, x1, y1, title="Comparison with the published observations")
    ax = sh.ax(3.36, 0.03, 3.8, 1.5); st.blank(ax); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    frac = comp.div(comp.sum(1), axis=0)
    P1216 = heat[KEY_PAIRS[0]][0]; P2022 = heat[KEY_PAIRS[2]][0]
    em = float(e[e.config_id == MAIN].enrichment.iloc[0]); n_above = int((e.enrichment > 1).sum())
    rows = [
        ("AR-high PRAD before castration,\nlost by 2 weeks",
         f"{frac.loc['w00_preCX_1', 'AR-high PRAD']:.0%} of cells at pre;\n{comp.loc['w02_2wkCX_2', 'AR-high PRAD']} cells at 2 weeks (data)",
         "consistent (data)"),
        ("Transient intermediate state\n(cluster 17) at 12–16 weeks",
         f"Int.→Int. P = {P1216.loc['intermediate EMT-like', 'intermediate EMT-like']:.2f}, {em:.0f}× enriched;\n> 1 in {n_above} of {len(e)} configurations",
         "consistent (OT)"),
        ("ASCL1+ (cluster 10, 20 weeks)\nthen ASCL1− (cluster 1, 22 weeks)",
         f"ASCL1+→term. ASCL1− P = {P2022.loc['NEPC ASCL1+', 'terminal NEPC ASCL1-']:.2f}\nat 20R→22R († calibration flag)",
         "consistent;\npair flagged"),
        ("One clone reprogrammes\nPRAD → NEPC",
         "no lineage data;\ncoupling is not ancestry",
         "not testable"),
    ]
    for xx, t in ((0.0, "Published (as recorded in repo)"), (0.40, "This analysis"), (0.82, "Status")):
        ax.text(xx, 0.965, t, fontsize=st.FS_ANNOT, va="center", color=st.SUBTLE)
    ax.plot([0, 1], [0.915, 0.915], color=st.GUIDE, lw=0.6)
    stcol = {"consistent (data)": st.RETAIN, "consistent (OT)": st.RETAIN, "consistent;\npair flagged": st.FALSE_REJ, "not testable": st.SUBTLE}
    for i, (a, b, c) in enumerate(rows):
        yy = 0.83 - i * 0.165
        ax.text(0.0, yy, a, fontsize=st.FS_SMALL, va="center", linespacing=1.15)
        ax.text(0.40, yy, b, fontsize=st.FS_SMALL, va="center", linespacing=1.15)
        ax.text(0.82, yy, c, fontsize=st.FS_SMALL, va="center", color=stcol[c], linespacing=1.15)
        ax.plot([0, 1], [yy - 0.0825, yy - 0.0825], color=st.GUIDE, lw=0.4)
    s1620 = rej[(rej.pair == "16→20R") & (rej.side == "source")].iloc[0]
    da = rej.set_index(["pair", "side"]).depth_auc_rejected_deeper
    a1, a2, a3 = da[("pre→2", "source")], da[("4→8", "source")], da[("16→20R", "source")]
    ax.text(0.0, 0.0, f"Limits: at 16→20R only {s1620.cells - s1620.rejected} of {s1620.cells:,} Source cells are retained.\n"
            f"Rejected cells are deeper (Source AUC {a1:.2f} pre→2, {a2:.2f} 4→8, {a3:.2f} 16→20R); unresolved.\n"
            "Rejection is not cell death or a state change; no PRAD→NEPC trajectory is reconstructed.",
            fontsize=st.FS_SMALL, va="bottom", color=st.SUBTLE, linespacing=1.15)


# ------------------------------------------------------------------------------------------------------------ text
def write_text(comp, design, rej, bys, heat, e):
    frac = comp.div(comp.sum(1), axis=0)
    em = float(e[e.config_id == MAIN].enrichment.iloc[0]); med = float(np.median(e.enrichment))
    q25, q75 = np.percentile(e.enrichment, [25, 75])
    P1216, P1620, P2022 = (heat[p][0] for p in KEY_PAIRS)
    ns1620 = heat[KEY_PAIRS[1]][1]
    r = rej.set_index(["pair", "side"])
    flagged = ", ".join(rej[~rej.m4e_inference_valid].pair.unique())
    tot = int(design.ot_cells.sum()); rem = int(design.removed_by_27_qc.sum())

    def bs(pl, side, s):
        x = bys[(bys.pair_label == pl) & (bys.side == side) & (bys.state == s)]
        return (float(x.rejected_fraction.iloc[0]), int(x.cells.iloc[0])) if len(x) else (np.nan, 0)

    cap = f"""# Figure 3 | PDX: can ConfidenceOT capture changes in cell states?

LTL331 patient-derived xenograft (Sar et al. 2026, GSE297328), sampled before host castration and 2, 4, 8, 12 and 16
weeks after it, and at 20 and 22 weeks after relapse as neuroendocrine LTL331R. RNA only; one sample per time point.
ConfidenceOT main configuration `{MAIN}` (chosen on Splatter only; not reselected here), frozen calibration v3
(within-side split null + mnncovlocal correction, acceptance 0.99), UOT backbone, M4-E. OT coupling is a correspondence
between cells of two samples; it is not ancestry, and rejection is not cell death or a state change.

**a**, Design. Eight samples; {tot:,} cells after QC (the project QC removed {rem:,} of {int(design.authors_qc_cells.sum()):,} cells
of the authors' set before any OT). Each of the 7 adjacent pairs is one comparison unit: Source = earlier sample,
Target = next sample. States follow the published map of Sar et al. (Seurat clusters in brackets).

**b**, Cell-state composition per sample (fraction of cells; n above bars). Data only, no OT. AR-high PRAD is
{frac.loc['w00_preCX_1', 'AR-high PRAD']:.1%} of cells before castration and absent at 2 weeks; AR-low PRAD is 95-99% at 2-8 weeks; the
intermediate EMT-like state (cluster 17) is {comp.loc['w12_12wkCX_2', 'intermediate EMT-like']} and {comp.loc['w16_16wkCX_1', 'intermediate EMT-like']} cells at 12 and 16 weeks; NEPC ASCL1+ is
{frac.loc['w20_20wkCX_1_R', 'NEPC ASCL1+']:.1%} at 20R, terminal NEPC ASCL1- {frac.loc['w22_22wkCX_2_R', 'terminal NEPC ASCL1-']:.1%} at 22R.

**c**, Left: rejected fraction of Source and Target cells per pair (n cells per side in `data/c_rejected_fraction.tsv`).
Dagger: the calibration reports M4-E inference as not valid ({flagged}); these pairs are shown, not removed. Right:
retained block of the coupling for 12->16, 16->20R and 20R->22R, as P(Target state | Source state) in percent; values >= 20% printed; grey numbers are
retained Source cells per state; pale rows have no retained cell. Retained Target cells per state are in `data/c_transition_*.tsv`. At 16->20R only {int(ns1620.sum())} of {int(r.loc[('16→20R', 'source')].cells):,} Source cells are retained,
so these links rest on few cells.

**d**, Left: Source-side rejected fraction by state at the three later pairs (colour: rejected fraction; dot size grows with log10 cells). At
12->16 the intermediate state is rejected least ({bs('12→16', 'source', 'intermediate EMT-like')[0]:.3f}, n = {bs('12→16', 'source', 'intermediate EMT-like')[1]}) and AR-low most
({bs('12→16', 'source', 'AR-low PRAD')[0]:.3f}, n = {bs('12→16', 'source', 'AR-low PRAD')[1]:,}); at 16->20R ASCL1+ keeps a counterpart ({bs('16→20R', 'source', 'NEPC ASCL1+')[0]:.3f} rejected) while AR-low
does not ({bs('16→20R', 'source', 'AR-low PRAD')[0]:.3f}). Right: enrichment of intermediate -> intermediate coupling at 12->16
(mass x total / (row x column), retained block) over all {len(e)} configurations of the grid: median {med:.1f}x (IQR {q25:.1f}-{q75:.1f}),
> 1 in {float((e.enrichment > 1).mean()):.0%}; main configuration {em:.1f}x (red line). Robustness across preprocessing, not proof of a transition.

**e**, Published observations (as recorded in the repository: `ltl331_nepc/README.md`, `STATE_BIOLOGY_REPORT.md`,
`PAPER_CLAIMS_CHECK.tsv`) against this analysis. Consistent: loss of AR-high PRAD (data); intermediate state matched to
itself at 12->16 (OT); ASCL1+ at 20R matched mainly to terminal ASCL1- at 22R (P = {P2022.loc['NEPC ASCL1+', 'terminal NEPC ASCL1-']:.2f}; pair flagged by the calibration).
Not testable: single-clone reprogramming (no lineage data). Rejected cells are deeper than retained cells in several
pairs (Source depth AUC {r.loc[('pre→2', 'source')].depth_auc_rejected_deeper:.2f} at pre->2, {r.loc[('4→8', 'source')].depth_auc_rejected_deeper:.2f} at 4->8, {r.loc[('16→20R', 'source')].depth_auc_rejected_deeper:.2f} at 16->20R); whether the gate follows depth is unresolved.
No full PRAD -> NEPC trajectory is reconstructed.
"""
    (OUT / "caption.md").write_text(cap, encoding="utf-8")
    commit = subprocess.run(["git", "-C", str(REPO), "log", "-1", "--format=%h", "--", "results/v3_grid_20261104"], capture_output=True, text=True).stdout.strip()
    src = f"""# Figure 3 sources (all read-only; derived tables in out/Figure3/data/)

| Panel | Source files | Columns / filter | Derived table |
|---|---|---|---|
| a | `results/v3_grid_20261104/figures/pdx/data/cells.tsv.gz` (authors' QC cells), `gates_splatter_l5_best.tsv.gz` (OT cells) | sample_id; OT cells = cells with an M4-E gate | `a_design.tsv` |
| a | `realdata_benchmark/v3_grid/figures/pdx_prepare.py` `STATE` | published state map (Sar et al. Additional files 1/2) | legend |
| b | `cells.tsv.gz` joined to `gates_splatter_l5_best.tsv.gz` on (sample_id, cell) | state, de-duplicated per sample | `b_composition_counts.tsv` |
| c left | `gates_splatter_l5_best.tsv.gz`; `results/v3_grid_20261104/ltl331/FIT_METRICS_v3tacc.tsv.gz` | config_id = `{MAIN}`, method M4-E; `retained`; `calibration_m4e_inference_valid`; check: gate fraction = `*_final_rejection_rate` | `c_rejected_fraction.tsv` |
| c right | `results/v3_grid_20261104/figures/pdx/data/transitions_splatter_l5_best.tsv` | block = retained; row-normalised | `c_transition_*.tsv` |
| d left | gates + cells | rejected fraction by pair x side x state | `d_rejected_by_state.tsv` |
| d right | `results/ltl331_nepc/v3_intermediate_check_20261117/state_coupling_all_configs.tsv.gz` | pair 12->16, src = tgt = intermediate EMT-like, `enrichment`, 296 configurations | `d_intermediate_enrichment_296.tsv` |
| e | `ltl331_nepc/README.md`, `ltl331_nepc/STATE_BIOLOGY_REPORT.md`, `results/ltl331_nepc/state_biology_20261101/nonharmony_v4/PAPER_CLAIMS_CHECK.tsv`; numbers from panels b-d; depth from `results/ltl331_nepc/tacc_20261031/ltl331_confidenceot/inputs/ds/predownsample_depth.csv.gz` (original counts) | depth AUC = P(rejected deeper) per pair x side | `c_rejected_fraction.tsv` (`depth_auc_rejected_deeper`) |

Result tables last changed in commit {commit or 'n/a'} (results/v3_grid_20261104). Not used: TACC job 1111186 and the 64-configuration
benchmark (older calibration), stale logs `pdx_prepare.log` / `pdx_figures.log`, the withdrawn rejected-vs-retained GSEA, and
the fate GSEA (logCPM arm only, not the main configuration).
"""
    (OUT / "sources.md").write_text(src, encoding="utf-8")
    notes = {
        "a": {"title": "PDX experimental design",
              "compared": f"Eight samples of one PDX before and after host castration; each adjacent pair (7) is a Source -> Target unit; {tot:,} cells.",
              "observe": "The sampling spans the androgen-dependent tumour (pre), regression after castration (2-16 weeks) and the neuroendocrine relapse (20R, 22R).",
              "meaning": "There is no lineage truth here; the question is whether matching is consistent with the published state changes."},
        "b": {"title": "Cell-state composition changes with time",
              "compared": "Fraction of cells in each published state per sample (data, before any OT).",
              "observe": f"AR-high PRAD {frac.loc['w00_preCX_1', 'AR-high PRAD']:.0%} before castration, AR-low PRAD dominates 2-12 weeks, the intermediate EMT-like state appears at 12-16 weeks, NEPC dominates the relapse.",
              "meaning": "Adjacent samples differ in which states they contain, so some cells should have no counterpart."},
        "c": {"title": "Rejection and retained state-to-state matching",
              "compared": "Rejected fraction per pair and side; for three later pairs, where retained Source cells of each state are matched in the Target.",
              "observe": f"Rejection is high where composition changes most (16->20R: {r.loc[('16→20R', 'source')].rejected_fraction:.0%} of Source cells); retained intermediate cells match intermediate cells (P = {P1216.loc['intermediate EMT-like', 'intermediate EMT-like']:.2f}); ASCL1+ at 20R matches terminal ASCL1- at 22R (P = {P2022.loc['NEPC ASCL1+', 'terminal NEPC ASCL1-']:.2f}).",
              "meaning": "The retained correspondences follow the published order of states, but some rest on few cells and three pairs carry a calibration flag."},
        "d": {"title": "Which states keep a counterpart, and is it robust?",
              "compared": "Source-side rejection by state; intermediate -> intermediate enrichment over all 296 preprocessing configurations.",
              "observe": f"The intermediate and ASCL1+ states keep counterparts while AR-low cells at 16 weeks mostly do not; enrichment > 1 in {float((e.enrichment > 1).mean()):.0%} of configurations (median {med:.1f}x, main {em:.1f}x).",
              "meaning": "The intermediate-state correspondence does not depend on one preprocessing choice; it is consistency, not a measured transition."},
        "e": {"title": "Comparison with the original paper",
              "compared": "Published observations (as recorded in the repository) against the data and the OT result.",
              "observe": "Composition claims and the ASCL1+ -> ASCL1- order agree; single-clone reprogramming cannot be tested with RNA alone; rejected cells tend to be deeper.",
              "meaning": "ConfidenceOT gives a consistent, cautious reading of a real state change; it does not reconstruct the PRAD -> NEPC trajectory. Independent lineage information is needed (Figure 4)."},
    }
    (OUT / "panel_notes.json").write_text(json.dumps(notes, indent=1, ensure_ascii=False), encoding="utf-8")


def main():
    st.apply()
    cells, g, pairs, fit, trans, sc = load()
    comp, design, rej, bys, heat, e = tables(cells, g, pairs, fit, trans, sc)
    sh = st.Sheet(7.2, 5.4, "Figure3")
    panel_a(sh, design); panel_b(sh, comp); panel_c(sh, rej, heat); panel_d(sh, bys, e); panel_e(sh, comp, rej, heat, e)
    write_text(comp, design, rej, bys, heat, e)
    sh.save()
    print(rej[["pair", "side", "cells", "rejected_fraction", "m4e_inference_valid", "depth_auc_rejected_deeper"]].to_string())


if __name__ == "__main__":
    main()
