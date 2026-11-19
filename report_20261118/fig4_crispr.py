"""Figure 4 - CRISPR: does RNA-based matching agree with lineage information?

Read-only on every repository result; writes report_20261118/out/Figure4/ (figure, panel crops, data, caption, sources,
panel notes). No OT, no fits.

    python -I report_20261118/fig4_crispr.py
"""
from __future__ import annotations

import glob
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import figstyle as fs  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

RLD = REPO / "results" / "rna_lineage_discordance_20261110"
CMB = REPO / "results" / "crispr_matrix_benchmark_20261108"
LARRY = REPO / "results" / "app_search_20261117" / "larry_pilot" / "score"
MAIN = "ranknm256-midrank|none|scale|cos|pca50"
FAM = "3457_Apc_T1"
POOLED = "L23__KPTracer_3457__3457_Apc_T1__family_metastases"
OUT = fs.OUT / "Figure4"
DATA = OUT / "data"

CL_COL = {"3457-A": fs.POP[0], "3457-B": fs.POP[2], "3457-C": fs.POP[3], "minor": "#D3D7DA"}
RNA_OT = fs.METHOD["ConfidenceOT"]   # RNA-only ConfidenceOT
CRISPR_OT = fs.POP[7]                # CRISPR-guided OT (uses the edits)
SITE_COL = {"T1": "#E3E6E8", "K1": "#8F969B", "N2": "#4A5055"}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------------------------------------------------- data
def data_b():
    m = pd.read_csv(CMB / "l1" / "l1_metastasis.tsv.gz", sep="\t", dtype={"mouse": str})
    m = m[(m.config_id == MAIN) & (m.method == "M4-E")][["mouse", "metastasis", "d", "same_retained", "other_retained"]]
    f = pd.read_csv(CMB / "l1" / "l1_family.tsv", sep="\t")
    allcfg = f[f.method == "M4-E"][["config_id", "mean_mouse_d", "mice_d_pos", "wilcoxon_p_all"]]
    main = f[(f.config_id == MAIN) & (f.method == "M4-E")].iloc[0]
    m.to_csv(DATA / "b_metastasis_d.tsv", sep="\t", index=False)
    allcfg.to_csv(DATA / "b_all296_mean_mouse_d.tsv", sep="\t", index=False)
    return m, allcfg, main


def data_c():
    AC = pd.read_csv(RLD / "state_lineage_ext" / "all_cells__original" / "cells.tsv.gz", sep="\t", dtype={"cell": str, "mouse": str})
    lab = pd.read_csv(RLD / "family_check" / "clade_labels.tsv", sep="\t")
    lab = lab[lab.family == FAM].set_index("clade").label
    g = AC[AC.family == FAM].copy()
    g["clade_label"] = g.clade.map(lab)
    g["site_short"] = g.site.str.split("_").str[-1]
    cs = pd.read_csv(RLD / "family_check" / FAM / "character_states.tsv", sep="\t")
    cot = load_module("cot", REPO / "lineage_validation" / "rna_lineage_discordance_20261110" / "10_crispr_ot.py")
    bb = cot.bundles_module()
    t, _ = bb.load_tables(CMB / "inputs")["KPTracer"]
    t = t[(t.lineage == FAM) & t.cell.isin(set(g.cell))].copy()
    M = t.melt(id_vars=["cell", "intBC"], value_vars=["r1", "r2", "r3"], var_name="r", value_name="state")
    M["pos"] = M.intBC + ":" + M.r
    W = M.pivot_table(index="cell", columns="pos", values="state", aggfunc="first")
    intbcs = sorted(t.intBC.unique())
    pos = [f"{i}:{r}" for i in intbcs for r in ("r1", "r2", "r3")]
    W = W.reindex(columns=pos)
    cs = cs.set_index("position").reindex(pos)
    founder = cs.founder.fillna(False).astype(bool)
    topA = cs["3457-A · T1 (primary) | top"]; topB = cs["3457-B · K1 | top"]; famtop = cs.family_top_state
    # row order: clade A, B, C, minor; within clade site (primary first) then RNA state
    state_rank = g.state.value_counts()
    g["_c"] = g.clade_label.map({"3457-A": 0, "3457-B": 1, "3457-C": 2, "minor": 3})
    g["_s"] = g.site_short.map({"T1": 0, "K1": 1, "N2": 2})
    g["_st"] = g.state.map({s: i for i, s in enumerate(state_rank.index)})
    g = g.sort_values(["_c", "_s", "_st", "cell"])
    X = W.reindex(g.cell).to_numpy(object)
    # codes: 0 missing, 1 uncut, 2 other edit, 3 founder state, 4 3457-A state, 5 3457-B state, 6 A = B state (non-founder)
    code = np.zeros(X.shape, int)
    for j, p in enumerate(pos):
        v = X[:, j]
        miss = pd.isna(v)
        k = np.where(miss, 0, np.where(v == "None", 1, 2))
        vv = np.where(miss, "__", v).astype(str)
        if founder[p]:
            k = np.where(vv == str(famtop[p]), 3, k)
        else:
            a_, b_ = str(topA[p]), str(topB[p])
            isA = (vv == a_) & (vv != "None"); isB = (vv == b_) & (vv != "None")
            k = np.where(isA & isB, 6, np.where(isA, 4, np.where(isB, 5, k)))
        code[:, j] = k
    rows = g[["cell", "clade_label", "site_short", "state"]].reset_index(drop=True)
    rows.to_csv(DATA / "c_rows.tsv", sep="\t", index=False)
    pd.DataFrame(code, columns=pos).assign(cell=rows.cell).to_csv(DATA / "c_edit_codes.tsv.gz", sep="\t", index=False)
    cs.reset_index()[["position", "family_top_state", "family_top_share", "founder"]].to_csv(DATA / "c_positions.tsv", sep="\t", index=False)
    return rows, code, pos, founder, cs, intbcs


def data_d():
    b = pd.read_csv(RLD / "explore" / "baseline_independent.tsv", sep="\t")
    L = pd.concat([pd.read_csv(f, sep="\t") for f in sorted(glob.glob(str(RLD / "explore" / "levels_*.tsv")))])
    L10 = L[L.cells >= 10].copy()
    c = pd.read_csv(RLD / "compare" / f"{POOLED}__l5best" / "cells.tsv.gz", sep="\t")
    AC = pd.read_csv(RLD / "state_lineage_ext" / "all_cells__original" / "cells.tsv.gz", sep="\t", dtype={"cell": str, "mouse": str},
                     usecols=["cell", "family", "clade"])
    lab = pd.read_csv(RLD / "family_check" / "clade_labels.tsv", sep="\t")
    lab = lab[lab.family == FAM].set_index("clade").label
    s = c[c.side == "source"].merge(AC[AC.family == FAM][["cell", "clade"]], on="cell", how="left")
    s["clade_label"] = s.clade.map(lab)
    keep = s.groupby("clade_label").agg(cells=("cell", "size"), rna_kept=("rna_retained", "sum"), crispr_kept=("crispr_retained", "sum"))
    keep = keep.reindex(["3457-A", "3457-B", "3457-C", "minor"]).reset_index()
    b.to_csv(DATA / "d_clade_level_overlap_units.tsv", sep="\t", index=False)
    L10.to_csv(DATA / "d_cell_level_top15_unit_x_clade.tsv", sep="\t", index=False)
    keep.to_csv(DATA / "d_3457T1_pooled_primary_kept_by_clade.tsv", sep="\t", index=False)
    return b, L10, keep


def data_e():
    fl = pd.read_csv(RLD / "lineages" / "flags.tsv", sep="\t")
    fl = fl[fl.dataset == "KPTracer"].copy()
    fl["c_cells"] = fl.cells >= 50
    fl["c_events"] = fl.median_events_per_cell >= 3
    fl["c_distinct"] = fl.distinct_sets_share >= 0.1
    fl["c_iqr"] = fl.s_iqr >= 0.05
    mv = pd.read_csv(CMB / "summary" / "mouse_values.tsv.gz", sep="\t", dtype={"mouse": str})
    mv = mv[(mv.config_id == MAIN) & (mv.method == "M4-E")]
    l3 = mv[(mv.level == "L3_matrix")][["mouse", "dataset", "stratum", "value"]]
    l3 = l3[l3.stratum == "case"]
    qclone = mv[(mv.level == "L1_clone") & (mv.dataset == "Quinn")][["mouse", "value"]]
    cm = pd.read_csv(LARRY / "clone_mass.tsv", sep="\t")
    cm = cm[(cm.config == "l5best") & (cm.method == "M4-E")][["unit_id", "scored_cells", "median_m_over_uniform", "top10_hit_rate", "top10_chance"]]
    fate = pd.read_csv(LARRY / "fate.tsv", sep="\t")
    fate = fate[(fate.config == "l5best") & (fate.method == "M4-E")][["unit_id", "median_tv", "null_median_tv_mean", "p_null_le_obs", "neumono_auroc"]]
    fl.to_csv(DATA / "e_resolution_kp_families.tsv", sep="\t", index=False)
    l3.to_csv(DATA / "e_L3_matrix_dSnorm_cases_main.tsv", sep="\t", index=False)
    qclone.to_csv(DATA / "e_quinn_own_clone_gain_main.tsv", sep="\t", index=False)
    cm.merge(fate, on="unit_id").to_csv(DATA / "e_larry_pilot_main_m4e.tsv", sep="\t", index=False)
    return fl, l3, qclone, cm, fate


# ---------------------------------------------------------------------------------------------------------- panels
def legend_swatch(sh, x, y, color, label, fsz=None):
    sh.fig.patches.append(Rectangle((x / sh.W, (y - 0.035) / sh.H), 0.08 / sh.W, 0.08 / sh.H, transform=sh.fig.transFigure,
                                    fc=color, ec=fs.SUBTLE, lw=0.3))
    sh.text(x + 0.11, y + 0.005, label, fontsize=fsz or fs.FS_SMALL, va="center")


def panel_a(sh):
    x0, y0, x1, y1 = 0.0, 3.30, 2.62, 5.40
    sh.panel("a", x0, y0, x1, y1, title="Two measurements of the same cells")
    ax = sh.ax(x0 + 0.06, y0 + 0.04, x1 - x0 - 0.12, y1 - y0 - 0.30)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); fs.blank(ax)
    F = fs.FS_SMALL
    fs.box(ax, 1.6, 8.75, 6.8, 1.15, "KP-Tracer tumours, primary → metastases:\nthe same single cells, two read-outs", fc=fs.PALE, ec=fs.SUBTLE, fs=F)
    fs.arrow(ax, 3.0, 8.75, 2.3, 7.75); fs.arrow(ax, 7.0, 8.75, 7.7, 7.75)
    fs.box(ax, 0.0, 6.75, 4.6, 0.95, "scRNA counts", ec=RNA_OT, tc=RNA_OT, fs=F)
    fs.box(ax, 5.4, 6.75, 4.6, 0.95, "CRISPR edits (raw allele table)", ec=fs.INK, fs=F)
    fs.arrow(ax, 2.3, 6.75, 2.3, 5.9, color=RNA_OT)
    fs.box(ax, 0.0, 4.35, 4.6, 1.5, "RNA-only ConfidenceOT\nmain configuration, M4-E\n(never sees the edits)", ec=RNA_OT, tc=RNA_OT, lw=0.9, fs=F)
    fs.arrow(ax, 7.7, 6.75, 7.7, 5.9)
    fs.box(ax, 5.4, 4.35, 4.6, 1.5, "edit agreement s per cell pair\n→ clades within a family\n(founder edits removed)", ec=fs.INK, fs=F)
    fs.arrow(ax, 2.3, 4.35, 2.3, 3.15, color=RNA_OT); fs.arrow(ax, 6.6, 4.35, 3.6, 3.15)
    fs.box(ax, 0.0, 2.05, 4.6, 1.05, "evaluation: does RNA matching\nfollow families and clades?", fc=fs.PALE, ec=fs.INK, fs=F)
    fs.arrow(ax, 8.6, 4.35, 8.6, 3.15, color=CRISPR_OT)
    pb = fs.box(ax, 5.4, 1.7, 4.6, 1.45, "CRISPR-guided OT\n(same solver, cost 1 − s)\nuses the edits: reference only", ec=CRISPR_OT, tc=CRISPR_OT, fs=F)
    pb.set_linestyle((0, (2.5, 1.5)))
    ax.text(0.0, 0.6, "schematic", fontsize=F, color=fs.SUBTLE, style="italic", ha="left", va="center")


def panel_b(sh, m, allcfg, main):
    x0, y0, x1, y1 = 2.62, 3.30, 4.55, 5.40
    sh.panel("b", x0, y0, x1, y1, title="Tumour level: own family kept more")
    ax = sh.ax(x0 + 0.52, y0 + 0.50, 0.98, 1.30)
    mice = m.groupby("mouse").d.mean().sort_values()
    order = list(mice.index)
    for i, mo in enumerate(order):
        v = m[m.mouse == mo].d.to_numpy()
        jit = (np.arange(len(v)) - (len(v) - 1) / 2) * 0.06
        ax.scatter(np.full(len(v), i) + jit, v, s=6, color=fs.NEUTRAL, lw=0, zorder=2)
        ax.plot([i - 0.3, i + 0.3], [mice[mo]] * 2, color=RNA_OT, lw=1.3, zorder=3)
    ax.axhline(0, **fs.REF_KW)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=90, fontsize=fs.FS_SMALL)
    ax.set_xlim(-0.6, len(order) - 0.4); ax.set_ylim(-0.25, 0.95); ax.set_yticks([-0.2, 0, 0.4, 0.8])
    ax.set_ylabel("d = kept, own-family primary\n− other primaries (same mouse)", fontsize=fs.FS_SMALL)
    fs.tidy(ax)
    ax.text(0.02, 0.99, f"mean over mice {main.mean_mouse_d:.3f}\n{int(main.mice_d_pos)} of {int(main.mice)} mice > 0\n"
            f"one-sided Wilcoxon, {int(main.metastases)} mets:\np = {main.wilcoxon_p_all:.1e}",
            transform=ax.transAxes, va="top", ha="left", fontsize=fs.FS_SMALL, color=fs.INK)
    ax2 = sh.ax(x0 + 1.60, y0 + 0.50, 0.28, 1.30)
    v = allcfg.mean_mouse_d.to_numpy()
    rng = np.random.default_rng(1)
    ax2.scatter(rng.uniform(-0.25, 0.25, len(v)), v, s=2, color=fs.NEUTRAL, lw=0, alpha=0.8)
    ax2.scatter([0], [main.mean_mouse_d], s=14, color=RNA_OT, zorder=3, lw=0)
    ax2.plot([-0.32, 0.32], [np.median(v)] * 2, color=fs.INK, lw=0.8)
    ax2.axhline(0, **fs.REF_KW)
    ax2.set_ylim(-0.25, 0.95); ax2.set_xlim(-0.45, 0.45); ax2.set_xticks([0]); ax2.set_xticklabels(["296\nconfigs"], fontsize=fs.FS_SMALL)
    ax2.set_yticks([]); fs.tidy(ax2, left=False)
    ax2.text(0.5, 0.99, f"median\n{np.median(v):.3f}", transform=ax2.transAxes, ha="center", va="top", fontsize=fs.FS_SMALL, color=fs.SUBTLE)
    sh.text(x0 + 0.42, y0 + 0.04, f"grey: {len(m)} metastases; red: mouse means", fontsize=fs.FS_SMALL,
            color=fs.SUBTLE, va="bottom")


def panel_c(sh, rows, code, pos, founder, intbcs):
    x0, y0, x1, y1 = 4.55, 0.0, 7.20, 5.40
    sh.panel("c", x0, y0, x1, y1, title="Edits of family 3457 T1 (885 cells)")
    n, p = code.shape
    hx, hy, hw, hh = x0 + 0.70, y0 + 1.36, 1.40, 3.30
    ax = sh.ax(hx, hy, hw, hh)
    cmap = ListedColormap(["#FFFFFF", "#EDEFF1", fs.NEUTRAL, "#3B3F42", CL_COL["3457-A"], CL_COL["3457-B"], "#7E6A9A"])
    ax.imshow(code, aspect="auto", interpolation="nearest", cmap=cmap, vmin=-0.5, vmax=6.5)
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_linewidth(0.4); s_.set_color(fs.SUBTLE)
    for k in range(1, len(intbcs)):
        ax.axvline(3 * k - 0.5, color="white", lw=0.6)
    fpos = [j for j, q in enumerate(pos) if founder[q]]
    for j in fpos:
        ax.plot(j, -12, marker="v", ms=2.6, color="#3B3F42", clip_on=False)
    ax.set_ylim(n - 0.5, -0.5); ax.set_xlim(-0.5, p - 0.5)
    ax.set_xlabel(f"{p} positions (10 target sites × 3 cut sites)\n▼ founder: state in ≥ 90% of cells ({len(fpos)})",
                  fontsize=fs.FS_SMALL, labelpad=2)
    states = rows.state.value_counts()
    top_states = list(states.index[:6])
    st_pal = ["#6E8E7D", "#BCA177", "#8FB0C6", "#C98B6B", "#9DB57A", "#B58DB0"]
    st_col = {s: c for s, c in zip(top_states, st_pal)}
    from matplotlib.colors import to_rgb
    tr = [("clade", rows.clade_label.map(CL_COL)), ("site", rows.site_short.map(SITE_COL)),
          ("RNA state", rows.state.map(lambda s: st_col.get(s, "#D9D9D9")))]
    for i, (nm, cols) in enumerate(tr):
        axt = sh.ax(hx - 0.60 + i * 0.19, hy, 0.15, hh)
        img = np.array([to_rgb(c) for c in cols])[:, None, :]
        axt.imshow(img, aspect="auto", interpolation="nearest")
        axt.set_xticks([0]); axt.set_xticklabels([nm], rotation=90, fontsize=fs.FS_SMALL); axt.set_yticks([])
        axt.xaxis.set_ticks_position("top"); axt.tick_params(axis="x", length=0, pad=1)
        for s_ in axt.spines.values():
            s_.set_visible(False)
    axr = sh.ax(hx + hw + 0.03, hy, 0.12, hh); axr.set_xlim(0, 1); axr.set_ylim(n, 0); fs.blank(axr)
    for lab_ in ["3457-A", "3457-B", "3457-C", "minor"]:
        idx = np.flatnonzero(rows.clade_label.to_numpy() == lab_)
        if not len(idx):
            continue
        a_, b_ = idx.min(), idx.max() + 1
        axr.plot([0.1, 0.1], [a_ + 2, b_ - 2], color=CL_COL[lab_] if lab_ != "minor" else fs.SUBTLE, lw=1.4, solid_capstyle="butt")
        yy = (a_ + b_) / 2
        if lab_ == "3457-C":
            yy -= 30
        if lab_ == "minor":
            yy += 30
        axr.text(0.35, yy, f"{lab_}\n{len(idx)}",
                 fontsize=fs.FS_SMALL, va="center", ha="left", color=fs.INK)
    items = [("missing", "#FFFFFF"), ("uncut", "#EDEFF1"), ("founder state", "#3B3F42"), ("3457-A state", CL_COL["3457-A"]),
             ("3457-B state", CL_COL["3457-B"]), ("A = B state", "#7E6A9A"), ("other edit", fs.NEUTRAL)]
    for k, (t_, c_) in enumerate(items):
        legend_swatch(sh, x0 + 0.10 + (k % 3) * 0.84, 1.02 - (k // 3) * 0.13, c_, t_)
    sh.text(x0 + 0.10, 0.63, "site:", fontsize=fs.FS_SMALL, va="center", color=fs.SUBTLE)
    for k, (t_, c_) in enumerate([("T1 primary", SITE_COL["T1"]), ("K1 met", SITE_COL["K1"]), ("N2 met", SITE_COL["N2"])]):
        legend_swatch(sh, x0 + 0.40 + k * 0.70, 0.63, c_, t_)
    sh.text(x0 + 0.10, 0.49, "RNA state (cells):", fontsize=fs.FS_SMALL, va="center", color=fs.SUBTLE)
    other_n = int((~rows.state.isin(top_states)).sum())
    labs = [(f"{s} ({int(states[s])})", st_col[s]) for s in top_states] + [(f"other ({other_n})", "#D9D9D9")]
    for k, (t_, c_) in enumerate(labs):
        legend_swatch(sh, x0 + 0.10 + (k % 3) * 0.86, 0.36 - (k // 3) * 0.13, c_, t_)
    return top_states


def panel_d(sh, b, L10, keep):
    x0, y0, x1, y1 = 0.0, 0.0, 2.45, 3.30
    sh.panel("d", x0, y0, x1, y1, title="RNA-OT vs CRISPR-OT: clade level ≠ cell level")
    ax = sh.ax(x0 + 1.12, y0 + 1.80, 1.22, 1.08)
    groups = [(f"clade level:\nRNA-OT vs CRISPR-OT\n({len(b)} units)", b.shared_rna_crispr.to_numpy(), CRISPR_OT),
              (f"clade level:\nRNA-OT vs size-only\n({len(b)} units)", b.shared_rna_independent.to_numpy(), fs.NEUTRAL),
              (f"exact cell partners:\ntop-15 overlap\n({len(L10)} unit × clade)", L10.cell_top15_overlap_mean.to_numpy(), RNA_OT)]
    rng = np.random.default_rng(2)
    for i, (nm, v, c) in enumerate(groups):
        ax.scatter(v, i + rng.uniform(-0.2, 0.2, len(v)), s=4, color=c, alpha=0.6, lw=0)
        med = np.median(v)
        ax.plot([med, med], [i - 0.3, i + 0.3], color=fs.INK, lw=1.1)
        ax.text(med + (0.03 if med < 0.5 else 0), i - 0.36, f"{med:.2f}", fontsize=fs.FS_SMALL, va="bottom", ha="left" if med < 0.5 else "center")
    ax.set_yticks(range(3)); ax.set_yticklabels([g[0] for g in groups], fontsize=fs.FS_SMALL)
    ax.set_ylim(2.55, -0.65); ax.set_xlim(0, 1.03); ax.set_xticks([0, 0.5, 1])
    ax.set_xlabel("overlap (fraction; line = median)", fontsize=fs.FS_SMALL, labelpad=1); fs.tidy(ax)
    ax.tick_params(axis="y", length=0)
    ax2 = sh.ax(x0 + 0.42, y0 + 0.28, 1.92, 0.98)
    k = keep.copy()
    xs = np.arange(len(k)); w = 0.36
    r1 = (k.rna_kept / k.cells).to_numpy(); r2 = (k.crispr_kept / k.cells).to_numpy()
    ax2.bar(xs - w / 2, r1, w, color=RNA_OT, lw=0, label="RNA-OT")
    ax2.bar(xs + w / 2, r2, w, color=CRISPR_OT, lw=0, label="CRISPR-OT")
    for i, row in k.iterrows():
        ax2.text(i - w / 2, r1[i] + 0.04, f"{int(row.rna_kept)}/{int(row.cells)}", fontsize=fs.FS_SMALL, ha="center", va="bottom", rotation=90)
        ax2.text(i + w / 2, r2[i] + 0.04, f"{int(row.crispr_kept)}/{int(row.cells)}", fontsize=fs.FS_SMALL, ha="center", va="bottom", rotation=90)
    ax2.set_xticks(xs); ax2.set_xticklabels(k.clade_label, fontsize=fs.FS_SMALL)
    ax2.set_ylim(0, 2.0); ax2.set_yticks([0, 0.5, 1]); ax2.set_ylabel("primary cells kept", fontsize=fs.FS_SMALL)
    fs.tidy(ax2)
    ax2.legend(loc="upper right", fontsize=fs.FS_SMALL, handlelength=0.8, ncol=2, borderaxespad=0.0, columnspacing=0.8)
    ax2.text(0.0, 1.03, "3457 T1 → K1 + N2 (pooled): primary cells kept, by clade", transform=ax2.transAxes,
             fontsize=fs.FS_SMALL, va="bottom", ha="left")


def panel_e(sh, fl, l3, qclone, cm):
    x0, y0, x1, y1 = 2.45, 0.0, 4.55, 3.30
    sh.panel("e", x0, y0, x1, y1, title="Limits and other lineage systems")
    ax = sh.ax(x0 + 0.78, y0 + 1.95, 0.95, 0.90)
    crit = [("cells", "c_cells"), ("edits", "c_events"), ("distinct", "c_distinct"), ("s IQR", "c_iqr"), ("resolved", "resolved")]
    f = fl.sort_values(["resolved", "lineage"], ascending=[False, True]).reset_index(drop=True)
    for i, r in f.iterrows():
        for j, (_, c) in enumerate(crit):
            ok = bool(r[c])
            col = (fs.RETAIN if ok else "white") if c != "resolved" else (fs.INK if ok else "white")
            ax.scatter(j, i, s=12, color=col, edgecolor=fs.SUBTLE if not ok else col, lw=0.5)
    ax.set_xticks(range(len(crit))); ax.set_xticklabels([c[0] for c in crit], rotation=90, fontsize=fs.FS_SMALL)
    ax.xaxis.set_ticks_position("top"); ax.tick_params(axis="x", length=0, pad=1)
    ax.set_yticks(range(len(f))); ax.set_yticklabels([s.replace("_", " ") for s in f.lineage], fontsize=fs.FS_SMALL)
    ax.set_ylim(len(f) - 0.5, -0.5); ax.set_xlim(-0.6, len(crit) - 0.4)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.tick_params(axis="y", length=0)
    nres = int(f.resolved.sum())
    sh.text(x0 + 1.78, y0 + 2.40, f"{nres} of {len(f)}\nresolved", fontsize=fs.FS_SMALL, va="center", ha="left")
    ax2 = sh.ax(x0 + 0.78, y0 + 0.92, 1.20, 0.78)
    d2 = l3.copy()
    d2["name"] = np.where(d2.dataset == "KP", "KP " + d2.mouse.astype(str), "Quinn " + d2.mouse.astype(str))
    d2 = d2.sort_values(["dataset", "value"], ascending=[True, False]).reset_index(drop=True)
    ax2.barh(range(len(d2)), d2.value, color=np.where(d2.dataset == "KP", RNA_OT, fs.NEUTRAL), height=0.65, lw=0)
    ax2.set_yticks(range(len(d2))); ax2.set_yticklabels(d2.name, fontsize=fs.FS_SMALL)
    ax2.invert_yaxis(); ax2.axvline(0, **fs.REF_KW); ax2.set_xlim(-0.05, 0.55); ax2.set_xticks([0, 0.25, 0.5])
    fs.tidy(ax2); ax2.tick_params(axis="y", length=0)
    ax2.set_title("cell-level edit gain ΔS_norm (pooled cases)", fontsize=fs.FS_SMALL, loc="left", x=-0.62, pad=2)
    ax3 = sh.ax(x0 + 0.78, y0 + 0.24, 1.20, 0.34)
    c3 = cm.reset_index(drop=True)
    names = c3.unit_id.str.replace("LARRY_", "").str.replace("_to_", "→").str.replace("_", " ")
    ax3.scatter(c3.median_m_over_uniform, range(len(c3)), s=12, color=fs.FALSE_RET, lw=0, zorder=3)
    ax3.axvline(1, **fs.REF_KW)
    ax3.set_yticks(range(len(c3))); ax3.set_yticklabels(names, fontsize=fs.FS_SMALL); ax3.set_ylim(len(c3) - 0.4, -0.6)
    ax3.set_xlim(0, 1.3); ax3.set_xticks([0, 0.5, 1]); fs.tidy(ax3); ax3.tick_params(axis="y", length=0)
    ax3.set_title("LARRY pilot: clone mass / uniform", fontsize=fs.FS_SMALL, loc="left", x=-0.62, pad=2)


# ---------------------------------------------------------------------------------------------------------- text outputs
def write_text(m, allcfg, main, rows, founder, pos, b, L10, keep, fl, l3, qclone, cm, fate, top_states):
    k = keep.set_index("clade_label")
    kp_l3 = l3[l3.dataset == "KP"].set_index("mouse").value
    q_l3 = l3[l3.dataset == "Quinn"].set_index("mouse").value
    nres = int(fl.resolved.sum()); nf = len(fl)
    clade_n = rows.clade_label.value_counts()
    cap = f"""# Figure 4 | Does RNA-based matching agree with independent CRISPR lineage information?

**a**, Design (schematic). KP-Tracer tumours carry a Cas9 lineage recorder. The same single cells give scRNA counts and CRISPR
editing records. RNA counts alone enter ConfidenceOT (main configuration `{MAIN}`, UOT backbone, frozen calibration v3, M4-E);
the edits never enter the RNA-only OT. The raw allele table gives an edit agreement s for each cell pair (prior-weighted Jaccard
of indel sets; founder characters removed) and clades (average linkage on 1 − s within a family, cut at 0.5; clusters < 20 cells
pooled as minor). CRISPR-guided OT runs the same solver and calibration with cost 1 − s: it uses the edits, so its agreement with
the edits is a reference, not an independent validation.

**b**, Tumour (family) level. For each metastasis, d = retained source fraction of its own-family primary minus the mean of the
other-family primaries of the same mouse (separate primary → metastasis fits, M4-E). Grey dots: {len(m)} metastases; red lines:
mouse means ({m.mouse.nunique()} mice; mouse 3724 has no other-family primary and is not included). Mean of mouse means
{main.mean_mouse_d:.3f}, {int(main.mice_d_pos)} of {int(main.mice)} mice > 0, one-sided Wilcoxon over metastases p = {main.wilcoxon_p_all:.1e}
(descriptive; metastases of one mouse are not independent). The comparison is the other primaries of the same mouse; there is no
permutation null. Right strip: the same quantity for all {len(allcfg)} preprocessing configurations (median
{allcfg.mean_mouse_d.median():.3f}, range {allcfg.mean_mouse_d.min():.3f} to {allcfg.mean_mouse_d.max():.3f}); red dot, main configuration.

**c**, Editing structure of family 3457 T1 (mouse 3457, primary T1 and metastases K1, N2; {len(rows)} eligible cells). Rows: cells, ordered
by clade, site and RNA state; columns: {len(pos)//3} target sites × 3 cut sites. Colours give each cell's state at a position: missing,
uncut, the family founder state ({int(founder.sum())} founder positions, ▼, state in ≥ 90% of observed cells; only
TTACATGCTACCCC r1/r2 are carried by 100%), the state typical of 3457-A, the state typical of 3457-B, a state typical of both (A = B, non-founder), or another edit. Tracks: clade
(3457-A {clade_n.get('3457-A', 0)}, 3457-B {clade_n.get('3457-B', 0)}, 3457-C {clade_n.get('3457-C', 0)}, minor {clade_n.get('minor', 0)} cells), site, and the authors'
RNA state (top {len(top_states)} shown). Clade letters are ranked by size within this family (3457 T1); other families of mouse 3457
reuse the same letters. Every metastatic cell is in 3457-B. Caveat: 3457-C carries different states at almost every founder position
(same-state share with 3457-A 0.06), so it may not descend from the same founder as 3457-A and 3457-B.

**d**, RNA-OT vs CRISPR-OT on the same cells. Top: overlap of the two couplings summed into clade × clade blocks (on cells both gates
keep; median {b.shared_rna_crispr.median():.2f}, {len(b)} KP units), overlap of the RNA-OT coupling with a size-only coupling that matches clades only in
proportion to their size (median {b.shared_rna_independent.median():.2f}; closer to RNA-OT than CRISPR-OT is in
{int((b.shared_rna_independent > b.shared_rna_crispr).sum())} of {len(b)} units), and the exact cell-level overlap of the top-15 source partners of the same
target cells (median {L10.cell_top15_overlap_mean.median():.2f}; {len(L10)} unit × target-clade rows with ≥ 10 cells). High clade-level overlap therefore
does not mean the same cell partners. Bottom: 3457 T1 pooled case (primary → K1 + N2), primary cells kept by each gate, per clade:
RNA-OT keeps 3457-B {int(k.loc['3457-B','rna_kept'])}/{int(k.loc['3457-B','cells'])} and 3457-A {int(k.loc['3457-A','rna_kept'])}/{int(k.loc['3457-A','cells'])};
CRISPR-OT keeps 3457-B {int(k.loc['3457-B','crispr_kept'])}/{int(k.loc['3457-B','cells'])} and 3457-A {int(k.loc['3457-A','crispr_kept'])}/{int(k.loc['3457-A','cells'])}
(OT cells; {int(keep.cells.sum())} primary cells).

**e**, Limits and other lineage systems. Top: frozen resolution rule for the {nf} KP families in the OT units (filled = criterion
met); {nres} of {nf} are resolved. Middle: cell-level edit gain of the RNA coupling over an independent coupling (L3 matrix ΔS_norm,
main configuration, pooled cases): KP 3724 {kp_l3.get('3724', np.nan):.3f}, other KP cases {', '.join(f'{v:.3f}' for mo, v in kp_l3.items() if mo != '3724')};
Quinn A549 xenografts {', '.join(f'{v:.3f}' for v in q_l3)}. One family (3724, 20,992 cells) carries the cell-level signal. Bottom: LARRY in vitro
(lentiviral barcodes, not CRISPR) local pilot, 3,000 cells per side, M4-E: median clone mass of retained cells relative to a uniform
coupling, {', '.join(f'{v:.2f}' for v in cm.median_m_over_uniform)} (below 1 = no cell-level clone matching), while fate composition is recovered
(fate TV below the clone-label null, p = {fate.p_null_le_obs.iloc[0]:.4f} in every unit). The full TACC LARRY run is complete but its score tables were
not read for this figure. Median missing intBC share is 0.26 (KP); edits are near-saturated, founder characters are many (e.g. 3457 T4 23 of 30).
"""
    (OUT / "caption.md").write_text(cap, encoding="utf-8")
    src = f"""# Figure 4 sources (all read-only; derived tables in out/Figure4/data/)

| Panel | Source | Filter / columns | Derived table |
|---|---|---|---|
| a | schematic; definitions from lineage_validation/crispr_matrix_benchmark_20261108/MATRIX_METRICS.md, lineage_validation/rna_lineage_discordance_20261110/10_crispr_ot.py | — | — |
| b | results/crispr_matrix_benchmark_20261108/l1/l1_metastasis.tsv.gz; l1/l1_family.tsv | config_id = {MAIN}, method = M4-E; d per metastasis; mean_mouse_d, mice_d_pos, wilcoxon_p_all; all 296 configs (M4-E) | b_metastasis_d.tsv, b_all296_mean_mouse_d.tsv |
| c | results/rna_lineage_discordance_20261110/state_lineage_ext/all_cells__original/cells.tsv.gz (clade, site, state); family_check/clade_labels.tsv (3457-A/B/C, commit a235997); family_check/3457_Apc_T1/character_states.tsv (founder flag, family top state, 3457-A T1 and 3457-B K1 top states); allele table results/crispr_matrix_benchmark_20261108/inputs (KPTracer.alleleTable FINAL + unfiltered, loaded with 01_build_bundles.load_tables) | family = 3457_Apc_T1, all eligible cells | c_rows.tsv, c_edit_codes.tsv.gz, c_positions.tsv |
| d top | results/rna_lineage_discordance_20261110/explore/baseline_independent.tsv (shared_rna_crispr, shared_rna_independent; 45 KP units); explore/levels_*.tsv (cell_top15_overlap_mean, rows with cells ≥ 10) | — | d_clade_level_overlap_units.tsv, d_cell_level_top15_unit_x_clade.tsv |
| d bottom | results/rna_lineage_discordance_20261110/compare/{POOLED}__l5best/cells.tsv.gz (rna_retained, crispr_retained; side = source) joined to clades | [audit]-computed counts (not in a report) | d_3457T1_pooled_primary_kept_by_clade.tsv |
| e top | results/rna_lineage_discordance_20261110/lineages/flags.tsv (dataset = KPTracer; frozen rule ≥ 50 cells, median events ≥ 3, distinct sets ≥ 0.1, s IQR ≥ 0.05) | 11 KP OT families | e_resolution_kp_families.tsv |
| e middle | results/crispr_matrix_benchmark_20261108/summary/mouse_values.tsv.gz | level = L3_matrix, stratum = case, config = main, M4-E | e_L3_matrix_dSnorm_cases_main.tsv |
| e bottom | results/app_search_20261117/larry_pilot/score/clone_mass.tsv, fate.tsv | config = l5best, method = M4-E (local 3,000-cell pilot; TACC full-run tables not read: ssh sessions returned 255 on 2026-11-18) | e_larry_pilot_main_m4e.tsv |

Numbers quoted only in the caption: Quinn own-clone gain (L1_clone, main config, M4-E) {', '.join(f'{r.mouse} {r.value:.3f}' for r in qclone.itertuples())}
(mouse_values.tsv.gz); missing intBC share 0.26 and 3457 T4 founder 23/30 from lineages/flags.tsv / resolution.tsv; 3457-C founder
mismatch (0.06) from family_check/3457_Apc_T1/agreement.tsv as reported in the CRISPR audit.
Fixed inconsistencies: size-only control closer than CRISPR-OT in {int((b.shared_rna_independent > b.shared_rna_crispr).sum())} of 45 units (deck said 44);
clade-level overlap = median over units of the shared share of each coupling (0.90); clade labels from clade_labels.tsv only.
"""
    (OUT / "sources.md").write_text(src, encoding="utf-8")
    notes = {
        "a": {"title": "RNA and CRISPR measurements", "compared": "Two read-outs of the same cells: RNA goes into ConfidenceOT; CRISPR edits are kept aside for evaluation.",
              "observe": "The RNA-only OT never sees edits. CRISPR-guided OT uses the edits as its cost, so it is a reference, not a test.",
              "meaning": "Agreement between RNA-OT and the edits is independent evidence; agreement of CRISPR-OT with the edits is not."},
        "b": {"title": "Tumour-level matching", "compared": "For each metastasis, how much of its own-family primary is kept versus the other primaries of the same mouse.",
              "observe": f"Own family kept more: mean d {main.mean_mouse_d:.2f}, {int(main.mice_d_pos)} of {int(main.mice)} mice above 0; positive in almost all {len(allcfg)} preprocessing configurations.",
              "meaning": "At the family level, RNA-only matching follows lineage."},
        "c": {"title": "CRISPR editing structure (family 3457 T1)", "compared": f"{len(rows)} cells: their edit states, clades from the edits, site and RNA state.",
              "observe": "Founder edits are shared by the family; 3457-A stays in the primary; every metastatic cell is 3457-B.",
              "meaning": "The edits resolve sub-clones inside one tumour family; 3457-C may be a separate lineage."},
        "d": {"title": "RNA-OT versus CRISPR-OT", "compared": "Overlap of the two couplings at clade level and at the level of exact cell partners.",
              "observe": f"Clade level {b.shared_rna_crispr.median():.2f} (size-only {b.shared_rna_independent.median():.2f}); exact partners {L10.cell_top15_overlap_mean.median():.2f}. In 3457 T1, RNA-OT keeps 3457-B like CRISPR-OT but also keeps {int(k.loc['3457-A','rna_kept'])} of {int(k.loc['3457-A','cells'])} 3457-A cells.",
              "meaning": "Agreement is mostly proportional to clade size; RNA matching does not recover cell-to-cell lineage partners."},
        "e": {"title": "Limits and controls", "compared": "Edit resolution per family, cell-level edit gain, and two other lineage systems.",
              "observe": f"{nres} of {nf} families resolved; cell-level gain only in 3724; LARRY pilot clone mass below uniform.",
              "meaning": "Lineage supports RNA matching at family and composition level, not at single-cell level."},
    }
    (OUT / "panel_notes.json").write_text(json.dumps(notes, indent=1, ensure_ascii=False), encoding="utf-8")


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    fs.apply()
    m, allcfg, main_ = data_b()
    rows, code, pos, founder, cs, intbcs = data_c()
    b, L10, keep = data_d()
    fl, l3, qclone, cm, fate = data_e()
    sh = fs.Sheet(7.2, 5.4, "Figure4")
    panel_a(sh)
    panel_b(sh, m, allcfg, main_)
    top_states = panel_c(sh, rows, code, pos, founder, intbcs)
    panel_d(sh, b, L10, keep)
    panel_e(sh, fl, l3, qclone, cm)
    sh.save()
    write_text(m, allcfg, main_, rows, founder, pos, b, L10, keep, fl, l3, qclone, cm, fate, top_states)


if __name__ == "__main__":
    main()
