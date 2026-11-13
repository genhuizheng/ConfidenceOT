"""Slide versions of the RNA-OT vs CRISPR-OT figures: large type, plain labels, one message each. Reads existing tables
only (no recomputation). Writes ROOT/figures/deck_20261112/<name>.png (+ .pdf).

    python lineage_validation/rna_lineage_discordance_20261110/111_deck_figures.py --root results/rna_lineage_discordance_20261110 \
        --inputs results/crispr_matrix_benchmark_20261108/inputs --bundles results/crispr_matrix_benchmark_20261108/bundles
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "realdata_benchmark" / "v3_grid" / "figures"))
import nbt_style as st  # noqa: E402

RNA, CRISPR, GREY, INK = "#2C6E9B", "#D98C2B", "#A6A6A6", "#262626"
KPC, QUC = "#2C6E9B", "#B4702A"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def slide_style():
    matplotlib.rcParams.update({"font.family": "Arial", "font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14, "xtick.labelsize": 13,
                                "ytick.labelsize": 13, "legend.fontsize": 13, "axes.spines.top": False, "axes.spines.right": False,
                                "axes.linewidth": 1.0, "xtick.major.width": 1.0, "ytick.major.width": 1.0, "pdf.fonttype": 42})


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=250 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def scale_nbt(f=1.6):
    for k in [k for k in dir(st) if k.startswith("FS_")]:
        setattr(st, k, getattr(st, k) * f)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--bundles", type=Path, required=True)
    a = p.parse_args()
    out = a.root / "figures" / "deck_20261112"; out.mkdir(parents=True, exist_ok=True)
    rd = lambda f: pd.read_csv(f, sep="\t", dtype={"mouse": str}, low_memory=False)  # noqa: E731
    slide_style()
    ab = rd(a.root / "summary" / "a_vs_b_units.tsv"); ab = ab[ab.tag == "l5best"]

    # D1 what each OT matches: edit similarity and RNA similarity of each OT's top-15 sources
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (cr, cc, title) in zip(axs, (("s_top_rnaot", "s_top_crisprot", "How related are the matched cells?\n(CRISPR edit similarity, 0-1)"),
                                         ("r_top_rnaot", "r_top_crisprot", "How alike are the matched cells?\n(RNA similarity)"))):
        for ds, colr in (("KP", KPC), ("Quinn", QUC)):
            d = ab[ab.dataset_short == ds]
            ax.scatter(d[cr], d[cc], s=45, color=colr, label=f"{ds} ({len(d)} units)", alpha=0.85, edgecolor="white", lw=0.5)
        lo = min(ab[[cr, cc]].min().min(), 0); hi = max(ab[[cr, cc]].max().max(), 1 if cr.startswith("s") else 0.8)
        ax.plot([lo, hi], [lo, hi], ls="--", color="#777777", lw=1); ax.set_xlim(lo - 0.03, hi + 0.03); ax.set_ylim(lo - 0.03, hi + 0.03)
        ax.set_xlabel("matched by RNA-OT"); ax.set_ylabel("matched by CRISPR-OT"); ax.set_title(title)
    axs[0].legend(frameon=False, loc="lower right")
    axs[0].text(0.25, 0.71, "above the line:\nCRISPR-OT partners\nare closer relatives", va="center", fontsize=12, color="#555555")  # data coords: empty area above the diagonal
    axs[1].text(0.97, 0.58, "below the line:\nRNA-OT partners\nlook more alike", transform=axs[1].transAxes, ha="right", va="center", fontsize=12, color="#555555")
    fig.tight_layout(); save(fig, out, "D1_what_each_OT_matches")

    # D2 do the two OTs keep the same cells? (Cohen's kappa of the gates)
    grp = np.where(ab.dataset_short == "Quinn", "Quinn\n(all metastases)", np.where(ab.stratum == "pair", "KP\n(one metastasis)", "KP\n(all metastases)"))
    ab = ab.assign(group=grp)
    order = ["KP\n(one metastasis)", "KP\n(all metastases)", "Quinn\n(all metastases)"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, g in enumerate(order):
        d = ab[ab.group == g]
        for j, (side, colr) in enumerate((("source", "#7F7F7F"), ("target", INK))):
            v = d[f"kappa_{side}"].dropna().to_numpy(); x = i * 3 + j + (np.random.default_rng(i * 3 + j).random(len(v)) - 0.5) * 0.4
            ax.scatter(x, v, s=40, color=colr, alpha=0.75, label=("primary-tumour cells" if side == "source" else "metastasis cells") if i == 0 else None)
            if len(v):
                ax.plot([i * 3 + j - 0.3, i * 3 + j + 0.3], [np.median(v)] * 2, color=colr, lw=2.5)
    ax.axhline(0, ls="--", color="#777777", lw=1)
    ax.set_xticks([i * 3 + 0.5 for i in range(len(order))]); ax.set_xticklabels(order)
    ax.set_ylabel("agreement on which cells to keep\n(Cohen's kappa: 0 = chance, 1 = identical)")
    ax.legend(frameon=False, loc="upper right"); ax.set_ylim(-0.15, 1.05)
    fig.tight_layout(); save(fig, out, "D2_same_cells_kept")

    # D3 clade level vs cell level
    bv = pd.read_csv(a.root / "explore" / "baseline_independent.tsv", sep="\t")
    import glob
    lv = pd.concat([pd.read_csv(f, sep="\t") for f in glob.glob(str(a.root / "explore" / "levels_*.tsv"))]); lv = lv[lv.cells >= 10]
    vals = [("RNA-OT vs CRISPR-OT\n(same clade pairs)", bv.shared_rna_crispr.median(), GREY),
            ("RNA-OT vs a coupling that\nonly uses clade sizes", bv.shared_rna_independent.median(), RNA),
            ("RNA-OT vs CRISPR-OT\n(same cells: top-15 partners)", lv.cell_top15_overlap_mean.median(), INK)]
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (lab, v, colr) in enumerate(vals):
        ax.bar(i, v, color=colr, width=0.6); ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=15, fontweight="bold")
    ax.set_xticks(range(3)); ax.set_xticklabels([v[0] for v in vals]); ax.set_ylim(0, 1.1); ax.set_ylabel("overlap (median over KP units)")
    ax.axvline(1.5, color="#BBBBBB", lw=1, ls=":")
    ax.text(0.75, 1.05, "clade level", ha="center", fontsize=13, color="#555555"); ax.text(2, 1.05, "cell level", ha="center", fontsize=13, color="#555555")
    fig.tight_layout(); save(fig, out, "D3_clade_vs_cell_level")

    # D4 which states are tied to few clades: per state, mice where it is concentrated / at chance / spread
    S = rd(a.root / "ext_summary" / "states_runs.tsv"); S = S[(S.run == "all_cells__original") & (S.scope == "sensitivity_all")]
    focus = ["Early EMT-1", "Apc Early", "Mesenchymal-2", "Lung progenitor-like", "Endoderm-like", "Early EMT-2", "Pre-EMT", "High plasticity", "AT2-like"]
    rows = []
    for x in focus:
        d = S[S.state == x]
        rows.append((x, int((d.hill_z <= -2).sum()), int(((d.hill_z > -2) & (d.hill_z < 2)).sum()), int((d.hill_z >= 2).sum())))
    R = pd.DataFrame(rows, columns=["state", "few", "chance", "many"]).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    y = np.arange(len(R))
    ax.barh(y, R.few, color="#A23B5B", label="in fewer clades than chance")
    ax.barh(y, R.chance, left=R.few, color="#D9D9D9", label="as expected by chance")
    ax.barh(y, R.many, left=R.few + R.chance, color="#5E8FA8", label="in more clades than chance")
    for i, r in enumerate(R.itertuples()):
        ax.text(r.few + r.chance + r.many + 0.2, i, f"fewer: {r.few} / chance: {r.chance} / more: {r.many}", va="center", fontsize=12)
    ax.set_yticks(y); ax.set_yticklabels(R.state); ax.set_xlabel("number of mice (state with >= 50 cells)")
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, fontsize=12)
    ax.set_xlim(0, max(R[["few", "chance", "many"]].sum(1)) + 6.5); ax.set_xticks(range(0, int(max(R[["few", "chance", "many"]].sum(1))) + 1, 5))
    fig.tight_layout(); save(fig, out, "D4_states_tied_to_few_clades")

    # D5 KP 3513: which tumours (families) carry each state
    AC = rd(a.root / "state_lineage_ext" / "all_cells__original" / "cells.tsv.gz"); g = AC[(AC.mouse == "3513") & (AC.clade != "minor")]
    vc = g.state.value_counts(); states = list(vc[vc >= 50].index)
    fams = sorted(g.family.unique()); fcol = dict(zip(fams, ["#2C6E9B", "#B4702A", "#4E8A6B", "#7D3C98", "#8C7A5B", "#C0392B"]))
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for i, s_ in enumerate(states[::-1]):
        h = g[g.state == s_]; comp = h.family.value_counts(normalize=True).reindex(fams).fillna(0); x = 0
        for f_, v in comp.items():
            ax.barh(i, v, left=x, color=fcol[f_], edgecolor="white", lw=0.8); x += v
        ax.text(1.02, i, f"{int((h.family.value_counts() >= 5).sum())} tumours, largest {comp.max():.0%}", va="center", fontsize=12)
    ax.set_yticks(range(len(states))); ax.set_yticklabels(states[::-1]); ax.set_xlim(0, 1.36); ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel("share of the state's cells")
    handles = [plt.Rectangle((0, 0), 1, 1, color=fcol[f]) for f in fams]
    ax.legend(handles, [f"tumour {f.split('_')[-1]}" for f in fams], frameon=False, ncol=len(fams), loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=12)
    fig.tight_layout(); save(fig, out, "D5_states_across_tumours_3513")

    # D6 KP 3724 sites: clades and states per site (simple)
    ACm = AC[(AC.mouse == "3724")]
    sites = sorted(ACm.site.unique(), key=lambda s: (not ACm[ACm.site == s].primary.iloc[0], s))
    STATE_COL = load("f92", HERE / "92_flow_figure.py").STATE_COL
    fig, ax = plt.subplots(figsize=(12, 4.8)); ax.axis("off"); wsite = 0.17; gap = (1 - wsite * len(sites)) / (len(sites) - 1)
    for si, s_ in enumerate(sites):
        h = ACm[ACm.site == s_]; x0 = si * (wsite + gap); cl = h.clade.value_counts(); cl = cl[cl / len(h) >= 0.02]; x = x0
        for c_, n_ in cl.items():
            w_ = n_ / cl.sum() * wsite; comp = h[h.clade == c_].state.value_counts(normalize=True); y = 0
            for stt, fr in comp.items():
                ax.add_patch(plt.Rectangle((x, y), w_, fr, color=STATE_COL.get(stt, "#E3E6E8"))); y += fr
            ax.add_patch(plt.Rectangle((x, 0), w_, 1, fill=False, edgecolor="white", lw=1.5))
            if w_ > 0.04:
                lab = "small clades" if c_ == "minor" else f"clade {c_.split('|')[-1][1:]}"
                ax.text(x + w_ / 2, -0.06, lab, ha="center", va="top", fontsize=12)
            x += w_
        kind = "primary" if h.primary.iloc[0] else "metastasis"
        ax.text(x0 + wsite / 2, 1.04, f"{s_.split('_')[-1]} ({kind})\n{len(h):,} cells", ha="center", va="bottom", fontsize=13)
    present = [s for s in STATE_COL if s in set(ACm.state) and (ACm.state == s).mean() > 0.005]
    handles = [plt.Rectangle((0, 0), 1, 1, color=STATE_COL[s]) for s in present]
    ax.legend(handles, present, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=12)
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.2, 1.25)
    fig.tight_layout(); save(fig, out, "D6_primary_vs_metastasis_3724")

    # cell-level trees of the two case families (96_ layout, "CRISPR clade" track name)
    T96 = load("t96b", HERE / "96_cell_tree.py")
    st.apply()
    cot = load("cot", HERE / "10_crispr_ot.py"); bb = cot.bundles_module(); t, prior = bb.load_tables(a.inputs)["KPTracer"]
    res = rd(a.root / "state_lineage_ext" / "all_cells__original" / "resolution.tsv").set_index("lineage")
    for fam in ("3457_Apc_T1", "3724_NT_T1"):
        rr = res.loc[fam]
        flag = f"{rr.median_events_per_cell:.0f} edits per cell; " + ("resolved" if rr.resolved else "low CRISPR resolution")
        g2 = AC[AC.family == fam]
        T96.draw_cell_tree(fam, g2, fam, t, prior, flag, out, 900, STATE_COL, clade_title="CRISPR clade")
    print("deck figures written to", out)


if __name__ == "__main__":
    main()
