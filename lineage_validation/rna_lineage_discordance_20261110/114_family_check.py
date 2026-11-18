"""Primary / metastasis check of one KP family on the raw allele table (no OT, no new clustering): which clades each
site holds, intBC identity, missingness, informative edits, and how far the cells of each clade x site group carry the
same indel states. Founder characters (one state in >= 90% of the family's observed cells, which matrix_core.Lineage
removes before computing s and drawing the edit matrix) are shown here, not removed.

    python lineage_validation/rna_lineage_discordance_20261110/114_family_check.py --root results/rna_lineage_discordance_20261110 \
        --inputs results/crispr_matrix_benchmark_20261108/inputs --family 3457_Apc_T1

Clade labels: within each family, named clades ranked by cell count (all eligible cells) and called <mouse>-A, -B, ...;
"minor" stays minor. Written for every family to ROOT/family_check/clade_labels.tsv (used by the figures).
Writes ROOT/family_check/<family>/: groups.tsv (cells, sites, missingness, events), intbc_observed.tsv,
character_states.tsv (per intBC x site: top state per group and its share), agreement.tsv (group x group: share of
jointly observed intBC x site positions with the same state, with and without founder positions), and
ROOT/figures/family_check/<family>_{composition,states}.{png,pdf}.
"""

from __future__ import annotations

import argparse
import importlib.util
import string
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "realdata_benchmark" / "v3_grid" / "figures"))
import nbt_style as st  # noqa: E402

SITES = ("r1", "r2", "r3")
FOUNDER, SEED, PER_GROUP = 0.9, 20261117, 120


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def clade_labels(AC):
    rows = []
    for fam, g in AC.groupby("family"):
        mouse = str(g.mouse.iloc[0])
        vc = g[g.clade != "minor"].clade.value_counts()
        for k, (c, n) in enumerate(vc.items()):
            lab = f"{mouse}-{string.ascii_uppercase[k]}" if k < 26 else f"{mouse}-Z{k}"
            rows.append({"family": fam, "clade": c, "label": lab, "cells": int(n), "cluster_id": c.split("|")[-1]})
        rows.append({"family": fam, "clade": "minor", "label": "minor", "cells": int((g.clade == "minor").sum()), "cluster_id": "minor"})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--family", default="3457_Apc_T1")
    a = p.parse_args()
    out = a.root / "family_check"; fo = out / a.family; fo.mkdir(parents=True, exist_ok=True)
    fig_out = a.root / "figures" / "family_check"; fig_out.mkdir(parents=True, exist_ok=True)
    AC = pd.read_csv(a.root / "state_lineage_ext" / "all_cells__original" / "cells.tsv.gz", sep="\t", dtype={"cell": str, "mouse": str})
    CL = clade_labels(AC); CL.to_csv(out / "clade_labels.tsv", sep="\t", index=False)
    lab = CL[CL.family == a.family].set_index("clade").label
    g = AC[AC.family == a.family].copy(); g["clade_label"] = g.clade.map(lab)
    prim = g[g.primary.astype(bool)].site.unique()
    g["site_kind"] = np.where(g.primary.astype(bool), "primary", "metastasis")
    g["group"] = g.clade_label + " · " + g.site.str.split("_").str[-1] + np.where(g.primary.astype(bool), " (primary)", "")
    cot = load("cot", HERE / "10_crispr_ot.py"); bb = cot.bundles_module(); t, _ = bb.load_tables(a.inputs)["KPTracer"]
    t = t[(t.lineage == a.family) & t.cell.isin(set(g.cell))].copy()
    t = t.merge(g[["cell", "group", "clade_label", "site_kind"]], on="cell")
    order = (g.groupby("group").agg(n=("cell", "size"), lab=("clade_label", "first"), prim=("primary", "first"))
             .reset_index().sort_values(["lab", "prim", "group"], ascending=[True, False, True]))
    order = order[order.lab != "minor"].group.tolist() + order[order.lab == "minor"].group.tolist()
    intbcs = sorted(t.intBC.unique())

    # groups: cells, missingness, events
    obs = t.groupby("cell").intBC.nunique()
    G = g.assign(intbc_observed=g.cell.map(obs).fillna(0).astype(int)).groupby("group").agg(
        clade=("clade_label", "first"), site=("site", "first"), kind=("site_kind", "first"), cells=("cell", "size"),
        median_intbc_observed=("intbc_observed", "median"), median_missing_share=("missing_share", "median"),
        median_informative_events=("events", "median")).reindex(order)
    G["intbcs_in_family"] = len(intbcs); G.to_csv(fo / "groups.tsv", sep="\t")
    # intBC identity: observed share per group
    n_g = g.groupby("group").size()
    IO = (t.groupby(["intBC", "group"]).cell.nunique().unstack(fill_value=0) / n_g).reindex(columns=order).fillna(0)
    IO.round(3).to_csv(fo / "intbc_observed.tsv", sep="\t")
    # character states per intBC x site
    M = t.melt(id_vars=["cell", "intBC", "group"], value_vars=list(SITES), var_name="site_r", value_name="state")
    M["pos"] = M.intBC + ":" + M.site_r
    fam_top = M.groupby("pos").state.agg(lambda s: s.value_counts().index[0]); fam_share = M.groupby("pos").state.agg(lambda s: s.value_counts(normalize=True).iloc[0])
    founder = {p_: s_ for p_, s_ in fam_top.items() if s_ != "None" and fam_share[p_] >= FOUNDER}
    rows = []
    for pos, h in M.groupby("pos"):
        rec = {"position": pos, "family_top_state": fam_top[pos], "family_top_share": round(float(fam_share[pos]), 3), "founder": pos in founder}
        for grp in order:
            v = h[h.group == grp].state.value_counts(normalize=True)
            rec[f"{grp} | top"] = v.index[0] if len(v) else ""; rec[f"{grp} | share"] = round(float(v.iloc[0]), 3) if len(v) else np.nan
        rows.append(rec)
    CS = pd.DataFrame(rows); CS.to_csv(fo / "character_states.tsv", sep="\t", index=False)
    # agreement between groups: share of jointly observed positions with the same state (cells subsampled per group)
    rng = np.random.default_rng(SEED)
    W = M.pivot_table(index="cell", columns="pos", values="state", aggfunc="first")
    pick = {grp: rng.choice(g.cell[g.group == grp].to_numpy(), min(PER_GROUP, int((g.group == grp).sum())), replace=False) for grp in order}
    fpos = np.array([c in founder for c in W.columns])
    AG = []
    for g1 in order:
        A1 = W.reindex(pick[g1]).to_numpy(object)
        for g2 in order:
            A2 = W.reindex(pick[g2]).to_numpy(object)
            for tag, cols in (("all positions", np.ones(len(fpos), bool)), ("without founder positions", ~fpos)):
                x, y = A1[:, cols], A2[:, cols]
                both = (pd.notna(x)[:, None, :] & pd.notna(y)[None, :, :])
                same = (x[:, None, :] == y[None, :, :]) & both
                if g1 == g2:
                    k = np.arange(len(x)); both[k, k] = False; same[k, k] = False
                AG.append({"group_a": g1, "group_b": g2, "positions": tag, "same_state_share": float(same.sum() / max(both.sum(), 1))})
    AGt = pd.DataFrame(AG); AGt.to_csv(fo / "agreement.tsv", sep="\t", index=False)

    # figure 1: composition, cells per clade x site
    st.apply()
    sites = list(prim) + sorted(set(g.site) - set(prim))
    labs = [l_ for l_ in CL[CL.family == a.family].label if l_ in set(g.clade_label)]
    cols = dict(zip([l_ for l_ in labs if l_ != "minor"], ["#2C6E9B", "#B4702A", "#4E8A6B", "#7D3C98", "#C0392B", "#5E8FA8", "#8C7A5B"])); cols["minor"] = "#D3D7DA"
    fig, ax = plt.subplots(figsize=(3.6, 2.2))
    ct = pd.crosstab(g.site, g.clade_label).reindex(index=sites, columns=labs).fillna(0)
    left = np.zeros(len(sites))
    for c in labs:
        v = ct[c].to_numpy(); ax.barh(range(len(sites)), v, left=left, color=cols[c], label=c, height=0.6, edgecolor="white", lw=0.5)
        for i, (l0, vv) in enumerate(zip(left, v)):
            if vv >= 0.06 * ct.sum(1).max():
                ax.text(l0 + vv / 2, i, f"{int(vv)}", ha="center", va="center", fontsize=st.FS_MIN, color="white" if c != "minor" else st.INK)
        left += v
    ax.set_yticks(range(len(sites))); ax.set_yticklabels([f"{s_.split('_')[-1]} ({'primary' if s_ in prim else 'metastasis'})" for s_ in sites], fontsize=st.FS_TICK)
    ax.invert_yaxis(); ax.set_xlabel("cells", fontsize=st.FS_AXLAB); st.tidy(ax)
    ax.legend(frameon=False, fontsize=st.FS_MIN, ncol=len(labs), loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.set_title(f"{a.family}: clade composition of each site (clades from CRISPR edits)", fontsize=st.FS_TITLE, loc="left", pad=14)
    fig.savefig(fig_out / f"{a.family}_composition.png", dpi=400, bbox_inches="tight"); fig.savefig(fig_out / f"{a.family}_composition.pdf", bbox_inches="tight"); plt.close(fig)

    # figure 2: per-cell character states, coloured by whose dominant state they carry
    big = [l_ for l_ in labs if l_ != "minor"][:2]
    refA = g.group[(g.clade_label == big[0])].value_counts().index[0]
    met_groups = [x for x in order if x.startswith(big[1]) and "(primary)" not in x] if len(big) > 1 else []
    refB_cells = g.cell[g.group.isin(met_groups)] if met_groups else g.cell[g.clade_label == (big[1] if len(big) > 1 else big[0])]
    topA = W.reindex(g.cell[g.group == refA]).apply(lambda s: s.dropna().value_counts().index[0] if s.notna().any() else None)
    topB = W.reindex(refB_cells).apply(lambda s: s.dropna().value_counts().index[0] if s.notna().any() else None)
    posord = sorted(W.columns, key=lambda c: (intbcs.index(c.split(":")[0]), c.split(":")[1]))
    rows_c, ybreak, ylab = [], [], []
    for grp in order:
        cells = np.sort(pick[grp]); rows_c += list(cells); ybreak.append(len(rows_c)); ylab.append((grp, len(cells), int((g.group == grp).sum())))
    X = W.reindex(index=rows_c, columns=posord)
    code = np.zeros(X.shape, dtype=int)  # 0 missing, 1 unedited, 2 other edit, 3 = A's state, 4 = B's state, 5 = shared (A's = B's)
    for j, c in enumerate(posord):
        v = X[c].to_numpy(object); a_, b_ = topA.get(c), topB.get(c)
        k = np.where(pd.isna(v), 0, np.where(v == "None", 1, 2))
        isA = (v == a_) & (v != "None"); isB = (v == b_) & (v != "None")
        k = np.where(isA & isB, 5, np.where(isA, 3, np.where(isB, 4, k)))
        code[:, j] = k
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#FFFFFF", "#EEEEEE", "#B0B5B9", "#2C6E9B", "#D98C2B", "#262626"])
    fig = plt.figure(figsize=(7.2, 4.8)); ax = fig.add_axes([0.24, 0.09, 0.56, 0.80])
    ax.imshow(code, aspect="auto", cmap=cmap, vmin=0, vmax=5, interpolation="nearest")
    for yb in ybreak[:-1]:
        ax.axhline(yb - 0.5, color=st.INK, lw=0.6)
    xb = [j for j in range(1, len(posord)) if posord[j].split(":")[0] != posord[j - 1].split(":")[0]]
    for x_ in xb:
        ax.axvline(x_ - 0.5, color="white", lw=1.6)
    starts = [0] + xb; ends = xb + [len(posord)]
    ax.set_xticks([(s0 + e0 - 1) / 2 for s0, e0 in zip(starts, ends)])
    ax.set_xticklabels([f"intBC {intbcs.index(posord[s0].split(':')[0]) + 1}" for s0 in starts], rotation=60, ha="right", fontsize=st.FS_MIN)
    y0 = 0; yt, yl = [], []
    for (grp, n_draw, n_all), yb in zip(ylab, ybreak):
        yt.append((y0 + yb - 1) / 2); yl.append(f"{grp}\n{n_all} cells" + (f" ({n_draw} drawn)" if n_draw < n_all else "")); y0 = yb
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=st.FS_MIN)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    lg = fig.add_axes([0.82, 0.35, 0.17, 0.5]); lg.axis("off"); lg.set_xlim(0, 1); lg.set_ylim(0, 1)
    items = [("#262626", f"same state in {big[0]} and {big[1] if len(big) > 1 else ''} (shared)"),
             ("#2C6E9B", f"main state of {big[0]}"), ("#D98C2B", f"main state of {big[1] if len(big) > 1 else ''} metastasis cells"),
             ("#B0B5B9", "other edit"), ("#EEEEEE", "unedited"), ("#FFFFFF", "intBC not detected")]
    for i, (c_, t_) in enumerate(items):
        lg.add_patch(plt.Rectangle((0, 0.95 - i * 0.12), 0.1, 0.07, facecolor=c_, edgecolor=st.INK, lw=0.3))
        lg.text(0.14, 0.985 - i * 0.12, t_, fontsize=st.FS_MIN, va="center")
    fig.text(0.01, 0.985, f"{a.family}: every intBC x target site (3 columns per intBC), raw allele table, founder states included",
             fontsize=st.FS_TITLE, va="top")
    fig.savefig(fig_out / f"{a.family}_states.png", dpi=400); fig.savefig(fig_out / f"{a.family}_states.pdf"); plt.close(fig)

    # console digest
    print(G.to_string())
    print("founder positions:", len(founder), "of", W.shape[1])
    piv = AGt.pivot_table(index=["positions", "group_a"], columns="group_b", values="same_state_share").reindex(columns=order)
    print(piv.round(2).to_string())


if __name__ == "__main__":
    main()
