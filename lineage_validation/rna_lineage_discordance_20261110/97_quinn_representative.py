"""Quinn 2021 (A549 xenograft) representative figures, the counterparts of the KP set (no refits, no new thresholds).

    python lineage_validation/rna_lineage_discordance_20261110/97_quinn_representative.py --root results/rna_lineage_discordance_20261110 \
        --inputs results/crispr_matrix_benchmark_20261108/inputs --bundles results/crispr_matrix_benchmark_20261108/bundles

Per experiment (5k, 10k, 30k, 100k; one L2/L3 unit each, source = LL, targets = metastatic sites):
  lineage  the LineageGroup (LG): a separate lentiviral integration, so LGs are independent lineages (as KP families)
  state    Leiden clusters on the L5-best RNA coordinates (35_quinn_states.py). Quinn has no authors' states; state ids
           are per experiment and not comparable between experiments
  Q1_cell_tree_<exp>_<LG>  the two largest resolved LGs of each experiment (lineages/resolution.tsv): cell-level tree
           with clades within the LG (the A2 rule: UPGMA on 1 - s, cut 0.5, clusters < 20 cells = minor), RNA state, site
  Q2_state_lineages_<exp>  per state (>= 50 cells): its cells split by LG (the 8 largest coloured, the rest grey) with
           the within-site null's expected largest-LG share (199 permutations, seed 20261111) as a tick
  Q3_state_concentration   per experiment and state: largest LG share minus that null; dot area = cells
  Q4_sites_<exp>           per site (LL first): LG x state mosaic; LGs < 2% of a site pooled as "other LGs"; a site with
           < 2 LGs or < 2 states (>= 5 cells) is "not identifiable"
  Q5_flow_<exp>            92_flow_figure.py's difference flow with LGs as the lineage units (no tree: LGs are independent);
           LGs < 2% of the common-support cells on both sides pooled as Other
Tables: ROOT/quinn_rep/ (cells.tsv.gz, flows/, units.tsv, gates.tsv, state_concentration.tsv).
"""

from __future__ import annotations

import argparse
import importlib.util
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "realdata_benchmark" / "v3_grid" / "figures"))
sys.path.insert(0, str(REPO / "lineage_validation" / "crispr_matrix_benchmark_20261108"))
import matrix_core as mc  # noqa: E402
import nbt_style as st  # noqa: E402

SEED, NPERM, MIN_STATE, OTHER = 20261111, 199, 50, 0.02
EXPS = ["5k", "10k", "30k", "100k"]
LG_COL = ["#2C6E9B", "#B4702A", "#4E8A6B", "#7D3C98", "#C0392B", "#5E8FA8", "#8C7A5B", "#A3A86B"]
STATE_PAL = ["#8FB996", "#C9A227", "#7D3C98", "#A23B5B", "#4E8A6B", "#8C7A5B", "#6F9FB0", "#D1495B", "#E3D08A", "#5B4A8A",
             "#BE98CF", "#9C6644", "#A3A86B", "#DB9BAE", "#3F7F7F", "#5E4030", "#CBB295", "#A9C8D6"]
SITE_PAL = ["#2C6E9B", "#B4702A", "#4E8A6B", "#7D3C98", "#C0392B", "#8C7A5B", "#A3A86B", "#5E8FA8"]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def save(fig, out, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"{name}.{ext}", dpi=400 if ext == "png" else None)
    plt.close(fig)


def permute_within(states, site, rng):
    out = states.copy()
    for s in np.unique(site):
        i = np.flatnonzero(site == s); out[i] = states[rng.permutation(i)]
    return out


def legend_row(fig, items, y, x0=0.02, h=0.06):
    lx = fig.add_axes([0, y, 1, h]); lx.axis("off"); lx.set_xlim(0, 10); lx.set_ylim(0, 2)
    x, row = x0 * 10, 1.5
    for name, colr in items:
        w_ = 0.3 + 0.055 * len(name)
        if x + w_ > 9.95:
            x, row = x0 * 10, row - 1
        lx.add_patch(Rectangle((x, row - 0.3), 0.13, 0.6, facecolor=colr, edgecolor="none")); lx.text(x + 0.17, row, name, fontsize=st.FS_MIN, va="center")
        x += w_


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--bundles", type=Path, required=True)
    a = p.parse_args()
    out = a.root / "figures" / "quinn_rep"; out.mkdir(parents=True, exist_ok=True)
    tab = a.root / "quinn_rep"; (tab / "flows").mkdir(parents=True, exist_ok=True)
    st.apply()
    F92 = load("f92", HERE / "92_flow_figure.py"); T96 = load("t96", HERE / "96_cell_tree.py")
    cot = load("cot", HERE / "10_crispr_ot.py"); bb = cot.bundles_module(); tabs = bb.load_tables(a.inputs)
    res = pd.read_csv(a.root / "lineages" / "resolution.tsv", sep="\t", dtype={"mouse": str})
    res = res[res.dataset != "KPTracer"]
    A = a.root / "inputs_A"; cid = F92.CONFIGS["l5best"]
    conc_rows, urows, grows, allcells = [], [], [], []
    for exp in EXPS:
        u = f"L23__Quinn2021_{exp}__LL__metastases"
        b = pickle.load(open(a.bundles / f"{u}.pkl", "rb"))
        t, prior = tabs[f"Quinn2021:{exp}"]
        ids = np.r_[b["source_ids"], b["target_ids"]].astype(str)
        C = pd.DataFrame({"cell": ids, "side": np.r_[np.repeat("source", len(b["source_ids"])), np.repeat("target", len(b["target_ids"]))],
                          "site": np.r_[np.repeat("LL", len(b["source_ids"])), np.asarray(b["target_site"]).astype(str)],
                          "lineage": pd.Series(np.r_[b["source_lineage"], b["target_lineage"]]).astype(object).fillna("none").astype(str).to_numpy()})
        C["table_cell"] = bb.map_ids(C.cell, pd.Index(t.drop_duplicates("cell").cell))
        sts = pd.read_csv(a.root / "states" / f"{u}__l5best.tsv.gz", sep="\t", dtype={"cell": str})
        C["state"] = C.cell.map(sts.drop_duplicates("cell").set_index("cell").state).fillna("NA")
        C["primary"] = C.site == "LL"
        C = C[C.lineage.str.startswith("LG")].reset_index(drop=True)
        allcells.append(C.assign(experiment=exp))
        state_order = sorted(C.state.unique(), key=lambda s: (len(s), s))
        scol = {s: STATE_PAL[i % len(STATE_PAL)] for i, s in enumerate(state_order)}
        sites = ["LL"] + [s for s in C.site.value_counts().index if s != "LL"]
        sitecol = {"LL": "#5F666B", **{s: SITE_PAL[i % len(SITE_PAL)] for i, s in enumerate(sites[1:])}}
        lg_order = list(C.lineage.value_counts().index); lgcol = {g: LG_COL[i] for i, g in enumerate(lg_order[:len(LG_COL)])}
        # Q1 cell trees: two largest resolved LGs, clades within the LG by the A2 rule
        rr = res[(res.mouse == exp) & res.resolved].sort_values("cells", ascending=False).head(2)
        for _, r in rr.iterrows():
            g = C[(C.lineage == r.lineage) & C.table_cell.notna()].copy()
            L = mc.Lineage(t[(t.lineage == r.lineage) & t.cell.isin(set(g.table_cell))], g.table_cell.to_numpy(), np.repeat("source", len(g)), prior)
            s, ok = L.agreement(np.arange(len(g)), np.arange(len(g)))
            d = squareform(1 - np.asarray(s, dtype=np.float64), checks=False); okv = squareform(np.asarray(ok, dtype=np.uint8), checks=False).astype(bool)
            del s, ok
            d[~okv] = np.median(d[okv]) if okv.any() else 1.0
            lab = fcluster(linkage(d, "average"), t=0.5, criterion="distance"); del d
            vc = pd.Series(lab).value_counts(); big = set(vc[vc >= 20].index)
            g["clade"] = [f"{r.lineage}|c{x}" if x in big else "minor" for x in lab]
            g2 = g.rename(columns={"cell": "rna_cell"}).rename(columns={"table_cell": "cell"})
            flag = (f"median edits/cell {r.median_events_per_cell:.0f}, distinct edit sets {r.distinct_sets_share:.2f}, resolved; "
                    f"RNA states = our Leiden clusters ({exp})")
            T96.draw_cell_tree(f"Q1_cell_tree_{exp}_{r.lineage}", g2, r.lineage, t, prior, flag, out, 1200, scol, site_col=sitecol,
                               clade_title="clade within LG")
        # Q2 per state: cells by LG, null largest-LG share
        stv = C.state.value_counts(); states = [s for s in state_order if stv.get(s, 0) >= MIN_STATE]
        rng = np.random.default_rng(SEED)
        st_arr, site_arr, lg_arr = C.state.to_numpy(), C.site.to_numpy(), C.lineage.to_numpy()
        lgi = pd.Index(lg_order).get_indexer(lg_arr)

        def max_share(sa):
            o = {}
            for s_ in states:
                m = sa == s_
                o[s_] = np.bincount(lgi[m], minlength=len(lg_order)).max() / m.sum()
            return o
        obs = max_share(st_arr); nul = [max_share(permute_within(st_arr, site_arr, rng)) for _ in range(NPERM)]
        for s_ in states:
            nv = np.array([n[s_] for n in nul])
            conc_rows.append({"experiment": exp, "state": s_, "cells": int((st_arr == s_).sum()), "lgs_ge5": int((C[C.state == s_].lineage.value_counts() >= 5).sum()),
                              "max_share": obs[s_], "max_share_null": nv.mean(), "excess": obs[s_] - nv.mean(),
                              "z": (obs[s_] - nv.mean()) / nv.std() if nv.std() > 0 else np.nan})
        H = 1.0 + 0.24 * len(states)
        fig = plt.figure(figsize=(7.2, H)); ax = fig.add_axes([1.0 / 7.2, 0.55 / H, 4.8 / 7.2, 0.24 * len(states) / H])
        ax.set_xlim(0, 1); ax.set_ylim(len(states) - 0.5, -0.5)
        for s_ in ("top", "right", "left"):
            ax.spines[s_].set_visible(False)
        cr = {r_["state"]: r_ for r_ in conc_rows if r_["experiment"] == exp}
        for i, s_ in enumerate(states):
            h = C[C.state == s_]; comp = h.lineage.value_counts()
            comp = comp.reindex([g_ for g_ in lg_order if g_ in comp.index])
            x = 0
            for g_, n_ in comp.items():
                w_ = n_ / len(h)
                ax.add_patch(Rectangle((x, i - 0.32), w_, 0.64, facecolor=lgcol.get(g_, "#C9CDD1"), edgecolor="white", lw=0.4)); x += w_
            ax.plot([cr[s_]["max_share_null"]] * 2, [i - 0.42, i + 0.42], color=st.INK, lw=0.9)
            ax.text(1.02, i, f"{cr[s_]['lgs_ge5']} LGs  (n={len(h)})", fontsize=st.FS_MIN, va="center")
        ax.set_yticks(range(len(states))); ax.set_yticklabels(states, fontsize=st.FS_TICK); ax.tick_params(axis="y", length=0)
        ax.set_xlabel("share of the state's cells", fontsize=st.FS_AXLAB)
        hl = [Rectangle((0, 0), 1, 1, facecolor=lgcol[g_]) for g_ in lgcol] + [Rectangle((0, 0), 1, 1, facecolor="#C9CDD1")]
        ax.legend(hl, list(lgcol) + ["other LGs"], frameon=False, fontsize=st.FS_MIN, ncol=9, loc="lower left", bbox_to_anchor=(0, 1.0))
        fig.text(0.01, 1 - 0.04 / H, f"Quinn {exp}: which lineages (LineageGroups, independent integrations) carry each RNA state (our Leiden clusters); "
                 "tick = largest-LG share expected under the within-site null", fontsize=st.FS_MIN, va="top")
        save(fig, out, f"Q2_state_lineages_{exp}")
        # Q4 site mosaics
        sites_d = [s for s in sites if (C.site == s).sum() >= 20]
        fig = plt.figure(figsize=(7.2, 2.7)); ax = fig.add_axes([0.04, 0.26, 0.94, 0.56]); ax.axis("off")
        gapw = 0.03; wsite = (1 - gapw * (len(sites_d) - 1)) / len(sites_d)
        for si, s_ in enumerate(sites_d):
            h = C[C.site == s_]; x0 = si * (wsite + gapw); vc = h.lineage.value_counts(normalize=True)
            keep = list(vc[vc >= OTHER].index); lab_ = h.lineage.where(h.lineage.isin(keep), "other LGs")
            order_ = keep + (["other LGs"] if (lab_ == "other LGs").any() else [])
            x = x0
            for g_ in order_:
                hh = h[lab_ == g_]; w_ = len(hh) / len(h) * wsite; comp = hh.state.value_counts(normalize=True); y = 0
                for s2, fr in comp.items():
                    ax.add_patch(Rectangle((x, y), w_, fr, facecolor=scol.get(s2, "#E3E6E8"), edgecolor="none")); y += fr
                ax.add_patch(Rectangle((x, 0), w_, 1, facecolor="none", edgecolor="white", lw=0.8))
                if w_ > 0.045:
                    ax.text(x + w_ / 2, -0.03, g_ if g_ != "other LGs" else "other", ha="center", va="top", fontsize=st.FS_MIN)
                x += w_
            ncl = int((h.lineage.value_counts() >= 5).sum()); nst = int((h.state.value_counts() >= 5).sum()); idf = ncl >= 2 and nst >= 2
            ax.text(x0 + wsite / 2, 1.04, f"{s_} ({'pri.' if s_ == 'LL' else 'met.'})\nn={len(h)}", ha="center", va="bottom", fontsize=st.FS_MIN)
            ax.text(x0 + wsite / 2, -0.14, "identifiable" if idf else "not identifiable", ha="center", va="top", fontsize=st.FS_MIN,
                    color=st.INK if idf else st.SUBTLE, style="normal" if idf else "italic")
        ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.25, 1.25)
        fig.text(0.01, 0.985, f"Quinn {exp}: within each site, LineageGroups (column width = cells) and their RNA states (fill; our Leiden clusters)",
                 fontsize=st.FS_TITLE, va="top")
        legend_row(fig, [(s_, scol[s_]) for s_ in state_order if s_ in set(C.state)], 0.04)
        fig.text(0.01, 0.01, "LGs < 2% of a site pooled as other; not identifiable = fewer than 2 LGs or 2 states (>= 5 cells) in the site",
                 fontsize=st.FS_MIN, color=st.SUBTLE)
        save(fig, out, f"Q4_sites_{exp}")
        # Q5 difference flow with LGs as lineage units
        sid, tid = np.asarray(b["source_ids"]).astype(str), np.asarray(b["target_ids"]).astype(str)
        row = A / "v3_grid_20261104" / "kp" / "rows" / u / (cid.split("|")[0] + "__none") / cid.replace("|", "__")
        z = np.load(row / "coupling_m4e.npz")
        P = z["coupling"][np.ix_(pd.Index(z["source_ids"].astype(str)).get_indexer(sid), pd.Index(z["target_ids"].astype(str)).get_indexer(tid))].astype(np.float64)
        crd = a.root / "crispr_ot" / u; Pc = np.load(crd / "coupling_m4e.npz")["coupling"].astype(np.float64)
        cc = pd.read_csv(row / "cell_confidence.csv.gz", usecols=["method", "side", "observation_id", "rejected"])
        cc = cc[cc.method == "M4-E"]; cc["observation_id"] = cc.observation_id.astype(str)
        gt = pd.read_csv(crd / "gates.tsv.gz", sep="\t", dtype={"observation_id": str}); gt = gt[gt.method == "M4-E"]
        gate = {}
        for side, ids_ in (("source", sid), ("target", tid)):
            r_ = ~cc[cc.side == side].set_index("observation_id").rejected.reindex(ids_).to_numpy(bool)
            c_ = gt[gt.side == side].set_index("observation_id").retained.reindex(ids_).to_numpy(bool)
            gate[side] = (r_, c_)
            grows.append({"unit_id": u, "side": side, "cells": len(ids_), "both retained": float(np.mean(r_ & c_)), "RNA-only retained": float(np.mean(r_ & ~c_)),
                          "CRISPR-only retained": float(np.mean(~r_ & c_)), "both rejected": float(np.mean(~r_ & ~c_))})
        S = np.flatnonzero(gate["source"][0] & gate["source"][1]); T = np.flatnonzero(gate["target"][0] & gate["target"][1])
        lin = pd.Series(np.r_[b["source_lineage"], b["target_lineage"]]).astype(object).fillna("unassigned").astype(str).to_numpy()
        lab_s, lab_t = lin[:len(sid)][S], lin[len(sid):][T]
        stmap = sts.drop_duplicates("cell").set_index("cell").state
        st_s, st_t = pd.Series(sid[S]).map(stmap).fillna("NA").to_numpy(), pd.Series(tid[T]).map(stmap).fillna("NA").to_numpy()
        fs = pd.Series(lab_s).value_counts(normalize=True); ft = pd.Series(lab_t).value_counts(normalize=True)
        small = [c for c in set(fs.index) | set(ft.index) if fs.get(c, 0) < OTHER and ft.get(c, 0) < OTHER]
        mp = {c: ("Other" if c in small else c) for c in set(fs.index) | set(ft.index)}
        lab_s = np.array([mp[c] for c in lab_s]); lab_t = np.array([mp[c] for c in lab_t])
        named = [c for c in lg_order if c in set(lab_s) | set(lab_t)]
        order = named + [c for c in ("Other", "unassigned") if c in set(lab_s) | set(lab_t)]
        k = len(order); ix = {c: i for i, c in enumerate(order)}
        bs, bt = np.array([ix[c] for c in lab_s]), np.array([ix[c] for c in lab_t])
        Fm = {}
        for name, Q in (("rna", P), ("crispr", Pc)):
            B = Q[np.ix_(S, T)]; B = B / B.sum(); Fm[name] = np.eye(k)[bs].T @ B @ np.eye(k)[bt]
        sh_, re_, ce_ = np.minimum(Fm["rna"], Fm["crispr"]), np.maximum(Fm["rna"] - Fm["crispr"], 0), np.maximum(Fm["crispr"] - Fm["rna"], 0)
        un = np.maximum(Fm["rna"], Fm["crispr"])
        pd.DataFrame([{"source_lg": order[i], "target_lg": order[j], "F_rna": Fm["rna"][i, j], "F_crispr": Fm["crispr"][i, j], "shared": sh_[i, j],
                       "rna_enriched": re_[i, j], "crispr_enriched": ce_[i, j]} for i in range(k) for j in range(k)]).to_csv(
            tab / "flows" / f"{u}.tsv", sep="\t", index=False)
        diag = [i for i in range(k) if order[i] not in ("Other", "unassigned")]
        rec = {"unit_id": u, "mouse": exp, "common_sources": len(S), "common_targets": len(T), "clades": k,
               "bar1_shared": sh_.sum(), "bar1_rna_enriched": re_.sum(), "bar2_shared": sh_.sum(), "bar2_crispr_enriched": ce_.sum(),
               "bar3_shared": sh_.sum() / un.sum(), "bar3_rna_enriched": re_.sum() / un.sum(), "bar3_crispr_enriched": ce_.sum() / un.sum(),
               "same_lg_mass_rna": float(sum(Fm["rna"][i, i] for i in diag)), "same_lg_mass_crispr": float(sum(Fm["crispr"][i, i] for i in diag)),
               "crispr_enriched_same_lg": float(sum(ce_[i, i] for i in diag) / ce_.sum()) if ce_.sum() > 0 else np.nan,
               "rna_enriched_same_lg": float(sum(re_[i, i] for i in diag) / re_.sum()) if re_.sum() > 0 else np.nan}
        ps_ = pd.Series(lab_s).value_counts(normalize=True).reindex(order).fillna(0).to_numpy(); pt_ = pd.Series(lab_t).value_counts(normalize=True).reindex(order).fillna(0).to_numpy()
        Fi = np.outer(ps_, pt_)
        rec.update(shared_rna_independent=float(np.minimum(Fm["rna"], Fi).sum()), shared_crispr_independent=float(np.minimum(Fm["crispr"], Fi).sum()))
        urows.append(rec)
        F92.STATE_COL.update({s_: scol[s_] for s_ in scol})
        F92.draw(out, u, b, order, None, Fm, sh_, re_, ce_, un, lab_s, lab_t, st_s, st_t, gate, rec, sid, tid)
        for ext in ("png", "pdf", "svg"):
            src = out / f"{u}.{ext}"
            if src.exists():
                src.replace(out / f"Q5_flow_{exp}.{ext}")
        for s_ in scol:
            F92.STATE_COL.pop(s_, None)
        print(f"{exp}: {len(C)} cells, {C.lineage.nunique()} LGs, {len(states)} states; flow shared {rec['bar3_shared']:.2f}", flush=True)
    pd.concat(allcells).to_csv(tab / "cells.tsv.gz", sep="\t", index=False, compression="gzip")
    pd.DataFrame(urows).to_csv(tab / "units.tsv", sep="\t", index=False); pd.DataFrame(grows).to_csv(tab / "gates.tsv", sep="\t", index=False)
    CR = pd.DataFrame(conc_rows); CR.to_csv(tab / "state_concentration.tsv", sep="\t", index=False)
    # Q3 across experiments
    fig = plt.figure(figsize=(3.6, 2.8)); ax = fig.add_axes([0.17, 0.15, 0.78, 0.68]); st.tidy(ax)
    for j, exp in enumerate(EXPS):
        d = CR[CR.experiment == exp]
        xj = j + (np.random.default_rng(1).random(len(d)) - 0.5) * 0.35
        ax.scatter(xj, d.excess, s=6 + 40 * np.sqrt(d.cells / CR.cells.max()), color=st.PAL["tcr"], alpha=0.75, lw=0.3, edgecolor=st.INK)
    ax.axhline(0, **st.REF_KW); ax.set_xticks(range(len(EXPS))); ax.set_xticklabels([f"{e}\n({(CR.experiment == e).sum()} states)" for e in EXPS], fontsize=st.FS_MIN)
    ax.set_ylabel("largest LG share - null", fontsize=st.FS_AXLAB)
    fig.text(0.02, 0.98, "Quinn: per RNA state, lineage concentration beyond the within-site null\n(dot = one Leiden state; area = cells)",
             fontsize=st.FS_TITLE, va="top")
    save(fig, out, "Q3_state_concentration")
    print(pd.DataFrame(urows).round(3).to_string(index=False))
    print(CR.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
