"""RNA-OT clade flow for every unit (KP and Quinn), CRISPR-OT removed (user 2026-11-23). Read-only, no refit.

    python lineage_validation/clade_overlap_20261119/03_rna_flow_all.py \
        --root results/rna_lineage_discordance_20261110 --bundles results/crispr_matrix_benchmark_20261108/bundles \
        --inputs results/crispr_matrix_benchmark_20261108/inputs --out results/clade_overlap_20261119 [--tag l5best]

Per unit (every bundle with an RNA-OT row):
  cells    the RNA-OT M4-E retained block only (its own gates; no CRISPR gate, unlike 92_'s common support)
  clades   KP: all-eligible-cell A2 clades (as 92_); Quinn: the authors' LineageGroup. A clade with < 2% of the retained
           cells on both sides is merged into Other (92_'s rule); "minor", "Other" and "unassigned" are no clade
  flow     F = RNA-OT coupling on the block, normalised to 1, summed into source-clade x target-clade blocks
  ribbons  blue = same named clade on both ends, grey = different clade or no clade; width = coupling mass
  numbers  same-clade mass = sum of the named-clade diagonal of F; chance = sum over named clades of
           (source row mass) x (target column mass), the same-clade mass of a coupling with these clade marginals and
           no structure inside them; excess = same-clade - chance
  tree     KP: the family's cached clade tree (lineage_tree.py), pruned; Quinn: none (clones are separate integrations)
Writes OUT/rna_flow/{units.tsv, flows/<unit>.tsv}, OUT/figures/rna_flow/<unit>.{png,pdf} (one panel per file), the
all-unit summary panel OUT/figures/rna_flow/_summary_same_clade_vs_chance.{png,pdf}, and OUT/figures/rna_flow/CAPTIONS.md.
Refuses to overwrite OUT/rna_flow/units.tsv.
"""

from __future__ import annotations

import argparse
import importlib.util
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
RLD = HERE.parent / "rna_lineage_discordance_20261110"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


FL = load("flow92", RLD / "92_flow_figure.py")          # ribbon, spread, STATE_COL, CONFIGS, st (nbt_style), LT, mc
st, LT = FL.st, FL.LT
C_SAME, C_DIFF = "#2C6E9B", "#BFC4C9"
SPECIAL = ("minor", "Other", "unassigned")
OTHER_SHARE, MIN_RIBBON = 0.02, 0.002


def nice(c):
    return c.split("|")[-1] if "|" in c else c


def draw(path, u, b, order, F, lab_s, lab_t, st_s, st_t, Z, rec):
    k = len(order)
    fig = plt.figure(figsize=(7.2, 4.6))
    ax = fig.add_axes([0.0, 0.12, 1.0, 0.88]); ax.set_xlim(0, 10); ax.set_ylim(0, 1); ax.axis("off")
    XS0, XS1, XT0, XT1 = 2.25, 2.5, 7.5, 7.75
    gap = 0.012 if k < 25 else 0.004
    top = 0.80; Htot = 0.74 - gap * (k - 1); scale = Htot / F.sum()
    hs, ht = F.sum(1) * scale, F.sum(0) * scale
    ys = top - np.r_[0, np.cumsum(hs[:-1] + gap)]; yt = top - np.r_[0, np.cumsum(ht[:-1] + gap)]
    kp = b["dataset"] == "KPTracer"

    def col_bar(x0, x1, ytops, hh, labs, states):
        for i, c in enumerate(order):
            if hh[i] <= 0:
                continue
            m = labs == c
            comp = pd.Series(states[m]).value_counts(normalize=True) if (kp and m.any()) else pd.Series({"NA": 1.0})
            y = ytops[i]
            for s_, fr in comp.items():
                ax.add_patch(Rectangle((x0, y - fr * hh[i]), x1 - x0, fr * hh[i],
                                       facecolor=FL.STATE_COL.get(s_, "#D5D8DC") if kp else "#D5D8DC", edgecolor="none"))
                y -= fr * hh[i]
            ax.add_patch(Rectangle((x0, ytops[i] - hh[i]), x1 - x0, hh[i], facecolor="none", edgecolor=st.INK, lw=0.5))
    col_bar(XS0, XS1, ys, hs, lab_s, st_s); col_bar(XT0, XT1, yt, ht, lab_t, st_t)
    same = lambda i, j: i == j and order[i] not in SPECIAL  # noqa: E731
    sp, tp = {}, {}
    y = ys.copy()
    for i in range(k):
        for j in range(k):
            h = F[i, j] * scale; sp[(i, j)] = (y[i] - h, y[i]); y[i] -= h
    y = yt.copy()
    for j in range(k):
        for i in range(k):
            h = F[i, j] * scale; tp[(i, j)] = (y[j] - h, y[j]); y[j] -= h
    undrawn = 0.0
    for want_same in (False, True):                       # same-clade ribbons drawn on top
        for i in range(k):
            for j in range(k):
                v = F[i, j]
                if v <= 0 or same(i, j) != want_same:
                    continue
                if v < MIN_RIBBON:
                    undrawn += v; continue
                a0, a1 = sp[(i, j)]; b0, b1 = tp[(i, j)]
                FL.ribbon(ax, XS1, XT0, a1, a0, b1, b0, C_SAME if want_same else C_DIFF, alpha=0.85 if want_same else 0.7)
    rec["undrawn"] = undrawn
    # labels (only clades with >= 1.5% of the bar height get a label when there are many)
    TIP = 1.25; LG = 0.03 if k < 25 else 0.018
    minh = 0.0 if k < 25 else 0.015 * Htot
    src = [i for i in range(k) if hs[i] > minh]; tgt = [i for i in range(k) if ht[i] > minh]
    ly_s = dict(zip(src, FL.spread([ys[i] - hs[i] / 2 for i in src], LG, 0.03, top)))
    ly_t = dict(zip(tgt, FL.spread([yt[i] - ht[i] / 2 for i in tgt], LG, 0.03, top)))
    fs = st.FS_TICK if k < 25 else st.FS_MIN
    for i in src:
        c = order[i]; yc = ys[i] - hs[i] / 2; yl = ly_s[i]
        ax.text(TIP + 0.05, yl, f"{nice(c)} ({int((lab_s == c).sum())})", ha="left", va="center", fontsize=fs, color=st.INK)
        ax.plot([XS0 - 0.32, XS0 - 0.06, XS0], [yl, yl, yc], color="#9AA0A6", lw=0.4)
    for i in tgt:
        c = order[i]; yc = yt[i] - ht[i] / 2; yl = ly_t[i]
        ax.text(XT1 + 0.12, yl, f"{nice(c)} ({int((lab_t == c).sum())})", ha="left", va="center", fontsize=fs, color=st.INK)
        if abs(yl - yc) > 1e-6:
            ax.plot([XT1, XT1 + 0.04, XT1 + 0.1], [yc, yl, yl], color="#9AA0A6", lw=0.4)
    if Z is not None and Z[0] is not None and Z[0][0] != "leaf":
        y_of = {order[i]: ly_s[i] for i in src if order[i] not in SPECIAL}
        for c in [c for c in order if c not in SPECIAL and c not in y_of and order.index(c) in ly_t]:
            y_of[c] = ly_t[order.index(c)]
        if all(nm in y_of for nm in [Z[1][x] for x in LT._leaves(Z[0])]):
            LT.draw_pruned(ax, Z[0], Z[1], y_of, 0.1, TIP, y_bar=0.02, color=st.INK, fontsize=st.FS_MIN)
    tsites = sorted(pd.Series(np.asarray(b["target_site"]).astype(str)).unique())
    tsite = tsites[0] if len(tsites) == 1 else f"{len(tsites)} sites"
    src_name = u.split("__")[2] if kp else f"{b['mouse']} LL"
    ax.text(1.9, 0.985, f"Source: {src_name} (primary)", ha="center", va="top", fontsize=st.FS_TITLE, color=st.INK)
    ax.text(8.1, 0.985, f"Target: {tsite} (metastasis)", ha="center", va="top", fontsize=st.FS_TITLE, color=st.INK)
    ax.text(5.0, 0.935, f"RNA-OT same-clade mass {rec['same_clade']:.0%}   |   expected without pairing {rec['chance']:.0%}"
            f"   |   excess {rec['excess']:+.0%}", ha="center", va="top", fontsize=st.FS_TITLE, color=st.INK)
    x = 3.3
    for nm, colr in (("same CRISPR clade", C_SAME), ("different clade / no clade", C_DIFF)):
        ax.add_patch(Rectangle((x, 0.86), 0.2, 0.024, facecolor=colr, edgecolor="none"))
        ax.text(x + 0.26, 0.872, nm, fontsize=st.FS_TICK, va="center", color=st.INK)
        x += 0.5 + 0.075 * len(nm)
    if kp:
        present = [s_ for s_ in FL.STATE_COL if s_ in set(st_s) | set(st_t)]
        lx = fig.add_axes([0.0, 0.0, 1.0, 0.10]); lx.axis("off"); lx.set_xlim(0, 10); lx.set_ylim(0, 2)
        lx.text(0.15, 1.5, "Bar fill = RNA state:", fontsize=st.FS_MIN, va="center", color=st.INK)
        x, row = 1.35, 1.5
        for s_ in present:
            w_ = 0.32 + 0.058 * len(s_)
            if x + w_ > 9.9:
                x, row = 1.35, row - 1
            lx.add_patch(Rectangle((x, row - 0.3), 0.14, 0.6, facecolor=FL.STATE_COL[s_], edgecolor="none"))
            lx.text(x + 0.18, row, s_, fontsize=st.FS_MIN, va="center", color=st.INK)
            x += w_
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300 if ext == "png" else None)
    plt.close(fig)


def summary(path, t):
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    grp = {("KPTracer", "L1"): ("KP primary -> one metastasis", "#2C6E9B", "o"),
           ("KPTracer", "L23"): ("KP primary -> pooled metastases", "#A23B5B", "s"),
           ("Quinn2021", "L23"): ("Quinn LL -> pooled tissues", "#C9A227", "D")}
    for (ds, lv), (lab, colr, mk) in grp.items():
        s = t[(t.dataset == ds) & (t.level == lv)]
        ax.scatter(s.chance, s.same_clade, s=14, color=colr, marker=mk, edgecolor="white", lw=0.4, label=f"{lab} (n={len(s)})", zorder=3)
    ax.plot([0, 1], [0, 1], color="#9AA0A6", lw=0.6, zorder=1)
    for _, r in t[t.excess > 0.1].iterrows():
        nm = r.unit.replace("KPTracer_", "").replace("Quinn2021_", "Quinn ").replace("__family_metastases", " pooled").replace("__LL__metastases", "")
        nm = nm.replace("L23__", "").replace("L1__", "")
        ax.annotate(nm.split("__")[0], (r.chance, r.same_clade), xytext=(6, 6), textcoords="offset points", fontsize=st.FS_MIN,
                    color=st.INK, ha="left", va="bottom")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("Same-clade mass expected without pairing", fontsize=st.FS_AXLAB)
    ax.set_ylabel("RNA-OT same-clade mass", fontsize=st.FS_AXLAB)
    ax.set_title("Every unit: RNA-OT vs no-pairing expectation", fontsize=st.FS_TITLE)
    ax.legend(fontsize=st.FS_MIN, frameon=False, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300 if ext == "png" else None)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--bundles", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag", default="l5best")
    a = ap.parse_args()
    out_t = a.out / "rna_flow"; out_f = a.out / "figures" / "rna_flow"
    if (out_t / "units.tsv").exists():
        raise SystemExit(f"{out_t}/units.tsv exists; not overwritten")
    (out_t / "flows").mkdir(parents=True, exist_ok=True); out_f.mkdir(parents=True, exist_ok=True)
    st.apply()
    cl = pd.read_csv(a.root / "state_lineage_ext" / "all_cells__original" / "cells.tsv.gz", sep="\t",
                     dtype={"cell": str, "mouse": str}).set_index("cell")
    cl_fam = cl.reset_index()[["cell", "family", "clade"]]
    cid = FL.CONFIGS[a.tag]
    rows_dir = a.root / "inputs_A" / "v3_grid_20261104" / "kp" / "rows"
    tables = None
    recs = []
    for udir in sorted(rows_dir.iterdir()):
        u = udir.name
        row = udir / (cid.split("|")[0] + "__none") / cid.replace("|", "__")
        if not (row / "coupling_m4e.npz").exists():
            continue
        b = pickle.load(open(a.bundles / f"{u}.pkl", "rb")); kp = b["dataset"] == "KPTracer"
        z = np.load(row / "coupling_m4e.npz", allow_pickle=True)
        P, sid, tid = z["coupling"].astype(np.float64), z["source_ids"].astype(str), z["target_ids"].astype(str)
        g = pd.read_csv(row / "cell_confidence.csv.gz", usecols=["method", "side", "observation_id", "retained"], dtype={"observation_id": str})
        g = g[g.method == "M4-E"]
        sret = g[g.side == "source"].drop_duplicates("observation_id").set_index("observation_id").retained.reindex(sid).fillna(False).to_numpy(bool)
        tret = g[g.side == "target"].drop_duplicates("observation_id").set_index("observation_id").retained.reindex(tid).fillna(False).to_numpy(bool)
        S, T = np.flatnonzero(sret), np.flatnonzero(tret)
        if len(S) == 0 or len(T) == 0:
            recs.append({"unit": u, "dataset": b["dataset"], "level": u.split("__")[0], "mouse": str(b["mouse"]), "status": "empty block"}); continue
        if kp:
            lab_s = cl.clade.reindex(sid[S]).fillna("unassigned").to_numpy(object); lab_t = cl.clade.reindex(tid[T]).fillna("unassigned").to_numpy(object)
            st_s = cl.state.reindex(sid[S]).fillna("NA").to_numpy(object); st_t = cl.state.reindex(tid[T]).fillna("NA").to_numpy(object)
            fam_of = cl.family.reindex(sid[S]).dropna()
        else:
            ls = pd.Series(np.asarray(b["source_lineage"]).astype(str), index=np.asarray(b["source_ids"]).astype(str))
            lt = pd.Series(np.asarray(b["target_lineage"]).astype(str), index=np.asarray(b["target_ids"]).astype(str))
            ls, lt = ls[~ls.index.duplicated()], lt[~lt.index.duplicated()]
            lab_s = ls.reindex(sid[S]).fillna("unassigned").replace({"nan": "unassigned", "": "unassigned"}).to_numpy(object)
            lab_t = lt.reindex(tid[T]).fillna("unassigned").replace({"nan": "unassigned", "": "unassigned"}).to_numpy(object)
            st_s, st_t = np.repeat("NA", len(S)).astype(object), np.repeat("NA", len(T)).astype(object)
        fs_ = pd.Series(lab_s).value_counts(normalize=True); ft_ = pd.Series(lab_t).value_counts(normalize=True)
        allc = sorted(set(fs_.index) | set(ft_.index))
        small = [c for c in allc if c not in SPECIAL and fs_.get(c, 0) < OTHER_SHARE and ft_.get(c, 0) < OTHER_SHARE]
        mp = {c: ("Other" if c in small else c) for c in allc}
        lab_s = np.array([mp[c] for c in lab_s], dtype=object); lab_t = np.array([mp[c] for c in lab_t], dtype=object)
        named = sorted({c for c in set(lab_s) | set(lab_t) if c not in SPECIAL})
        Z = None
        if kp and named:
            fam = named[0].split("|")[0]
            cache = a.root / "explore" / "trees"
            if not (cache / f"{fam}.npz").exists() and tables is None:
                bb = load("cot", RLD / "10_crispr_ot.py").bundles_module(); tables = bb.load_tables(a.inputs)["KPTracer"]
            tt, prior = tables if tables is not None else (None, None)
            tree = LT.family_tree(fam, cl_fam[cl_fam.family == fam], tt, prior, FL.mc, cache)
            src_named = [c for c in named if (lab_s == c).any()]   # tree leaves only where there is a source bar to hang them on
            order, pr = LT.pruned_order(tree, src_named)
            Z = (pr, tree["names"]) if tree is not None else None
            order = order + [c for c in named if c not in order]
        else:                                              # Quinn: clones ordered by source share, largest first
            order = sorted(named, key=lambda c: (-(lab_s == c).mean(), -(lab_t == c).mean(), c))
        order = order + [c for c in SPECIAL if c in set(lab_s) | set(lab_t)]
        ix = {c: i for i, c in enumerate(order)}; k = len(order)
        Bk = P[np.ix_(S, T)]; Bk = Bk / Bk.sum()
        F = np.eye(k)[[ix[c] for c in lab_s]].T @ Bk @ np.eye(k)[[ix[c] for c in lab_t]]
        diag = [i for i in range(k) if order[i] not in SPECIAL]
        same_c = float(F[diag, diag].sum()); chance = float(sum(F.sum(1)[i] * F.sum(0)[i] for i in diag))
        rec = {"unit": u, "dataset": b["dataset"], "level": u.split("__")[0], "mouse": str(b["mouse"]), "status": "ok",
               "retained_sources": len(S), "retained_targets": len(T), "source_retained_frac": float(sret.mean()),
               "target_retained_frac": float(tret.mean()), "clades_drawn": k, "merged_into_other": len(small),
               "same_clade": same_c, "chance": chance, "excess": same_c - chance}
        pd.DataFrame([{"source_clade": order[i], "target_clade": order[j], "F_rna": F[i, j]} for i in range(k) for j in range(k)]
                     ).to_csv(out_t / "flows" / f"{u}.tsv", sep="\t", index=False)
        draw(out_f / u, u, b, order, F, lab_s, lab_t, st_s, st_t, Z, rec)
        recs.append(rec)
        print(f"{u}: {len(S)} x {len(T)}, {k} clades, same {same_c:.2f} chance {chance:.2f}", flush=True)
    t = pd.DataFrame(recs)
    t.to_csv(out_t / "units.tsv", sep="\t", index=False)
    ok = t[t.status == "ok"]
    summary(out_f / "_summary_same_clade_vs_chance", ok)
    lines = ["# RNA-OT clade flow, every unit (03_rna_flow_all.py)", "",
             "Each figure: one Source -> Target unit. Cells are the RNA-OT M4-E retained block only (no CRISPR gate). "
             "Ribbon width = RNA-OT coupling mass between a source clade and a target clade, the block normalised to 1. "
             "Blue = same CRISPR clade on both ends; grey = different clade, or a cell without a named clade "
             "(minor = UPGMA clusters < 20 cells; Other = clades < 2% of the retained cells on both sides; unassigned). "
             f"Ribbons under {MIN_RIBBON:.1%} of the mass are not drawn but are counted in the bars and the numbers. "
             "Clades: KP = all-eligible-cell A2 clades (UPGMA on 1 - s, cut 0.5), with the family clade tree (a summary "
             "of the edits, not the authors' Cassiopeia tree); Quinn = the authors' LineageGroup, no tree. "
             "'Expected without pairing' = the same-clade mass of a coupling with the same clade totals on each side and "
             "no structure inside them. Bar fill (KP) = the authors' RNA state of the clade's cells.", "",
             "| Unit | Retained sources x targets | Clades drawn | Same-clade mass | Expected without pairing | Excess |",
             "|---|---|---|---|---|---|"]
    for _, r in ok.sort_values(["dataset", "level", "unit"]).iterrows():
        lines.append(f"| {r.unit} | {r.retained_sources} x {r.retained_targets} | {r.clades_drawn} | {r.same_clade:.3f} | {r.chance:.3f} | {r.excess:+.3f} |")
    (out_f / "CAPTIONS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(ok)} figures -> {out_f}")


if __name__ == "__main__":
    main()
