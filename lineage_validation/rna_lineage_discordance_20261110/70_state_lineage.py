"""PLAN_v2.md amendment A2 / A3: KP state-lineage association and primary vs metastasis (raw CRISPR edits, view C).

    python lineage_validation/rna_lineage_discordance_20261110/70_state_lineage.py --root results/rna_lineage_discordance_20261110 \
        --inputs results/crispr_matrix_benchmark_20261108/inputs [--cells CELLS.tsv.gz]

CELLS (default: every OT cell of the KP units, states and depth from the v3 inputs' obs) has cell, mouse, site, state,
depth; the lineage (MetFamily) and edits come from the allele tables (FINAL + unfiltered-only cells).
Per mouse:
  clades   per family, UPGMA on 1 - s (non-evaluable pairs: the family median, for clustering only), cut at distance
           0.5; clusters < 20 cells pooled as "minor"
  flags    per family: the step-1 resolution flag; per cell: informative events, missing intBC share, depth
  A2       per state (>= 50 cells): families and clades with >= 5 cells, Shannon / Hill diversity and the largest share
           of its clade composition; per clade: state Shannon / Hill; per mouse: Cramer's V of clade x state and the
           standardised residuals. Null: states permuted within site, 999 draws, seed 20261111
  A3       primary (tumour name with a T-numbered suffix) vs metastatic cells, states with >= 20 cells on both sides:
           Cramer's V of clade x state on each side after subsampling both to the smaller n (200 draws), each draw with
           one within-site permutation; repeated within tertiles of informative events and of depth
  diagnostic permutable_share = share of cells in sites holding >= 2 states and >= 2 clades (what the within-site null
           can move); a
           null equal to the observed value with a low share is a null that cannot move, not an absence of coupling
Writes ROOT/<out-name>/{cells, clades, states, clade_diversity, mouse_cramer, residuals, primary_vs_met}.tsv(.gz).

Extension 2026-11-11 (PLAN_v2.md E3), all optional and off by default so the frozen run is unchanged:
  --out-name     output directory under ROOT (default state_lineage, the frozen A2 / A3 run)
  --state-map    TSV original -> harmonized; the state column becomes the harmonized label, state_original keeps the
                 authors' label
  --resolution recompute: the frozen step-1 rule on the analysed cell set, per family, from the same s matrix as
                 the clades (default frozen: lineages/resolution.tsv)
  extra outputs  (any --out-name other than state_lineage) site_identifiability.tsv (per site: non-minor clades and
                 analysed states with >= 5 cells there; < 2 of either = not identifiable), state_side.tsv (per state
                 with >= 20 cells on both sides: clade Hill diversity and largest share per side at equal n, 200 draws,
                 each with a within-site permutation), identifiable-share columns on primary_vs_met. They draw from
                 their own generator (seed 20261111), so the main null stream is the frozen one
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MX = REPO / "lineage_validation" / "crispr_matrix_benchmark_20261108"
sys.path.insert(0, str(MX))
import matrix_core as mc  # noqa: E402

SEED, NPERM, NSUB = 20261111, 999, 200
CUT, MIN_CLADE, MIN_STATE, MIN_SIDE = 0.5, 20, 50, 20
MIN_IN_SITE = 5                                                   # extension: a clade / state counts in a site with >= 5 cells
RES_CELLS, RES_EVENTS, RES_DISTINCT, RES_IQR = 50, 3, 0.1, 0.05   # the frozen step-1 rule (01_lineage_resolution.py)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def default_cells(root: Path) -> pd.DataFrame:
    fr = []
    for f in glob.glob(str(root / "inputs_A" / "rna_lineage_discordance_20261110" / "rna_coords" / "*" / "obs.tsv.gz")):
        o = pd.read_csv(f, sep="\t", dtype={"cell": str, "patient_id": str})
        if "cell_type" not in o:
            continue
        fr.append(o[["cell", "patient_id", "sample_id", "cell_type", "total_counts_raw"]])
    c = pd.concat(fr).drop_duplicates("cell")
    return c.rename(columns={"patient_id": "mouse", "sample_id": "site", "cell_type": "state", "total_counts_raw": "depth"})


def is_primary(site: str) -> bool:
    return bool(re.search(r"_T\d+$", site))


def entropy(p):
    p = p[p > 0]; p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def cramer(tab: np.ndarray) -> float:
    tab = tab[tab.sum(1) > 0][:, tab.sum(0) > 0]
    n = tab.sum()
    if n == 0 or min(tab.shape) < 2:
        return np.nan
    e = np.outer(tab.sum(1), tab.sum(0)) / n
    chi2 = ((tab - e) ** 2 / np.where(e > 0, e, 1)).sum()
    return float(np.sqrt(chi2 / (n * (min(tab.shape) - 1))))


def crosstab(a: np.ndarray, b: np.ndarray, na: int, nb: int) -> np.ndarray:
    t = np.zeros((na, nb)); np.add.at(t, (a, b), 1); return t


def permutable_share(states, site, clades) -> float:
    """Share of cells in sites holding >= 2 states and >= 2 clades: only there can a within-site permutation of the
    states change the clade x state table (diagnostic)."""
    d = pd.DataFrame({"s": np.asarray(states), "x": np.asarray(site), "c": np.asarray(clades)})
    ns, nc = d.groupby("x").s.transform("nunique"), d.groupby("x").c.transform("nunique")
    return float(((ns >= 2) & (nc >= 2)).mean())


def permute_within(states: np.ndarray, site: np.ndarray, rng) -> np.ndarray:
    out = states.copy()
    for s in np.unique(site):
        i = np.flatnonzero(site == s); out[i] = states[rng.permutation(i)]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--cells", type=Path)
    p.add_argument("--out-name", default="state_lineage")
    p.add_argument("--state-map", type=Path)
    p.add_argument("--resolution", choices=["frozen", "recompute"], default="frozen")
    p.add_argument("--reuse-clades", action="store_true",
                   help="read clades / events / missingness from an existing state_lineage/cells.tsv.gz instead of recomputing "
                        "them (deterministic; the null draws are unchanged because the clade step uses no random numbers)")
    a = p.parse_args()
    out = a.root / a.out_name; out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED); rng_e = np.random.default_rng(SEED)
    ext = a.out_name != "state_lineage"
    cot = load("cot", HERE / "10_crispr_ot.py"); bb = cot.bundles_module()
    tabs = bb.load_tables(a.inputs); t, prior = tabs["KPTracer"]
    cells = pd.read_csv(a.cells, sep="\t", dtype={"cell": str, "mouse": str}) if a.cells else default_cells(a.root)
    lin = t.drop_duplicates("cell").set_index("cell").lineage
    cells["family"] = cells.cell.map(lin)
    cells = cells.dropna(subset=["family", "state"]).copy()
    if a.state_map:
        cells["state_original"] = cells.state
        mp = pd.read_csv(a.state_map, sep="\t").set_index("original").harmonized
        cells["state"] = cells.state.map(lambda x: mp.get(x, x))
        print(f"state map: {int((cells.state != cells.state_original).sum())} cells relabelled", flush=True)
    res = pd.read_csv(a.root / "lineages" / "resolution.tsv", sep="\t", dtype={"mouse": str})
    resolved = res[res.dataset == "KPTracer"].set_index("lineage").resolved
    # clades, events, missingness per family
    rows, res_rows = [], []
    if a.reuse_clades:
        prev = pd.read_csv(out / "cells.tsv.gz", sep="	", dtype={"cell": str})[["cell", "clade", "events", "missing_share"]]
        rows.append(prev); print(f"reused clades of {len(prev)} cells", flush=True)
    for fam, g in ([] if a.reuse_clades else cells.groupby("family")):
        sub = t[t.lineage == fam]
        L = mc.Lineage(sub[sub.cell.isin(set(g.cell))], g.cell.to_numpy(), np.repeat("source", len(g)), prior)
        ev = np.asarray(L.incidence.sum(1)).ravel()
        miss = 1 - np.asarray(L.observed.sum(1)).ravel() / max(len(L.intbcs), 1)
        clade = np.full(len(g), "minor", dtype=object)
        rr = {"lineage": fam, "cells": len(g), "median_events_per_cell": float(np.median(ev)),
              "distinct_sets_share": pd.Series([tuple(r.indices) for r in L.incidence.tocsr()]).nunique() / len(g), "s_iqr": np.nan}
        if len(g) >= 2 * MIN_CLADE:
            s, ok = L.agreement(np.arange(len(g)), np.arange(len(g)))
            d = squareform(1 - np.asarray(s, dtype=np.float64), checks=False); del s       # upper triangle, row order
            okv = squareform(np.asarray(ok, dtype=np.uint8), checks=False).astype(bool); del ok
            if okv.any():
                rr["s_iqr"] = float(np.subtract(*np.percentile(1 - d[okv], [75, 25])))
            d[~okv] = np.median(d[okv]) if okv.any() else 1.0
            lab = fcluster(linkage(d, method="average"), t=CUT, criterion="distance"); del d
            vc = pd.Series(lab).value_counts()
            big = vc[vc >= MIN_CLADE].index
            clade = np.where(np.isin(lab, big), [f"{fam}|c{x}" for x in lab], "minor")
        rr["resolved"] = bool(len(g) >= RES_CELLS and rr["median_events_per_cell"] >= RES_EVENTS
                              and rr["distinct_sets_share"] >= RES_DISTINCT and rr["s_iqr"] >= RES_IQR)
        res_rows.append(rr)
        rows.append(pd.DataFrame({"cell": g.cell.to_numpy(), "clade": clade, "events": ev, "missing_share": miss}))
        print(f"{fam}: {len(g)} cells, {len(set(clade) - {'minor'})} clades, minor {np.mean(clade == 'minor'):.2f}", flush=True)
    cells = cells.merge(pd.concat(rows), on="cell")
    if a.resolution == "recompute":
        if not res_rows:
            sys.exit("--resolution recompute needs the clade step (not --reuse-clades)")
        R_ = pd.DataFrame(res_rows); R_.to_csv(out / "resolution.tsv", sep="\t", index=False)
        resolved = R_.set_index("lineage").resolved
    cells["resolved"] = cells.family.map(resolved)
    cells["primary"] = cells.site.map(is_primary)
    cells.to_csv(out / "cells.tsv.gz", sep="\t", index=False, compression="gzip")
    st_rows, cl_rows, mv_rows, rs_rows, pm_rows = [], [], [], [], []
    id_rows, ss_rows = [], []
    for scope in ("primary_analysis", "sensitivity_all"):
        C = cells[cells.resolved.fillna(False)] if scope == "primary_analysis" else cells
        for mouse, g in C.groupby("mouse"):
            g = g[g.clade != "minor"]
            if g.clade.nunique() < 2 or len(g) < MIN_STATE:
                continue
            states = g.state.value_counts(); keep_states = states[states >= MIN_STATE].index
            g = g[g.state.isin(keep_states)]
            if g.state.nunique() < 2:
                continue
            cl = pd.Index(sorted(g.clade.unique())); stx = pd.Index(sorted(g.state.unique()))
            ci, si = cl.get_indexer(g.clade), stx.get_indexer(g.state); site = g.site.to_numpy()
            fam_of = pd.Series(g.family.to_numpy(), index=g.clade.to_numpy()).groupby(level=0).first().reindex(cl).to_numpy()

            def metrics(si_):
                T = crosstab(ci, si_, len(cl), len(stx))
                per_state = {}
                for k in range(len(stx)):
                    col = T[:, k]
                    per_state[k] = (int((col >= 5).sum()), len(set(fam_of[col >= 5])), entropy(col), np.exp(entropy(col)),
                                    col.max() / col.sum() if col.sum() else np.nan)
                per_clade = [entropy(T[j]) for j in range(len(cl))]
                return T, per_state, per_clade, cramer(T)

            T, ps, pc, V = metrics(si)
            nulls = [metrics(permute_within(si, site, rng)) for _ in range(NPERM)]
            for k, x in enumerate(stx):
                obs = ps[k]; nul = np.array([n[1][k] for n in nulls])
                rec = {"scope": scope, "mouse": mouse, "state": x, "cells": int(T[:, k].sum()), "clades_ge5": obs[0], "families_ge5": obs[1],
                       "shannon": obs[2], "hill": obs[3], "max_share": obs[4]}
                for j, name in ((0, "clades_ge5"), (2, "shannon"), (3, "hill"), (4, "max_share")):
                    mu, sd = np.nanmean(nul[:, j]), np.nanstd(nul[:, j])
                    rec[f"{name}_null"] = mu; rec[f"{name}_z"] = (obs[j] - mu) / sd if sd > 0 else np.nan
                    rec[f"{name}_p"] = float((1 + np.sum(np.abs(nul[:, j] - mu) >= abs(obs[j] - mu))) / (NPERM + 1))
                st_rows.append(rec)
            nul_c = np.array([n[2] for n in nulls])
            for j, x in enumerate(cl):
                mu, sd = nul_c[:, j].mean(), nul_c[:, j].std()
                cl_rows.append({"scope": scope, "mouse": mouse, "clade": x, "family": fam_of[j], "cells": int(T[j].sum()),
                                "state_shannon": pc[j], "state_hill": np.exp(pc[j]), "shannon_null": mu, "shannon_z": (pc[j] - mu) / sd if sd > 0 else np.nan})
            Vn = np.array([n[3] for n in nulls])
            mv_rows.append({"scope": scope, "mouse": mouse, "clades": len(cl), "families": len(set(fam_of)), "states": len(stx), "cells": len(g),
                            "cramer_v": V, "cramer_v_null": Vn.mean(), "cramer_v_z": (V - Vn.mean()) / Vn.std() if Vn.std() > 0 else np.nan,
                            "cramer_v_p": float((1 + np.sum(Vn >= V)) / (NPERM + 1)),
                            "permutable_share": permutable_share(g.state.to_numpy(), site, g.clade.to_numpy())})
            n = T.sum(); e = np.outer(T.sum(1), T.sum(0)) / n
            R = (T - e) / np.sqrt(np.where(e > 0, e, 1))
            for j, x in enumerate(cl):
                for k, y in enumerate(stx):
                    rs_rows.append({"scope": scope, "mouse": mouse, "clade": x, "state": y, "count": T[j, k], "residual": R[j, k]})
            if ext:
                # extension: site identifiability
                ident = {}
                for sname, hs in g.groupby("site"):
                    ncl = int((hs.clade.value_counts() >= MIN_IN_SITE).sum()); nst = int((hs.state.value_counts() >= MIN_IN_SITE).sum())
                    ident[sname] = ncl >= 2 and nst >= 2
                    id_rows.append({"scope": scope, "mouse": mouse, "site": sname, "primary": bool(hs.primary.iloc[0]), "cells": len(hs),
                                    "clades_ge5": ncl, "states_ge5": nst, "identifiable": ident[sname]})
                g = g.assign(_ident=g.site.map(ident).astype(bool))
                # extension: per state, primary vs metastasis at equal n
                for x in stx:
                    sp_, sm_ = g[(g.state == x) & g.primary], g[(g.state == x) & ~g.primary]
                    if len(sp_) < MIN_SIDE or len(sm_) < MIN_SIDE:
                        continue
                    nmin = min(len(sp_), len(sm_))
                    rec = {"scope": scope, "mouse": mouse, "state": x, "n_each": nmin, "primary_cells": len(sp_), "metastasis_cells": len(sm_)}
                    for side, hh_side in (("primary", g[g.primary]), ("metastasis", g[~g.primary])):
                        st_side = hh_side.state.to_numpy(); site_side = hh_side.site.to_numpy()
                        cl_side = pd.Index(sorted(hh_side.clade.unique())); ci_s = cl_side.get_indexer(hh_side.clade)
                        pos = np.flatnonzero(st_side == x)
                        vals = []
                        for _ in range(NSUB):
                            pick = rng_e.choice(pos, nmin, replace=False)
                            comp = np.bincount(ci_s[pick], minlength=len(cl_side)).astype(float)
                            perm = permute_within(st_side, site_side, rng_e)
                            pick_p = rng_e.choice(np.flatnonzero(perm == x), nmin, replace=False)
                            comp_p = np.bincount(ci_s[pick_p], minlength=len(cl_side)).astype(float)
                            vals.append((np.exp(entropy(comp)), comp.max() / nmin, np.exp(entropy(comp_p)), comp_p.max() / nmin))
                        v = np.array(vals); hx = hh_side[hh_side.state == x]
                        rec.update({f"{side}_hill": v[:, 0].mean(), f"{side}_hill_null": v[:, 2].mean(),
                                    f"{side}_hill_excess": v[:, 0].mean() - v[:, 2].mean(),
                                    f"{side}_max_share": v[:, 1].mean(), f"{side}_max_share_null": v[:, 3].mean(),
                                    f"{side}_max_share_excess": v[:, 1].mean() - v[:, 3].mean(),
                                    f"{side}_identifiable_share": float(hx._ident.mean()), f"{side}_sites": int(hx.site.nunique())})
                    ss_rows.append(rec)
            # A3 primary vs metastasis
            pri, met = g[g.primary], g[~g.primary]
            both_states = [x for x in stx if (pri.state == x).sum() >= MIN_SIDE and (met.state == x).sum() >= MIN_SIDE]
            if len(both_states) < 2 or pri.clade.nunique() < 2 or met.clade.nunique() < 2:
                continue
            for strat_name, strat in (("all", None), ("events", "events"), ("depth", "depth")):
                groups = [("all", g)] if strat is None else list(g.assign(_t=pd.qcut(g[strat].rank(method="first"), 3, labels=["low", "mid", "high"])).groupby("_t", observed=True))
                for tert, h in groups:
                    h = h[h.state.isin(both_states)]
                    hp, hm = h[h.primary], h[~h.primary]
                    nmin = min(len(hp), len(hm))
                    if nmin < 50 or hp.clade.nunique() < 2 or hm.clade.nunique() < 2:
                        continue
                    vals = {"primary": [], "metastasis": []}
                    for _ in range(NSUB):
                        for side, hh in (("primary", hp), ("metastasis", hm)):
                            idx = rng.choice(len(hh), nmin, replace=False); x = hh.iloc[idx]
                            ci2 = pd.Index(sorted(x.clade.unique())).get_indexer(x.clade); si2 = pd.Index(sorted(x.state.unique())).get_indexer(x.state)
                            vo = cramer(crosstab(ci2, si2, ci2.max() + 1, si2.max() + 1))
                            vp = cramer(crosstab(ci2, permute_within(si2, x.site.to_numpy(), rng), ci2.max() + 1, si2.max() + 1))
                            vals[side].append((vo, vp))
                    rec = {"scope": scope, "mouse": mouse, "stratum": strat_name, "tertile": tert, "n_each": nmin, "states": len(both_states)}
                    for side, v in vals.items():
                        v = np.array(v); rec[f"{side}_v"] = np.nanmean(v[:, 0]); rec[f"{side}_v_null"] = np.nanmean(v[:, 1])
                        rec[f"{side}_excess"] = rec[f"{side}_v"] - rec[f"{side}_v_null"]
                    rec["met_minus_primary_excess"] = rec["metastasis_excess"] - rec["primary_excess"]
                    rec["primary_permutable_share"] = permutable_share(hp.state.to_numpy(), hp.site.to_numpy(), hp.clade.to_numpy())
                    rec["metastasis_permutable_share"] = permutable_share(hm.state.to_numpy(), hm.site.to_numpy(), hm.clade.to_numpy())
                    rec["primary_sites"], rec["metastasis_sites"] = hp.site.nunique(), hm.site.nunique()
                    if ext:
                        rec["primary_identifiable_share"] = float(hp._ident.mean()); rec["metastasis_identifiable_share"] = float(hm._ident.mean())
                    pm_rows.append(rec)
    tables = [("states", st_rows), ("clade_diversity", cl_rows), ("mouse_cramer", mv_rows), ("residuals", rs_rows), ("primary_vs_met", pm_rows)]
    if ext:
        tables += [("site_identifiability", id_rows), ("state_side", ss_rows)]
    for name, rows_ in tables:
        pd.DataFrame(rows_).to_csv(out / f"{name}.tsv", sep="\t", index=False)
    print(pd.DataFrame(mv_rows).round(3).to_string(index=False))
    if pm_rows:
        print(pd.DataFrame(pm_rows).query("stratum == 'all'").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
