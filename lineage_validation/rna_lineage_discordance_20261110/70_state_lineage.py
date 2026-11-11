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
Writes ROOT/state_lineage/{cells, clades, states, clade_diversity, mouse_cramer, residuals, primary_vs_met}.tsv(.gz).
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MX = REPO / "lineage_validation" / "crispr_matrix_benchmark_20261108"
sys.path.insert(0, str(MX))
import matrix_core as mc  # noqa: E402

SEED, NPERM, NSUB = 20261111, 999, 200
CUT, MIN_CLADE, MIN_STATE, MIN_SIDE = 0.5, 20, 50, 20


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
    p.add_argument("--reuse-clades", action="store_true",
                   help="read clades / events / missingness from an existing state_lineage/cells.tsv.gz instead of recomputing "
                        "them (deterministic; the null draws are unchanged because the clade step uses no random numbers)")
    a = p.parse_args()
    out = a.root / "state_lineage"; out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    cot = load("cot", HERE / "10_crispr_ot.py"); bb = cot.bundles_module()
    tabs = bb.load_tables(a.inputs); t, prior = tabs["KPTracer"]
    cells = pd.read_csv(a.cells, sep="\t", dtype={"cell": str, "mouse": str}) if a.cells else default_cells(a.root)
    lin = t.drop_duplicates("cell").set_index("cell").lineage
    cells["family"] = cells.cell.map(lin)
    cells = cells.dropna(subset=["family", "state"]).copy()
    res = pd.read_csv(a.root / "lineages" / "resolution.tsv", sep="\t", dtype={"mouse": str})
    resolved = res[res.dataset == "KPTracer"].set_index("lineage").resolved
    # clades, events, missingness per family
    rows = []
    if a.reuse_clades:
        prev = pd.read_csv(out / "cells.tsv.gz", sep="	", dtype={"cell": str})[["cell", "clade", "events", "missing_share"]]
        rows.append(prev); print(f"reused clades of {len(prev)} cells", flush=True)
    for fam, g in ([] if a.reuse_clades else cells.groupby("family")):
        sub = t[t.lineage == fam]
        L = mc.Lineage(sub[sub.cell.isin(set(g.cell))], g.cell.to_numpy(), np.repeat("source", len(g)), prior)
        ev = np.asarray(L.incidence.sum(1)).ravel()
        miss = 1 - np.asarray(L.observed.sum(1)).ravel() / max(len(L.intbcs), 1)
        clade = np.full(len(g), "minor", dtype=object)
        if len(g) >= 2 * MIN_CLADE:
            s, ok = L.agreement(np.arange(len(g)), np.arange(len(g)))
            iu = np.triu_indices(len(g), 1)
            d = (1 - s[iu]).astype(np.float64); okv = ok[iu]; del s, ok
            d[~okv] = np.median(d[okv]) if okv.any() else 1.0
            lab = fcluster(linkage(d, method="average"), t=CUT, criterion="distance"); del d
            vc = pd.Series(lab).value_counts()
            big = vc[vc >= MIN_CLADE].index
            clade = np.where(np.isin(lab, big), [f"{fam}|c{x}" for x in lab], "minor")
        rows.append(pd.DataFrame({"cell": g.cell.to_numpy(), "clade": clade, "events": ev, "missing_share": miss}))
        print(f"{fam}: {len(g)} cells, {len(set(clade) - {'minor'})} clades, minor {np.mean(clade == 'minor'):.2f}", flush=True)
    cells = cells.merge(pd.concat(rows), on="cell")
    cells["resolved"] = cells.family.map(resolved)
    cells["primary"] = cells.site.map(is_primary)
    cells.to_csv(out / "cells.tsv.gz", sep="\t", index=False, compression="gzip")
    st_rows, cl_rows, mv_rows, rs_rows, pm_rows = [], [], [], [], []
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
                    pm_rows.append(rec)
    for name, rows_ in (("states", st_rows), ("clade_diversity", cl_rows), ("mouse_cramer", mv_rows), ("residuals", rs_rows), ("primary_vs_met", pm_rows)):
        pd.DataFrame(rows_).to_csv(out / f"{name}.tsv", sep="\t", index=False)
    print(pd.DataFrame(mv_rows).round(3).to_string(index=False))
    if pm_rows:
        print(pd.DataFrame(pm_rows).query("stratum == 'all'").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
