"""PLAN_v2.md extension E3 / reporting: compare the four A2 / A3 runs (cell set x annotation), per mouse and state,
pooled with and without KP 3724.

    python lineage_validation/rna_lineage_discordance_20261110/88_ext_summary.py --root results/rna_lineage_discordance_20261110

Runs: state_lineage_ext/{ot_units, all_cells}__{original, harmonized}. Writes ROOT/ext_summary/:
  mouse_cramer_runs.tsv      per run, scope and mouse: Cramer's V, null, excess, z, permutable share
  mouse_cramer_pooled.tsv    mean over mice per run and scope, with / without 3724
  states_runs.tsv            per run, scope, mouse and state: cells, families / clades with >= 5 cells, Hill and largest
                             share with their null and z
  states_across_mice.tsv     per run, scope and state: mice analysed, mice with Hill z <= -2 (fewer clades than the
                             null), mice with Hill z >= 2 (more), median Hill z, mean families_ge5, with / without 3724
                             (descriptive counts, no pass / fail)
  state_wide_hill_z.tsv      mouse x state rows, the four runs' Hill z side by side (sensitivity scope)
  primary_vs_met_runs.tsv    per run: the A3 rows (all cells and tertiles) with identifiable shares
  primary_vs_met_pooled.tsv  mean over mice per run / stratum / tertile, with / without 3724
  state_side_runs.tsv        per run: per state primary vs metastasis Hill / largest-share excess, identifiable shares
  identifiability.tsv        per run, scope and mouse: sites, identifiable sites, cells in identifiable sites (per side)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = ["ot_units__original", "ot_units__harmonized", "all_cells__original", "all_cells__harmonized"]


def pooled(df, by, cols):
    out = []
    for excl in (False, True):
        d = df[df.mouse != "3724"] if excl else df
        g = d.groupby(by, dropna=False)
        r = g[cols].mean().add_suffix("_mean"); r["mice"] = g.mouse.nunique(); r["without_3724"] = excl
        out.append(r.reset_index())
    return pd.concat(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    out = a.root / "ext_summary"; out.mkdir(exist_ok=True)
    rd = lambda f: pd.read_csv(f, sep="\t", dtype={"mouse": str})  # noqa: E731
    mv, stt, pm, ss, idn = [], [], [], [], []
    for r in RUNS:
        d = a.root / "state_lineage_ext" / r
        if not (d / "mouse_cramer.tsv").exists():
            print(f"{r}: missing"); continue
        cs, an = r.split("__")
        for name, lst in (("mouse_cramer", mv), ("states", stt), ("primary_vs_met", pm), ("state_side", ss), ("site_identifiability", idn)):
            f = d / f"{name}.tsv"
            if f.exists() and f.stat().st_size > 1:
                try:
                    x = rd(f)
                except pd.errors.EmptyDataError:
                    continue
                x.insert(0, "annotation", an); x.insert(0, "cell_set", cs); x.insert(0, "run", r); lst.append(x)
    mv, stt = pd.concat(mv), pd.concat(stt)
    mv["cramer_v_excess"] = mv.cramer_v - mv.cramer_v_null
    mv.to_csv(out / "mouse_cramer_runs.tsv", sep="\t", index=False)
    pooled(mv, ["run", "scope"], ["cramer_v", "cramer_v_null", "cramer_v_excess", "cramer_v_z", "permutable_share"]).to_csv(
        out / "mouse_cramer_pooled.tsv", sep="\t", index=False)
    stt.to_csv(out / "states_runs.tsv", sep="\t", index=False)
    rows = []
    for excl in (False, True):
        d = stt[stt.mouse != "3724"] if excl else stt
        for (r, sc, s), g in d.groupby(["run", "scope", "state"]):
            rows.append({"run": r, "scope": sc, "state": s, "without_3724": excl, "mice": g.mouse.nunique(),
                         "mice_hill_z_le_-2": int((g.hill_z <= -2).sum()), "mice_hill_z_ge_2": int((g.hill_z >= 2).sum()),
                         "median_hill_z": float(g.hill_z.median()), "mean_families_ge5": float(g.families_ge5.mean()),
                         "mean_clades_ge5": float(g.clades_ge5.mean()), "cells": int(g.cells.sum())})
    pd.DataFrame(rows).to_csv(out / "states_across_mice.tsv", sep="\t", index=False)
    w = stt[stt.scope == "sensitivity_all"].pivot_table(index=["mouse", "state"], columns="run", values="hill_z")
    w = w[[c for c in RUNS if c in w.columns]]; w.to_csv(out / "state_wide_hill_z.tsv", sep="\t")
    if pm:
        pm = pd.concat(pm); pm.to_csv(out / "primary_vs_met_runs.tsv", sep="\t", index=False)
        pooled(pm, ["run", "scope", "stratum", "tertile"], ["primary_excess", "metastasis_excess", "met_minus_primary_excess"]).to_csv(
            out / "primary_vs_met_pooled.tsv", sep="\t", index=False)
    if ss:
        pd.concat(ss).to_csv(out / "state_side_runs.tsv", sep="\t", index=False)
    if idn:
        idn = pd.concat(idn)
        g = idn.groupby(["run", "scope", "mouse", "primary"])
        I = g.agg(sites=("site", "size"), identifiable_sites=("identifiable", "sum"), cells=("cells", "sum")).reset_index()
        I["cells_in_identifiable"] = g.apply(lambda x: int(x.loc[x.identifiable, "cells"].sum())).to_numpy()
        I.to_csv(out / "identifiability.tsv", sep="\t", index=False)
    print(pd.read_csv(out / "mouse_cramer_pooled.tsv", sep="\t").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
