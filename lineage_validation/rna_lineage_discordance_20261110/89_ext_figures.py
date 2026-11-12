"""PLAN_v2.md extension: figure panels of the four A2 / A3 runs, one PDF + PNG each (nbt style).

    python lineage_validation/rna_lineage_discordance_20261110/89_ext_figures.py --root results/rna_lineage_discordance_20261110

Reads ROOT/ext_summary (88_) and state_lineage_ext/<run>/residuals.tsv. Writes ROOT/figures/ext/:
  h1_residuals_<run>_<mouse>  clade x state standardised residuals (sensitivity scope; all-cell runs)
  h2_hill_z_<run>_<scope>     state x mouse: z of the clade Hill diversity vs the within-site null (clipped +/-10)
  h3_hill_z_runs[_no3724]     state x run: median Hill z over mice (sensitivity scope)
  h4_primary_vs_met_<run>     per mouse: Cramer's V excess, primary vs metastasis, with each side's identifiable share
  h5_state_side_<run>         per mouse and state: clade Hill excess, primary vs metastasis, with identifiable shares
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "realdata_benchmark" / "v3_grid" / "figures"))
import nbt_style as st  # noqa: E402

RUNS = ["ot_units__original", "ot_units__harmonized", "all_cells__original", "all_cells__harmonized"]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def pairs(ax, labels, a_, b_, la, lb):
    x = np.arange(len(labels))
    ax.vlines(x, a_, b_, color=st.GUIDE, lw=st.LW_GUIDE * 2)
    ax.scatter(x, a_, s=10, color=st.PAL["tcr"], lw=0, label=la, zorder=3)
    ax.scatter(x, b_, s=10, color=st.PAL["peptide"], lw=0, label=lb, zorder=3)
    ax.axhline(0, **st.REF_KW)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=70, ha="right", rotation_mode="anchor", fontsize=st.FS_MIN)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    out = a.root / "figures" / "ext"; out.mkdir(parents=True, exist_ok=True)
    st.apply()
    heat = load("f80", HERE / "80_amendment_figures.py").heat
    E = a.root / "ext_summary"
    rd = lambda f: pd.read_csv(f, sep="\t", dtype={"mouse": str})  # noqa: E731
    for r in ("all_cells__original", "all_cells__harmonized"):
        R = rd(a.root / "state_lineage_ext" / r / "residuals.tsv"); R = R[R.scope == "sensitivity_all"]
        for mouse, g in R.groupby("mouse"):
            M = g.pivot(index="clade", columns="state", values="residual"); N = g.pivot(index="clade", columns="state", values="count")
            M.index = [c.split("|")[-1] + " (" + c.split("|")[0] + ")" for c in M.index]; N.index = M.index
            heat(M, "residual", out, f"h1_residuals_{r}_{mouse}", f"KP {mouse}, {r.replace('__', ', ')}: clade x state residual (counts)",
                 "(obs - exp) / sqrt(exp)", annot=N)
    S = rd(E / "states_runs.tsv")
    for r in RUNS:
        for sc in ("primary_analysis", "sensitivity_all"):
            d = S[(S.run == r) & (S.scope == sc)]
            if d.empty:
                continue
            heat(d.pivot(index="state", columns="mouse", values="hill_z").clip(-10, 10), "hill_z", out, f"h2_hill_z_{r}_{sc}",
                 f"{r.replace('__', ', ')}, {sc}: z of clade Hill diversity vs within-site null", "z")
    for excl in (False, True):
        d = S[S.scope == "sensitivity_all"]; d = d[d.mouse != "3724"] if excl else d
        M = d.groupby(["state", "run"]).hill_z.median().unstack()[[c for c in RUNS if c in set(d.run)]]
        M.columns = [c.replace("__", "\n") for c in M.columns]
        n = d.groupby(["state", "run"]).mouse.nunique().unstack()[[c for c in RUNS if c in set(d.run)]]
        n.columns = M.columns
        heat(M.clip(-10, 10), "median_z", out, "h3_hill_z_runs" + ("_no3724" if excl else ""),
             "Median over mice of the clade Hill z" + (" (without 3724)" if excl else "") + "; numbers = mice", "median z", annot=n)
    pm = rd(E / "primary_vs_met_runs.tsv") if (E / "primary_vs_met_runs.tsv").exists() else pd.DataFrame()
    for r in RUNS:
        d = pm[(pm.run == r) & (pm.stratum == "all") & (pm.scope == "sensitivity_all")] if len(pm) else pd.DataFrame()
        if d.empty:
            continue
        d = d.sort_values("mouse")
        sh = st.Sheet(1.2 + 0.32 * len(d), 2.5); ax = sh.ax(0.6, 0.75, 0.32 * len(d), 1.35); st.tidy(ax)
        labs = [f"{m}\n{ip:.2f}/{im:.2f}" for m, ip, im in zip(d.mouse, d.primary_identifiable_share, d.metastasis_identifiable_share)]
        pairs(ax, labs, d.primary_excess, d.metastasis_excess, "primary", "metastasis")
        ax.set_ylabel("Cramer's V - null V", fontsize=st.FS_AXLAB); ax.legend(frameon=False, fontsize=st.FS_LEGEND)
        sh.text(0.05, 2.45, f"{r.replace('__', ', ')}: coupling beyond site, equal n (labels: mouse, identifiable share pri/met)",
                fontsize=st.FS_TITLE, va="top"); sh.save(out, f"h4_primary_vs_met_{r}")
    ss = rd(E / "state_side_runs.tsv") if (E / "state_side_runs.tsv").exists() else pd.DataFrame()
    for r in RUNS:
        d = ss[(ss.run == r) & (ss.scope == "sensitivity_all")] if len(ss) else pd.DataFrame()
        if d.empty:
            continue
        d = d.sort_values(["state", "mouse"])
        sh = st.Sheet(1.2 + 0.3 * len(d), 2.8); ax = sh.ax(0.6, 1.05, 0.3 * len(d), 1.4); st.tidy(ax)
        labs = [f"{s_} {m}\n{ip:.2f}/{im:.2f}" for s_, m, ip, im in zip(d.state, d.mouse, d.primary_identifiable_share, d.metastasis_identifiable_share)]
        pairs(ax, labs, d.primary_hill_excess, d.metastasis_hill_excess, "primary", "metastasis")
        ax.set_ylabel("clade Hill - null (neg. = fewer clades)", fontsize=st.FS_AXLAB); ax.legend(frameon=False, fontsize=st.FS_LEGEND)
        sh.text(0.05, 2.75, f"{r.replace('__', ', ')}: per state, lineage concentration by side at equal n (identifiable share pri/met)",
                fontsize=st.FS_TITLE, va="top"); sh.save(out, f"h5_state_side_{r}")


if __name__ == "__main__":
    main()
