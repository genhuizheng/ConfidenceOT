"""PLAN_v2.md amendment A2-A4: summary tables, per mouse and pooled (mean over mice, with and without KP 3724).

    python lineage_validation/rna_lineage_discordance_20261110/78_amendment_summary.py --root results/rna_lineage_discordance_20261110 [--tag l5best]

Reads state_lineage/ (70_), common_support/ (75_), summary/ (40_) and lineages/flags.tsv (72_). Writes
ROOT/amendment_summary/:
  a2_mouse_cramer_pooled.tsv  Cramer's V observed / null / excess, mean over mice, per scope, with / without 3724
  a2_states_by_mouse.tsv      per state: mice, and per mouse its Hill z and largest-share z (no pooling across states)
  a3_primary_vs_met_pooled.tsv  per stratum and tertile: mean over mice of primary / metastasis excess, with / without 3724
  a3_viewc_side_state.tsv     KP view C: per mouse, side (source = primary, target = metastasis) and authors' state
                              (>= 20 distinct cells): divergence and convergence excess (observed - null), cell-unit mean
  a4_common_support_mice.tsv  per dataset and mouse: unit medians of the common-block TV, top-k overlap and its chance
                              level, mass kept on the block, beside the gate kappa (40_ a_vs_b_units), with / without 3724
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def pooled(df: pd.DataFrame, by: list, cols: list) -> pd.DataFrame:
    out = []
    for excl in (False, True):
        d = df[df.mouse != "3724"] if excl else df
        g = d.groupby(by, dropna=False)
        r = g[cols].mean().add_suffix("_mean")
        r["mice"] = g.mouse.nunique()
        r["without_3724"] = excl
        out.append(r.reset_index())
    return pd.concat(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--tag", default="l5best")
    a = p.parse_args()
    SL, out = a.root / "state_lineage", a.root / "amendment_summary"; out.mkdir(exist_ok=True)
    rd = lambda f: pd.read_csv(f, sep="\t", dtype={"mouse": str}, low_memory=False)  # noqa: E731
    mv = rd(SL / "mouse_cramer.tsv"); mv["cramer_v_excess"] = mv.cramer_v - mv.cramer_v_null
    pooled(mv, ["scope"], ["cramer_v", "cramer_v_null", "cramer_v_excess", "cramer_v_z"]).to_csv(out / "a2_mouse_cramer_pooled.tsv", sep="\t", index=False)
    stt = rd(SL / "states.tsv")
    stt[["scope", "mouse", "state", "cells", "clades_ge5", "clades_ge5_null", "families_ge5", "hill", "hill_null", "hill_z", "hill_p",
         "max_share", "max_share_null", "max_share_z", "max_share_p"]].sort_values(["scope", "state", "mouse"]).to_csv(
        out / "a2_states_by_mouse.tsv", sep="\t", index=False)
    pm = rd(SL / "primary_vs_met.tsv")
    if len(pm):
        pooled(pm, ["scope", "stratum", "tertile"], ["primary_excess", "metastasis_excess", "met_minus_primary_excess"]).to_csv(
            out / "a3_primary_vs_met_pooled.tsv", sep="\t", index=False)
    C = rd(a.root / "summary" / "cells_all.tsv.gz")
    C = C[(C.tag == a.tag) & (C.dataset == "KP")].copy()
    C["div_excess"], C["conv_excess"] = C["div"] - C.div_null, C.conv - C.conv_null
    g = C.groupby(["mouse", "side", "cell_type"])
    v = g.agg(cells=("cell", "nunique"), div_excess=("div_excess", "mean"), conv_excess=("conv_excess", "mean")).reset_index()
    v[v.cells >= 20].to_csv(out / "a3_viewc_side_state.tsv", sep="\t", index=False)
    cs = rd(a.root / "common_support" / "units.tsv"); cs = cs[cs.config == a.tag]
    ab = rd(a.root / "summary" / "a_vs_b_units.tsv"); ab = ab[ab.tag == a.tag][["unit_id", "kappa_source", "kappa_target", "tv_median"]]
    cs = cs.merge(ab.rename(columns={"tv_median": "tv_median_own_blocks"}), on="unit_id", how="left")
    m = cs.groupby(["dataset", "mouse"]).agg(units=("unit_id", "size"), kappa_source=("kappa_source", "median"), kappa_target=("kappa_target", "median"),
                                             tv_own_blocks=("tv_median_own_blocks", "median"), tv_common=("tv_median", "median"),
                                             overlap_common=("overlap_mean", "median"), chance_overlap=("chance_overlap", "median"),
                                             rna_mass_on_block=("rna_mass_on_block", "median"), crispr_mass_on_block=("crispr_mass_on_block", "median"),
                                             evaluable_share=("evaluable_share", "median")).reset_index()
    m.to_csv(out / "a4_common_support_mice.tsv", sep="\t", index=False)
    for f in sorted(out.glob("*.tsv")):
        print(f"== {f.name}"); print(rd(f).round(3).to_string(index=False, max_rows=80))


if __name__ == "__main__":
    main()
