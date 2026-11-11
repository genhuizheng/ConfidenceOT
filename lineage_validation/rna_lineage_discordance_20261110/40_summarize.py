"""Summary tables of the three-view comparison (PLAN_v2.md): CRISPR-OT audit, A vs B gates and couplings, discordance
against its permutation null, its concentration by site / state / lineage, covariates, gate relation, 3724.

    python lineage_validation/rna_lineage_discordance_20261110/40_summarize.py --root results/rna_lineage_discordance_20261110

Flags (exploratory, PLAN.md section 4 carried into v2): a cell is "divergent" when its div is in the top 10% of its
lineage within the unit, "convergent" likewise for conv; "concordant" when both are within the 40-60% band.
Mouse = KP mouse / Quinn experiment. Writes ROOT/summary/*.tsv and prints the main tables.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def unit_meta(u):
    if u.startswith("L23__Quinn2021_"):
        return "Quinn", u.replace("L23__Quinn2021_", "").split("__")[0], "case"
    if u.startswith("L23__KPTracer_"):
        return "KP", u.replace("L23__KPTracer_", "").split("__")[0], "case"
    return "KP", u.split("__")[1], "pair"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    out = a.root / "summary"; out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    aud = pd.DataFrame([json.load(open(f)) for f in glob.glob(str(a.root / "crispr_ot" / "*" / "audit.json"))])
    aud = aud.drop(columns=["calibration_warnings"], errors="ignore").sort_values("unit_id")
    aud.to_csv(out / "crispr_ot_audit.tsv", sep="\t", index=False)
    units, cells = [], []
    for d in sorted(glob.glob(str(a.root / "compare" / "*" / "unit.json"))):
        j = json.load(open(d)); ds, mouse, st = unit_meta(j["unit_id"])
        tag = Path(d).parent.name.rsplit("__", 1)[1]
        j.update(dataset_short=ds, mouse=mouse, stratum=st, tag=tag)
        units.append(j)
        c = pd.read_csv(Path(d).parent / "cells.tsv.gz", sep="\t", dtype={"cell": str})
        cells.append(c.assign(unit_id=j["unit_id"], dataset=ds, mouse=mouse, stratum=st, tag=tag))
    U = pd.DataFrame(units); C = pd.concat(cells, ignore_index=True)
    # flags within unit x lineage
    grp = C.groupby(["tag", "unit_id", "lineage"])
    C["div_rank"] = grp["div"].rank(pct=True); C["conv_rank"] = grp["conv"].rank(pct=True)
    C["divergent"] = C.div_rank >= 0.9; C["convergent"] = C.conv_rank >= 0.9
    C["concordant"] = C.div_rank.between(0.4, 0.6) & C.conv_rank.between(0.4, 0.6)
    C.to_csv(out / "cells_all.tsv.gz", sep="\t", index=False, compression="gzip")
    # 1. A vs B: gates and couplings per unit
    cols = ["tag", "dataset_short", "stratum", "mouse", "unit_id", "kappa_source", "kappa_target", "lineages", "cells"]
    ab = U[cols].copy()
    t = C[C.side == "target"]
    cm = t.groupby(["tag", "unit_id"]).agg(tv_median=("tv", "median"), top15_overlap=("top15_overlap", "median"),
                                           s_top_rnaot=("s_top_rnaot", "mean"), s_top_crisprot=("s_top_crisprot", "mean"),
                                           r_top_rnaot=("r_top_rnaot", "mean"), r_top_crisprot=("r_top_crisprot", "mean"),
                                           lcrf_rna=("lcrf_rna", "mean"), lcrf_ind=("lcrf_ind_rna", "mean"), lcrf_cr=("lcrf_cr", "mean"),
                                           rclf_rna=("rclf_rna", "mean"), rclf_ind=("rclf_ind_rna", "mean"), rclf_cr=("rclf_cr", "mean")).reset_index()
    ret = C.groupby(["tag", "unit_id", "side"])[["rna_retained", "crispr_retained"]].mean().unstack("side")
    ret.columns = [f"{v}_{s}" for v, s in ret.columns]
    ab = ab.merge(cm, on=["tag", "unit_id"]).merge(ret.reset_index(), on=["tag", "unit_id"])
    ab.to_csv(out / "a_vs_b_units.tsv", sep="\t", index=False)
    # 2. discordance vs null per unit x lineage (and per mouse)
    lin = C.groupby(["tag", "dataset", "stratum", "mouse", "unit_id", "lineage"]).agg(
        cells=("cell", "size"), div=("div", "mean"), div_null=("div_null", "mean"), conv=("conv", "mean"), conv_null=("conv_null", "mean"),
        agree=("agree", "mean")).reset_index()
    lin["div_excess"] = lin["div"] - lin.div_null; lin["conv_excess"] = lin.conv - lin.conv_null
    res = a.root / "lineages" / "resolution.tsv"
    if res.exists():
        R = pd.read_csv(res, sep="\t", dtype={"mouse": str})
        R["lineage_key"] = np.where(R.dataset == "KPTracer", "unit", R.lineage)
        lin = lin.merge(R[["mouse", "lineage", "resolved"]].assign(lineage=lambda d: d.lineage), on=["mouse", "lineage"], how="left")
        kp_res = R[R.dataset == "KPTracer"].set_index("lineage").resolved      # KP family = the unit's primary tumour
        fam = lin.unit_id.str.split("__").str[2]
        lin.loc[lin.dataset == "KP", "resolved"] = fam[lin.dataset == "KP"].map(kp_res)
        lin["family"] = np.where(lin.dataset == "KP", fam, lin.lineage)
    lin.to_csv(out / "discordance_lineages.tsv", sep="\t", index=False)
    w = lambda g, c: np.average(g[c], weights=g.cells)  # noqa: E731
    mouse = lin.groupby(["tag", "dataset", "stratum", "mouse"]).apply(lambda g: pd.Series({
        "lineages": len(g), "cells": g.cells.sum(), "div": w(g, "div"), "div_null": w(g, "div_null"), "conv": w(g, "conv"),
        "conv_null": w(g, "conv_null"), "agree": w(g, "agree")}), include_groups=False).reset_index()
    mouse["div_excess"] = mouse["div"] - mouse.div_null; mouse["conv_excess"] = mouse.conv - mouse.conv_null
    mouse.to_csv(out / "discordance_mice.tsv", sep="\t", index=False)
    # 3. concentration: by site and (KP) cell state
    site = C.groupby(["tag", "dataset", "mouse", "site"]).agg(cells=("cell", "size"), div=("div", "mean"), div_null=("div_null", "mean"),
                                                            conv=("conv", "mean"), conv_null=("conv_null", "mean"),
                                                            divergent=("divergent", "mean"), convergent=("convergent", "mean")).reset_index()
    site.to_csv(out / "discordance_sites.tsv", sep="\t", index=False)
    if "cell_type" in C:
        stt = C.dropna(subset=["cell_type"]).groupby(["tag", "mouse", "cell_type"]).agg(
            cells=("cell", "size"), div=("div", "mean"), div_null=("div_null", "mean"), conv=("conv", "mean"), conv_null=("conv_null", "mean"),
            divergent=("divergent", "mean"), convergent=("convergent", "mean")).reset_index()
        stt.to_csv(out / "discordance_states_kp.tsv", sep="\t", index=False)
    # 4. covariates: Spearman of div / conv with depth and detected genes, per unit x lineage (>= 50 cells)
    cov = []
    for (tag, u, l), g in C.groupby(["tag", "unit_id", "lineage"]):
        if len(g) < 50:
            continue
        r = {"tag": tag, "unit_id": u, "lineage": l, "cells": len(g)}
        for sc in ("div", "conv"):
            for cv in ("total_counts_raw", "detected_genes"):
                if cv in g and g[cv].notna().sum() > 20:
                    r[f"rho_{sc}_{cv}"] = spearmanr(g[sc], g[cv], nan_policy="omit")[0]
        cov.append(r)
    pd.DataFrame(cov).to_csv(out / "covariates.tsv", sep="\t", index=False)
    # 5. gate relation (AUROC of div / conv for rejected vs retained)
    gcols = [c for c in U.columns if c.startswith("auc_")]
    U[["tag", "dataset_short", "stratum", "mouse", "unit_id", *gcols]].to_csv(out / "gate_vs_discordance.tsv", sep="\t", index=False)
    # print the main views
    print("== CRISPR-OT audit"); print(aud[["unit_id", "rejection_cost", "m4e_inference_valid", "cross_pairs_not_evaluable", "cross_pairs_D0",
                                            "m4e_retained_source", "m4e_retained_target", "m4e_outer_converged"]].round(3).to_string(index=False))
    print("== A vs B per dataset/stratum (median over units)")
    print(ab.groupby(["tag", "dataset_short", "stratum"])[["kappa_source", "kappa_target", "tv_median", "top15_overlap", "s_top_rnaot",
                                                           "s_top_crisprot", "r_top_rnaot", "r_top_crisprot"]].median().round(3).to_string())
    print("== discordance per mouse"); print(mouse.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
