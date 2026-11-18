"""Balanced vs unbalanced OT across the cost families C1-C4 (PLAN.md post-hoc sections of 2026-11-18).

    python ot_depth_benchmark_20261117/compare_bot_uot.py

Fair contrast: entropic balanced OT (Sinkhorn, ε 0.1) vs entropic UOT (ε 0.1, λ 1); exact balanced OT and partial OT are
listed beside them; ConfidenceOT is read on its retained x retained block with its rejected fractions.
Writes results/ot_depth_benchmark_20261117/bot_uot/{tables,figures}.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "realdata_benchmark" / "v3_grid" / "figures"))
import nbt_style as ns  # noqa: E402
import fig_splatter as fs  # noqa: E402
from fig_s2b import shares, LEVELS  # noqa: E402

RES = HERE.parent / "results" / "ot_depth_benchmark_20261117"
OUT = RES / "bot_uot"
COSTS = {"C1": "C1 ranknm256 scale cos pca50 (main)", "C2": "C2 log1p-PCA sqeuc (standard)",
         "C3": "C3 ranknm512 noscale cos pca30", "C4": "C4 logCPM noscale cos pca30"}


def load(kind: str) -> pd.DataFrame:
    if kind == "s1":
        parts = [pd.read_csv(RES / "splatter" / "METRICS_s1.tsv", sep="\t"), pd.read_csv(RES / "splatter_best" / "s1" / "METRICS_s1.tsv", sep="\t")]
        return pd.concat(parts, ignore_index=True)
    roots = [RES / "splatter", RES / "splatter_s2b", RES / "splatter_best" / "s2", RES / "splatter_best" / "s2b"]
    return pd.concat([shares(r) for r in roots], ignore_index=True)


def main():
    ns.apply()
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    fs.FIG = OUT / "figures"
    m1 = load("s1")
    # S1 native: correct share and coupling-depth correlation, all-shared, by level x cost x solver
    a = m1[m1.case == "all_shared"]
    t1 = a.pivot_table(index=["cost", "level"], columns="solver", values=["correct_mass", "depth_corr_count"]).round(3)
    t1.to_csv(OUT / "tables" / "S1_allshared_correct_and_depthcorr.tsv", sep="\t")
    # unmatched populations: transported share of the unmatched population (gate-free) and gate rejection
    rows = []
    for case, side in (("population_lost", "source"), ("population_emerged", "target")):
        d = m1[m1.case == case]
        g = d.groupby(["cost", "solver"])[[f"transported_share_unmatched_{side}", f"transported_share_shared_{side}"]].mean()
        g.columns = ["unmatched", "shared"]; g["case"] = case; rows.append(g.reset_index())
    pd.concat(rows).round(3).to_csv(OUT / "tables" / "S1_unmatched_transported_share.tsv", sep="\t", index=False)
    c = m1[m1.solver == "confidenceot"]
    c.groupby(["cost", "case"])[["rejected_fraction_unmatched_source", "rejected_fraction_unmatched_target",
                                 "rejected_fraction_shared_source", "rejected_fraction_shared_target", "false_asserted_share"]].mean().round(3) \
        .to_csv(OUT / "tables" / "S1_confidenceot_gate.tsv", sep="\t")
    # S2 / S2b: conflict, aligned, global
    s = load("s2")
    s.groupby(["cost", "arm", "level", "f", "solver"]).mean(numeric_only=True).round(4).drop(columns=["replicate"], errors="ignore") \
        .to_csv(OUT / "tables" / "S2_all_costs_group12_shares.tsv", sep="\t")
    for arm in ("conflict", "aligned", "global"):
        d = s[(s.arm == arm) & (s.f == 0.1)]
        for solver in ("sinkhorn", "uot", "emd", "partial"):
            M = d[d.solver == solver].groupby(["level", "cost"]).correct.mean().unstack().reindex(index=LEVELS, columns=list(COSTS))
            fs.heat(M, f"{arm.capitalize()} arm, f = 0.1: {fs.LABEL[solver]}", f"{arm}_f0.1_{solver}_correct_by_cost", vmin=0, vmax=1,
                    clabel="Correct-group share of the\ntransported Group1/2 mass")
        w = d.pivot_table(index=["cost", "level", "replicate"], columns="solver", values="correct").reset_index()
        w["uot_minus_balanced"] = w.uot - w.sinkhorn
        M = w.groupby(["level", "cost"]).uot_minus_balanced.mean().unstack().reindex(index=LEVELS, columns=list(COSTS))
        fs.heat(M, f"{arm.capitalize()} arm, f = 0.1: UOT − entropic balanced (ε 0.1)", f"{arm}_f0.1_uot_minus_balanced_by_cost",
                vmin=-1, vmax=1, cmap="RdBu", clabel="Difference in correct-group share")
        M.round(3).to_csv(OUT / "tables" / f"{arm}_f0.1_uot_minus_balanced.tsv", sep="\t")
    # compact headline table: per cost, mean over levels L1-L5 / A4 / A5 at f 0.1, and L0
    hl = s[(s.f == 0.1) & s.solver.isin(["emd", "sinkhorn", "uot", "partial"])].copy()
    hl["levelset"] = np.where(hl.level == "L0_matched", "L0", "L1-L5,A4,A5")
    H = hl.groupby(["arm", "levelset", "cost", "solver"]).correct.mean().unstack().round(2)
    H.to_csv(OUT / "tables" / "HEADLINE_f0.1_correct_share.tsv", sep="\t")
    print(H.to_string())
    print("->", OUT)


if __name__ == "__main__" and "--gates" not in sys.argv:
    main()


# ------------------------------------------------------------------------------- ConfidenceOT: UOT vs balanced backbone
def gate_compare():
    """confidenceot (UOT backbone) vs confidenceot_bot (balanced backbone), C1-C4: gate rejections and the retained x
    retained block, S1 by case and S2 / S2b by arm at f 0.1. Writes bot_uot/tables/GATE_*.tsv."""
    t = OUT / "tables"; t.mkdir(parents=True, exist_ok=True)
    s1 = pd.concat([pd.read_csv(RES / "splatter" / "METRICS_s1.tsv", sep="\t"), pd.read_csv(RES / "splatter_best" / "s1" / "METRICS_s1.tsv", sep="\t"),
                    pd.read_csv(RES / "splatter_cotbot" / "s1" / "METRICS_s1.tsv", sep="\t")], ignore_index=True)
    s1 = s1[s1.solver.isin(["confidenceot", "confidenceot_bot"])]
    cols = ["rejected_fraction_unmatched_source", "rejected_fraction_unmatched_target", "rejected_fraction_shared_source",
            "rejected_fraction_shared_target", "false_asserted_share", "asserted_mass"]
    G1 = s1.groupby(["cost", "case", "solver"])[cols].mean().round(3)
    G1.to_csv(t / "GATE_S1_uot_vs_balanced_backbone.tsv", sep="\t")
    roots = [RES / "splatter", RES / "splatter_s2b", RES / "splatter_best" / "s2", RES / "splatter_best" / "s2b",
             RES / "splatter_cotbot" / "s2", RES / "splatter_cotbot" / "s2b"]
    s2 = pd.concat([pd.read_csv(r / "METRICS_s2.tsv", sep="\t") for r in roots], ignore_index=True)
    s2 = s2[s2.solver.isin(["confidenceot", "confidenceot_bot"]) & (s2.f == 0.1)].copy()
    s2["levelset"] = np.where(s2.level == "L0_matched", "L0", "L1-L5,A4,A5")
    G2 = s2.groupby(["arm", "levelset", "cost", "solver"])[["rejected_fraction_shared_source", "rejected_fraction_shared_target",
                                                            "asserted_mass", "false_asserted_share"]].mean().round(3)
    G2.to_csv(t / "GATE_S2_f0.1_uot_vs_balanced_backbone.tsv", sep="\t")
    print(G1.to_string()); print(G2.to_string())


if __name__ == "__main__" and "--gates" in sys.argv:
    gate_compare()
