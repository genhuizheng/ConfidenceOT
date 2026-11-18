"""S2b figures (PLAN.md post-hoc extension 2026-11-18): the depth-identity conflict on L0 (S2) and on L1-L5, A4, A5 (S2b),
read for the Group1/2 source cells; balanced vs unbalanced OT at the same ε.

    python ot_depth_benchmark_20261117/fig_s2b.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "realdata_benchmark" / "v3_grid" / "figures"))
sys.path.insert(0, str(HERE))
import nbt_style as ns  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from fig_splatter import COL, LABEL, THIN, heat  # noqa: E402
import fig_splatter as fs  # noqa: E402

RES = HERE.parent / "results" / "ot_depth_benchmark_20261117"
LEVELS = ["L0_matched", "L1_depth", "L2_depth_dropout", "L3_batch", "L4_batch_perturb", "L5_batch_perturb_imbalance",
          "A4_perturb_only", "A5_imbalance_only"]
SOLV = ["emd", "sinkhorn", "uot", "partial", "confidenceot"]


def shares(root: Path) -> pd.DataFrame:
    t = pd.read_csv(root / "TRANSITIONS_s2.tsv", sep="\t")
    g = pd.read_csv(root / "GATES_s2.tsv.gz", sep="\t")
    k = ["level", "replicate", "arm", "f", "cost", "solver"]
    x = t[t.source_group.isin(THIN)].copy()
    x["cat"] = np.where(x.source_group == x.target_group, "correct",
                        np.where(x.target_group.isin(THIN), "wrong_dissimilar_depth", "wrong_similar_depth"))
    s = x.groupby(k + ["cat"]).mass.sum().unstack(fill_value=0.0)
    tot = s.sum(1)
    marg = g[(g.side == "source") & (g.cost == "C1")].groupby(["level", "replicate", "arm", "f"]).group.apply(lambda v: v.isin(THIN).mean())
    s = s.div(tot, axis=0)
    s["transported_fraction"] = tot.values / marg.reindex(pd.MultiIndex.from_frame(tot.reset_index()[["level", "replicate", "arm", "f"]])).values
    return s.reset_index()


def main():
    ns.apply()
    out = RES / "splatter_s2b" / "figures"; fs.FIG = out
    tab = RES / "splatter_s2b" / "tables"; tab.mkdir(parents=True, exist_ok=True)
    s = pd.concat([shares(RES / "splatter"), shares(RES / "splatter_s2b")], ignore_index=True)
    s.groupby(["level", "arm", "f", "cost", "solver"]).agg(["mean", "sem"]).round(4).to_csv(tab / "S2_S2b_group12_mass_shares.tsv", sep="\t")
    for cost in ("C1", "C2"):
        for arm in ("conflict", "aligned", "global"):
            for q, lab in (("correct", "Correct-group share of the\ntransported Group1/2 mass"),
                           ("wrong_similar_depth", "Share on wrong groups\nwith source-like depth")):
                if arm != "conflict" and q != "correct":
                    continue
                d = s[(s.arm == arm) & (s.cost == cost) & (s.f == 0.1)]
                M = d.groupby(["level", "solver"])[q].mean().unstack().reindex(index=LEVELS, columns=SOLV)
                M.columns = [LABEL[c] for c in M.columns]
                heat(M, f"{arm.capitalize()} arm, f = 0.1, {fs.COSTNAME[cost]}", f"S2b_{arm}_{cost}_f0.1_{q}", vmin=0, vmax=1, clabel=lab)
            # UOT minus entropic balanced OT (same ε), every f: the balanced-vs-unbalanced contrast
            d = s[(s.arm == arm) & (s.cost == cost)].pivot_table(index=["level", "replicate", "f"], columns="solver", values="correct").reset_index()
            d["uot_minus_balanced"] = d["uot"] - d["sinkhorn"]
            M = d.groupby(["level", "f"]).uot_minus_balanced.mean().unstack().reindex(LEVELS)
            M.columns = [f"f {c:g}" for c in M.columns]
            M = M[M.columns[::-1]]
            heat(M, f"{arm.capitalize()} arm, {fs.COSTNAME[cost]}: UOT − entropic balanced (ε 0.1)",
                 f"S2b_{arm}_{cost}_uot_minus_balanced", vmin=-1, vmax=1, cmap="RdBu", clabel="Difference in correct-group share")
    # per-level line panels (conflict arm, correct share vs f)
    for lv in LEVELS:
        for cost in ("C1", "C2"):
            d = s[(s.level == lv) & (s.arm == "conflict") & (s.cost == cost)]
            fs.line_panel(d, "correct", f"{lv}, conflict, {fs.COSTNAME[cost]}", f"S2b_conflict_{lv}_{cost}_correct_share",
                          "Correct-group share of the\ntransported Group1/2 mass", solvers=["independent"] + SOLV)
    print("->", out)


if __name__ == "__main__":
    main()
