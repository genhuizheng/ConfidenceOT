"""Summaries of the local benchmark under SELECTION_RULES.md (M4-E).

    python realdata_benchmark/splatter_local/summarize_local.py --root ROOT --stage calibration [--tag validation] [--freeze]
    python realdata_benchmark/splatter_local/summarize_local.py --root ROOT --stage grid

calibration: per arm x level the three metrics (Case 1 FRR, Case 2 F1, Case 3 recall; mean over pairs and replicates,
SD over replicate means), the one-side reject-all F1 of each level (from the truth), the "handles batch" check at
L3-L5, the Pareto filter and the choice; writes CALIBRATION_SELECTION<_tag>.{tsv,md}. --freeze writes
FROZEN_CALIBRATION.json from the selection(s) as the rules say (needs both the selection and the validation tables).
grid: per level x configuration the three metrics; best set per level (within one replicate SD of the top Case 2
F1); writes GRID_SUMMARY.tsv and GRID_SUMMARY.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LEVELS = ["L0_matched", "L1_depth", "L2_depth_dropout", "L3_batch", "L4_batch_perturb", "L5_batch_perturb_imbalance",
          "A4_perturb_only", "A5_imbalance_only"]
BATCH = ["L3_batch", "L4_batch_perturb", "L5_batch_perturb_imbalance"]


def per_fit(r: pd.DataFrame, group: str) -> pd.DataFrame:
    """One row per fit: the metric its case contributes, and the one-side reject-all F1 for Case 2."""
    e = r[(r.method == "M4-E") & (r.status == "ok")].drop_duplicates(["level", "replicate", "pair_id", group, "side"])
    rows = []
    for k, g in e.groupby(["level", "replicate", "case", "pair_id", group]):
        lv, rep, case, pair, gid = k
        rec = {"level": lv, "replicate": rep, "case": case, group: gid,
               "source_rejected": float(g[g.side == "source"].rejected_fraction.iloc[0]),
               "target_rejected": float(g[g.side == "target"].rejected_fraction.iloc[0]),
               "rejection_cost": float(g.rejection_cost.iloc[0])}
        if case.startswith("Case1"):
            rec["case1_frr"] = float(g.false_rejection_rate.mean())
        elif case.startswith("Case2"):
            s = g[g.to_reject > 0].iloc[0]
            rec["case2_f1"] = float(s.f1); rec["case2_precision"] = float(s.precision); rec["case2_recall"] = float(s.recall)
            p = s.to_reject / s.cells; rec["reject_all_f1"] = 2 * p / (1 + p)
            rec["case2_other_frr"] = float(g[g.to_reject == 0].false_rejection_rate.iloc[0])
        elif case.startswith("Case3"):
            rec["case3_recall"] = float(g.recall.mean()); rec["case3_false_asserted_mass"] = float(g.false_asserted_mass.iloc[0])
        rec["one_side_all"] = (rec["source_rejected"] >= 0.99 and rec["target_rejected"] <= 0.01) or (rec["target_rejected"] >= 0.99 and rec["source_rejected"] <= 0.01)
        rows.append(rec)
    return pd.DataFrame(rows)


def by_level(f: pd.DataFrame, group: str) -> pd.DataFrame:
    metrics = ["case1_frr", "case2_f1", "case3_recall", "reject_all_f1", "case2_precision", "case2_recall", "case2_other_frr",
               "case3_false_asserted_mass", "rejection_cost", "one_side_all"]
    m = f.groupby([group, "level"])[metrics].mean()
    rep = f.groupby([group, "level", "replicate"])[["case1_frr", "case2_f1", "case3_recall"]].mean()
    sd = rep.groupby([group, "level"]).std().add_suffix("_rep_sd")
    return m.join(sd).reset_index()


def calibration(a) -> None:
    sfx = f"_{a.tag}" if a.tag else ""
    r = pd.read_csv(a.root / f"RESULTS_calibration{sfx}.tsv", sep="\t")
    f = per_fit(r, "arm"); t = by_level(f, "arm")
    t["handles_batch_level"] = (t.case2_f1 > t.reject_all_f1) & (t.case1_frr < 0.5)
    overall = t.groupby("arm")[["case1_frr", "case2_f1", "case3_recall"]].mean()
    batch = t[t.level.isin(BATCH)].groupby("arm")[["case1_frr", "case2_f1", "case3_recall"]].mean().add_suffix("_batch")
    handles = t[t.level.isin(BATCH)].groupby("arm").handles_batch_level.all().rename("handles_batch_L3_L5")
    sdf1 = f.groupby(["arm", "replicate"]).case2_f1.mean().groupby("arm").std().rename("case2_f1_rep_sd")
    s = overall.join(batch).join(handles).join(sdf1)
    def dominated(i):
        x = s.loc[i]
        for j in s.index:
            if j == i:
                continue
            y = s.loc[j]
            no_worse = y.case1_frr <= x.case1_frr and y.case2_f1 >= x.case2_f1 and y.case3_recall >= x.case3_recall
            better = y.case1_frr < x.case1_frr or y.case2_f1 > x.case2_f1 or y.case3_recall > x.case3_recall
            if no_worse and better:
                return True
        return False
    s["pareto_dominated"] = [dominated(i) for i in s.index]
    cand = s[~s.pareto_dominated].sort_values("case2_f1", ascending=False)
    top = cand.index[0]
    close = cand[(cand.case2_f1 >= cand.case2_f1.iloc[0] - cand.case2_f1_rep_sd.iloc[0])]
    choice = close.sort_values("case1_frr").index[0]
    s["chosen"] = s.index == choice
    s.to_csv(a.root / f"CALIBRATION_SELECTION{sfx}.tsv", sep="\t")
    t.to_csv(a.root / f"CALIBRATION_BY_LEVEL{sfx}.tsv", sep="\t", index=False)
    md = [f"# L−1 calibration selection{(' (' + a.tag + ')') if a.tag else ''}", "",
          "Means over the 8 levels (equal weight) and over the batch levels L3-L5; M4-E.", "",
          s.round(3).to_markdown(), "", f"**Chosen: {choice}.** Handles batch at L3-L5 (any arm): {bool(s.handles_batch_L3_L5.any())}.", ""]
    for col in ("case1_frr", "case2_f1", "reject_all_f1", "case3_recall", "one_side_all", "rejection_cost"):
        v = t.pivot_table(index="arm", columns="level", values=col).reindex(columns=[l for l in LEVELS if l in set(t.level)])
        md += [f"**{col}**", "", v.round(3).to_markdown(), ""]
    (a.root / f"CALIBRATION_SELECTION{sfx}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:8]))
    if a.freeze:
        sel = pd.read_csv(a.root / (f"CALIBRATION_SELECTION_{a.selection}.tsv" if a.selection else "CALIBRATION_SELECTION.tsv"), sep="\t").set_index("arm")
        val = pd.read_csv(a.root / "CALIBRATION_SELECTION_validation.tsv", sep="\t").set_index("arm")
        c1, c2 = sel.index[sel.chosen][0], val.index[val.chosen][0]
        if c1 == c2:
            arm, note = c1, "replicates 1-2 and the independent replicates 3-4 choose the same arm"
        else:
            both = pd.concat([pd.read_csv(a.root / "RESULTS_calibration.tsv", sep="\t"), pd.read_csv(a.root / "RESULTS_calibration_validation.tsv", sep="\t")])
            ff = per_fit(both, "arm"); tt = by_level(ff, "arm")
            pooled = tt.groupby("arm")[["case1_frr", "case2_f1", "case3_recall"]].mean().sort_values("case2_f1", ascending=False)
            arm, note = pooled.index[0], f"replicates 1-2 chose {c1}, replicates 3-4 chose {c2}; frozen by replicates 1-4 pooled"
        (a.root / "FROZEN_CALIBRATION.json").write_text(json.dumps({"arm": arm, "note": note, "rules": "SELECTION_RULES.md"}, indent=1))
        print("FROZEN:", arm, "-", note)


def grid(a) -> None:
    r = pd.read_csv(a.root / "RESULTS_grid.tsv", sep="\t")
    f = per_fit(r, "config_id"); t = by_level(f, "config_id")
    t.to_csv(a.root / "GRID_SUMMARY.tsv", sep="\t", index=False)
    md = ["# L1-L5 preprocessing grid (M4-E; frozen calibration)", ""]
    for lv in [l for l in LEVELS if l in set(t.level)]:
        x = t[t.level == lv].sort_values("case2_f1", ascending=False)
        top = x.case2_f1.iloc[0]; sd = x.case2_f1_rep_sd.iloc[0] if np.isfinite(x.case2_f1_rep_sd.iloc[0]) else 0
        best = x[x.case2_f1 >= top - sd]
        md += [f"## {lv}: best set {len(best)} configurations (Case 2 F1 >= {top - sd:.3f})", "",
               x.head(15)[["config_id", "case2_f1", "case2_f1_rep_sd", "case1_frr", "case3_recall", "one_side_all"]].round(3).to_markdown(index=False), ""]
    (a.root / "GRID_SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:30]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--stage", choices=("calibration", "grid"), required=True)
    p.add_argument("--tag", default="")
    p.add_argument("--freeze", action="store_true")
    p.add_argument("--selection", default="", help="tag of the replicates 1-2 selection table to freeze against (all = the 4 arms + exploration)")
    a = p.parse_args()
    calibration(a) if a.stage == "calibration" else grid(a)
