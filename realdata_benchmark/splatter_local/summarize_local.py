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


def md_table(t: pd.DataFrame, index: bool = True) -> str:
    """A Markdown table without the optional tabulate dependency (absent on TACC)."""
    t = t.reset_index() if index else t
    head = [str(c) for c in t.columns]
    rows = [["" if pd.isna(v) else str(v).replace("|", "\\|") for v in r] for r in t.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows])


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
        # effective readout (frozen calibration v3): when one side retains no cell at all, the other side's retained cells
        # have no possible partner (their asserted mass is exactly 0), so they count as unmatched; nothing else changes
        eff = {}
        for side, other in (("source", "target"), ("target", "source")):
            d = g[g.side == side].iloc[0]
            if float(g[g.side == other].rejected_fraction.iloc[0]) == 1.0:
                n, pos = int(d.cells), int(d.to_reject)
                eff[side] = {"rec": 1.0 if pos else np.nan, "prec": pos / n if n else np.nan, "frr": 1.0 if n - pos else np.nan}
            else:
                eff[side] = {"rec": float(d.recall), "prec": float(d.precision), "frr": float(d.false_rejection_rate)}
            r_, p_ = eff[side]["rec"], eff[side]["prec"]
            eff[side]["f1"] = np.nan if not np.isfinite(r_) else (0.0 if r_ == 0 or not np.isfinite(p_) else 2 * p_ * r_ / (p_ + r_))
        if case.startswith("Case1"):
            rec["case1_frr_eff"] = float(np.nanmean([eff["source"]["frr"], eff["target"]["frr"]]))
        elif case.startswith("Case2"):
            pos_side = "source" if int(g[g.side == "source"].to_reject.iloc[0]) > 0 else "target"
            rec["case2_f1_eff"] = eff[pos_side]["f1"]
        elif case.startswith("Case3"):
            rec["case3_recall_eff"] = float(np.nanmean([eff["source"]["rec"], eff["target"]["rec"]]))
        rows.append(rec)
    return pd.DataFrame(rows)


def by_level(f: pd.DataFrame, group: str) -> pd.DataFrame:
    metrics = ["case1_frr", "case2_f1", "case3_recall", "case1_frr_eff", "case2_f1_eff", "case3_recall_eff", "reject_all_f1",
               "case2_precision", "case2_recall", "case2_other_frr", "case3_false_asserted_mass", "rejection_cost", "one_side_all"]
    f = f.reindex(columns=list(dict.fromkeys(list(f.columns) + metrics)))  # a metric whose case is absent stays NaN
    m = f.groupby([group, "level"])[metrics].mean()
    rep = f.groupby([group, "level", "replicate"])[["case1_frr", "case2_f1", "case3_recall", "case1_frr_eff", "case2_f1_eff", "case3_recall_eff"]].mean()
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
          md_table(s.round(3)), "", f"**Chosen: {choice}.** Handles batch at L3-L5 (any arm): {bool(s.handles_batch_L3_L5.any())}.", ""]
    for col in ("case1_frr", "case2_f1", "reject_all_f1", "case3_recall", "one_side_all", "rejection_cost"):
        v = t.pivot_table(index="arm", columns="level", values=col).reindex(columns=[l for l in LEVELS if l in set(t.level)])
        md += [f"**{col}**", "", md_table(v.round(3)), ""]
    (a.root / f"CALIBRATION_SELECTION{sfx}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:8]))
    if a.freeze:
        sel = pd.read_csv(a.root / (f"CALIBRATION_SELECTION_{a.selection}.tsv" if a.selection else "CALIBRATION_SELECTION.tsv"), sep="\t").set_index("arm")
        val = pd.read_csv(a.root / f"CALIBRATION_SELECTION_{a.tag}.tsv", sep="\t").set_index("arm")
        c1, c2 = sel.index[sel.chosen][0], val.index[val.chosen][0]
        if c1 == c2:
            arm, note = c1, f"the development selection ({a.selection or 'default'}) and the independent seeds ({a.tag}) choose the same arm"
        else:
            both = pd.concat([pd.read_csv(a.root / (f"RESULTS_calibration_{a.selection}.tsv" if a.selection else "RESULTS_calibration.tsv"), sep="\t"),
                              pd.read_csv(a.root / f"RESULTS_calibration_{a.tag}.tsv", sep="\t")])
            ff = per_fit(both, "arm"); tt = by_level(ff, "arm")
            pooled = tt.groupby("arm")[["case1_frr", "case2_f1", "case3_recall"]].mean().sort_values("case2_f1", ascending=False)
            arm, note = pooled.index[0], f"development chose {c1}, independent seeds chose {c2}; frozen by both pooled"
        (a.root / a.freeze_name).write_text(json.dumps({"arm": arm, "note": note, "rules": "SELECTION_RULES.md", "selection": a.selection, "validation": a.tag}, indent=1))
        print("FROZEN:", arm, "-", note)


def pareto(x: pd.DataFrame) -> pd.Series:
    """Not dominated on (lower case1_frr_eff, higher case2_f1_eff, higher case3_recall_eff)."""
    v = x[["case1_frr_eff", "case2_f1_eff", "case3_recall_eff"]].to_numpy(float)
    out = []
    for k in range(len(v)):
        a_, others = v[k], np.delete(v, k, axis=0)
        dom = (others[:, 0] <= a_[0]) & (others[:, 1] >= a_[1]) & (others[:, 2] >= a_[2]) & \
              ((others[:, 0] < a_[0]) | (others[:, 1] > a_[1]) | (others[:, 2] > a_[2]))
        out.append(not dom.any())
    return pd.Series(out, index=x.index)


def grid(a) -> None:
    sfx = f"_{a.tag}" if a.tag else ""
    r = pd.read_csv(a.root / f"RESULTS_grid{sfx}.tsv", sep="\t")
    base = json.loads((a.root / a.frozen).read_text()).get("config", "") if a.frozen else ""
    f = per_fit(r, "config_id"); t = by_level(f, "config_id")
    t["pareto"] = False
    for lv in set(t.level):
        m = t.level == lv
        t.loc[m, "pareto"] = pareto(t[m]).to_numpy()
    t.to_csv(a.root / f"GRID_SUMMARY{sfx}.tsv", sep="\t", index=False)
    cols = ["config_id", "case1_frr_eff", "case2_f1_eff", "case3_recall_eff", "case2_f1_rep_sd", "one_side_all", "case1_frr", "case3_recall"]
    levels = [l for l in LEVELS if l in set(t.level)]
    md = [f"# L1-L5 preprocessing grid (M4-E; frozen calibration {a.frozen}; effective readout)", "",
          "Each level: the Pareto front on (Case 1 FRR low, Case 2 Rejection F1 high, Case 3 unmatched recall high), "
          "sorted by Case 2 F1; the baseline is shown whether or not it is on the front. Raw-gate FRR / recall beside.", ""]
    for lv in levels:
        x = t[t.level == lv]
        front = x[x.pareto].sort_values("case2_f1_eff", ascending=False)
        bl = x[x.config_id == base]
        md += [f"## {lv}: {len(front)} configurations on the front (of {len(x)})", "",
               md_table(pd.concat([front.head(20), bl]).drop_duplicates("config_id")[cols].round(3), index=False), ""]
    # robust view over L1-L5: on the front at how many levels, and the mean of each metric
    agg = t.groupby("config_id").agg(levels_on_front=("pareto", "sum"), case1_frr_eff=("case1_frr_eff", "mean"),
                                     case2_f1_eff=("case2_f1_eff", "mean"), case3_recall_eff=("case3_recall_eff", "mean"),
                                     worst_case2_f1=("case2_f1_eff", "min"), one_side_all=("one_side_all", "mean")).reset_index()
    agg["pareto_overall"] = pareto(agg).to_numpy()
    agg = agg.sort_values(["levels_on_front", "case2_f1_eff"], ascending=False)
    agg.to_csv(a.root / f"GRID_OVERALL{sfx}.tsv", sep="\t", index=False)
    md += ["## Over L1-L5", "", "Configurations on the Pareto front at the most levels, then by mean Case 2 F1; the overall front "
           "(on the three means) is marked. Baseline included.", "",
           md_table(pd.concat([agg.head(25), agg[agg.config_id == base]]).drop_duplicates("config_id").round(3), index=False), ""]
    (a.root / f"GRID_SUMMARY{sfx}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[-30:]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--stage", choices=("calibration", "grid"), required=True)
    p.add_argument("--tag", default="")
    p.add_argument("--freeze", action="store_true")
    p.add_argument("--freeze-name", default="FROZEN_CALIBRATION.json")
    p.add_argument("--frozen", default="", help="grid: the frozen calibration file (its config is the baseline)")
    p.add_argument("--selection", default="", help="tag of the replicates 1-2 selection table to freeze against (all = the 4 arms + exploration)")
    a = p.parse_args()
    calibration(a) if a.stage == "calibration" else grid(a)
