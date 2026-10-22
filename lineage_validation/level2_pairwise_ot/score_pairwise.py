"""Score Level 2 runs (Method A_standard_OT, Method B_our_OT) against the CRISPR subclones (PLAN.md, Metrics).

    python lineage_validation/level2_pairwise_ot/score_pairwise.py --runs RUNS_ROOT --out OUT

RUNS_ROOT holds run_pairwise.py's <case_id>/ folders (the small files suffice: diagnostics.json,
source_groups.tsv, target_mass_*.tsv.gz, FAILED.json). Every case of inputs/cases.tsv is listed in
run_status.tsv as ok, failed (with the recorded error) or missing; nothing is substituted or dropped. Target
labels come from inputs/family_cells.tsv.gz.

For target cell j and source group g of family F: S_gj = mass_gj / column mass_j (= sum over i in g of
q_ij), S~_gj = S_gj / sum over F (subclones + family unlabelled), pi_g = row mass of g / row mass of F
(the plan's realised source marginal; cell share under BOT), E_gj = S~_gj / pi_g, defined for subclones
with pi_g >= 0.01 and >= 5 source cells (the candidates). Scored: target cells of F whose label at the
level is a subclone with >= 1 source cell. Rules: enrichment argmax (primary) and raw-mass argmax over
the candidates. References: chance 1 / candidates, size rule (largest pi among candidates).
evaluation_status = not_evaluable (kept, with its reason) when the scored cells hold < 2 true subclones or
there are < 2 candidates; with one true subclone, under BOT, the mean own enrichment is 1 by construction
(PLAN.md, identity). Coverage: coverage_scored = scored / family target cells; mean_family_mass_share =
mean over scored cells of the share of their incoming mass that comes from their own family / clone.
Writes OUT/: run_status.tsv, diagnostics.tsv, summary.tsv, comparison.tsv (A and B side by side, B - A),
comparison_overview.tsv (per level and metric, over evaluable families: medians, B higher / A higher),
per_subclone.tsv, per_site.tsv, confusion.tsv, cells.tsv.gz.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from methods import METHODS  # noqa: E402  (A_standard_OT, B_our_OT)
UNLABELLED, OTHER = "family unlabelled", "other"
MIN_PI, MIN_CELLS, RARE = 0.01, 5, 0.10
COMPARED = ["balanced_accuracy_enrichment", "balanced_accuracy_raw", "accuracy_enrichment", "rare_recall_enrichment", "mean_own_log2E",
            "mean_family_mass_share"]


def score_one(tm: pd.DataFrame, src: pd.DataFrame, labels: pd.Series, subclones: list[str]):
    fam_groups = subclones + [UNLABELLED]
    mass = tm[[f"mass__{g}" for g in fam_groups]].to_numpy(float)
    col = tm["column_mass"].to_numpy(float)
    S = mass / np.where(col > 0, col, np.nan)[:, None]
    fam_total = S.sum(1, keepdims=True)
    St = S / np.where(fam_total > 0, fam_total, np.nan)
    src = src.set_index("group")
    row = src.loc[fam_groups, "row_mass"].to_numpy(float)
    n = src.loc[fam_groups, "cells"].to_numpy(int)
    pi, count_share = row / row.sum(), n / n.sum()
    cand = np.array([(k < len(subclones)) and pi[k] >= MIN_PI and n[k] >= MIN_CELLS for k in range(len(fam_groups))])
    E = St / pi[None, :]
    names = np.array(fam_groups, dtype=object)
    lab = labels.reindex(tm["cell"]).to_numpy(dtype=object)
    in_family = pd.notna(lab)
    has_source = {g: n[k] > 0 for k, g in enumerate(subclones)}
    scored = np.array([isinstance(x, str) and x in has_source and has_source[x] for x in lab]) & np.isfinite(St).all(1)
    unmatched = np.array([isinstance(x, str) and x in has_source and not has_source[x] for x in lab])
    cells = pd.DataFrame({"cell": tm["cell"], "site": tm["site"] if "site" in tm else "", "true": lab, "scored": scored})
    if cand.any():
        cells["pred_enrichment"] = names[cand][np.argmax(np.where(np.isfinite(E[:, cand]), E[:, cand], -np.inf), 1)]
        cells["pred_raw"] = names[cand][np.argmax(np.nan_to_num(St[:, cand], nan=-np.inf), 1)]
    else:
        cells["pred_enrichment"] = cells["pred_raw"] = ""
    gi = {g: k for k, g in enumerate(fam_groups)}
    own = np.array([St[j, gi[t]] if scored[j] else np.nan for j, t in enumerate(lab)])
    own_pi = np.array([pi[gi[t]] if scored[j] else np.nan for j, t in enumerate(lab)])
    cells["own_S"], cells["own_pi"] = own, own_pi
    cells["own_log2E"] = np.log2(np.maximum(own, 1e-300) / own_pi)
    sc = cells[scored]
    size_pred = names[cand][np.argmax(pi[cand])] if cand.any() else ""
    per = []
    for t in subclones:
        rows = sc[sc["true"].eq(t)]
        k = gi[t]
        per.append({"subclone": t, "candidate": bool(cand[k]), "source_cells": int(n[k]), "pi": pi[k], "count_share": count_share[k],
                    "rare": bool(pi[k] < RARE), "target_cells_scored": len(rows),
                    "mean_own_S": rows["own_S"].mean() if len(rows) else np.nan,
                    "E_own": rows["own_S"].mean() / pi[k] if len(rows) and pi[k] > 0 else np.nan,
                    "mean_own_log2E": rows["own_log2E"].mean() if len(rows) else np.nan,
                    "recall_enrichment": rows["pred_enrichment"].eq(t).mean() if len(rows) else np.nan,
                    "recall_raw": rows["pred_raw"].eq(t).mean() if len(rows) else np.nan,
                    "recall_size_rule": float(size_pred == t) if len(rows) else np.nan})
    per = pd.DataFrame(per)
    present = per[per["target_cells_scored"] > 0]
    # Evaluable needs >= 2 true subclones among the scored cells and >= 2 candidates; otherwise the case is kept
    # and reported as not_evaluable (under BOT a one-subclone target has mean own enrichment 1 by construction).
    reason = ("no scored target cell" if not scored.any() else "one true subclone among the scored cells" if len(present) < 2
              else "fewer than 2 candidate subclones" if cand.sum() < 2 else "")
    fam_share = mass.sum(1) / np.where(col > 0, col, np.nan)
    summ = {"evaluation_status": "not_evaluable" if reason else "evaluable", "not_evaluable_reason": reason,
            "target_cells": len(tm), "target_family_cells": int(in_family.sum()), "scored": int(scored.sum()),
            "coverage_scored": scored.sum() / in_family.sum() if in_family.any() else np.nan,
            "mean_family_mass_share": float(np.nanmean(fam_share[scored])) if scored.any() else np.nan,
            "unmatched_observed": int(unmatched.sum()), "true_classes": len(present), "single_true_class": len(present) == 1, "candidates": int(cand.sum()),
            "candidates_list": ";".join(names[cand]),
            "true_not_candidate_cells": int(sc["true"].map(lambda t: not cand[gi[t]]).sum()) if len(sc) else 0,
            "accuracy_enrichment": sc["pred_enrichment"].eq(sc["true"]).mean() if len(sc) else np.nan,
            "accuracy_raw": sc["pred_raw"].eq(sc["true"]).mean() if len(sc) else np.nan,
            "balanced_accuracy_enrichment": present["recall_enrichment"].mean() if len(present) else np.nan,
            "balanced_accuracy_raw": present["recall_raw"].mean() if len(present) else np.nan,
            "balanced_accuracy_size_rule": present["recall_size_rule"].mean() if len(present) else np.nan,
            "chance": 1 / cand.sum() if cand.any() else np.nan,
            "mean_own_log2E": sc["own_log2E"].mean() if len(sc) else np.nan,
            "rare_recall_enrichment": present.loc[present["rare"], "recall_enrichment"].mean() if present["rare"].any() else np.nan}
    conf = pd.crosstab(sc["true"], sc["pred_enrichment"]).stack().rename("cells").reset_index() if len(sc) else pd.DataFrame()
    return summ, per, conf, cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, default=HERE / "inputs")
    args = parser.parse_args()
    pairs = pd.read_csv(args.inputs / "cases.tsv", sep="\t", dtype=str)
    fam_cells = pd.read_csv(args.inputs / "family_cells.tsv.gz", sep="\t", dtype=str, keep_default_na=False)
    subs = pd.read_csv(args.inputs / "subclones.tsv", sep="\t", dtype={"level": int})
    status, diag, summ, per, conf, cells, sites = [], [], [], [], [], [], []
    for _, pair in pairs.iterrows():
        root = args.runs / pair["case_id"]
        base = {"index": int(pair["index"]), "case_id": pair["case_id"], "families": pair["families"]}
        if not (root / "SUCCESS").exists():
            failed = root / "FAILED.json"
            reason = json.loads(failed.read_text(encoding="utf-8"))["error"] if failed.exists() else \
                "no SUCCESS and no FAILED.json (not run, still running, or killed by SLURM: see the task log)"
            for method in METHODS:
                status.append({**base, "method": method, "status": "failed" if failed.exists() else "missing", "reason": reason})
            print(f"NOT DONE {pair['case_id']}: {reason}")
            continue
        for method in METHODS:
            arm_dir = root / method
            if not (arm_dir / "diagnostics.json").exists():
                status.append({**base, "method": method, "status": "missing", "reason": "SUCCESS but no diagnostics.json"})
                print(f"MISSING {pair['case_id']} {method}")
                continue
            d = json.loads((arm_dir / "diagnostics.json").read_text(encoding="utf-8"))
            status.append({**base, "method": method, "status": "ok", "reason": "", "converged": d.get("converged")})
            diag.append({"case_id": pair["case_id"], **{k: v for k, v in d.items() if k != "warnings"},
                         "n_warnings": len(d.get("warnings", []))})
            src_all = pd.read_csv(arm_dir / "source_groups.tsv", sep="\t", dtype={"level": int})
            for fam in pair["families"].split(";"):
                table = fam_cells[fam_cells["name"].eq(fam)].set_index("cell_id")
                for level in (1, 2):
                    allowed = sorted(subs.loc[subs["name"].eq(fam) & subs["level"].eq(level), "subclone"])
                    if not allowed:
                        continue
                    tm = pd.read_csv(arm_dir / f"target_mass_{fam}_L{level}.tsv.gz", sep="\t")
                    src = src_all[src_all["family"].eq(fam) & src_all["level"].eq(level)]
                    labels = table[f"L{level}"].replace("", "ambiguous")
                    s, p, c, cl = score_one(tm, src, labels, allowed)
                    key = {"case_id": pair["case_id"], "method": method, "family": fam, "level": level}
                    summ.append({**key, **s})
                    per.append(p.assign(**key))
                    conf.append(c.assign(**key))
                    cells.append(cl[cl["true"].notna()].assign(**key))
                    for (site, true), g in cl[cl["scored"]].groupby(["site", "true"]):
                        sites.append({**key, "site": site, "true": true, "cells": len(g), "mean_own_log2E": g["own_log2E"].mean(),
                                      "recall_enrichment": g["pred_enrichment"].eq(true).mean(), "recall_raw": g["pred_raw"].eq(true).mean()})
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(status).to_csv(args.out / "run_status.tsv", sep="\t", index=False)
    if not summ:
        print("no completed case to score; see run_status.tsv")
        return
    pd.DataFrame(diag).to_csv(args.out / "diagnostics.tsv", sep="\t", index=False)
    pd.DataFrame(summ).to_csv(args.out / "summary.tsv", sep="\t", index=False)
    pd.concat(per).to_csv(args.out / "per_subclone.tsv", sep="\t", index=False)
    pd.concat(conf).to_csv(args.out / "confusion.tsv", sep="\t", index=False)
    pd.DataFrame(sites).to_csv(args.out / "per_site.tsv", sep="\t", index=False)
    pd.concat(cells).to_csv(args.out / "cells.tsv.gz", sep="\t", index=False, compression="gzip")
    # A vs B side by side, one row per case, family and level (both methods score the same cells).
    s = pd.DataFrame(summ)
    keys = ["case_id", "family", "level"]
    wide = s.set_index(keys + ["method"])[COMPARED].unstack("method")  # keeps all-NaN columns (no rare subclone)
    wide.columns = [f"{v}__{m}" for v, m in wide.columns]
    for v in COMPARED:
        if all(f"{v}__{m}" in wide for m in METHODS):
            wide[f"{v}__B_minus_A"] = wide[f"{v}__B_our_OT"] - wide[f"{v}__A_standard_OT"]
    flags = s.groupby(keys)[["evaluation_status", "not_evaluable_reason", "scored", "coverage_scored", "true_classes", "chance"]].first()
    wide = flags.join(wide).reset_index()
    wide.to_csv(args.out / "comparison.tsv", sep="\t", index=False)
    # Overview over every family / level with both methods done: evaluable ones counted, the rest listed by
    # status. Descriptive only (PLAN.md: no test statistic).
    over = []
    for level, g in wide.groupby("level"):
        ev = g[g["evaluation_status"].eq("evaluable")]
        for v in COMPARED:
            a, b = f"{v}__A_standard_OT", f"{v}__B_our_OT"
            if a not in ev or b not in ev:
                continue
            both = ev[[a, b]].dropna()
            d = both[b] - both[a]
            over.append({"level": level, "metric": v, "families_total": len(g), "not_evaluable": int((g["evaluation_status"] != "evaluable").sum()),
                         "evaluable_with_value": len(both), "median_A": both[a].median(), "median_B": both[b].median(),
                         "median_B_minus_A": d.median(), "B_higher": int((d > 0).sum()), "A_higher": int((d < 0).sum()), "equal": int((d == 0).sum())})
    pd.DataFrame(over).to_csv(args.out / "comparison_overview.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(s[["case_id", "method", "family", "level", "evaluation_status", "scored", "candidates", "balanced_accuracy_enrichment",
             "balanced_accuracy_raw", "chance", "mean_own_log2E", "rare_recall_enrichment"]].to_string(index=False))
    print(pd.DataFrame(over).to_string(index=False))


if __name__ == "__main__":
    main()
