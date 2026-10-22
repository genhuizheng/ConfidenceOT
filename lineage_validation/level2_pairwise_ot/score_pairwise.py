"""Score Level 2 pair-wise runs against the CRISPR subclones (PLAN.md, Metrics).

    python lineage_validation/level2_pairwise_ot/score_pairwise.py --runs RUNS_ROOT --out OUT

RUNS_ROOT holds run_pairwise.py's <pair_id>/ folders (the small files suffice: diagnostics.json,
source_groups.tsv, target_mass_*.tsv.gz). Target labels come from inputs/family_cells.tsv.gz.

For target cell j and source group g of family F: S_gj = mass_gj / column mass_j (= sum over i in g of
q_ij), S~_gj = S_gj / sum over F (subclones + family unlabelled), pi_g = row mass of g / row mass of F
(the plan's realised source marginal; cell share under BOT), E_gj = S~_gj / pi_g, defined for subclones
with pi_g >= 0.01 and >= 5 source cells (the candidates). Scored: target cells of F whose label at the
level is a subclone with >= 1 source cell. Rules: enrichment argmax (primary) and raw-mass argmax over
the candidates. References: chance 1 / candidates, size rule (largest pi among candidates).
Writes OUT/: diagnostics.tsv, summary.tsv, per_subclone.tsv, confusion.tsv, cells.tsv.gz.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
UNLABELLED, OTHER = "family unlabelled", "other"
MIN_PI, MIN_CELLS, RARE = 0.01, 5, 0.10


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
    cells = pd.DataFrame({"cell": tm["cell"], "true": lab, "scored": scored})
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
    summ = {"target_cells": len(tm), "target_family_cells": int(in_family.sum()), "scored": int(scored.sum()),
            "unmatched_observed": int(unmatched.sum()), "true_classes": len(present), "candidates": int(cand.sum()),
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
    pairs = pd.read_csv(args.inputs / "pilot_pairs.tsv", sep="\t", dtype=str)
    fam_cells = pd.read_csv(args.inputs / "family_cells.tsv.gz", sep="\t", dtype=str, keep_default_na=False)
    subs = pd.read_csv(args.inputs / "subclones.tsv", sep="\t", dtype={"level": int})
    diag, summ, per, conf, cells = [], [], [], [], []
    for _, pair in pairs.iterrows():
        root = args.runs / pair["pair_id"]
        if not (root / "SUCCESS").exists():
            print(f"MISSING {pair['pair_id']}")
            continue
        for arm_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            arm = arm_dir.name
            d = json.loads((arm_dir / "diagnostics.json").read_text(encoding="utf-8"))
            diag.append({"pair_id": pair["pair_id"], **{k: v for k, v in d.items() if k != "warnings"},
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
                    key = {"pair_id": pair["pair_id"], "arm": arm, "family": fam, "level": level}
                    summ.append({**key, **s})
                    per.append(p.assign(**key))
                    conf.append(c.assign(**key))
                    cells.append(cl[cl["true"].notna()].assign(**key))
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(diag).to_csv(args.out / "diagnostics.tsv", sep="\t", index=False)
    pd.DataFrame(summ).to_csv(args.out / "summary.tsv", sep="\t", index=False)
    pd.concat(per).to_csv(args.out / "per_subclone.tsv", sep="\t", index=False)
    pd.concat(conf).to_csv(args.out / "confusion.tsv", sep="\t", index=False)
    pd.concat(cells).to_csv(args.out / "cells.tsv.gz", sep="\t", index=False, compression="gzip")
    pd.set_option("display.width", 250)
    print(pd.DataFrame(summ)[["pair_id", "arm", "family", "level", "scored", "candidates", "balanced_accuracy_enrichment",
                              "balanced_accuracy_raw", "chance", "mean_own_log2E", "rare_recall_enrichment"]].to_string(index=False))


if __name__ == "__main__":
    main()
