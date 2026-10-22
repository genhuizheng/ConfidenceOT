"""OT-based subclone classification of metastatic cells (SUBCLONE_RULES.md §5-§6, fixed at a545f1c).

    # on TACC, from the runs as written
    python lineage_validation/within_family_subclones/classify_subclones.py --defs DEFS --out OUT \\
        --runs /scratch/.../lineage_validation/runs/acc0.99 --labels /scratch/.../lineage_validation/labels \\
        --kp-meta /scratch/.../KPTracer-Data/KPTracer_meta.csv [--umap-root /scratch/.../lineage_validation/umap]
    # locally, from the downloaded export (gates + couplings)
    python lineage_validation/within_family_subclones/classify_subclones.py --defs DEFS --out OUT --export EXPORT

DEFS is the output of define_subclones.py. Every eligible family / clone is classified in its fixed run
(KP-Tracer D_B of the family's primary, Quinn D_B), for all five strategies and M4-E / M4-R, at level 1
and level 2. Nothing is re-fitted. Writes OUT/<family>/:
  target_<method>_<strategy>.csv.gz   per metastatic cell of the family: true L1/L2, state, family weight,
                                      within-family weight of every subclone, prediction and outcome
  source_<method>_<strategy>.csv.gz   per source cell: subclone label, state
  summary.tsv                         per method x strategy x level: outcome counts, accuracy, coverage,
                                      size and chance references
  confusion.tsv                       per method x strategy x level: true x predicted counts
  heatmap.tsv                         per method x strategy x level: mean within-family weight
                                      (true subclone x source subclone) over retained metastatic cells
  umap.csv.gz                         the fixed joint UMAP of the run's pair (if available)
and OUT/eligible_runs.tsv (every eligible family, its run and whether it was classified).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import lv_common as lv  # noqa: E402

AMBIGUOUS_MARGIN = 0.05
UNLABELLED, OTHER = "family unlabelled", "other"


def load_export_run(export: Path, strategy: str, block: str, pair: str, method: str) -> dict:
    """lv.load_run on the export layout: gates in ot/gates/, couplings in couplings/."""
    gate_dir = export / "ot" / "gates" / block / pair / strategy
    npz = export / "couplings" / block / pair / strategy / f"coupling_{method.lower().replace('-', '')}.npz"
    if not (gate_dir / "cell_confidence.csv").exists() or not npz.exists():
        raise lv.RunMissing(f"{strategy} {pair}: gate or coupling not in the export")
    gate = lv.read_gate(gate_dir, method)
    src, tgt = gate[gate["side"].eq("source")], gate[gate["side"].eq("target")]
    coupling = np.load(npz)["coupling"]
    if coupling.shape != (len(src), len(tgt)):
        raise ValueError("coupling shape does not match the gate")
    if strategy in lv.STEP1:
        base = lv.read_gate(export / "ot" / "gates" / block / pair / lv.STEP1[strategy], "M4-E")
        bs, bt = base[base["side"].eq("source")], base[base["side"].eq("target")]
    else:
        bs, bt = src, tgt
    s0, t0 = bs["observation_id"].to_numpy(), bt["observation_id"].to_numpy()
    sp = pd.Index(s0).get_indexer(src["observation_id"]); tp = pd.Index(t0).get_indexer(tgt["observation_id"])
    rho = np.zeros(len(s0), bool); rho[sp] = src["retained"].to_numpy()
    tau = np.zeros(len(t0), bool); tau[tp] = tgt["retained"].to_numpy()
    P = np.zeros((len(s0), len(t0))); P[np.ix_(sp, tp)] = coupling
    ms = np.full(len(s0), np.inf); ms[sp] = src["signed_rejection_margin"].to_numpy(float)
    mt = np.full(len(t0), np.inf); mt[tp] = tgt["signed_rejection_margin"].to_numpy(float)
    return {"source": s0, "target": t0, "rho": rho, "tau": tau, "P": P, "source_margin": ms, "target_margin": mt,
            "source_site": bs["site"].to_numpy(), "target_site": bt["site"].to_numpy()}


def run_for(row: pd.Series) -> tuple[str, str]:
    mouse = str(row["mouse"])
    if row["dataset"] == "KPTracer":
        return "kp_D_B", f"{mouse}__D_B__{row['family_or_clone']}__metastases"
    return "quinn_D_B", f"{mouse}__D_B__LL__metastases"


def state_of(kept, margin):
    return np.where(kept, "retained", np.where(np.isinf(margin), "excluded", "rejected"))


def classify(run, src_label, tgt_label, family_groups, level, ref_counts):
    """One level. src_label: per source cell group name; tgt_label: per target cell (family cells only, else None)."""
    P, rho, tau = run["P"], run["rho"], run["tau"]
    groups = list(family_groups) + [UNLABELLED, OTHER]
    gindex = {g: k for k, g in enumerate(groups)}
    gs = np.array([gindex.get(x, gindex[OTHER]) for x in src_label])
    GP = lv.onehot(gs, len(groups), rho.astype(float)) @ P
    D = GP.sum(axis=0)
    fam_rows = [gindex[g] for g in family_groups] + [gindex[UNLABELLED]]
    wF = np.where(D > 0, GP[fam_rows].sum(axis=0) / np.where(D > 0, D, 1), 0.0)
    fam_mass = GP[fam_rows].sum(axis=0)
    Wt = np.zeros((len(fam_rows), P.shape[1]))
    ok = fam_mass > 0
    Wt[:, ok] = GP[fam_rows][:, ok] / fam_mass[ok]
    analysed = {g: int((gs == gindex[g]).sum()) for g in family_groups}
    retained = {g: int(((gs == gindex[g]) & rho).sum()) for g in family_groups}
    out = pd.DataFrame({"cell_id": run["target"], "site": run["target_site"],
                        "state": state_of(tau, run["target_margin"]), "family_weight": wF})
    for k, g in enumerate(family_groups):
        out[f"S__{g}"] = Wt[k]
    out[f"S__{UNLABELLED}"] = Wt[-1]
    order = np.argsort(-Wt, axis=0, kind="stable")
    top, second = order[0], order[1] if len(fam_rows) > 1 else order[0]
    cols = np.arange(P.shape[1])
    top_w, second_w = Wt[top, cols], Wt[second, cols]
    names = np.array(list(family_groups) + [UNLABELLED], dtype=object)
    pred = names[top]
    outcome = []
    for j, t in enumerate(tgt_label):
        if t is None:
            outcome.append("not in family")
        elif t in ("", "ambiguous") or t not in family_groups:
            outcome.append("no CRISPR label")
        elif analysed[t] == 0:
            outcome.append("unmatched_observed")
        elif retained[t] == 0:
            outcome.append("counterpart rejected")
        elif not tau[j] or D[j] <= 0 or fam_mass[j] <= 0:
            outcome.append("abstain")
        elif top_w[j] - second_w[j] < AMBIGUOUS_MARGIN or pred[j] == UNLABELLED:
            outcome.append("ambiguous")
        else:
            outcome.append("correct" if pred[j] == t else "incorrect")
    out["true"] = [t if t is not None else "" for t in tgt_label]
    out["predicted"] = np.where(np.isin(outcome, ["correct", "incorrect", "ambiguous"]), pred, "")
    out["top_weight"] = top_w
    out["outcome"] = outcome
    out["level"] = level
    fam = out[out["outcome"] != "not in family"]
    counts = fam["outcome"].value_counts().to_dict()
    scored = fam[fam["outcome"].isin(["correct", "incorrect"])]
    total_ret = sum(retained.values())
    size_g = max(family_groups, key=lambda g: retained[g]) if family_groups else ""
    labelled_counterpart = fam[~fam["outcome"].isin(["no CRISPR label", "unmatched_observed"])]
    summary = {"level": level, "subclones": len(family_groups),
               **{f"n_{k}": int(counts.get(k, 0)) for k in ("correct", "incorrect", "ambiguous", "abstain", "counterpart rejected",
                                                             "unmatched_observed", "no CRISPR label")},
               "accuracy": float((scored["outcome"] == "correct").mean()) if len(scored) else np.nan,
               "coverage": float(len(scored) / len(labelled_counterpart)) if len(labelled_counterpart) else np.nan,
               "size_reference_subclone": size_g,
               "size_reference_accuracy": float((scored["true"] == size_g).mean()) if len(scored) else np.nan,
               "chance_accuracy": float(np.mean([retained[t] / total_ret for t in scored["true"]])) if len(scored) and total_ret else np.nan,
               "source_cells": json.dumps(analysed), "source_retained": json.dumps(retained)}
    conf = pd.crosstab(fam["true"], fam["predicted"].replace("", "-") + " | " + fam["outcome"]).reset_index()
    heat = []
    ret = fam[fam["state"].eq("retained") & fam["true"].isin(family_groups)]
    for t, part in ret.groupby("true"):
        for g in list(family_groups) + [UNLABELLED]:
            heat.append({"level": level, "true": t, "source": g, "cells": len(part), "mean_weight": float(part[f"S__{g}"].mean())})
    return out, summary, conf, pd.DataFrame(heat)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--defs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--export", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--kp-meta", type=Path)
    parser.add_argument("--umap-root", type=Path)
    parser.add_argument("--only", nargs="*", default=None, help="family names to classify (default: every eligible)")
    args = parser.parse_args()
    if (args.runs is None) == (args.export is None):
        sys.exit("pass --runs (TACC) or --export (local), not both")
    labels_dir = args.labels or (args.export / "labels")
    umap_root = args.umap_root or (args.export / "umap" if args.export else None)
    cells_lab = pd.read_csv(labels_dir / "cells.tsv.gz", sep="\t", dtype=str, keep_default_na=False)
    lineage = {k: g.set_index("cell")["lineage"] for k, g in cells_lab.groupby(["dataset", "mouse"])}
    if args.kp_meta:
        meta = pd.read_csv(args.kp_meta, dtype=str, low_memory=False)
        meta = meta.rename(columns={meta.columns[0]: "cell_id"}) if "cell_id" not in meta else meta
    else:
        meta = pd.concat([pd.read_csv(p, dtype=str) for p in sorted((args.export / "meta").glob("KPTracer_meta_*.csv.gz"))])
    family_of = dict(zip(meta["cell_id"], meta["MetFamily"]))
    feas = pd.read_csv(args.defs / "feasibility.tsv", sep="\t", dtype={"mouse": str})
    feas = feas[feas["subclone_structure_eligible"].astype(str).str.lower().eq("true")]
    args.out.mkdir(parents=True, exist_ok=True)
    subs = pd.read_csv(args.defs / "subclones.tsv", sep="\t", dtype=str)
    families = []
    for _, row in feas.iterrows():
        dataset, mouse, family = row["dataset"], str(row["mouse"]), row["family_or_clone"]
        name = f"{dataset}_{mouse}_{family}"
        if args.only and name not in args.only:
            continue
        block, pair = run_for(row)
        cells = pd.read_csv(args.defs / "cells" / f"{name}.csv.gz", dtype=str, keep_default_na=False)
        mine = subs[subs["name"] == name]
        if dataset == "KPTracer":
            in_family = (lambda family: lambda c: family_of.get(c, "") == family)(family)
            other_src = lambda c: False
        else:
            lin = lineage[("Quinn2021", mouse)]
            in_family = (lambda lin, clone: lambda c: lin.get(c, "") == clone)(lin, family[2:])
            other_src = (lambda lin, clone: lambda c: lin.get(c, "") != clone)(lin, family[2:])
        families.append({"name": name, "block": block, "pair": pair,
                         "l1": dict(zip(cells["cell_id"], cells["L1"])),
                         "l2": dict(zip(cells["cell_id"], cells["L2"].replace("", "ambiguous"))),
                         "g1": mine.loc[mine["level"] == "1", "subclone"].tolist(),
                         "g2": mine.loc[mine["level"] == "2", "subclone"].tolist(),
                         "in_family": in_family, "other_src": other_src,
                         "summaries": [], "confs": [], "heats": [], "reason": "", "classified": False})
    # One load of each run (pair x method x strategy) serves every family that uses it.
    for (block, pair), group in pd.DataFrame([{"block": f["block"], "pair": f["pair"], "k": k} for k, f in enumerate(families)]).groupby(["block", "pair"]):
        members = [families[k] for k in group["k"]]
        for method in lv.METHODS:
            for strategy in lv.STRATEGIES:
                try:
                    run = (lv.load_run(args.runs, strategy, block, pair, method) if args.runs
                           else load_export_run(args.export, strategy, block, pair, method))
                except lv.RunMissing as error:
                    for f in members:
                        f["reason"] = str(error)
                    continue
                tag = f"{method.replace('-', '')}_{strategy}"
                for f in members:
                    l1, l2, g1, g2 = f["l1"], f["l2"], f["g1"], f["g2"]
                    src1 = [OTHER if f["other_src"](c) else (l1.get(c, "") if l1.get(c, "") in g1 else UNLABELLED) for c in run["source"]]
                    src2 = [OTHER if f["other_src"](c) else (l2.get(c, "") if l2.get(c, "") in g2 else UNLABELLED) for c in run["source"]]
                    tgt1 = [l1.get(c, "ambiguous") if f["in_family"](c) else None for c in run["target"]]
                    tgt2 = [l2.get(c, "ambiguous") if f["in_family"](c) else None for c in run["target"]]
                    dest = args.out / f["name"]
                    dest.mkdir(parents=True, exist_ok=True)
                    tables = []
                    for level, src, tgt, groups in ((1, src1, tgt1, g1), (2, src2, tgt2, g2)):
                        if not groups:
                            continue
                        out, summary, conf, heat = classify(run, src, tgt, groups, level, None)
                        tables.append(out[out["outcome"] != "not in family"])
                        f["summaries"].append({"method": method, "strategy": strategy, **summary})
                        f["confs"].append(conf.assign(method=method, strategy=strategy, level=level))
                        f["heats"].append(heat.assign(method=method, strategy=strategy))
                    pd.concat(tables).to_csv(dest / f"target_{tag}.csv.gz", index=False, compression="gzip")
                    keep = [f["in_family"](c) or not f["other_src"](c) for c in run["source"]]
                    pd.DataFrame({"cell_id": run["source"], "site": run["source_site"], "L1": src1, "L2": src2,
                                  "state": state_of(run["rho"], run["source_margin"])})[keep].to_csv(
                        dest / f"source_{tag}.csv.gz", index=False, compression="gzip")
                    f["classified"] = True
                del run
                print(f"{pair} {method} {strategy}: {len(members)} families", flush=True)
    runs_rows = []
    for f in families:
        dest = args.out / f["name"]
        if f["summaries"]:
            pd.DataFrame(f["summaries"]).to_csv(dest / "summary.tsv", sep="\t", index=False)
            pd.concat(f["confs"]).to_csv(dest / "confusion.tsv", sep="\t", index=False)
            pd.concat(f["heats"]).to_csv(dest / "heatmap.tsv", sep="\t", index=False)
            if umap_root is not None:
                for cand in (umap_root / f["block"] / f["pair"] / "reference_umap.csv.gz",
                             umap_root / f["block"] / "pairs" / f["pair"] / "reference_umap.csv.gz"):
                    if cand.exists():
                        shutil.copy2(cand, dest / "umap.csv.gz")
                        break
        runs_rows.append({"name": f["name"], "block": f["block"], "pair": f["pair"], "classified": f["classified"],
                          "runs_classified": len({(r["method"], r["strategy"]) for r in f["summaries"]}),
                          "missing": "" if f["classified"] else f["reason"]})
        print(f"{f['name']}: {'classified' if f['classified'] else 'not classified: ' + f['reason']}", flush=True)
    pd.DataFrame(runs_rows).to_csv(args.out / "eligible_runs.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
