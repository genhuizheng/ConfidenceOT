"""Rejection-coupling mass audit of the within-family subclone analysis, on every coupling available locally.

    python lineage_validation/within_family_subclones/audit/run_audit.py \\
        --export lineage_validation/local_lineage_visualization/data/local_lineage_visualization_export \\
        --defs lineage_validation/within_family_subclones/output --out lineage_validation/within_family_subclones/audit

Read only: loads the saved couplings (byte copies of the TACC files) and gates, re-indexes them by cell id
with its own code (not classify_subclones.py), and writes:
  mass_conservation.tsv            every local run (pair x strategy x method): cell counts, input measure,
                                   transport mass by retained / rejected / excluded rows and columns, the four
                                   gate blocks, effective mass, marginal and normalisation checks, the rank-one
                                   alignment test of the gated block and its control
  subclone_rejection.tsv           per family x run x level x source group: eligible / retained cell counts,
                                   retention rate, raw row mass, effective retained mass M_g, pi_g (over the
                                   whole source) and pi_g^F (within the family), CRISPR lineage-support class
  subclone_mass_decomposition.tsv  the same groups as three compositions: eligible cells, retained cells,
                                   effective transport mass (each sums to 1 over the family)
  ot_vs_independent.tsv            per family x run x level x true target group x source group: retained
                                   target cells, mean S_gj, mean within-family weight, pi_g^F, enrichment
                                   S/pi (undefined below pi 0.01 or 5 retained cells), CRISPR correspondence
  classifier_comparison.tsv        per family x run x level: accuracy on the same cells of raw-mass argmax,
                                   independent-coupling argmax, retained-count (size) argmax, enrichment argmax
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))
import audit_core as ac  # noqa: E402
import lv_common as lv  # noqa: E402

UNLAB = "family unlabelled"
OTHER = "other"
RNG = np.random.default_rng(20261022)


def gate(export, block, pair, strategy, method):
    path = export / "ot" / "gates" / block / pair / strategy / "cell_confidence.csv"
    g = pd.read_csv(path, usecols=["method", "side", "observation_id", "retained", "signed_rejection_margin", "site"], dtype={"observation_id": str})
    g = g[g["method"] == method]
    g["retained"] = g["retained"].astype(str).str.lower().eq("true")
    return g[g["side"] == "source"].reset_index(drop=True), g[g["side"] == "target"].reset_index(drop=True)


def load(export, block, pair, strategy, method):
    """The saved coupling indexed by the cell ids of the step-1 (or own) gate; rho/tau = final gates, excluded flags."""
    src, tgt = gate(export, block, pair, strategy, method)
    raw = np.load(export / "couplings" / block / pair / strategy / f"coupling_{method.lower().replace('-', '')}.npz")["coupling"]
    if raw.shape != (len(src), len(tgt)):
        raise ValueError(f"{pair} {strategy} {method}: coupling {raw.shape} vs gate {len(src)} x {len(tgt)}")
    ac.check_unique(src["observation_id"], "source gate"); ac.check_unique(tgt["observation_id"], "target gate")
    if strategy in lv.STEP1:
        bs, bt = gate(export, block, pair, lv.STEP1[strategy], "M4-E")
    else:
        bs, bt = src, tgt
    s_index = pd.Series(np.arange(len(bs)), index=bs["observation_id"])
    t_index = pd.Series(np.arange(len(bt)), index=bt["observation_id"])
    rows = s_index.reindex(src["observation_id"]).to_numpy()
    cols = t_index.reindex(tgt["observation_id"]).to_numpy()
    if np.isnan(rows.astype(float)).any() or np.isnan(cols.astype(float)).any():
        raise ValueError("step-2 cell not in step 1")
    P = np.zeros((len(bs), len(bt)))
    P[np.ix_(rows.astype(int), cols.astype(int))] = raw
    rho = np.zeros(len(bs), bool); rho[rows.astype(int)] = src["retained"].to_numpy()
    tau = np.zeros(len(bt), bool); tau[cols.astype(int)] = tgt["retained"].to_numpy()
    ex_s = np.ones(len(bs), bool); ex_s[rows.astype(int)] = False
    ex_t = np.ones(len(bt), bool); ex_t[cols.astype(int)] = False
    return {"P": P, "raw": raw, "rho": rho, "tau": tau, "ex_s": ex_s, "ex_t": ex_t, "src_gate": src, "tgt_gate": tgt,
            "source": bs["observation_id"].to_numpy(object), "target": bt["observation_id"].to_numpy(object),
            "source_site": bs["site"].astype(str).to_numpy(object), "target_site": bt["site"].astype(str).to_numpy(object)}


def mass_row(run, ident):
    P, rho, tau, ex_s, ex_t, raw = run["P"], run["rho"], run["tau"], run["ex_s"], run["ex_t"], run["raw"]
    rej_s = ~rho & ~ex_s; rej_t = ~tau & ~ex_t
    m = P.sum()
    Pt = ac.effective(P, rho, tau)
    q, ok = ac.matching(Pt, tau)
    colsum = q.sum(axis=0)
    src, tgt = run["src_gate"], run["tgt_gate"]
    rs, rt = src["retained"].to_numpy(), tgt["retained"].to_numpy()
    res = {}
    for label, mask, axis in (("rejected_target_columns", ~rt, 1), ("rejected_source_rows", ~rs, 0)):
        k = int(mask.sum())
        if k >= 2 and (~mask).sum() >= k:
            pick = RNG.choice(np.flatnonzero(~mask), size=k, replace=False)
            res[f"rank1_{label}"] = ac.rank_one_residual(raw[:, mask] if axis else raw[mask])
            res[f"rank1_control_{label}"] = ac.rank_one_residual(raw[:, pick] if axis else raw[pick])
    return {**ident,
            "n_source": len(rho), "n_source_retained": int(rho.sum()), "n_source_rejected": int(rej_s.sum()), "n_source_excluded": int(ex_s.sum()),
            "n_target": len(tau), "n_target_retained": int(tau.sum()), "n_target_rejected": int(rej_t.sum()), "n_target_excluded": int(ex_t.sum()),
            "input_source_measure": 1.0, "input_target_measure": 1.0,
            "coupling_mass": m,
            "mass_rows_retained": P[rho].sum(), "mass_rows_rejected": P[rej_s].sum(), "mass_rows_excluded": P[ex_s].sum(),
            "mass_cols_retained": P[:, tau].sum(), "mass_cols_rejected": P[:, rej_t].sum(), "mass_cols_excluded": P[:, ex_t].sum(),
            "block_retained_retained": P[np.ix_(rho, tau)].sum(), "block_retained_gated": P[np.ix_(rho, ~tau)].sum(),
            "block_gated_retained": P[np.ix_(~rho, tau)].sum(), "block_gated_gated": P[np.ix_(~rho, ~tau)].sum(),
            "blocks_sum_minus_total": P[np.ix_(rho, tau)].sum() + P[np.ix_(rho, ~tau)].sum() + P[np.ix_(~rho, tau)].sum()
            + P[np.ix_(~rho, ~tau)].sum() - m,
            "effective_mass": Pt.sum(), "effective_row_marginal_sum": Pt.sum(axis=1).sum(), "effective_col_marginal_sum": Pt.sum(axis=0).sum(),
            "retained_targets_with_effective_mass": int(ok.sum()), "retained_targets_zero_effective_mass": int((tau & ~ok).sum()),
            "max_abs_q_colsum_minus_1_retained": float(np.abs(colsum[ok] - 1).max()) if ok.any() else np.nan,
            "max_q_rejected_or_excluded_targets": float(q[:, ~tau].max()) if (~tau).any() else 0.0,
            **res}


def support_class(sub_row):
    met = int(sub_row["metastatic"])
    return "lineage-supported (>= 5 metastatic cells)" if met >= 5 else ("no observed metastatic counterpart" if met == 0 else "insufficient (1-4 metastatic cells)")


def group_tables(run, ident, level, src_groups, tgt_true, groups, family_groups, support, sizes_note):
    """src_groups: per source cell its group (one of groups); tgt_true: per target its true group or None."""
    P, rho, tau = run["P"], run["rho"], run["tau"]
    Pt = ac.effective(P, rho, tau)
    q, ok = ac.matching(Pt, tau)
    S = ac.group_sum(q, src_groups, groups)                 # groups x targets
    row_raw = ac.group_sum(P.sum(axis=1)[:, None], src_groups, groups)[:, 0]
    M = ac.group_sum(Pt.sum(axis=1)[:, None], src_groups, groups)[:, 0]
    n_all = ac.group_sum(np.ones((len(rho), 1)), src_groups, groups)[:, 0]
    n_ret = ac.group_sum(rho[:, None].astype(float), src_groups, groups)[:, 0]
    fam = np.array([g in family_groups for g in groups])
    rej_rows = []
    for k, g in enumerate(groups):
        if not fam[k]:
            continue
        rej_rows.append({**ident, "level": level, "group": g, "support": support.get(g, ""),
                         "eligible_cells": int(n_all[k]), "retained_cells": int(n_ret[k]),
                         "retention_rate": n_ret[k] / n_all[k] if n_all[k] else np.nan,
                         "raw_row_mass": row_raw[k], "effective_mass_M": M[k],
                         "pi_whole_source": M[k] / M.sum() if M.sum() > 0 else np.nan,
                         "pi_within_family": M[k] / M[fam].sum() if M[fam].sum() > 0 else np.nan})
    dec_rows = []
    for k, g in enumerate(groups):
        if fam[k]:
            for comp, v, tot in (("eligible cells", n_all[k], n_all[fam].sum()), ("retained cells", n_ret[k], n_ret[fam].sum()),
                                 ("effective transport mass", M[k], M[fam].sum())):
                dec_rows.append({**ident, "level": level, "group": g, "support": support.get(g, ""), "composition": comp,
                                 "value": v, "share": v / tot if tot > 0 else np.nan})
    # targets
    fam_mass = S[fam].sum(axis=0)
    St = np.zeros((fam.sum(), S.shape[1]))
    has = ok & (fam_mass > 0)
    St[:, has] = S[fam][:, has] / fam_mass[has]
    piF = M[fam] / M[fam].sum() if M[fam].sum() > 0 else np.full(fam.sum(), np.nan)
    fam_names = [g for g in groups if g in family_groups]
    labelled = [g for g in fam_names if g != UNLAB]
    lab_idx = [fam_names.index(g) for g in labelled]
    E = ac.enrichment(St[lab_idx].T, piF[lab_idx], retained_cells=n_ret[[groups.index(g) for g in labelled]]).T  # labelled x targets
    vs_rows, cmp_cells = [], []
    true = np.array([t if t is not None else "" for t in tgt_true], dtype=object)
    scored = has & np.isin(true, labelled) & np.array([n_ret[groups.index(t)] > 0 if t in labelled else False for t in true])
    for t in labelled:
        cols = np.flatnonzero(has & (true == t))
        if not len(cols):
            continue
        for k, g in enumerate(fam_names):
            gi = groups.index(g)
            mean_S = float(S[gi, cols].mean())
            mean_St = float(St[k, cols].mean())
            e = mean_St / piF[k] if (g in labelled and piF[k] >= 0.01 and n_ret[gi] >= 5) else np.nan
            vs_rows.append({**ident, "level": level, "true_group": t, "source_group": g, "crispr_correspondence": int(t == g),
                            "retained_target_cells": len(cols), "mean_S_whole_source": mean_S, "mean_S_within_family": mean_St,
                            "pi_within_family": piF[k], "independent_expectation": piF[k], "enrichment": e})
    # four classifiers over the family's labelled subclones with >= 1 retained source cell, on the same cells
    cand = [g for g in labelled if n_ret[groups.index(g)] > 0]
    result = {**ident, "level": level, "candidates": len(cand), "scored_target_cells": int(scored.sum())}
    if len(cand) >= 2 and scored.any():
        ci = [labelled.index(g) for g in cand]
        Sc = St[[lab_idx[i] for i in ci]][:, scored]
        t_sc = true[scored]
        raw_pred = np.array(cand, dtype=object)[Sc.argmax(axis=0)]
        indep_pred = cand[int(np.argmax(piF[[lab_idx[i] for i in ci]]))]
        size_pred = cand[int(np.argmax([n_ret[groups.index(g)] for g in cand]))]
        Ec = E[ci][:, scored]
        valid_e = np.isfinite(Ec).any(axis=0)
        e_pred = np.array(cand, dtype=object)[np.nanargmax(np.where(np.isfinite(Ec), Ec, -np.inf), axis=0)]
        result.update(raw_mass_argmax=float((raw_pred == t_sc).mean()), independent_argmax=float((indep_pred == t_sc).mean()),
                      independent_prediction=indep_pred, size_argmax=float((size_pred == t_sc).mean()), size_prediction=size_pred,
                      enrichment_argmax=float((e_pred[valid_e] == t_sc[valid_e]).mean()) if valid_e.any() else np.nan,
                      enrichment_cells=int(valid_e.sum()),
                      raw_equals_independent=float((raw_pred == indep_pred).mean()),
                      per_true_group=json.dumps({t: {"cells": int((t_sc == t).sum()), "raw_correct": int(((raw_pred == t) & (t_sc == t)).sum()),
                                                     "enrichment_correct": int(((e_pred == t) & (t_sc == t) & valid_e).sum())}
                                                 for t in cand}))
    return rej_rows, dec_rows, vs_rows, [result]


def families(defs, export):
    feas = pd.read_csv(defs / "feasibility.tsv", sep="\t", dtype={"mouse": str})
    subs = pd.read_csv(defs / "subclones.tsv", sep="\t", dtype={"mouse": str})
    out = []
    for _, r in feas.iterrows():
        name = f"{r['dataset']}_{r['mouse']}_{r['family_or_clone']}"
        path = defs / "cells" / f"{name}.csv.gz"
        if not path.exists() or subs[subs["name"] == name].empty:
            continue
        if r["dataset"] == "KPTracer" and r["mouse"] in ("3508", "3515"):
            runs = [("kp_D_B", f"{r['mouse']}__D_B__{r['family_or_clone']}__metastases"),
                    ("kp_D_C3", f"{r['mouse']}__D_C3__primaries__metastases")]
        elif r["dataset"] == "Quinn2021" and r["mouse"] == "10k":
            runs = [("quinn_D_B", "10k__D_B__LL__metastases")]
        else:
            continue
        runs = [(b, p) for b, p in runs if (export / "couplings" / b / p).exists()]
        if runs:
            out.append((name, r, runs))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--defs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    export, defs, out = args.export, args.defs, args.out
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.concat([pd.read_csv(p, dtype=str) for p in sorted((export / "meta").glob("KPTracer_meta_*.csv.gz"))])
    family_of = dict(zip(meta["cell_id"], meta["MetFamily"]))
    lab = pd.read_csv(export / "labels" / "cells.tsv.gz", sep="\t", dtype=str, keep_default_na=False)
    clone_of = dict(zip(lab.loc[lab["mouse"] == "10k", "cell"], lab.loc[lab["mouse"] == "10k", "lineage"]))
    subs = pd.read_csv(defs / "subclones.tsv", sep="\t", dtype={"mouse": str})

    # 1. every local run
    mass = []
    for npz in sorted((export / "couplings").rglob("coupling_m4e.npz")):
        strategy, pair, block = npz.parent.name, npz.parent.parent.name, npz.parent.parent.parent.name
        for method in lv.METHODS:
            run = load(export, block, pair, strategy, method)
            mass.append(mass_row(run, {"block": block, "pair": pair, "strategy": strategy, "method": method}))
    pd.DataFrame(mass).to_csv(out / "mass_conservation.tsv", sep="\t", index=False)
    print(f"mass_conservation: {len(mass)} runs", flush=True)

    # 2. families
    rej, dec, vs, cmp_ = [], [], [], []
    for name, r, runs in families(defs, export):
        cells = pd.read_csv(defs / "cells" / f"{name}.csv.gz", dtype=str, keep_default_na=False)
        ac.check_unique(cells["cell_id"], f"{name} cells table")
        mine = subs[subs["name"] == name]
        sup = {row["subclone"]: support_class(row) for _, row in mine.iterrows()}
        L = {lvl: dict(zip(cells["cell_id"], cells[col].replace("", "ambiguous"))) for lvl, col in (("L1", "L1"), ("L2", "L2"))}
        gl = {lvl: mine.loc[mine["level"] == (1 if lvl == "L1" else 2), "subclone"].tolist() for lvl in ("L1", "L2")}
        fam_label = r["family_or_clone"]
        for block, pair in runs:
            for strategy in lv.STRATEGIES:
                for method in lv.METHODS:
                    try:
                        run = load(export, block, pair, strategy, method)
                    except FileNotFoundError:
                        continue
                    ident = {"family": name, "block": block, "pair": pair, "strategy": strategy, "method": method}
                    if r["dataset"] == "KPTracer":
                        in_fam_src = np.array([family_of.get(c, "") == fam_label for c in run["source"]])
                        in_fam_tgt = np.array([family_of.get(c, "") == fam_label for c in run["target"]])
                        # level "family": the source primary tumours of the run
                        site_groups = sorted(set(run["source_site"]))
                        fam_site = fam_label  # a family is named after its primary tumour
                        if len(site_groups) >= 2:
                            t_site = [fam_site if in_fam_tgt[j] else None for j in range(len(run["target"]))]
                            a, b_, c_, d_ = group_tables(run, ident, "family (primary site)", run["source_site"], t_site, site_groups,
                                                         site_groups, {s: ("lineage-supported" if s == fam_site else "other family") for s in site_groups}, "")
                            rej += a; dec += b_; vs += c_; cmp_ += d_
                    else:
                        in_fam_src = np.array([clone_of.get(c, "") == fam_label[2:] for c in run["source"]])
                        in_fam_tgt = np.array([clone_of.get(c, "") == fam_label[2:] for c in run["target"]])
                        clones = sorted(set(clone_of.get(c, "") for c in run["source"]))
                        t_clone = [clone_of.get(c, "") if clone_of.get(c, "") in clones else None for c in run["target"]]
                        if fam_label == "LG2":  # the clone level is the same for every clone of the mouse; written once
                            a, b_, c_, d_ = group_tables(run, {**ident, "family": "Quinn2021_10k (all clones)"}, "family (clone)",
                                                         [clone_of.get(c, "") for c in run["source"]], t_clone, clones, clones, {}, "")
                            rej += a; dec += b_; vs += c_; cmp_ += d_
                    for lvl in ("L1", "L2"):
                        groups_l = gl[lvl]
                        if not groups_l:
                            continue
                        src_g = [(L[lvl].get(c, "") if L[lvl].get(c, "") in groups_l else UNLAB) if in_fam_src[i] else OTHER
                                 for i, c in enumerate(run["source"])]
                        tgt_t = [(L[lvl].get(c, "ambiguous")) if in_fam_tgt[j] else None for j, c in enumerate(run["target"])]
                        groups_all = groups_l + [UNLAB, OTHER]
                        a, b_, c_, d_ = group_tables(run, ident, f"subclone {lvl}", src_g, tgt_t, groups_all, set(groups_l + [UNLAB]),
                                                     {**sup, UNLAB: "ambiguous / unlabelled"}, "")
                        rej += a; dec += b_; vs += c_; cmp_ += d_
        print(f"{name}: audited on {len(runs)} pair(s)", flush=True)
    pd.DataFrame(rej).to_csv(out / "subclone_rejection.tsv", sep="\t", index=False)
    pd.DataFrame(dec).to_csv(out / "subclone_mass_decomposition.tsv", sep="\t", index=False)
    pd.DataFrame(vs).to_csv(out / "ot_vs_independent.tsv", sep="\t", index=False)
    pd.DataFrame(cmp_).to_csv(out / "classifier_comparison.tsv", sep="\t", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
