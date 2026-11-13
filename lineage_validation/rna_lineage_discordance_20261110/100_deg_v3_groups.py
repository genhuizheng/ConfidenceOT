"""PLAN_v3_deg.md: contrast groups and QC for the OT-preference (A1 / A2), state-within-clade (B1) and
clade-within-state (C1) DEG, with the RNA-proximity baselines and random-split nulls. No expression is read here.

    python lineage_validation/rna_lineage_discordance_20261110/100_deg_v3_groups.py --root results/rna_lineage_discordance_20261110 \
        --bundles results/crispr_matrix_benchmark_20261108/bundles

Writes ROOT/deg_v3/:
  groups.tsv      contrast_id, unit_id (= the counts file key KPTracer_<mouse>), cell, group (case / reference); the
                  input of 60_deg_gsea.py
  contrasts.tsv   one row per contrast: type (A1_rna_vs_crispr, A1_rna_vs_both, A1_crispr_vs_both, A2_cross_vs_own,
                  A1_baseline, A2_baseline, B1_state, C1_clade, *_rand<k>), mouse, unit, clade, site, groups' sizes and
                  QC (median depth / detected genes per group and their ratio, informative edits, missing share, clone
                  size, state composition, gate fractions of the unit, mean q_R / q_C or s_R / s_C per group)
  cells_A.tsv.gz  per A1 / A2 cell: scores (q_R, q_C, d or s_R, s_C) and group
"""

from __future__ import annotations

import argparse
import importlib.util
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MIN_G, NRAND = 20, 5
CONFIG = "ranknm256-midrank|none|scale|cos|pca50"
COORD = "ranknm256-midrank__none__scale__cos__pca50"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def comp(states):
    vc = pd.Series(states).value_counts(normalize=True)
    return "; ".join(f"{k} {v:.2f}" for k, v in vc.head(3).items())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--bundles", type=Path, required=True)
    a = p.parse_args()
    out = a.root / "deg_v3"; out.mkdir(exist_ok=True)
    AC = pd.read_csv(a.root / "state_lineage_ext" / "all_cells__original" / "cells.tsv.gz", sep="\t", dtype={"cell": str, "mouse": str}).set_index("cell")
    gates = pd.read_csv(a.root / "flow" / "gates.tsv", sep="\t").set_index(["unit_id", "side"])
    A = a.root / "inputs_A"
    G, CT, CA = [], [], []

    def add(cid, mouse, case, ref, meta):
        key = f"KPTracer_{mouse}"
        G.append(pd.DataFrame({"contrast_id": cid, "unit_id": key, "cell": np.r_[case, ref], "group": np.r_[np.repeat("case", len(case)), np.repeat("reference", len(ref))]}))
        rec = {"contrast_id": cid, "mouse": mouse, "case_n": len(case), "reference_n": len(ref), **meta}
        for nm, ids in (("case", case), ("reference", ref)):
            h = AC.reindex(ids)
            rec.update({f"{nm}_depth": h.depth.median(), f"{nm}_detected": h.detected_genes.median(), f"{nm}_events": h.events.median(),
                        f"{nm}_missing": h.missing_share.median(), f"{nm}_states": comp(h.state.dropna())})
        rec["depth_ratio"] = rec["case_depth"] / rec["reference_depth"] if rec["reference_depth"] else np.nan
        CT.append(rec)

    def add_with_rand(cid, mouse, case, ref, meta, rng_seed):
        add(cid, mouse, case, ref, meta)
        pool = np.r_[case, ref]
        for k in range(NRAND):
            rng = np.random.default_rng(rng_seed + k)
            perm = rng.permutation(pool)
            add(f"{cid}__rand{k + 1}", mouse, perm[:len(case)], perm[len(case):], {**meta, "type": meta["type"] + f"_rand{k + 1}", "parent": cid})

    seed = 20261112
    # ---------- A: OT preference ----------
    for f in sorted((a.bundles).glob("*.pkl")):
        b = pickle.load(open(f, "rb")); u = b["unit_id"]
        if b["dataset"] != "KPTracer":
            continue
        crd = a.root / "crispr_ot" / u
        row = A / "v3_grid_20261104" / "kp" / "rows" / u / (CONFIG.split("|")[0] + "__none") / CONFIG.replace("|", "__")
        if not (crd / "audit.json").exists() or not (row / "coupling_m4e.npz").exists():
            continue
        mouse = b["mouse"]
        sid, tid = np.asarray(b["source_ids"]).astype(str), np.asarray(b["target_ids"]).astype(str)
        z = np.load(row / "coupling_m4e.npz")
        P = z["coupling"][np.ix_(pd.Index(z["source_ids"].astype(str)).get_indexer(sid), pd.Index(z["target_ids"].astype(str)).get_indexer(tid))].astype(np.float64)
        Pc = np.load(crd / "coupling_m4e.npz")["coupling"].astype(np.float64)
        cc = pd.read_csv(row / "cell_confidence.csv.gz", usecols=["method", "side", "observation_id", "rejected"])
        cc = cc[cc.method == "M4-E"]; cc["observation_id"] = cc.observation_id.astype(str)
        gt = pd.read_csv(crd / "gates.tsv.gz", sep="\t", dtype={"observation_id": str}); gt = gt[gt.method == "M4-E"]
        keep = {}
        for side, ids in (("source", sid), ("target", tid)):
            r = ~cc[cc.side == side].set_index("observation_id").rejected.reindex(ids).to_numpy(bool)
            c = gt[gt.side == side].set_index("observation_id").retained.reindex(ids).to_numpy(bool)
            keep[side] = r & c
        S, T = np.flatnonzero(keep["source"]), np.flatnonzero(keep["target"])
        if len(S) == 0 or len(T) == 0:
            continue
        BR, BC = P[np.ix_(S, T)], Pc[np.ix_(S, T)]; BR, BC = BR / BR.sum(), BC / BC.sum()
        cs = AC.clade.reindex(sid[S]).fillna("unassigned").to_numpy(); ct = AC.clade.reindex(tid[T]).fillna("unassigned").to_numpy()
        tsite = np.asarray(b["target_site"]).astype(str)[T]
        zc = np.load(A / "rna_lineage_discordance_20261110" / "rna_coords" / u / f"{COORD}.npz", allow_pickle=True)
        X = zc["X"][pd.Index(zc["source_ids"].astype(str)).get_indexer(sid[S])]; Y = zc["Y"][pd.Index(zc["target_ids"].astype(str)).get_indexer(tid[T])]
        X = X / np.linalg.norm(X, axis=1, keepdims=True); Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
        g_src = gates.loc[(u, "source")]; g_tgt = gates.loc[(u, "target")]
        gmeta = {"gate_src_both": g_src["both retained"], "gate_tgt_both": g_tgt["both retained"]}
        for cl in sorted(set(cs) & set(ct)):
            if cl in ("minor", "unassigned"):
                continue
            si, ti = np.flatnonzero(cs == cl), np.flatnonzero(ct == cl)
            # A1 source side
            if len(si) >= 4 * MIN_G and len(ti) > 0:
                mR, mC = BR[np.ix_(si, ti)].sum(1), BC[np.ix_(si, ti)].sum(1)
                if mR.sum() > 0 and mC.sum() > 0:
                    qR, qC = mR / mR.sum(), mC / mC.sum(); n = len(si); d = n * (qR - qC)
                    lo, hi = np.quantile(d, [0.25, 0.75])
                    grp = np.where(d >= hi, "RNA-preferred", np.where(d <= lo, "CRISPR-preferred",
                                                                       np.where((qR >= 1 / n) & (qC >= 1 / n), "both-supported", "neither")))
                    ids = sid[S][si]
                    CA.append(pd.DataFrame({"cell": ids, "unit_id": u, "kind": "A1", "clade": cl, "q_rna": qR, "q_crispr": qC, "d": d, "group": grp}))
                    meta = {"unit": u, "clade": cl, "site": u.split("__")[2], "side": "source", "clone_size": n,
                            "block_targets": len(ti), "mean_qR_x_n_case": None, **gmeta}
                    pairs = (("A1_rna_vs_crispr", "RNA-preferred", "CRISPR-preferred"), ("A1_rna_vs_both", "RNA-preferred", "both-supported"),
                             ("A1_crispr_vs_both", "CRISPR-preferred", "both-supported"))
                    for typ, g1, g2 in pairs:
                        c1, c2 = ids[grp == g1], ids[grp == g2]
                        if len(c1) >= MIN_G and len(c2) >= MIN_G:
                            m_ = {**meta, "type": typ, "case_group": g1, "reference_group": g2,
                                  "case_qR_n": float(n * qR[grp == g1].mean()), "case_qC_n": float(n * qC[grp == g1].mean()),
                                  "ref_qR_n": float(n * qR[grp == g2].mean()), "ref_qC_n": float(n * qC[grp == g2].mean())}
                            cid = f"{u}__{cl.split('|')[-1]}__{typ}"
                            if typ == "A1_rna_vs_crispr":
                                add_with_rand(cid, mouse, c1, c2, m_, seed)
                            else:
                                add(cid, mouse, c1, c2, m_)
                    # baseline: RNA proximity to the block's targets
                    prox = X[si] @ Y[ti].mean(0)
                    lo, hi = np.quantile(prox, [0.25, 0.75])
                    c1, c2 = ids[prox >= hi], ids[prox <= lo]
                    if len(c1) >= MIN_G and len(c2) >= MIN_G:
                        add(f"{u}__{cl.split('|')[-1]}__A1_baseline", mouse, c1, c2, {**meta, "type": "A1_baseline", "case_group": "RNA-closest",
                                                                                      "reference_group": "RNA-farthest", "parent": f"{u}__{cl.split('|')[-1]}__A1_rna_vs_crispr"})
            # A2 target side, per target site
            for site in sorted(set(tsite[ti])):
                tj = ti[tsite[ti] == site]
                if len(tj) < 3 * MIN_G:
                    continue
                colR, colC = BR[:, tj].sum(0), BC[:, tj].sum(0)
                okc = (colR > 0) & (colC > 0); tj = tj[okc]
                if len(tj) < 3 * MIN_G:
                    continue
                sR = BR[np.ix_(si, tj)].sum(0) / BR[:, tj].sum(0); sC = BC[np.ix_(si, tj)].sum(0) / BC[:, tj].sum(0)
                lo, hi = np.quantile(sR, [1 / 3, 2 / 3])
                grp = np.where(sR <= lo, "RNA-cross", np.where(sR >= hi, "RNA-own", "middle"))
                ids = tid[T][tj]
                CA.append(pd.DataFrame({"cell": ids, "unit_id": u, "kind": "A2", "clade": cl, "s_rna": sR, "s_crispr": sC, "group": grp, "site": site}))
                c1, c2 = ids[grp == "RNA-cross"], ids[grp == "RNA-own"]
                meta = {"unit": u, "clade": cl, "site": site, "side": "target", "clone_size": len(tj), **gmeta,
                        "case_sR": float(sR[grp == "RNA-cross"].mean()), "ref_sR": float(sR[grp == "RNA-own"].mean()),
                        "case_sC": float(sC[grp == "RNA-cross"].mean()), "ref_sC": float(sC[grp == "RNA-own"].mean())}
                cid = f"{u}__{cl.split('|')[-1]}__{site}__A2_cross_vs_own"
                if len(c1) >= MIN_G and len(c2) >= MIN_G:
                    add_with_rand(cid, mouse, c1, c2, {**meta, "type": "A2_cross_vs_own", "case_group": "RNA-cross", "reference_group": "RNA-own"}, seed)
                    oth = np.setdiff1d(np.arange(len(S)), si)
                    if len(oth):
                        diff = Y[tj] @ X[oth].mean(0) - Y[tj] @ X[si].mean(0)
                        lo, hi = np.quantile(diff, [1 / 3, 2 / 3])
                        add(f"{cid}__baseline".replace("A2_cross_vs_own__baseline", "A2_baseline"), mouse, ids[diff >= hi], ids[diff <= lo],
                            {**meta, "type": "A2_baseline", "case_group": "RNA-closer-to-other-clades", "reference_group": "RNA-closer-to-own-clade", "parent": cid})
        print(f"{u}: A done", flush=True)
    # ---------- B1 / C1 on all eligible cells of the KP mice in OT units ----------
    ot_mice = sorted({pickle.load(open(f, "rb"))["mouse"] for f in a.bundles.glob("L*KPTracer*.pkl")} |
                     {pickle.load(open(f, "rb"))["mouse"] for f in a.bundles.glob("L1__*.pkl")})
    ACr = AC.reset_index()
    for mouse in ot_mice:
        g = ACr[(ACr.mouse == mouse) & (ACr.clade != "minor")]
        for (cl, site), h in g.groupby(["clade", "site"]):
            vc = h.state.value_counts()
            if len(vc) >= 2 and vc.iloc[1] >= MIN_G:
                x1, x2 = vc.index[0], vc.index[1]
                cid = f"KP_{mouse}__{cl.split('|')[-1]}__{site}__B1_state"
                add_with_rand(cid, mouse, h.cell[h.state == x1].to_numpy(), h.cell[h.state == x2].to_numpy(),
                              {"type": "B1_state", "clade": cl, "site": site, "case_group": x1, "reference_group": x2, "clone_size": len(h)}, seed)
        for (stt, site), h in g.groupby(["state", "site"]):
            for fam, hf in h.groupby("family"):
                vc = hf.clade.value_counts()
                if len(vc) >= 2 and vc.iloc[1] >= MIN_G:
                    c1, c2 = vc.index[0], vc.index[1]
                    cid = f"KP_{mouse}__{stt.replace(' ', '_')}__{site}__{c1.split('|')[-1]}_vs_{c2.split('|')[-1]}__C1_clade"
                    add_with_rand(cid, mouse, hf.cell[hf.clade == c1].to_numpy(), hf.cell[hf.clade == c2].to_numpy(),
                                  {"type": "C1_clade", "state": stt, "site": site, "family": fam, "case_group": c1, "reference_group": c2,
                                   "clone_size": len(hf)}, seed)
    Gd = pd.concat(G); Gd.to_csv(out / "groups.tsv", sep="\t", index=False)
    Ct = pd.DataFrame(CT); Ct.to_csv(out / "contrasts.tsv", sep="\t", index=False)
    pd.concat(CA).to_csv(out / "cells_A.tsv.gz", sep="\t", index=False, compression="gzip")
    Ct["base_type"] = Ct.type.str.replace(r"_rand\d+$", "", regex=True)
    print(Ct.groupby(["base_type", "mouse"]).size().unstack(fill_value=0).to_string())
    print(f"{len(Ct)} contrasts, {Gd.cell.nunique()} distinct cells, {Gd.unit_id.nunique()} mice")


if __name__ == "__main__":
    main()
