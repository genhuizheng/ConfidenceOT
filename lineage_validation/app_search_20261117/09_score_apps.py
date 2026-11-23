"""Score the application OT rows (PLAN_TACC.md scoring sections; definitions fixed here, 2026-11-23, before any result of
the ot_20261123 runs is read). Both configurations, M4-E primary and M4-R reported. LARRY in vivo is scored by
larry_20261112/03_score.py (--units, --non-mature Undifferentiated Prog), not here.

    python -I 09_score_apps.py --app $SCRATCH/OT_project/app_search_20261117 --dataset kang [--gmt h.all...gmt]

For every unit x configuration x method (rows: <dataset>/ot_20261123/rows/<unit>/<key>/<config>/):
  gates.tsv       rejected share per side, overall and per label value (label column per dataset, below)
  coupling.tsv    retained x retained block, label x label: mass, P(target label | source label), enrichment
                  = mass * total / (row * column)
  same_label.tsv  share of retained mass between equal labels vs the independent expectation sum_k r_k c_k (labels on
                  both sides only)
  depth.tsv       AUROC of per-cell total counts, rejected vs retained (> 0.5: rejected cells are deeper), per side
Dataset-specific (PLAN_TACC):
  kang        label cell_type. ifn_shift.tsv: per retained source cell, the coupling-weighted mean over its retained
              targets minus its own value, of the mean log1p(CP10k) over HALLMARK_INTERFERON_ALPHA_RESPONSE and
              _GAMMA_RESPONSE genes (the authors' 767-gene list is not in their supplement files we hold); summary =
              mean per source cell type
  sciplex2    no cell-type label. dose.tsv: per drug, Spearman over the 7 doses of rejected share (source, target) vs dose,
              and target cells per dose (the paper's viability proxy); cycle.tsv: mean S and G2M score (mean
              log1p(CP10k) over the Tirosh 2016 lists) of rejected vs retained cells, per side
  schiebinger label stage (one per side, so per-age rejection only); WOT cell sets / signatures are not held: not scored
  celegans    labels cell_type and lineage. lineage.tsv: among source and target cells that both carry a lineage name,
              the share of retained coupling mass on lineage-compatible pairs (target name descends from or equals the
              source name: Sulston prefix, x = any letter, '/' alternatives compatible if any is) vs the independent
              expectation over the same annotated cells; gates per lineage depth (name length)
  pancreas    label cell_type. emerging.tsv: target rejection of types with < 1% of source cells (absent or rare the day
              before) vs the other target types
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

CONFIGS = {"l5best": "ranknm256-midrank|none|scale|cos|pca50", "logcpm_pca": "logcpm|none|noscale|sqeuc|pca30"}
LABEL = {"kang": "cell_type", "sciplex2": None, "schiebinger": "stage", "celegans": "cell_type", "pancreas": "cell_type"}
S_GENES = ["MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2", "MCM6", "CDCA7", "DTL", "PRIM1", "UHRF1",
           "MLF1IP", "CENPU", "HELLS", "RFC2", "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76", "SLBP", "CCNE2", "UBR7", "POLD3",
           "MSH2", "ATAD2", "RAD51", "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", "BLM", "CASP8AP2", "USP1", "CLSPN",
           "POLA1", "CHAF1B", "BRIP1", "E2F8"]
G2M_GENES = ["HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80", "CKS2", "NUF2", "CKS1B", "MKI67", "TMPO",
             "CENPF", "TACC3", "FAM64A", "PIMREG", "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB", "BUB1", "KIF11", "ANP32E",
             "TUBB4B", "GTSE1", "KIF20B", "HJURP", "CDCA3", "HN1", "JPT1", "CDC20", "TTK", "CDC25C", "KIF2C", "RANGAP1",
             "NCAPD2", "DLGAP5", "CDCA2", "CDCA8", "ECT2", "KIF23", "HMMR", "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5", "CENPE",
             "CTCF", "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA"]


def auroc(score, lab):
    lab = np.asarray(lab, bool); r = rankdata(score); n1 = lab.sum(); n0 = len(lab) - n1
    return float((r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan


def program(A, genes):
    import scipy.sparse as sp
    X = A.X.tocsr() if sp.issparse(A.X) else sp.csr_matrix(A.X)
    tot = np.asarray(X.sum(1)).ravel()
    idx = [i for i, g in enumerate(A.var_names.astype(str)) if g in set(genes)]
    if not idx:
        return np.full(A.n_obs, np.nan), 0
    Y = X[:, idx].toarray() * (1e4 / np.maximum(tot, 1))[:, None]
    return np.log1p(Y).mean(1), len(idx)


def read_gmt(path, names):
    out = {}
    for line in Path(path).read_text().splitlines():
        f = line.split("\t")
        if f[0] in names:
            out[f[0]] = f[2:]
    return out


def compatible(src, tgt):
    """Target lineage name descends from or equals the source name (Sulston prefix; x = any; '/' alternatives)."""
    for s in src.split("/"):
        for t in tgt.split("/"):
            if len(t) >= len(s) and all(a == b or a == "x" or b == "x" for a, b in zip(s, t[:len(s)])):
                return True
    return False


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--app", type=Path, required=True)
    p.add_argument("--dataset", required=True, choices=list(LABEL))
    p.add_argument("--gmt", type=Path, default=None, help="hallmark gmt (Kang IFN sets)")
    a = p.parse_args()
    import anndata as ad
    ds = a.dataset; root = a.app / ds / "ot_20261123"; out = root / "score"; out.mkdir(exist_ok=True)
    man = pd.read_csv(a.app / ds / "manifest.csv", keep_default_na=False)
    wl = pd.read_csv(root / "WORKLIST.tsv", sep="	", dtype=str).set_index(["unit_id", "key"]).out  # the worker's row folders
    lab = LABEL[ds]
    ifn = None
    if ds == "kang":
        sets = read_gmt(a.gmt, {"HALLMARK_INTERFERON_ALPHA_RESPONSE", "HALLMARK_INTERFERON_GAMMA_RESPONSE"})
        ifn = sorted(set(sum(sets.values(), [])))
    G, C, SL, D, EX = [], [], [], [], []
    for _, r in man.iterrows():
        u = r.pair_id
        A = {s: ad.read_h5ad(r[f"{s}_h5ad"]) for s in ("source", "target")}
        for s in A:
            A[s].obs_names = A[s].obs_names.astype(str)
        tot = {s: np.asarray(A[s].X.sum(1)).ravel() for s in A}
        prog = {}
        if ds == "kang":
            prog = {s: program(A[s], ifn)[0] for s in A}
        if ds == "sciplex2":
            prog = {s: (program(A[s], S_GENES)[0], program(A[s], G2M_GENES)[0]) for s in A}
        for tag, cid in CONFIGS.items():
            d = Path(wl.loc[(u, cid.rsplit("|", 3)[0])]) / cid.replace("|", "__")
            if not (d / "coupling_m4e.npz").exists():
                print("missing", d, flush=True); continue
            cc = pd.read_csv(d / "cell_confidence.csv.gz", usecols=["method", "side", "observation_id", "rejected"])
            cc["observation_id"] = cc.observation_id.astype(str)
            for method in ("M4-E", "M4-R"):
                z = np.load(d / f"coupling_{method.lower().replace('-', '')}.npz")
                P = z["coupling"].astype(np.float64); sid, tid = z["source_ids"].astype(str), z["target_ids"].astype(str)
                g = cc[cc.method == method]
                rj = {"source": g[g.side == "source"].set_index("observation_id").rejected.reindex(sid).to_numpy(bool),
                      "target": g[g.side == "target"].set_index("observation_id").rejected.reindex(tid).to_numpy(bool)}
                pos = {"source": A["source"].obs_names.get_indexer(sid), "target": A["target"].obs_names.get_indexer(tid)}
                assert (pos["source"] >= 0).all() and (pos["target"] >= 0).all()
                key = {"dataset": ds, "unit_id": u, "config": tag, "config_id": cid, "method": method}
                for side in ("source", "target"):
                    G.append({**key, "side": side, "label": "ALL", "cells": len(rj[side]), "rejected_share": float(rj[side].mean())})
                    D.append({**key, "side": side, "depth_auroc_rejected_deeper": auroc(tot[side][pos[side]], rj[side]),
                              "rejected": int(rj[side].sum()), "cells": len(rj[side])})
                    if lab:
                        lv = A[side].obs[lab].astype(str).to_numpy()[pos[side]]
                        for v in sorted(set(lv)):
                            m = lv == v
                            G.append({**key, "side": side, "label": v, "cells": int(m.sum()), "rejected_share": float(rj[side][m].mean())})
                rs, rt = ~rj["source"], ~rj["target"]
                Q = P[np.ix_(rs, rt)]
                if lab and Q.sum() > 0:
                    ls = A["source"].obs[lab].astype(str).to_numpy()[pos["source"]][rs]
                    lt = A["target"].obs[lab].astype(str).to_numpy()[pos["target"]][rt]
                    us, ut = sorted(set(ls)), sorted(set(lt))
                    M = np.array([[Q[np.ix_(ls == x, lt == y)].sum() for y in ut] for x in us]); T = M.sum()
                    for i, x in enumerate(us):
                        for j, y in enumerate(ut):
                            if M[i, j] > 0:
                                C.append({**key, "source_label": x, "target_label": y, "mass": M[i, j] / T,
                                          "p_target_given_source": M[i, j] / M[i].sum(),
                                          "enrichment": M[i, j] * T / (M[i].sum() * M[:, j].sum())})
                    shared = sorted(set(us) & set(ut))
                    if shared:
                        obs_ = sum(M[us.index(k), ut.index(k)] for k in shared) / T
                        exp_ = sum((M[us.index(k)].sum() / T) * (M[:, ut.index(k)].sum() / T) for k in shared)
                        SL.append({**key, "labels_shared": len(shared), "same_label_mass": obs_, "independent_expectation": exp_})
                if ds == "kang" and Q.sum() > 0:
                    w = Q / Q.sum(1, keepdims=True).clip(1e-300)
                    ps, pt = prog["source"][pos["source"]][rs], prog["target"][pos["target"]][rt]
                    shift = w @ pt - ps
                    ct_ = A["source"].obs["cell_type"].astype(str).to_numpy()[pos["source"]][rs]
                    for v in sorted(set(ct_)):
                        m = ct_ == v
                        EX.append({**key, "table": "ifn_shift", "source_cell_type": v, "cells": int(m.sum()),
                                   "mean_shift": float(shift[m].mean()), "ifn_genes_present": int(len(ifn))})
                if ds == "sciplex2":
                    for side in ("source", "target"):
                        sc, g2 = (x[pos[side]] for x in prog[side])
                        for nm, v in (("S", sc), ("G2M", g2)):
                            EX.append({**key, "table": "cycle", "side": side, "score": nm,
                                       "rejected_mean": float(np.nanmean(v[rj[side]])) if rj[side].any() else np.nan,
                                       "retained_mean": float(np.nanmean(v[~rj[side]])) if (~rj[side]).any() else np.nan,
                                       "genes_present": int(program(A[side], S_GENES if nm == "S" else G2M_GENES)[1])})
                    EX.append({**key, "table": "dose", "drug": A["target"].obs["drug"].astype(str).iloc[0],
                               "dose": float(A["target"].obs["dose"].iloc[0]), "target_cells": int(len(tid)),
                               "source_rejected": float(rj["source"].mean()), "target_rejected": float(rj["target"].mean())})
                if ds == "celegans" and Q.sum() > 0:
                    ln_s = A["source"].obs["lineage"].astype(str).to_numpy()[pos["source"]][rs]
                    ln_t = A["target"].obs["lineage"].astype(str).to_numpy()[pos["target"]][rt]
                    ok_s = ~np.isin(ln_s, ["nan", "", "None"]); ok_t = ~np.isin(ln_t, ["nan", "", "None"])
                    if ok_s.any() and ok_t.any():
                        S_ = Q[np.ix_(ok_s, ok_t)]; T = S_.sum()
                        us_, ui = np.unique(ln_s[ok_s], return_inverse=True); ut_, uj = np.unique(ln_t[ok_t], return_inverse=True)
                        comp = np.array([[compatible(x, y) for y in ut_] for x in us_])
                        Cm = comp[np.ix_(ui, uj)]
                        rsum, csum = S_.sum(1) / T, S_.sum(0) / T
                        EX.append({**key, "table": "lineage", "annotated_source": int(ok_s.sum()), "annotated_target": int(ok_t.sum()),
                                   "mass_annotated_block": float(T / Q.sum()), "compatible_mass_share": float(S_[Cm].sum() / T),
                                   "independent_expectation": float((rsum[:, None] * csum[None, :])[Cm].sum())})
                    for side in ("source", "target"):
                        ln = A[side].obs["lineage"].astype(str).to_numpy()[pos[side]]
                        depth_ = np.array([len(x.split("/")[0]) if x not in ("nan", "", "None") else -1 for x in ln])
                        for dv in sorted(set(depth_)):
                            m = depth_ == dv
                            EX.append({**key, "table": "lineage_depth", "side": side, "name_length": int(dv), "cells": int(m.sum()),
                                       "rejected_share": float(rj[side][m].mean())})
                if ds == "pancreas":
                    st_ = A["source"].obs["cell_type"].astype(str).value_counts(normalize=True)
                    lt_ = A["target"].obs["cell_type"].astype(str).to_numpy()[pos["target"]]
                    rare = np.array([st_.get(v, 0.0) < 0.01 for v in lt_])
                    EX.append({**key, "table": "emerging", "rare_types": ";".join(sorted(set(lt_[rare]))), "rare_cells": int(rare.sum()),
                               "rare_rejected": float(rj["target"][rare].mean()) if rare.any() else np.nan,
                               "other_rejected": float(rj["target"][~rare].mean())})
        print("scored", u, flush=True)
    for name, rows in (("gates", G), ("coupling", C), ("same_label", SL), ("depth", D)):
        pd.DataFrame(rows).to_csv(out / f"{name}.tsv", sep="\t", index=False)
    ex = pd.DataFrame(EX)
    for t in (ex.table.unique() if len(ex) else []):
        ex[ex.table == t].dropna(axis=1, how="all").to_csv(out / f"{t}.tsv", sep="\t", index=False)
    if ds == "sciplex2" and len(ex):
        dd = ex[ex.table == "dose"]; rows = []
        for (cfg, mth, drug), g in dd.groupby(["config", "method", "drug"]):
            for side in ("source", "target"):
                rho = spearmanr(g.dose, g[f"{side}_rejected"]).correlation if len(g) > 2 else np.nan
                rows.append({"config": cfg, "method": mth, "drug": drug, "side": side, "doses": len(g), "spearman_rejected_vs_dose": rho})
            rows.append({"config": cfg, "method": mth, "drug": drug, "side": "target_cells", "doses": len(g),
                         "spearman_rejected_vs_dose": spearmanr(g.dose, g.target_cells).correlation if len(g) > 2 else np.nan})
        pd.DataFrame(rows).to_csv(out / "dose_spearman.tsv", sep="\t", index=False)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
