"""Schiebinger: rejection by the authors' gene signatures (PLAN_TACC.md section 3 fallback; fixed 2026-11-27 before
reading any signature result). The WOT cell sets cannot be used: the pertpy 16-day subset (GSM2836267-88, 14-nt
barcodes) shares no cell with the WOT tutorial data (236,285 cells, 16-nt barcodes; checked 2026-11-27: 0 ids and 0
barcodes in common). The tutorial's gene_sets.gmx signatures are gene-level and apply.

Per unit x configuration x method (M4-E primary, M4-R reported), per side and signature: score = mean log1p(CP10k)
over the signature's genes present; cells split into within-side quartiles of the score (Q1 low .. Q4 high);
rejected share per quartile, and the AUROC of the score for rejected vs retained cells (> 0.5: rejected cells score
higher).

    python -I 10_score_schiebinger_signatures.py --app $A --gmx $A/data/schiebinger_wot_tutorial/data/gene_sets.gmx
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

CONFIGS = {"l5best": "ranknm256-midrank|none|scale|cos|pca50", "logcpm_pca": "logcpm|none|noscale|sqeuc|pca30"}
SIGS = ["MEF.identity", "Pluripotency", "Epithelial.identity", "Neural.identity", "Trophoblast", "SASP", "Apoptosis", "Cell.cycle"]


def auroc(score, lab):
    lab = np.asarray(lab, bool); ok = np.isfinite(score); score, lab = score[ok], lab[ok]
    r = rankdata(score); n1 = lab.sum(); n0 = len(lab) - n1
    return float((r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--app", type=Path, required=True)
    p.add_argument("--gmx", type=Path, required=True)
    a = p.parse_args()
    import anndata as ad
    import scipy.sparse as sp
    g = pd.read_csv(a.gmx, sep="\t", header=None, dtype=str)
    sets = {g.iloc[0, j]: [x for x in g.iloc[2:, j].dropna().tolist() if x] for j in range(g.shape[1])}
    root = a.app / "schiebinger" / "ot_20261123"
    man = pd.read_csv(a.app / "schiebinger" / "manifest.csv", keep_default_na=False)
    wl = pd.read_csv(root / "WORKLIST.tsv", sep="\t", dtype=str).set_index(["unit_id", "key"]).out
    rows, genes = [], []
    for _, r in man.iterrows():
        u = r.pair_id
        A = {s: ad.read_h5ad(r[f"{s}_h5ad"]) for s in ("source", "target")}
        sc = {}
        for s in A:
            X = A[s].X.tocsr() if sp.issparse(A[s].X) else sp.csr_matrix(A[s].X)
            tot = np.asarray(X.sum(1)).ravel(); vn = pd.Index(A[s].var_names.astype(str))
            sc[s] = {}
            for k in SIGS:
                idx = vn.get_indexer([x for x in sets[k] if x in set(vn)]); idx = idx[idx >= 0]
                sc[s][k] = np.log1p(X[:, idx].toarray() * (1e4 / np.maximum(tot, 1))[:, None]).mean(1) if len(idx) else np.full(A[s].n_obs, np.nan)
                if s == "source":
                    genes.append({"unit_id": u, "signature": k, "genes_in_set": len(sets[k]), "genes_present": int(len(idx))})
            A[s].obs_names = A[s].obs_names.astype(str)
        for tag, cid in CONFIGS.items():
            d = Path(wl.loc[(u, cid.rsplit("|", 3)[0])]) / cid.replace("|", "__")
            cc = pd.read_csv(d / "cell_confidence.csv.gz", usecols=["method", "side", "observation_id", "rejected"])
            cc["observation_id"] = cc.observation_id.astype(str)
            for method in ("M4-E", "M4-R"):
                for side in ("source", "target"):
                    gg = cc[(cc.method == method) & (cc.side == side)].set_index("observation_id")
                    rj = gg.rejected.reindex(A[side].obs_names).to_numpy(bool)
                    for k in SIGS:
                        v = sc[side][k]
                        q = pd.qcut(pd.Series(v).rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"]).to_numpy()
                        rec = {"unit_id": u, "config": tag, "method": method, "side": side, "signature": k,
                               "cells": len(v), "rejected_share": float(rj.mean()), "auroc_rejected_higher": auroc(v, rj)}
                        for qq in ("Q1", "Q2", "Q3", "Q4"):
                            rec[f"rejected_{qq}"] = float(rj[q == qq].mean())
                        rows.append(rec)
        print("scored", u, flush=True)
    out = root / "score"; out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "signatures.tsv", sep="\t", index=False)
    pd.DataFrame(genes).to_csv(out / "signature_genes.tsv", sep="\t", index=False)
    print("wrote", out / "signatures.tsv")


if __name__ == "__main__":
    main()
