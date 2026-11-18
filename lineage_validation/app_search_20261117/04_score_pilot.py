"""Score the application pilots from the v3 worker rows (one configuration; M4-E primary, M4-R reported). Definitions
are fixed here before any pilot result is read.

    python lineage_validation/app_search_20261117/04_score_pilot.py --pilot results/app_search_20261117/kang_pilot \
        --label cell_type [--truth-from-manifest] [--programs]

Per unit and method:
  gates.tsv        cells and rejected share per side and per label
  coupling.tsv     retained block, label x label: mass, enrichment = mass * total / (row * column),
                   P(target label | source label)
  same_label.tsv   share of retained coupling mass between cells of the same label vs the independent expectation
                   sum_k r_k c_k (only for labels shared by both sides)
  truth.tsv        (--truth-from-manifest) the label removed from one side is the one-sided population: its rejection
                   rate vs the rest, precision / recall of rejection on that side, AUROC of the normalized rejection
                   score for it
  depth.tsv        AUROC of the per-cell total counts for rejected vs retained (rejection that follows depth)
  programs.tsv     (--programs) per side, rejected vs retained: mean authors' pseudotime (if in obs) and mean
                   log1p(CP10k) programs EMT (VIM, CDH2, FN1, ZEB1, SNAI2, TWIST1, SERPINE1, COL1A1) and epithelial
                   (CDH1, EPCAM, KRT8, KRT18, KRT19, CLDN4)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

CID = "ranknm256-midrank|none|scale|cos|pca50"
EMT = ["VIM", "CDH2", "FN1", "ZEB1", "SNAI2", "TWIST1", "SERPINE1", "COL1A1"]
EPI = ["CDH1", "EPCAM", "KRT8", "KRT18", "KRT19", "CLDN4"]


def auroc(score, lab):
    lab = np.asarray(lab, bool); r = rankdata(score); n1 = lab.sum(); n0 = len(lab) - n1
    return float((r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pilot", type=Path, required=True)
    p.add_argument("--label", default="cell_type")
    p.add_argument("--config", default=CID)
    p.add_argument("--truth-from-manifest", action="store_true")
    p.add_argument("--programs", action="store_true")
    a = p.parse_args()
    import anndata as ad
    man = pd.read_csv(a.pilot / "manifest.csv", keep_default_na=False)
    out = a.pilot / "score"; out.mkdir(exist_ok=True)
    G, C, S, T, D, PR = [], [], [], [], [], []
    for _, r in man.iterrows():
        u = r.pair_id
        d = a.pilot / "v3" / "rows" / u / a.config.rsplit("|", 3)[0].replace("|", "__") / a.config.replace("|", "__")
        if not (d / "DONE").exists():
            print("missing", d); continue
        A = {s: ad.read_h5ad(r[f"{s}_h5ad"]) for s in ("source", "target")}
        cc = pd.read_csv(d / "cell_confidence.csv.gz")
        cc["observation_id"] = cc.observation_id.astype(str)
        for method in ("M4-E", "M4-R"):
            z = np.load(d / f"coupling_{method.lower().replace('-', '')}.npz")
            P = z["coupling"].astype(np.float64); sid, tid = z["source_ids"].astype(str), z["target_ids"].astype(str)
            g = cc[cc.method == method]
            rej = {s: g[g.side == s].set_index("observation_id").rejected.reindex(ids).to_numpy(bool) for s, ids in (("source", sid), ("target", tid))}
            score = {s: g[g.side == s].set_index("observation_id").normalized_rejection_score.reindex(ids).to_numpy(float)
                     for s, ids in (("source", sid), ("target", tid))}
            lab = {s: A[s].obs[a.label].astype(str).reindex(ids).to_numpy() for s, ids in (("source", sid), ("target", tid))}
            depth = {s: np.asarray(A[s][ids].X.sum(1)).ravel() for s, ids in (("source", sid), ("target", tid))}
            key = {"unit": u, "method": method}
            for s in ("source", "target"):
                G.append({**key, "side": s, "label": "ALL", "cells": len(rej[s]), "rejected_share": float(rej[s].mean())})
                for l_ in sorted(set(lab[s])):
                    m = lab[s] == l_
                    G.append({**key, "side": s, "label": l_, "cells": int(m.sum()), "rejected_share": float(rej[s][m].mean())})
                D.append({**key, "side": s, "rejected_share": float(rej[s].mean()), "depth_auroc_rejected_vs_retained": auroc(depth[s], rej[s])})
            Q = P[np.ix_(~rej["source"], ~rej["target"])]
            ls_, lt_ = lab["source"][~rej["source"]], lab["target"][~rej["target"]]
            tot = Q.sum()
            if tot > 0:
                us, ut = sorted(set(ls_)), sorted(set(lt_))
                M = np.array([[Q[np.ix_(ls_ == x, lt_ == y)].sum() for y in ut] for x in us])
                rw, cl = M.sum(1), M.sum(0)
                for i, x in enumerate(us):
                    for j, y in enumerate(ut):
                        C.append({**key, "source_label": x, "target_label": y, "mass": M[i, j] / tot,
                                  "enrichment": M[i, j] * tot / (rw[i] * cl[j]) if rw[i] > 0 and cl[j] > 0 else np.nan,
                                  "p_target_given_source": M[i, j] / rw[i] if rw[i] > 0 else np.nan})
                shared = sorted(set(us) & set(ut))
                same = sum(M[us.index(k), ut.index(k)] for k in shared) / tot
                expect = sum((rw[us.index(k)] / tot) * (cl[ut.index(k)] / tot) for k in shared)
                S.append({**key, "same_label_mass": float(same), "independent_expectation": float(expect), "labels_shared": len(shared)})
            if a.truth_from_manifest:
                for s, col in (("source", "removed_from_target"), ("target", "removed_from_source")):
                    t_ = r[col]
                    if not t_:
                        continue
                    m = lab[s] == t_
                    T.append({**key, "one_sided_side": s, "one_sided_label": t_, "cells": int(m.sum()),
                              "rejected_share_one_sided": float(rej[s][m].mean()), "rejected_share_rest": float(rej[s][~m].mean()),
                              "precision": float(m[rej[s]].mean()) if rej[s].any() else np.nan, "recall": float(rej[s][m].mean()),
                              "auroc_rejection_score": auroc(score[s], m)})
            if a.programs:
                for s, ids in (("source", sid), ("target", tid)):
                    B = A[s][ids]; tot_ = np.asarray(B.X.sum(1)).ravel()
                    prog = {}
                    for nm, gl in (("EMT", EMT), ("epithelial", EPI)):
                        gl = [x for x in gl if x in B.var_names]
                        Y = B[:, gl].X; Y = Y.toarray() if hasattr(Y, "toarray") else np.asarray(Y)
                        prog[nm] = np.log1p(Y / tot_[:, None] * 1e4).mean(1)
                    pt = B.obs["Pseudotime"].to_numpy(float) if "Pseudotime" in B.obs else np.full(len(ids), np.nan)
                    for grp, m in (("rejected", rej[s]), ("retained", ~rej[s])):
                        PR.append({**key, "side": s, "group": grp, "cells": int(m.sum()), "pseudotime": float(np.nanmean(pt[m])) if m.any() else np.nan,
                                   "EMT": float(prog["EMT"][m].mean()) if m.any() else np.nan,
                                   "epithelial": float(prog["epithelial"][m].mean()) if m.any() else np.nan})
        print(u, "scored", flush=True)
    for nm, rows in (("gates", G), ("coupling", C), ("same_label", S), ("truth", T), ("depth", D), ("programs", PR)):
        if rows:
            pd.DataFrame(rows).to_csv(out / f"{nm}.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
