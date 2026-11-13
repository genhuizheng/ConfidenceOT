"""LARRY PLAN.md scoring (fixed 2026-11-12, before any OT): clone mass, fate prediction, gates, per unit x configuration
x method (M4-E primary, M4-R reported). Reads the v3 worker rows (coupling_m4e/m4r.npz, cell_confidence.csv.gz) and the
unit inputs (obs: clone, cell_type); clone labels are used here only.

    python lineage_validation/larry_20261112/03_score.py --v3-root /scratch/.../larry_20261112/v3 \
        --inputs /scratch/.../larry_20261112/inputs --out /scratch/.../larry_20261112/score

Writes OUT/clone_mass.tsv, fate.tsv, gates.tsv (summaries, one row per unit x config x method), gates_by_type.tsv and
per_cell.tsv.gz (scored day-2 cells: m_i, baselines, top-10 hit, predicted / true Neu share).

Definitions (PLAN.md):
  sisters T_c     target cells of the day-2 cell's clone; scored day-2 cells: retained, clone with >= 1 sister
  m_i             sum_{j in T_c} P(i, j) / sum_j P(i, j)
  uniform         |T_c| / n_target
  independent     sum_{j in T_c} b_j / sum_j b_j, b = column mass of P over retained sources, retained targets only
  top-10 hit      >= 1 sister among the 10 highest-mass targets of row i; chance = 1 - C(n - |T_c|, 10) / C(n, 10)
  fate            mature types = every cell type except "Undifferentiated"; predicted = row mass on mature targets by
                  type; true = mature-type composition of T_c (cells with >= 1 mature sister); TV against a null of clone
                  labels shuffled among scored cells (100 draws, seed 20261112; statistic = median TV)
  Neu vs Mono     clones with >= 2 Neutrophil + Monocyte sisters: Spearman of predicted vs true Neutrophil share among
                  Neu + Mono; AUROC of the predicted share for clone majority Neu (> 0.5) vs Mono (< 0.5), ties dropped
  gates           rejected share per side and cell type; day-2 rejection with vs without sisters on the target side
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import rankdata, spearmanr

CONFIGS = {"l5best": "ranknm256-midrank|none|scale|cos|pca50", "logcpm_pca": "logcpm|none|noscale|sqeuc|pca30"}
UNITS = ["LARRY_LSK_d2_to_d4", "LARRY_LSK_d2_to_d6", "LARRY_LK_d2_to_d4", "LARRY_LK_d2_to_d6"]
SEED, K, NULL = 20261112, 10, 100


def safe(x):
    return str(x).replace("|", "__").replace(":", "_").replace("/", "_")


def log_comb(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def auroc(score, label):
    r = rankdata(score); n1 = int(label.sum()); n0 = len(label) - n1
    return (r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--v3-root", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    import anndata as ad
    a.out.mkdir(parents=True, exist_ok=True)
    CM, FA, GA, GT, PC = [], [], [], [], []
    for u in UNITS:
        if not all((a.inputs / f"{u}__{s}.h5ad").exists() for s in ("source", "target")):
            print(f"missing inputs of {u}", flush=True); continue
        obs = {s: ad.read_h5ad(a.inputs / f"{u}__{s}.h5ad", backed="r").obs[["clone", "cell_type"]].copy() for s in ("source", "target")}
        for tag, cid in CONFIGS.items():
            d = a.v3_root / "rows" / safe(u) / safe(cid.rsplit("|", 3)[0]) / cid.replace("|", "__")
            if not (d / "DONE").exists() and not (d / "coupling_m4e.npz").exists():
                print(f"missing {d}", flush=True); continue
            cc = pd.read_csv(d / "cell_confidence.csv.gz", usecols=["method", "side", "observation_id", "rejected"])
            cc["observation_id"] = cc.observation_id.astype(str)
            for method in ("M4-E", "M4-R"):
                z = np.load(d / f"coupling_{method.lower().replace('-', '')}.npz")
                P = z["coupling"].astype(np.float64); sid, tid = z["source_ids"].astype(str), z["target_ids"].astype(str)
                g = cc[cc.method == method]
                rej_s = g[g.side == "source"].set_index("observation_id").rejected.reindex(sid).to_numpy(bool)
                rej_t = g[g.side == "target"].set_index("observation_id").rejected.reindex(tid).to_numpy(bool)
                os_, ot = obs["source"].reindex(sid), obs["target"].reindex(tid)
                cs, ct = os_.clone.to_numpy(int), ot.clone.to_numpy(int); typ_t = ot.cell_type.astype(str).to_numpy()
                key = {"unit_id": u, "config": tag, "config_id": cid, "method": method}
                # gates
                for side, rj, ob in (("source", rej_s, os_), ("target", rej_t, ot)):
                    GA.append({**key, "side": side, "cells": len(rj), "rejected_share": float(rj.mean())})
                    for ty, m in ob.groupby(ob.cell_type.astype(str)).groups.items():
                        ix = ob.index.get_indexer(m)
                        GT.append({**key, "side": side, "cell_type": ty, "cells": len(ix), "rejected_share": float(rj[ix].mean())})
                tclones = set(ct[ct >= 0])
                has_sis = np.array([c >= 0 and c in tclones for c in cs])
                GA[-2].update({"source_with_sisters": int(has_sis.sum()), "rejected_with_sisters": float(rej_s[has_sis].mean()) if has_sis.any() else np.nan,
                               "rejected_without_sisters": float(rej_s[~has_sis].mean())})
                # clone mass
                rows = np.flatnonzero(~rej_s & has_sis)
                rs = P.sum(1); b = P[~rej_s].sum(0) * (~rej_t); bt = b.sum(); n_t = len(tid)
                sis_of = {c: np.flatnonzero(ct == c) for c in set(cs[rows])}
                mature = typ_t != "Undifferentiated"; types = sorted(set(typ_t[mature]))
                tix = {t: np.flatnonzero(typ_t == t) for t in types}
                rec = []
                for i in rows:
                    if rs[i] <= 0:
                        continue
                    T = sis_of[cs[i]]; row = P[i]
                    m_i = row[T].sum() / rs[i]; uni = len(T) / n_t; ind = b[T].sum() / bt if bt > 0 else np.nan
                    top = np.argpartition(-row, K)[:K]; hit = bool(np.isin(top, T).any())
                    chance = 1 - np.exp(log_comb(n_t - len(T), K) - log_comb(n_t, K)) if n_t - len(T) >= K else 1.0
                    pm = np.array([row[tix[t]].sum() for t in types]); tm = np.array([np.isin(T, tix[t]).sum() for t in types], float)
                    rec.append({"cell": sid[i], "clone": int(cs[i]), "sisters": len(T), "m": m_i, "uniform": uni, "independent": ind,
                                "top10_hit": hit, "top10_chance": chance, "pred": pm / pm.sum() if pm.sum() > 0 else None,
                                "true": tm / tm.sum() if tm.sum() > 0 else None})
                R = pd.DataFrame(rec)
                if R.empty:
                    continue
                CM.append({**key, "scored_cells": len(R), "scored_clones": R.clone.nunique(), "median_m": R.m.median(),
                           "median_m_over_uniform": (R.m / R.uniform).median(), "median_m_over_independent": (R.m / R.independent).median(),
                           "share_m_gt_uniform": float((R.m > R.uniform).mean()), "share_m_gt_independent": float((R.m > R.independent).mean()),
                           "top10_hit_rate": float(R.top10_hit.mean()), "top10_chance": float(R.top10_chance.mean())})
                # fate: TV vs clone-label null
                F = R[R.pred.notna() & R.true.notna()].reset_index(drop=True)
                fr = {**key, "fate_cells": len(F), "fate_clones": F.clone.nunique(), "mature_types": len(types)}
                if len(F):
                    pred = np.vstack(F.pred); true_by_clone = {c: t for c, t in zip(F.clone, F.true)}
                    tv = 0.5 * np.abs(pred - np.vstack(F.true)).sum(1); fr["median_tv"] = float(np.median(tv))
                    rng = np.random.default_rng(SEED); nul = []
                    for _ in range(NULL):
                        perm = rng.permutation(F.clone.to_numpy())
                        nul.append(np.median(0.5 * np.abs(pred - np.vstack([true_by_clone[c] for c in perm])).sum(1)))
                    nul = np.array(nul)
                    fr.update({"null_median_tv_mean": float(nul.mean()), "null_median_tv_p05": float(np.percentile(nul, 5)),
                               "p_null_le_obs": float((np.sum(nul <= fr["median_tv"]) + 1) / (NULL + 1))})
                    F["tv"] = tv
                    # Neu vs Mono
                    if "Neutrophil" in types and "Monocyte" in types:
                        iN, iM = types.index("Neutrophil"), types.index("Monocyte")
                        nm_sis = np.array([np.isin(sis_of[c], np.r_[tix["Neutrophil"], tix["Monocyte"]]).sum() for c in F.clone])
                        tn = np.array([np.isin(sis_of[c], tix["Neutrophil"]).sum() for c in F.clone]) / np.maximum(nm_sis, 1)
                        pn = pred[:, iN] / np.maximum(pred[:, iN] + pred[:, iM], 1e-300)
                        ok = (nm_sis >= 2) & ((pred[:, iN] + pred[:, iM]) > 0)
                        fr["neumono_cells"] = int(ok.sum()); fr["neumono_clones"] = int(F.clone[ok].nunique())
                        fr["neumono_spearman"] = float(spearmanr(pn[ok], tn[ok])[0]) if ok.sum() >= 5 else np.nan
                        lab = ok & (tn != 0.5)
                        fr["neumono_auroc"] = float(auroc(pn[lab], tn[lab] > 0.5)) if lab.sum() >= 5 else np.nan
                        F["pred_neu_share"] = pn; F["true_neu_share"] = np.where(nm_sis >= 2, tn, np.nan)
                FA.append(fr)
                keep = R.drop(columns=["pred", "true"]).merge(F.drop(columns=["pred", "true"])[["cell"] + [c for c in ("tv", "pred_neu_share", "true_neu_share") if c in F]],
                                                          on="cell", how="left") if len(F) else R.drop(columns=["pred", "true"])
                PC.append(keep.assign(**key))
                print(f"{u} {tag} {method}: {len(R)} cells, median m/uniform {CM[-1]['median_m_over_uniform']:.2f}, "
                      f"top10 {CM[-1]['top10_hit_rate']:.2f} vs {CM[-1]['top10_chance']:.2f}, TV {fr.get('median_tv', np.nan):.3f} "
                      f"(null {fr.get('null_median_tv_mean', np.nan):.3f}), Neu/Mono AUROC {fr.get('neumono_auroc', np.nan):.2f}", flush=True)
    for name, rows in (("clone_mass", CM), ("fate", FA), ("gates", GA), ("gates_by_type", GT)):
        pd.DataFrame(rows).to_csv(a.out / f"{name}.tsv", sep="\t", index=False)
    if PC:
        pd.concat(PC).to_csv(a.out / "per_cell.tsv.gz", sep="\t", index=False, compression="gzip")


if __name__ == "__main__":
    main()
