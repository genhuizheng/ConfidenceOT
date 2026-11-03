"""Audit of the batch effect of the Splatter N = 5,000 benchmark, from the generated data (read only; no refit, no new
generation). Runs on TACC (one gg node; batch_effect_audit.slurm).

    python realdata_benchmark/splatter_n5000/27_batch_effect_audit.py --root N5000_ROOT --out OUT [--replicates 1 2 3]

Reads, per replicate and level, ROOT/generated/N5000/rep<r>/<level>/ (generation.json, genes_truth.csv,
cells_truth.csv) and the all_shared condition the benchmark fitted, ROOT/conditions/N5000/rep<r>/<level>/
N5000_rep<r>_<builder level>_all_shared/{source,target}.h5ad (every population on both sides; the files 02_run_pair.py
read). Writes OUT/ (refuses if it exists):
  PARAMETERS.tsv          per level x replicate: the generation record (batch strength, batch.facLoc / facScale, D, seed,
                          library size, dropout, perturbation, target proportions, Splatter version)
  IDENTITY_VS_L2.tsv      per level x replicate against L2 of the same replicate: record keys that differ; whether the
                          gene truth (GeneMean, BaseGeneMean, OutlierFactor, DEFacGroup*) and the cells (cell_id, group,
                          ExpLibSize) are identical; the batch factors; the share of identical count entries per side
  TRUE_BATCH_FACTORS.tsv  per level x replicate: the generator's per-gene log2(BatchFacBatch2 / BatchFacBatch1), all genes
                          and expressed genes: SD, median and P90 of |log2FC|, share of genes above 0.25 / 0.5 / 1
  GENE_LOG2FC.tsv         per level x replicate x population: observed pseudobulk log2((CPM_target + 1) / (CPM_source + 1))
                          over genes with source CPM >= 10 in that population: median / P90 |log2FC|, share above 0.5 / 1;
                          and against L2 (same replicate): median / P90 |log2FC_level - log2FC_L2| and its Pearson r with
                          the true log2 batch ratio
  TECHNICAL.tsv           per level x replicate x side: cells, median nCount / nFeature, zero fraction
  SEPARATION.tsv          per level x replicate x representation (the benchmark's production labels logcpm_noscale_cos and
                          ranknm256_noscale_cos, and ranknm256 after P10 equalisation as the worker does it), per population
                          and pooled: side AUC of a 5-fold cross-validated logistic regression on the 30 PCs; share of the
                          within-population variance explained by side; share of other-side cells among a cell's 15 nearest
                          same-population neighbours (0.5 = mixed)
  COSTS.tsv               same rows: cosine cost (2 - 2 cos on unit rows) divided by the median cross-side cost of the
                          subsample (the pipeline's scaling, sampled), on 200 cells per population per side: median cost of
                          same population within side (split halves), same population across sides, different populations
                          within side, different populations across sides; per source cell the share whose median
                          cross-side same-population cost exceeds its median within-side different-population cost, and
                          whose nearest same-population target cell is farther than its nearest different-population
                          source cell; the share of source cells whose nearest same-population target cell is farther than
                          the 99th percentile of the within-side (split-half) nearest-neighbour costs
Descriptive only: no threshold, filter or score is derived from these tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

import rb_common as rb

LEVELS = ["L0_matched", "L1_depth", "L2_depth_dropout", "L3_batch", "L4_batch_perturb", "L5_batch_perturb_imbalance",
          "A4_perturb_only", "A5_imbalance_only"]
NATIVE = {"L0_matched", "L1_depth", "L2_depth_dropout"}
GENE_TRUTH = ["GeneMean", "BaseGeneMean", "OutlierFactor"]
REPS = {"logcpm": dict(encoding="logcpm", K=np.nan, H=np.nan, tie=np.nan, scale=False, cost="cos", ds="none", label=""),
        "ranknm256": dict(encoding="ranknm", K=256, H=np.nan, tie="production", scale=False, cost="cos", ds="none", label=""),
        "ranknm256_P10s1": dict(encoding="ranknm", K=256, H=np.nan, tie="production", scale=False, cost="cos", ds="P10", label="")}
PER_GROUP = 200


def pair_dir(root: Path, rep: int, level: str) -> Path:
    builder = level if level in NATIVE else "L2_depth_dropout"
    return root / "conditions" / "N5000" / f"rep{rep}" / level / f"N5000_rep{rep}_{builder}_all_shared"


def counts_of(a: ad.AnnData) -> sp.csr_matrix:
    return sp.csr_matrix(a.X, dtype=np.float64)


def pseudobulk_cpm(x: sp.csr_matrix) -> np.ndarray:
    tot = np.asarray(x.sum(axis=0)).ravel()
    return tot / max(tot.sum(), 1.0) * 1e6


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def cos_cost(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(2.0 - 2.0 * unit(a) @ unit(b).T, 0.0)


def separation(X: np.ndarray, side: np.ndarray, group: np.ndarray, seed: int) -> list[dict]:
    out = []
    nn = NearestNeighbors(n_neighbors=16).fit(unit(X))
    _, idx = nn.kneighbors(unit(X))
    for g in ["pooled"] + sorted(set(group)):
        m = np.ones(len(group), bool) if g == "pooled" else group == g
        y = (side[m] == "target").astype(int)
        if y.min() == y.max():
            continue
        p = cross_val_predict(LogisticRegression(max_iter=2000), X[m], y, cv=5, method="predict_proba")[:, 1]
        Xm = X[m] - X[m].mean(0)
        within = sum(((Xm[y == k] - Xm[y == k].mean(0)) ** 2).sum() for k in (0, 1))
        r2 = 1.0 - within / max((Xm ** 2).sum(), 1e-12)
        if g == "pooled":
            mix = np.nan
        else:
            cells = np.where(m)[0]; frac = []
            for i in cells:
                nb = [j for j in idx[i, 1:] if group[j] == g]
                if nb:
                    frac.append(np.mean(side[nb] != side[i]))
            mix = float(np.mean(frac)) if frac else np.nan
        out.append({"population": g, "side_auc": roc_auc_score(y, p), "side_variance_share": r2, "knn15_other_side_share": mix})
    return out


def costs(X: np.ndarray, side: np.ndarray, group: np.ndarray, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    pick = []
    for s in ("source", "target"):
        for g in sorted(set(group)):
            c = np.where((side == s) & (group == g))[0]
            pick.append(rng.choice(c, min(PER_GROUP, c.size), replace=False))
    pick = np.concatenate(pick)
    Xs, ss, gs = X[pick], side[pick], group[pick]
    S, T = ss == "source", ss == "target"
    scale = float(np.median(cos_cost(Xs[S], Xs[T])))
    C = cos_cost(Xs, Xs) / scale
    half = rng.random(len(pick)) < 0.5  # split halves within a side
    same = gs[:, None] == gs[None, :]
    sameside = ss[:, None] == ss[None, :]
    split = half[:, None] != half[None, :]
    rows = []
    for g in ["pooled"] + sorted(set(gs)):
        r = (np.ones(len(gs), bool) if g == "pooled" else gs == g)
        rs = r & S
        def med(mask):
            v = C[np.ix_(rs, np.ones(len(gs), bool))][mask[rs]]
            return float(np.median(v)) if v.size else np.nan
        within_same = med(same & sameside & split)
        cross_same = med(same & ~sameside)
        within_diff = med(~same & sameside)
        cross_diff = med(~same & ~sameside)
        # per source cell
        cs, wd, nn_cs, nn_wd, nn_ws = [], [], [], [], []
        for i in np.where(rs)[0]:
            a = C[i, same[i] & T]; b = C[i, ~same[i] & S]; w = C[i, same[i] & S & (half != half[i])]
            cs.append(np.median(a)); wd.append(np.median(b)); nn_cs.append(a.min()); nn_wd.append(b.min()); nn_ws.append(w.min() if w.size else np.nan)
        cs, wd, nn_cs, nn_wd, nn_ws = map(np.asarray, (cs, wd, nn_cs, nn_wd, nn_ws))
        q99 = np.nanquantile(nn_ws, 0.99)
        rows.append({"population": g, "cost_scale_median_cross": scale, "median_same_population_within_side_split": within_same,
                     "median_same_population_cross_side": cross_same, "median_different_population_within_side": within_diff,
                     "median_different_population_cross_side": cross_diff,
                     "share_cells_cross_same_median_gt_within_different_median": float(np.mean(cs > wd)),
                     "share_cells_nn_cross_same_gt_nn_within_different": float(np.mean(nn_cs > nn_wd)),
                     "median_nn_within_side_same_population": float(np.nanmedian(nn_ws)),
                     "median_nn_cross_side_same_population": float(np.median(nn_cs)),
                     "share_cells_nn_cross_same_gt_q99_within_nn": float(np.mean(nn_cs > q99))})
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--replicates", type=int, nargs="*", default=[1, 2, 3])
    a = p.parse_args()
    if a.out.exists():
        sys.exit(f"{a.out} exists; not overwritten")
    a.out.mkdir(parents=True)
    par, ident, truebf, glfc, tech, sep, cst = [], [], [], [], [], [], []
    for rep in a.replicates:
        l2 = {}
        for level in LEVELS:
            gen = a.root / "generated" / "N5000" / f"rep{rep}" / level
            pdir = pair_dir(a.root, rep, level)
            if not (gen / "generation.json").exists() or not (pdir / "source.h5ad").exists():
                print("missing", gen, pdir, flush=True); continue
            rec = json.loads((gen / "generation.json").read_text())
            par.append({"level": level, "replicate": rep, **{k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in rec.items()}})
            genes = pd.read_csv(gen / "genes_truth.csv"); cells = pd.read_csv(gen / "cells_truth.csv")
            S, T = ad.read_h5ad(pdir / "source.h5ad"), ad.read_h5ad(pdir / "target.h5ad")
            xs, xt = counts_of(S), counts_of(T)
            gs, gt = S.obs["group"].astype(str).to_numpy(), T.obs["group"].astype(str).to_numpy()
            bcols = sorted(c for c in genes if c.startswith("BatchFacBatch"))
            lbf = np.log2(genes[bcols[1]].to_numpy(float) / genes[bcols[0]].to_numpy(float))
            expressed = pseudobulk_cpm(xs) >= 10
            for subset, m in (("all genes", np.ones(len(lbf), bool)), ("source CPM >= 10", expressed)):
                v = np.abs(lbf[m])
                truebf.append({"level": level, "replicate": rep, "genes": subset, "n_genes": int(m.sum()), "sd_log2": float(lbf[m].std()),
                               "median_abs_log2": float(np.median(v)), "p90_abs_log2": float(np.quantile(v, .9)),
                               "share_abs_gt_0.25": float((v > .25).mean()), "share_abs_gt_0.5": float((v > .5).mean()), "share_abs_gt_1": float((v > 1).mean())})
            for s, x in (("source", xs), ("target", xt)):
                tot = np.asarray(x.sum(1)).ravel(); det = np.diff(x.indptr)
                tech.append({"level": level, "replicate": rep, "side": s, "cells": x.shape[0], "median_nCount": float(np.median(tot)),
                             "median_nFeature": float(np.median(det)), "zero_fraction": 1.0 - x.nnz / (x.shape[0] * x.shape[1])})
            lfc = {}
            for g in sorted(set(gs)):
                cs_, ct_ = pseudobulk_cpm(xs[gs == g]), pseudobulk_cpm(xt[gt == g])
                keep = cs_ >= 10
                l = np.log2((ct_ + 1) / (cs_ + 1)); lfc[g] = (l, keep)
                row = {"level": level, "replicate": rep, "population": g, "n_genes": int(keep.sum()),
                       "median_abs_log2fc": float(np.median(np.abs(l[keep]))), "p90_abs_log2fc": float(np.quantile(np.abs(l[keep]), .9)),
                       "share_abs_gt_0.5": float((np.abs(l[keep]) > .5).mean()), "share_abs_gt_1": float((np.abs(l[keep]) > 1).mean())}
                if level != "L2_depth_dropout" and "lfc" in l2 and g in l2["lfc"]:
                    l0, k0 = l2["lfc"][g]; k = keep & k0; d = l[k] - l0[k]
                    row.update({"median_abs_change_vs_L2": float(np.median(np.abs(d))), "p90_abs_change_vs_L2": float(np.quantile(np.abs(d), .9)),
                                "pearson_change_vs_true_batch_log2": float(np.corrcoef(d, lbf[k])[0, 1]) if lbf[k].std() > 0 else np.nan})
                glfc.append(row)
            if level == "L2_depth_dropout":
                l2 = {"rec": rec, "genes": genes, "cells": cells, "xs": xs, "xt": xt, "lfc": lfc}
            elif l2:
                diff = sorted(k for k in set(rec) | set(l2["rec"]) if rec.get(k) != l2["rec"].get(k))
                de = sorted(c for c in genes if c.startswith("DEFacGroup"))
                same_gene = {c: bool(np.array_equal(genes[c].to_numpy(), l2["genes"][c].to_numpy())) for c in GENE_TRUTH + de}
                c2 = l2["cells"]
                row = {"level": level, "replicate": rep, "record_keys_differing": ";".join(diff),
                       "gene_truth_identical": all(same_gene.values()), "gene_truth_columns_differing": ";".join(k for k, v in same_gene.items() if not v),
                       "cell_ids_identical": bool(np.array_equal(cells.cell_id.to_numpy(), c2.cell_id.to_numpy())),
                       "groups_identical": bool(np.array_equal(cells.group.to_numpy(), c2.group.to_numpy())),
                       "explibsize_max_abs_diff": float(np.max(np.abs(cells.ExpLibSize.to_numpy(float) - c2.ExpLibSize.to_numpy(float)))) if len(cells) == len(c2) else np.nan,
                       "batch_factor_log2_sd": float(lbf.std())}
                for s, x, y in (("source", xs, l2["xs"]), ("target", xt, l2["xt"])):
                    row[f"{s}_identical_count_entries"] = float(1.0 - (x != y).nnz / (x.shape[0] * x.shape[1])) if x.shape == y.shape else np.nan
                    row[f"{s}_counts_bit_identical"] = bool(x.shape == y.shape and (x != y).nnz == 0)
                ident.append(row)
            # representations: the benchmark's production path (rb_common.build_config), seed as the fit's (run seed + index)
            man = pd.read_csv(pdir.parent / "manifest.csv"); idx = int(np.where(man.pair_id.astype(str) == pdir.name)[0][0])
            files = sorted(set(man.source_h5ad) | set(man.target_h5ad))
            p10 = int(np.quantile(np.concatenate([np.asarray(ad.read_h5ad(f).X.sum(1)).ravel() for f in files]), rb.DS_QUANTILES["P10"]))
            for name, spec in REPS.items():
                s_, t_ = (rb.equalised(S, p10, 1001), rb.equalised(T, p10, 1002)) if spec["ds"] == "P10" else (S, T)
                X, Y, _ = rb.build_config(dict(spec), s_, t_, 20260925 + idx)
                Z = np.vstack([X, Y]); side = np.array(["source"] * len(X) + ["target"] * len(Y)); grp = np.concatenate([gs, gt])
                for r in separation(Z, side, grp, rep):
                    sep.append({"level": level, "replicate": rep, "representation": name, **r})
                for r in costs(Z, side, grp, 20260925 + rep):
                    cst.append({"level": level, "replicate": rep, "representation": name, **r})
            print(level, "rep", rep, "done", flush=True)
    for name, rows in (("PARAMETERS", par), ("IDENTITY_VS_L2", ident), ("TRUE_BATCH_FACTORS", truebf), ("GENE_LOG2FC", glfc),
                       ("TECHNICAL", tech), ("SEPARATION", sep), ("COSTS", cst)):
        pd.DataFrame(rows).to_csv(a.out / f"{name}.tsv", sep="\t", index=False)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
