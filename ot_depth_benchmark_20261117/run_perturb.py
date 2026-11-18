"""Dataset B: control -> perturbed couplings on a perturbation scRNA-seq dataset used for virtual-cell training (PLAN.md).

    python ot_depth_benchmark_20261117/run_perturb.py --dataset norman --stage prepare
    python ot_depth_benchmark_20261117/run_perturb.py --dataset norman --stage fit [--jobs 12] [--units KLF1 BAK1]
    python ot_depth_benchmark_20261117/run_perturb.py --dataset vcc --h5ad PATH --out DIR --stage prepare|fit   (TACC)

prepare  reads the h5ad once (python -I recommended), keeps genes with at least one count, writes the counts as CSR
         arrays (memory-mapped by the workers), a cell table (perturbation, batch, nCount, nFeature, QC columns), and the
         unit plan: every perturbation (none filtered) x control references A and B, plus the control-vs-control unit.
         Control references: two disjoint seeded subsamples of 2,000 control cells; every set split 80/20 (seed).
fit      per unit: train cells only enter the fit; C1 and C2 costs; every fixed-cost solver; metrics:
         depth association, depth-biology association, ITE depth leakage, held-out distribution metrics, gate behaviour.
         Writes METRICS.tsv, GATES.tsv.gz (ConfidenceOT per-cell gates), ERRORS.txt.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import otdb_common as oc  # noqa: E402

DATASETS = {
    "norman": {"h5ad": "D:/Xia_lab/CellOT_data/norman2019/NormanWeissman2019_filtered.h5ad", "pert": "perturbation",
               "control": "control", "batch": "gemgroup", "qc": ["percent_mito", "percent_ribo", "nperts", "good_coverage",
                                                                   "number_of_cells", "guide_id"],
               "out": str(oc.REPO / "results" / "ot_depth_benchmark_20261117" / "norman"), "target_sep": "_"},
    # VCC 2025 H1 CRISPRi training file; column names are confirmed by the audit stage before any fit
    "vcc": {"h5ad": "", "pert": "target_gene", "control": "non-targeting", "batch": "batch", "qc": ["guide_id"],
            "out": "", "target_sep": None},
}
SEED = 20261117
N_CONTROL = 2000
TEST_FRACTION = 0.2
N_DE = 50


# --------------------------------------------------------------------------------------------------------- prepare
def prepare(cfg: dict, out: Path):
    import anndata as ad
    a = ad.read_h5ad(cfg["h5ad"])
    X = sp.csr_matrix(a.X, dtype=np.float32)
    sample = X[: min(2000, X.shape[0])].data
    integer = bool(np.allclose(sample, np.round(sample)))
    keep = np.asarray((X > 0).sum(0)).ravel() > 0
    X = X[:, keep]
    genes = a.var_names[keep].astype(str)
    obs = a.obs
    missing = [c for c in [cfg["pert"], cfg["batch"]] + cfg["qc"] if c not in obs.columns]
    cells = pd.DataFrame({"cell": a.obs_names.astype(str), "perturbation": obs[cfg["pert"]].astype(str).to_numpy(),
                          "ncount": np.asarray(X.sum(1)).ravel(), "nfeature": np.diff(X.indptr)})
    if cfg["batch"] in obs:
        cells["batch"] = obs[cfg["batch"]].astype(str).to_numpy()
    for c in cfg["qc"]:
        if c in obs:
            cells[c] = obs[c].to_numpy()
    out.mkdir(parents=True, exist_ok=True)
    for k in ("data", "indices", "indptr"):
        np.save(out / f"X_{k}.npy", getattr(X, k))
    json.dump({"shape": list(X.shape)}, open(out / "X_shape.json", "w"))
    pd.Series(genes).to_csv(out / "genes.tsv", sep="\t", index=False, header=False)
    rng = np.random.default_rng(SEED)
    is_c = cells.perturbation == cfg["control"]
    cidx = np.flatnonzero(is_c.to_numpy())
    pick = rng.permutation(cidx)
    ref = {"A": np.sort(pick[:N_CONTROL]), "B": np.sort(pick[N_CONTROL:2 * N_CONTROL])}
    cells["split"] = ""
    cells["control_ref"] = ""
    for name, idx in ref.items():
        cells.loc[idx, "control_ref"] = name
    # 80/20 split per condition (CellOT-style cell-level holdout); controls per reference
    for name, grp in [("ref" + k, ref[k]) for k in ref] + [(p, np.flatnonzero((cells.perturbation == p).to_numpy()))
                                                            for p in sorted(cells.perturbation[~is_c].unique())]:
        perm = np.random.default_rng([SEED, sum(map(ord, name))]).permutation(grp)
        nt = int(round(TEST_FRACTION * len(grp)))
        cells.loc[perm[:nt], "split"] = "test"
        cells.loc[perm[nt:], "split"] = "train"
    cells.to_csv(out / "cells.tsv.gz", sep="\t", index=False)
    perts = sorted(cells.perturbation[~is_c].unique())
    units = [{"unit": p, "control_ref": r, "kind": "perturbation"} for p in perts for r in ("A", "B")]
    units.append({"unit": "control_vs_control", "control_ref": "A", "kind": "negative_control"})
    pd.DataFrame(units).to_csv(out / "UNITS.tsv", sep="\t", index=False)
    audit = {"h5ad": cfg["h5ad"], "n_cells": int(X.shape[0]), "n_genes_kept": int(X.shape[1]), "n_genes_total": int(a.n_vars),
             "integer_counts": integer, "missing_columns": missing, "obs_columns": list(map(str, obs.columns)),
             "n_control": int(is_c.sum()), "n_perturbations": len(perts),
             "cells_per_perturbation": cells[~is_c].perturbation.value_counts().describe().to_dict(),
             "median_ncount_control": float(cells.ncount[is_c].median()), "median_ncount_perturbed": float(cells.ncount[~is_c].median()),
             "median_nfeature_control": float(cells.nfeature[is_c].median()), "batches": cells["batch"].value_counts().to_dict() if "batch" in cells else None,
             "n_units": len(units)}
    json.dump(audit, open(out / "AUDIT.json", "w"), indent=1, default=str)
    print(json.dumps(audit, indent=1, default=str))


# -------------------------------------------------------------------------------------------------------------- fit
_X = None


def counts(out: Path):
    global _X
    if _X is None:
        shape = tuple(json.load(open(out / "X_shape.json"))["shape"])
        arrs = [np.load(out / f"X_{k}.npy", mmap_mode="r") for k in ("data", "indices", "indptr")]
        _X = sp.csr_matrix(tuple(arrs), shape=shape)
    return _X


def adata_of(X, idx, genes):
    import anndata as ad
    return ad.AnnData(X=sp.csr_matrix(X[idx], dtype=np.float64), obs=pd.DataFrame(index=[f"c{i}" for i in idx]),
                      var=pd.DataFrame(index=genes))


def eval_space(X, train_idx, test_idx, seed):
    """C2 fitted on train cells and applied to test cells: (train coords, test coords, log1p CP10k function, HVG)."""
    import anndata as ad
    import scanpy as sc

    def lognorm(M):
        M = sp.csr_matrix(M, dtype=np.float32)
        tot = np.asarray(M.sum(1)).ravel()
        M = sp.diags(1e4 / np.maximum(tot, 1)) @ M
        M.data = np.log1p(M.data)
        return sp.csr_matrix(M)
    tr = ad.AnnData(X=lognorm(X[train_idx]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.pp.highly_variable_genes(tr, n_top_genes=2000, flavor="seurat")
    hv = np.flatnonzero(tr.var.highly_variable.to_numpy())
    D = tr.X[:, hv].toarray()
    mu, sd = D.mean(0), D.std(0, ddof=1); sd[sd == 0] = 1
    Z = np.clip((D - mu) / sd, None, 10)
    zm = Z.mean(0)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=50, svd_solver="arpack", random_state=seed).fit(Z - zm)
    Zt = np.clip((lognorm(X[test_idx])[:, hv].toarray() - mu) / sd, None, 10)
    return pca.transform(Z - zm).astype(np.float32), pca.transform(Zt - zm).astype(np.float32), lognorm


def energy(A, B, rng, cap=1500):
    """Energy distance between point sets (subsampled to cap)."""
    if len(A) > cap:
        A = A[rng.choice(len(A), cap, replace=False)]
    if len(B) > cap:
        B = B[rng.choice(len(B), cap, replace=False)]
    from scipy.spatial.distance import cdist
    return float(2 * cdist(A, B).mean() - cdist(A, A).mean() - cdist(B, B).mean())


def fit_unit(job: dict) -> dict:
    out = Path(job["out"])
    rows, gates, err = [], [], []
    try:
        X = counts(out)
        cells = pd.read_csv(out / "cells.tsv.gz", sep="\t", low_memory=False)
        genes = pd.read_csv(out / "genes.tsv", sep="\t", header=None)[0].astype(str).to_numpy()
        ref = job["control_ref"]
        src_all = cells.index[(cells.control_ref == ref)].to_numpy()
        if job["kind"] == "negative_control":
            tgt_all = cells.index[(cells.control_ref == "B")].to_numpy()
        else:
            tgt_all = cells.index[cells.perturbation == job["unit"]].to_numpy()
        s_tr, s_te = src_all[cells.split[src_all] == "train"], src_all[cells.split[src_all] == "test"]
        t_tr, t_te = tgt_all[cells.split[tgt_all] == "train"], tgt_all[cells.split[tgt_all] == "test"]
        n, m = len(s_tr), len(t_tr)
        nc = cells.ncount.to_numpy(float); nf = cells.nfeature.to_numpy(float)
        rng = np.random.default_rng(SEED)
        # evaluation space (C2 fitted on train cells), perturbation score and DE genes
        E_tr, E_te, lognorm = eval_space(X, np.concatenate([s_tr, t_tr]), np.concatenate([s_te, t_te]), SEED)
        Es, Et = E_tr[:n], E_tr[n:]
        Es_te, Et_te = E_te[: len(s_te)], E_te[len(s_te):]
        u = Et.mean(0) - Es.mean(0); shift_norm = float(np.linalg.norm(u)); u = u / max(shift_norm, 1e-12)
        score_t = Et @ u
        L_s, L_t = lognorm(X[s_tr]), lognorm(X[t_tr])
        L_ste, L_tte = lognorm(X[s_te]), lognorm(X[t_te])
        dmean = np.asarray(L_t.mean(0)).ravel() - np.asarray(L_s.mean(0)).ravel()
        de = np.argsort(-np.abs(dmean))[:N_DE]
        obs_shift = np.asarray(L_tte[:, de].mean(0)).ravel() - np.asarray(L_ste[:, de].mean(0)).ravel()
        from scipy.stats import spearmanr
        lcs, lct = np.log10(nc[s_tr]), np.log10(nc[t_tr])
        base = {"unit": job["unit"], "control_ref": ref, "kind": job["kind"], "n_source_train": n, "n_target_train": m,
                "n_source_test": len(s_te), "n_target_test": len(t_te), "shift_norm": shift_norm,
                "depth_shift_log2": float(np.log2(np.median(nc[tgt_all]) / np.median(nc[src_all]))),
                "nfeature_shift_log2": float(np.log2(np.median(nf[tgt_all]) / np.median(nf[src_all]))),
                "depth_state_corr": float(spearmanr(lct, score_t).statistic),
                "nfeature_state_corr": float(spearmanr(np.log10(nf[t_tr]), score_t).statistic)}
        # CRISPRa / CRISPRi target-gene expression per perturbed cell (single targets present in the gene list)
        tg = [g for g in str(job["unit"]).split(job["target_sep"] or "\x00") if g in set(genes)] if job["kind"] == "perturbation" else []
        if tg:
            gi = [int(np.flatnonzero(genes == g)[0]) for g in tg]
            tgx = np.asarray(L_t[:, gi].mean(1)).ravel()
            base["target_gene_depth_corr"] = float(spearmanr(lct, tgx).statistic) if tgx.std() > 0 else np.nan
            base["target_gene_score_corr"] = float(spearmanr(score_t, tgx).statistic) if tgx.std() > 0 else np.nan
            base["target_gene_log2fc"] = float(np.log2((np.expm1(np.asarray(L_t[:, gi].mean(0))).mean() + 1e-3) /
                                                       (np.expm1(np.asarray(L_s[:, gi].mean(0))).mean() + 1e-3)))
        S, T = adata_of(X, s_tr, genes), adata_of(X, t_tr, genes)
        # held-out baselines (no coupling)
        nn = None
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=1).fit(Es).kneighbors(Es_te, return_distance=False)[:, 0]
        bl = {"identity": Es_te, "mean_shift": Es_te + (Et.mean(0) - Es.mean(0)),
              "random_target": Et[rng.choice(m, len(s_te), replace=True)]}
        de_s_tr_mean = np.asarray(L_ste[:, de].mean(0)).ravel()
        for name, pred in bl.items():
            r = base | {"cost": "none", "solver": f"baseline_{name}", "status": "ok",
                        "heldout_energy": energy(pred, Et_te, rng)}
            pshift = {"identity": np.zeros(N_DE), "mean_shift": dmean[de],
                      "random_target": np.asarray(L_t[:, de].mean(0)).ravel() - np.asarray(L_s[:, de].mean(0)).ravel()}[name]
            r["heldout_de_corr"] = float(np.corrcoef(pshift, obs_shift)[0, 1]) if pshift.std() > 0 else np.nan
            rows.append(r)
        for cost in job.get("costs", ("C1", "C2")):
            if cost == "C2":   # C2 = the evaluation space (scanpy-default path fitted on the train cells)
                Xc0, Yc0, diag, cos = Es, Et, {}, False
            else:
                Xc0, Yc0, diag, cos = oc.coordinates(S, T, SEED, cost)
            C, Xc, Yc, scale, crng = oc.cost_matrix(Xc0, Yc0, cos, 0)
            res = oc.solve_all(C, Xc, Yc, scale, crng, 0, solvers=tuple(job.get("solvers", oc.SOLVERS)), device=job.get("device", "cpu"))
            for solver, r_ in res.items():
                key = base | {"cost": cost, "solver": solver, "mnncov_weight": diag.get("mnncred_weight", np.nan)}
                if "error" in r_:
                    rows.append(key | {"status": r_["error"]}); continue
                P = r_["coupling"]
                row = key | {"status": "ok", "transported_mass": float(P.sum()), "marginal_error": r_.get("marginal_error", np.nan)}
                row |= oc.depth_metrics(P, nc[s_tr], nc[t_tr], nf[s_tr], nf[t_tr])
                rs = P.sum(1)
                ok = rs > 1e-12 * P.sum()
                bary = np.zeros_like(Es); bary[ok] = (P[ok] @ Et) / rs[ok, None]
                # per-cell treatment effect (CINEMA-OT style) and its technical leakage
                ite = (bary - Es)[ok] @ u
                partner_lc = np.zeros(n); partner_lc[ok] = (P[ok] @ lct) / rs[ok]
                partner_lf = np.zeros(n); partner_lf[ok] = (P[ok] @ np.log10(nf[t_tr])) / rs[ok]
                F = np.column_stack([lcs[ok], np.log10(nf[s_tr])[ok], partner_lc[ok] - lcs[ok], partner_lf[ok] - np.log10(nf[s_tr])[ok]])
                if ok.sum() > 10 and ite.std() > 0:
                    A_ = np.column_stack([np.ones(ok.sum()), F])
                    beta, *_ = np.linalg.lstsq(A_, ite, rcond=None)
                    row["ite_depth_r2"] = float(1 - ((ite - A_ @ beta) ** 2).sum() / ((ite - ite.mean()) ** 2).sum())
                    row["ite_owndepth_corr"] = float(spearmanr(lcs[ok], ite).statistic)
                    row["ite_sd"] = float(ite.std())
                # held-out prediction: 1-NN train control -> its barycentric target (and a sampled partner)
                if ok[nn].all():
                    row["heldout_energy"] = energy(bary[nn], Et_te, rng)
                    W = P[nn] / rs[nn, None]
                    samp = np.array([rng.choice(m, p=w / w.sum()) for w in W])
                    row["heldout_energy_sampled"] = energy(Et[samp], Et_te, rng)
                    pred = (W @ L_t[:, de].toarray()).mean(0) - de_s_tr_mean
                    row["heldout_de_corr"] = float(np.corrcoef(pred, obs_shift)[0, 1]) if pred.std() > 0 else np.nan
                else:
                    row["heldout_uncovered_fraction"] = float((~ok[nn]).mean())
                if r_["source_gate"] is not None:
                    sg, tg_ = r_["source_gate"], r_["target_gate"]
                    row["rejection_cost"] = r_["rejection_cost"]
                    row["rejected_fraction_source"] = float((~sg).mean()); row["rejected_fraction_target"] = float((~tg_).mean())
                    for side, g, lc, sc_ in (("source", sg, lcs, Es @ u), ("target", tg_, lct, score_t)):
                        rej = (~g).astype(float)
                        if 0 < rej.sum() < len(rej):
                            row[f"rejection_depth_corr_{side}"] = float(spearmanr(lc, rej).statistic)
                            row[f"rejection_score_corr_{side}"] = float(spearmanr(sc_, rej).statistic)
                    gates.append(pd.DataFrame({"unit": job["unit"], "control_ref": ref, "cost": cost, "solver": solver, "side": ["source"] * n + ["target"] * m,
                                               "cell_index": np.concatenate([s_tr, t_tr]), "rejected": np.concatenate([~sg, ~tg_]),
                                               "score": np.concatenate([Es @ u, score_t]), "log10_ncount": np.concatenate([lcs, lct])}))
                rows.append(row)
    except Exception:
        err.append(f"{job['unit']} {job['control_ref']}: " + traceback.format_exc()[-1500:])
    return {"rows": rows, "gates": gates, "err": err}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    p.add_argument("--stage", choices=("prepare", "fit", "collect"), required=True)
    p.add_argument("--shard", type=int, default=0, help="this job's shard (units with index mod --nshards == shard)")
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--costs", nargs="+", default=["C1", "C2", "C3", "C4"])
    p.add_argument("--solvers", nargs="+", default=None)
    p.add_argument("--every", type=int, default=0, help="local sample: every k-th perturbation in sorted order (+ the negative control)")
    p.add_argument("--h5ad", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--units", nargs="*", default=None)
    p.add_argument("--jobs", type=int, default=12)
    p.add_argument("--tag", default="")
    a = p.parse_args()
    cfg = dict(DATASETS[a.dataset])
    if a.h5ad:
        cfg["h5ad"] = a.h5ad
    out = Path(a.out or cfg["out"])
    if a.stage == "prepare":
        prepare(cfg, out); return
    parts = out / f"parts{('_' + a.tag) if a.tag else ''}"
    if a.stage == "collect":
        oc.collect_parts(parts, out, "", a.tag); return
    units = pd.read_csv(out / "UNITS.tsv", sep="\t")
    if a.units:
        units = units[units.unit.isin(a.units)]
    if a.every:
        perts = sorted(units[units.kind == "perturbation"].unit.unique())[:: a.every]
        units = units[units.unit.isin(perts) | (units.kind == "negative_control")]
        print(f"local sample: every {a.every}th perturbation -> {len(perts)} perturbations", flush=True)
    jobs = [{"out": str(out), "unit": r.unit, "control_ref": r.control_ref, "kind": r.kind, "target_sep": cfg["target_sep"],
             "costs": a.costs, "solvers": a.solvers or list(oc.SOLVERS)} for r in units.itertuples()]
    oc.run_resumable(jobs, fit_unit, parts, lambda j: f"{j['unit']}__{j['control_ref']}", a.jobs, a.shard, a.nshards)
    print(f"-> {parts}; merge with --stage collect")


if __name__ == "__main__":
    main()
