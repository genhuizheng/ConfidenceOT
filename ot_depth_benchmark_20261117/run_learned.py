"""Comparison B (PLAN.md, METHODS.md): original implementations of CINEMA-OT and CellOT on the units prepared by
run_perturb.py, reported separately from the fixed-cost solvers. Runs in the otbench env.

    python -I ot_depth_benchmark_20261117/run_learned.py --perturb-root results/ot_depth_benchmark_20261117/norman
        [--methods cinemaot cellot] [--units KLF1] [--control-ref A] [--cellot-iters 100000] [--jobs 8] [--tag ""]
        [--cellot-repo D:/Xia_lab/CellOT_tools/cellot]

Per unit (perturbation x control reference; train cells fit, test cells held out, as run_perturb.py):
  CINEMA-OT  cinemaot.cinemaot.cinemaot_unweighted with package defaults on log1p CP10k / 2,000 HVG / PCA of the train
             cells; its matching matrix is read as a coupling (control rows x perturbed columns): coupling-depth
             association and ITE depth leakage exactly as otdb_common / run_perturb.
  CellOT     the repository's scRNA route: scGen autoencoder (configs/models/scgen.yaml) on the unit's train cells, then
             CellOT (configs/models/cellot.yaml) in its latent space; features = top-1,000 HVG log1p CP10k of the train
             cells. Held-out control cells are mapped; energy distance and DE-shift correlation against identity /
             mean-shift / random-target baselines in the same feature space; depth leakage of T(x) - x.
Writes <root>/LEARNED_METRICS<tag>.tsv and LEARNED_ERRORS<tag>.txt.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import shutil
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SEED = 20261117


def lognorm(M):
    M = sp.csr_matrix(M, dtype=np.float32)
    tot = np.asarray(M.sum(1)).ravel()
    M = sp.csr_matrix(sp.diags(1e4 / np.maximum(tot, 1)) @ M)
    M.data = np.log1p(M.data)
    return M


def unit_cells(root: Path, cells: pd.DataFrame, unit: str, ref: str, kind: str):
    src = cells.index[cells.control_ref == ref].to_numpy()
    tgt = cells.index[cells.control_ref == "B"].to_numpy() if kind == "negative_control" else cells.index[cells.perturbation == unit].to_numpy()
    sp_ = cells.split.to_numpy()
    return src[sp_[src] == "train"], src[sp_[src] == "test"], tgt[sp_[tgt] == "train"], tgt[sp_[tgt] == "test"]


def hvg(L, n_top, flavor="seurat"):
    import anndata as ad
    import scanpy as sc
    a = ad.AnnData(X=L)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.pp.highly_variable_genes(a, n_top_genes=n_top, flavor=flavor)
    return np.flatnonzero(a.var.highly_variable.to_numpy())


def leakage(ite, lc_own, lf_own, lc_partner, lf_partner):
    from scipy.stats import spearmanr
    out = {}
    if len(ite) > 10 and np.std(ite) > 0:
        F = np.column_stack([np.ones(len(ite)), lc_own, lf_own, lc_partner - lc_own, lf_partner - lf_own])
        beta, *_ = np.linalg.lstsq(F, ite, rcond=None)
        out["ite_depth_r2"] = float(1 - ((ite - F @ beta) ** 2).sum() / ((ite - ite.mean()) ** 2).sum())
        out["ite_owndepth_corr"] = float(spearmanr(lc_own, ite).statistic)
        out["ite_sd"] = float(np.std(ite))
    return out


def run_cinemaot(X, s_tr, t_tr, nc, nf, return_coupling: bool = False):
    """CINEMA-OT (package defaults) on the train cells; returns metrics."""
    import anndata as ad
    import scanpy as sc
    import otdb_common as oc
    from cinemaot import cinemaot as cm
    idx = np.concatenate([s_tr, t_tr]); n = len(s_tr)
    L = lognorm(X[idx])
    hv = hvg(L, 2000)
    a = ad.AnnData(X=L[:, hv].toarray(), obs=pd.DataFrame({"condition": ["control"] * n + ["perturbed"] * len(t_tr)}, index=[f"c{i}" for i in idx]))
    a.raw = a.copy()
    sc.pp.scale(a, max_value=10)
    sc.tl.pca(a, n_comps=50, svd_solver="arpack", random_state=SEED)
    a.X = a.raw.X.copy()
    # CINEMA-OT confounder step exactly as cinemaot_unweighted (FastICA, random_state 0, arbitrary-variance whitening, on
    # X_pca[:, :dim]; components with Chatterjee xi < thres with the condition are confounders). Its matching is the
    # matrix scaling of exp(-d^2 / e), e = smoothness x n_confounders, i.e. entropic OT with cost d^2 and regularisation
    # e. The package's linear-domain Sinkhorn-Knopp underflows to NaN on these data (PLAN post-hoc note), so the same
    # objective is solved in the log domain (POT sinkhorn_log).
    import ot as pot
    import sklearn.metrics
    from sklearn.decomposition import FastICA
    dim, thres, smoothness = 20, 0.15, 1e-4
    Xt = FastICA(n_components=dim, random_state=0, whiten="arbitrary-variance").fit_transform(a.obsm["X_pca"][:, :dim])
    grp = (a.obs["condition"] == "control").to_numpy().astype(int)
    xi = np.array([cm.Xi(col, grp).correlation for col in Xt.T])
    cf = Xt[:, xi < thres]
    if cf.shape[1] == 0:
        raise ValueError("CINEMA-OT: no confounder components below thres")
    d2 = sklearn.metrics.pairwise_distances(cf[:n], cf[n:]) ** 2          # control x perturbed
    e = smoothness * cf.shape[1]
    P = pot.sinkhorn(np.full(n, 1.0 / n), np.full(len(t_tr), 1.0 / len(t_tr)), d2, e, method="sinkhorn_log",
                     numItermax=100000, stopThr=1e-9)
    P = np.asarray(P, np.float64)
    # orientation: rows must be control cells (source) to match the fixed-cost metrics
    if P.shape == (len(t_tr), n):
        P = P.T
    assert P.shape == (n, len(t_tr)), P.shape
    lcs, lct = np.log10(nc[s_tr]), np.log10(nc[t_tr])
    row = {"transported_mass": float(P.sum()), "cinemaot_n_confounder_components": int(cf.shape[1]), "cinemaot_reg": e,
           "marginal_error": float(max(np.abs(P.sum(1) * n - 1).max(), np.abs(P.sum(0) * len(t_tr) - 1).max()))}
    row |= oc.depth_metrics(P, nc[s_tr], nc[t_tr], nf[s_tr], nf[t_tr])
    # ITE along the mean shift of the PCA space (as run_perturb): barycentric partner of each control cell
    E = a.obsm["X_pca"]; Es, Et = E[:n], E[n:]
    u = Et.mean(0) - Es.mean(0); u /= max(np.linalg.norm(u), 1e-12)
    rs = P.sum(1); ok = rs > 0
    bary = (P[ok] @ Et) / rs[ok, None]
    row |= leakage((bary - Es[ok]) @ u, lcs[ok], np.log10(nf[s_tr])[ok], (P[ok] @ lct) / rs[ok], (P[ok] @ np.log10(nf[t_tr])) / rs[ok])
    return (row, P) if return_coupling else row


def cellot_train(X, src_idx, tgt_idx, repo: Path, iters: int, work: Path, ae_iters: int = 250000) -> dict:
    """CellOT scRNA route with the authors' own entry point (scripts/train.py) and model configs: scGen autoencoder
    (configs/models/scgen.yaml), then CellOT (configs/models/cellot.yaml) on its latent (--config.data.ae_emb.path).
    The unit's split column is ours (no datasplit in the task config, so read_single_anndata keeps it)."""
    import subprocess
    import anndata as ad
    import torch
    import yaml
    from scipy.spatial.distance import cdist
    from scipy.stats import spearmanr
    sys.path.insert(0, str(repo))
    import sitecustomize_cellot  # noqa: F401  (collections aliases, see cellot_shim/sitecustomize.py)
    from cellot.models import load_autoencoder_model, load_cellot_model
    from cellot.utils.helpers import load_config
    from cellot.data.cell import split_cell_data
    # all cells of the unit; CellOT's own split (train_test, grouped by condition, test 0.2, random_state 0), recomputed
    # here with its split_cell_data on the same obs so that the evaluation uses exactly CellOT's held-out cells
    idx = np.concatenate([src_idx, tgt_idx])
    cond = np.array(["control"] * len(src_idx) + ["perturbed"] * len(tgt_idx))
    obs = pd.DataFrame({"condition": pd.Categorical(cond)}, index=[f"c{i}" for i in idx])
    split = split_cell_data(ad.AnnData(obs=obs), name="train_test", groupby="condition", test_size=0.2, random_state=0).to_numpy()
    L = lognorm(X[idx])
    hv = hvg(L[split == "train"], 1000)                      # the authors' hvg-top1k-train-only
    shutil.rmtree(work, ignore_errors=True); work.mkdir(parents=True)
    F = L[:, hv].toarray().astype(np.float32)
    h5 = work / "unit.h5ad"
    ad.AnnData(X=F, obs=obs, var=pd.DataFrame(index=[f"g{j}" for j in hv])).write_h5ad(h5)
    task = work / "task.yaml"
    yaml.safe_dump({"data": {"type": "cell", "source": "control", "target": "perturbed", "condition": "condition", "path": str(h5)},
                    "dataloader": {"batch_size": 256, "shuffle": True},
                    "datasplit": {"name": "train_test", "groupby": "condition", "test_size": 0.2, "random_state": 0}}, open(task, "w"))
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(HERE / "cellot_shim"), str(repo)])}
    ae_dir, ot_dir = work / "model-scgen", work / "model-cellot"
    logs = []
    for outdir, model_yaml, n_it, extra in ((ae_dir, "scgen.yaml", ae_iters, []), (ot_dir, "cellot.yaml", iters, [f"--config.data.ae_emb.path={ae_dir}"])):
        cmd = [sys.executable, str(repo / "scripts" / "train.py"), "--outdir", str(outdir), "--config", str(task),
               "--config", str(repo / "configs" / "models" / model_yaml), f"--config.training.n_iters={n_it}", *extra]
        r = subprocess.run(cmd, cwd=str(repo), env=env, capture_output=True, text=True)
        logs.append(r.stdout[-1500:] + r.stderr[-1500:])
        if r.returncode or not (outdir / "cache" / "model.pt").exists():
            raise RuntimeError(f"CellOT {model_yaml} failed ({r.returncode}): {logs[-1][-1500:]}")
    c_ae, c_ot = load_config(ae_dir / "config.yaml"), load_config(ot_dir / "config.yaml")
    ae, _ = load_autoencoder_model(c_ae, restore=ae_dir / "cache" / "model.pt", input_dim=F.shape[1])
    (f, g), _ = load_cellot_model(c_ot, restore=ot_dir / "cache" / "model.pt", input_dim=c_ae.model.latent_dim)
    ae.eval()
    enc = lambda M: ae.encode(torch.tensor(M)).detach()  # noqa: E731
    dec = lambda Z: ae.decode(Z).detach().numpy()  # noqa: E731

    def push(M):
        z = enc(M).clone().requires_grad_(True)
        return dec(g.transport(z).detach())
    diag = {}
    for nm, d_ in (("ae", ae_dir), ("cellot", ot_dir)):
        try:   # training trace written by CellOT's Logger (cache/scalars, HDF): first and last evaluation
            ev = pd.read_hdf(d_ / "cache" / "scalars", "eval")
            col = "mmd" if "mmd" in ev else "loss"
            diag |= {f"{nm}_eval_{col}_first": float(ev[col].iloc[0]), f"{nm}_eval_{col}_last": float(ev[col].iloc[-1]),
                     f"{nm}_eval_{col}_min": float(ev[col].min()), f"{nm}_eval_n": int(len(ev))}
        except Exception as e:  # recorded, not fatal
            diag[f"{nm}_trace_error"] = f"{type(e).__name__}: {e}"[:200]
    return {"push": push, "enc": enc, "F": F, "cond": cond, "split": split, "idx": idx, "work": work, "diag": diag}


def run_cellot(X, s_tr, s_te, t_tr, t_te, nc, nf, repo: Path, iters: int, work: Path, ae_iters: int = 250000):
    """CellOT on one perturbation unit (cellot_train), held-out distribution metrics and depth leakage."""
    from scipy.spatial.distance import cdist
    from scipy.stats import spearmanr
    m_ = cellot_train(X, np.concatenate([s_tr, s_te]), np.concatenate([t_tr, t_te]), repo, iters, work, ae_iters)
    push, F, cond, split, idx = m_["push"], m_["F"], m_["cond"], m_["split"], m_["idx"]
    isc, tr = cond == "control", split == "train"
    Fs_tr, Fs_te, Ft_tr, Ft_te = F[isc & tr], F[isc & ~tr], F[~isc & tr], F[~isc & ~tr]
    s_tr = idx[isc & tr]                                      # control train cells (depth leakage below)
    rng = np.random.default_rng(SEED)

    def energy(A, B, cap=1500):
        if len(A) > cap:
            A = A[rng.choice(len(A), cap, replace=False)]
        if len(B) > cap:
            B = B[rng.choice(len(B), cap, replace=False)]
        return float(2 * cdist(A, B).mean() - cdist(A, A).mean() - cdist(B, B).mean())
    from sklearn.decomposition import PCA
    pca = PCA(n_components=50, random_state=SEED).fit(np.vstack([Fs_tr, Ft_tr]))
    obs_shift = Ft_te.mean(0) - Fs_te.mean(0)
    de = np.argsort(-np.abs(Ft_tr.mean(0) - Fs_tr.mean(0)))[:50]
    preds = {"cellot": push(Fs_te), "baseline_identity": Fs_te, "baseline_mean_shift": Fs_te + (Ft_tr.mean(0) - Fs_tr.mean(0)),
             "baseline_random_target": Ft_tr[rng.choice(len(Ft_tr), len(Fs_te), replace=True)]}
    rows = []
    for k, v in preds.items():
        sh = v.mean(0) - Fs_te.mean(0)
        rows.append({"solver": k, "heldout_energy_featurespace": energy(pca.transform(v), pca.transform(Ft_te)),
                     "heldout_de_corr": float(np.corrcoef(sh[de], obs_shift[de])[0, 1]) if sh[de].std() > 0 else np.nan})
    # technical leakage of the learned map: T(x) - x along the train mean shift vs the control cell's own depth
    u = Ft_tr.mean(0) - Fs_tr.mean(0); u /= max(np.linalg.norm(u), 1e-12)
    ite = (push(Fs_tr) - Fs_tr) @ u
    lcs, lfs = np.log10(nc[s_tr]), np.log10(nf[s_tr])
    F2 = np.column_stack([np.ones(len(ite)), lcs, lfs])
    beta, *_ = np.linalg.lstsq(F2, ite, rcond=None)
    rows[0] |= {"ite_owndepth_corr": float(spearmanr(lcs, ite).statistic),
                "ite_depth_r2_own": float(1 - ((ite - F2 @ beta) ** 2).sum() / ((ite - ite.mean()) ** 2).sum()),
                "pred_depth_proxy_corr": float(spearmanr(lcs, push(Fs_tr).sum(1)).statistic)} | m_["diag"]
    shutil.rmtree(work, ignore_errors=True)
    return rows


def fit_unit(job):
    import run_perturb as rp
    root = Path(job["root"])
    rows, err = [], []
    try:
        X = rp.counts(root)
        cells = pd.read_csv(root / "cells.tsv.gz", sep="\t", low_memory=False)
        nc = cells.ncount.to_numpy(float); nf = cells.nfeature.to_numpy(float)
        s_tr, s_te, t_tr, t_te = unit_cells(root, cells, job["unit"], job["control_ref"], job["kind"])
        base = {"unit": job["unit"], "control_ref": job["control_ref"], "kind": job["kind"], "n_source_train": len(s_tr), "n_target_train": len(t_tr)}
        for method in job["methods"]:
            t0 = time.time()
            try:
                if method == "cinemaot":
                    rows.append(base | {"method": "cinemaot", "solver": "cinemaot", "status": "ok"} | run_cinemaot(X, s_tr, t_tr, nc, nf) | {"seconds": time.time() - t0})
                elif method == "cellot":
                    for r in run_cellot(X, s_tr, s_te, t_tr, t_te, nc, nf, Path(job["cellot_repo"]), job["cellot_iters"], ae_iters=job["ae_iters"], work=
                                        (Path(job["work"]) / f"{job['unit']}_{job['control_ref']}").resolve()):
                        rows.append(base | {"method": "cellot", "status": "ok", "cellot_iters": job["cellot_iters"], "ae_iters": job["ae_iters"]} | r | {"seconds": time.time() - t0})
            except Exception:
                rows.append(base | {"method": method, "status": "failed"})
                err.append(f"{job['unit']} {job['control_ref']} {method}: " + traceback.format_exc()[-2000:])
    except Exception:
        err.append(f"{job['unit']} {job['control_ref']}: " + traceback.format_exc()[-2000:])
    return {"rows": rows, "err": err}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--perturb-root", type=Path, required=True)
    p.add_argument("--methods", nargs="+", default=["cinemaot", "cellot"])
    p.add_argument("--units", nargs="*", default=None)
    p.add_argument("--control-ref", nargs="+", default=["A", "B"])
    p.add_argument("--cellot-iters", type=int, default=100000, help="configs/models/cellot.yaml value")
    p.add_argument("--ae-iters", type=int, default=250000, help="configs/models/scgen.yaml value")
    p.add_argument("--cellot-repo", default=os.environ.get("CELLOT_REPO", "D:/Xia_lab/CellOT_tools/cellot"))
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--tag", default="")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--collect", action="store_true", help="merge the parts into LEARNED_METRICS<tag>.tsv and stop")
    a = p.parse_args()
    import otdb_common as oc
    parts = a.perturb_root / f"learned_parts{('_' + a.tag) if a.tag else ''}"
    if a.collect:
        oc.collect_parts(parts, a.perturb_root, "LEARNED_", a.tag); return
    units = pd.read_csv(a.perturb_root / "UNITS.tsv", sep="\t")
    units = units[units.control_ref.isin(a.control_ref)]
    if a.units:
        units = units[units.unit.isin(a.units)]
    jobs = [{"root": str(a.perturb_root), "unit": r.unit, "control_ref": r.control_ref, "kind": r.kind, "methods": a.methods,
             "cellot_repo": a.cellot_repo, "cellot_iters": a.cellot_iters, "ae_iters": a.ae_iters, "work": str(a.perturb_root / "learned_work")} for r in units.itertuples()]
    print(f"{len(jobs)} units, methods {a.methods}", flush=True)
    oc.run_resumable(jobs, fit_unit, parts, lambda j: f"{j['unit']}__{j['control_ref']}", a.jobs, a.shard, a.nshards)
    print(f"-> {parts}; merge with --collect")


if __name__ == "__main__":
    main()
