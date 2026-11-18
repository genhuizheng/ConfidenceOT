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


def run_cinemaot(X, s_tr, t_tr, nc, nf):
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
    cf, ot, de = cm.cinemaot_unweighted(a, obs_label="condition", ref_label="perturbed", expr_label="control",
                                        dim=20, thres=0.15, smoothness=1e-4, eps=1e-3)
    P = np.asarray(ot, np.float64)
    # orientation: rows must be control cells (source) to match the fixed-cost metrics
    if P.shape == (len(t_tr), n):
        P = P.T
    assert P.shape == (n, len(t_tr)), P.shape
    lcs, lct = np.log10(nc[s_tr]), np.log10(nc[t_tr])
    row = {"transported_mass": float(P.sum()), "cinemaot_n_confounder_components": int(np.asarray(cf).shape[1]) if np.ndim(cf) == 2 else np.nan}
    row |= oc.depth_metrics(P, nc[s_tr], nc[t_tr], nf[s_tr], nf[t_tr])
    # ITE along the mean shift of the PCA space (as run_perturb): barycentric partner of each control cell
    E = a.obsm["X_pca"]; Es, Et = E[:n], E[n:]
    u = Et.mean(0) - Es.mean(0); u /= max(np.linalg.norm(u), 1e-12)
    rs = P.sum(1); ok = rs > 0
    bary = (P[ok] @ Et) / rs[ok, None]
    row |= leakage((bary - Es[ok]) @ u, lcs[ok], np.log10(nf[s_tr])[ok], (P[ok] @ lct) / rs[ok], (P[ok] @ np.log10(nf[t_tr])) / rs[ok])
    return row


def run_cellot(X, s_tr, s_te, t_tr, t_te, nc, nf, repo: Path, iters: int, work: Path):
    """CellOT scRNA route (scGen AE, then CellOT in its latent) with the repository's configs."""
    import anndata as ad
    from scipy.spatial.distance import cdist
    from scipy.stats import spearmanr
    sys.path.insert(0, str(repo))
    from cellot.train.train import train_auto_encoder, train_cellot
    from cellot.utils.helpers import load_config  # noqa: F401
    from cellot.utils import load_config as lc
    from ml_collections import ConfigDict
    idx = np.concatenate([s_tr, s_te, t_tr, t_te])
    L = lognorm(X[idx])
    n_tr = len(s_tr) + len(t_tr)
    tr_mask = np.r_[np.ones(len(s_tr), bool), np.zeros(len(s_te), bool), np.ones(len(t_tr), bool), np.zeros(len(t_te), bool)]
    hv = hvg(L[tr_mask], 1000)
    cond = ["control"] * (len(s_tr) + len(s_te)) + ["perturbed"] * (len(t_tr) + len(t_te))
    split = ["train"] * len(s_tr) + ["test"] * len(s_te) + ["train"] * len(t_tr) + ["test"] * len(t_te)
    shutil.rmtree(work, ignore_errors=True); work.mkdir(parents=True)
    h5 = work / "unit.h5ad"
    ad.AnnData(X=L[:, hv].toarray().astype(np.float32), obs=pd.DataFrame({"condition": cond, "split": split}, index=[f"c{i}" for i in idx])).write_h5ad(h5)

    def config(model_yaml):
        c = lc(repo / "configs" / "models" / model_yaml)
        c.data = ConfigDict({"type": "cell", "source": "control", "target": "perturbed", "condition": "condition", "path": str(h5)})
        c.dataloader = ConfigDict({"batch_size": 256, "shuffle": True})
        c.datasplit = ConfigDict({"name": "precomputed"})
        return c
    # scGen autoencoder
    ae_dir = work / "scgen"; c_ae = config("scgen.yaml"); c_ae.training.n_iters = min(int(c_ae.training.get("n_iters", iters)), iters)
    train_auto_encoder(ae_dir, c_ae)
    c_ot = config("cellot.yaml"); c_ot.training.n_iters = iters
    c_ot.data.ae_emb = ConfigDict({"path": str(ae_dir)})
    ot_dir = work / "cellot"
    train_cellot(ot_dir, c_ot)
    from cellot.train.utils import load_item_from_save  # noqa: F401
    from cellot.models import load_model
    from cellot.data.cell import load_cell_data
    import torch
    (f, g), _ = load_model(c_ot, restore=ot_dir / "cache" / "model.pt")
    ae, _ = load_model(c_ae, restore=ae_dir / "cache" / "model.pt")
    Z = lambda M: ae.encode(torch.tensor(M, dtype=torch.float32)).detach().numpy()  # noqa: E731
    D = lambda M: ae.decode(torch.tensor(M, dtype=torch.float32)).detach().numpy()  # noqa: E731
    F = L[:, hv].toarray().astype(np.float32)
    o = np.cumsum([0, len(s_tr), len(s_te), len(t_tr), len(t_te)])
    Fs_tr, Fs_te, Ft_tr, Ft_te = F[o[0]:o[1]], F[o[1]:o[2]], F[o[2]:o[3]], F[o[3]:o[4]]
    zs_te = torch.tensor(Z(Fs_te), requires_grad=True)
    zt_pred = g.transport(zs_te).detach().numpy()
    pred = D(zt_pred)
    zs_tr = torch.tensor(Z(Fs_tr), requires_grad=True)
    ite_lat = g.transport(zs_tr).detach().numpy() - zs_tr.detach().numpy()
    rng = np.random.default_rng(SEED)

    def energy(A, B, cap=1500):
        if len(A) > cap:
            A = A[rng.choice(len(A), cap, replace=False)]
        if len(B) > cap:
            B = B[rng.choice(len(B), cap, replace=False)]
        return float(2 * cdist(A, B).mean() - cdist(A, A).mean() - cdist(B, B).mean())
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(50, F.shape[1] - 1), random_state=SEED).fit(np.vstack([Fs_tr, Ft_tr]))
    P_ = pca.transform
    obs_shift = Ft_te.mean(0) - Fs_te.mean(0)
    preds = {"cellot": pred, "baseline_identity": Fs_te, "baseline_mean_shift": Fs_te + (Ft_tr.mean(0) - Fs_tr.mean(0)),
             "baseline_random_target": Ft_tr[rng.choice(len(Ft_tr), len(Fs_te), replace=True)]}
    de = np.argsort(-np.abs(Ft_tr.mean(0) - Fs_tr.mean(0)))[:50]
    rows = []
    for k, v in preds.items():
        sh = v.mean(0) - Fs_te.mean(0)
        r = {"solver": k, "heldout_energy_featurespace": energy(P_(v), P_(Ft_te)),
             "heldout_de_corr": float(np.corrcoef(sh[de], obs_shift[de])[0, 1]) if sh[de].std() > 0 else np.nan}
        rows.append(r)
    u = Ft_tr.mean(0) - Fs_tr.mean(0); u /= max(np.linalg.norm(u), 1e-12)
    ite = (D(g.transport(zs_tr).detach().numpy()) - Fs_tr) @ u
    lcs = np.log10(nc[s_tr]); lfs = np.log10(nf[s_tr])
    rows[0] |= {"ite_owndepth_corr": float(spearmanr(lcs, ite).statistic), "ite_latent_norm_depth_corr": float(spearmanr(lcs, np.linalg.norm(ite_lat, axis=1)).statistic)}
    F2 = np.column_stack([np.ones(len(ite)), lcs, lfs])
    beta, *_ = np.linalg.lstsq(F2, ite, rcond=None)
    rows[0]["ite_depth_r2_own"] = float(1 - ((ite - F2 @ beta) ** 2).sum() / ((ite - ite.mean()) ** 2).sum())
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
                    for r in run_cellot(X, s_tr, s_te, t_tr, t_te, nc, nf, Path(job["cellot_repo"]), job["cellot_iters"],
                                        Path(job["work"]) / f"{job['unit']}_{job['control_ref']}"):
                        rows.append(base | {"method": "cellot", "status": "ok", "cellot_iters": job["cellot_iters"]} | r | {"seconds": time.time() - t0})
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
    p.add_argument("--cellot-iters", type=int, default=100000)
    p.add_argument("--cellot-repo", default=os.environ.get("CELLOT_REPO", "D:/Xia_lab/CellOT_tools/cellot"))
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--tag", default="")
    a = p.parse_args()
    units = pd.read_csv(a.perturb_root / "UNITS.tsv", sep="\t")
    units = units[units.control_ref.isin(a.control_ref)]
    if a.units:
        units = units[units.unit.isin(a.units)]
    jobs = [{"root": str(a.perturb_root), "unit": r.unit, "control_ref": r.control_ref, "kind": r.kind, "methods": a.methods,
             "cellot_repo": a.cellot_repo, "cellot_iters": a.cellot_iters, "work": str(a.perturb_root / "learned_work")} for r in units.itertuples()]
    print(f"{len(jobs)} units, methods {a.methods}", flush=True)
    rows, errs = [], []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for k, fu in enumerate(as_completed([ex.submit(fit_unit, j) for j in jobs]), 1):
            r = fu.result(); rows += r["rows"]; errs += r["err"]
            if k % 10 == 0 or k == len(jobs):
                print(f"{k}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    tag = f"_{a.tag}" if a.tag else ""
    pd.DataFrame(rows).to_csv(a.perturb_root / f"LEARNED_METRICS{tag}.tsv", sep="\t", index=False)
    (a.perturb_root / f"LEARNED_ERRORS{tag}.txt").write_text("\n\n".join(errs))
    print(f"{len(rows)} rows, {len(errs)} errors")


if __name__ == "__main__":
    main()
