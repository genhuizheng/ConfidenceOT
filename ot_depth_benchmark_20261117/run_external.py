"""External OT methods on one Splatter pair, each with its authors' preprocessing (PLAN.md section "External OT methods",
METHOD_PREPROCESSING.md). One call = one pair x one method, run with the python of the method's own environment.

    <env python> -I ot_depth_benchmark_20261117/run_external.py --method wot --source S.h5ad --target T.h5ad \
        --truth population_truth.tsv --out ROW_DIR [--meta '{"level": ..., "replicate": ...}'] [--seed 20261118]

Methods: wot, moscot_bal, moscot_unbal, moscot_temporal, cinemaot_tutorial, cinemaot_default, cinemaotw, otcfm, sbcfm,
cellot. Writes ROW_DIR/metrics.tsv (one row; map methods also write their baselines), ROW_DIR/transitions.tsv
(couplings: source-group x target-group mass) and ROW_DIR/run.json.
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
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import otdb_common as oc  # noqa: E402  (metrics only; no solver import at module level)

COUPLING = {"wot", "moscot_bal", "moscot_unbal", "moscot_temporal", "cinemaot_tutorial", "cinemaot_default"}
MAPS = {"otcfm", "sbcfm", "cellot"}
WEIGHTS = {"cinemaotw"}


# --------------------------------------------------------------------------------------------------------- inputs
def load(a):
    import anndata as ad
    S, T = ad.read_h5ad(a.source), ad.read_h5ad(a.target)
    truth = pd.read_csv(a.truth, sep="\t").set_index(["side", "observation_id"])
    genes = S.var_names.intersection(T.var_names)
    X = sp.vstack([sp.csr_matrix(S[:, genes].X, dtype=np.float32), sp.csr_matrix(T[:, genes].X, dtype=np.float32)], format="csr")
    n = S.n_obs
    obs = pd.DataFrame({"side": pd.Categorical(["src"] * n + ["tgt"] * T.n_obs, categories=["src", "tgt"]),
                        "day": [0.0] * n + [1.0] * T.n_obs},
                       index=[f"s{i}" for i in range(n)] + [f"t{j}" for j in range(T.n_obs)])
    g = np.concatenate([S.obs.group.astype(str).to_numpy(), T.obs.group.astype(str).to_numpy()])
    rej = np.concatenate([truth.loc[[("source", x) for x in S.obs_names]].should_reject.to_numpy(bool),
                          truth.loc[[("target", x) for x in T.obs_names]].should_reject.to_numpy(bool)])
    A = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes.astype(str)))
    nc = np.asarray(X.sum(1)).ravel().astype(float); nf = np.diff(X.indptr).astype(float)
    return A, n, g, rej, nc, nf


def auroc(score, label):
    """P(score of a positive > score of a negative), ties half; NaN when one class is empty."""
    from scipy.stats import rankdata
    label = np.asarray(label, bool)
    if label.all() or (~label).all():
        return np.nan
    r = rankdata(score)
    n1 = label.sum(); n0 = (~label).sum()
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def coupling_metrics(P, n, g, rej, nc, nf):
    P = np.asarray(P, np.float64)
    gs, gt = g[:n], g[n:]
    row = {"transported_mass": float(P.sum())}
    row |= oc.group_metrics(P, gs, gt, nc[:n], nc[n:])
    row |= oc.depth_metrics(P, nc[:n], nc[n:], nf[:n], nf[n:])
    m = P.shape[1]
    tot = P.sum()
    rs, cs = P.sum(1) / tot * n, P.sum(0) / tot * m           # mass relative to the uniform marginal (1 = full)
    for side, s_, r_ in (("source", rs, rej[:n]), ("target", cs, rej[n:])):
        row[f"rel_mass_unmatched_{side}"] = float(s_[r_].mean()) if r_.any() else np.nan
        row[f"rel_mass_shared_{side}"] = float(s_[~r_].mean()) if (~r_).any() else np.nan
        row[f"unmatched_auroc_{side}"] = auroc(-s_, r_)       # low mass = unmatched
    M = pd.DataFrame(P, index=gs, columns=gt).T.groupby(level=0).sum().T.groupby(level=0).sum()
    trans = [{"source_group": a_, "target_group": b_, "mass": float(M.loc[a_, b_])} for a_ in M.index for b_ in M.columns]
    return row, trans


def map_metrics(Xs, Ts, Xt, gs, gt, lcs, lct):
    """Nearest target cell of each mapped source point, in the method's own space."""
    from sklearn.neighbors import NearestNeighbors
    j = NearestNeighbors(n_neighbors=1).fit(Xt).kneighbors(Ts, return_distance=False)[:, 0]
    out = {"nn_correct_share": float((gt[j] == gs).mean()),
           "nn_depth_corr_count": float(np.corrcoef(lcs, lct[j])[0, 1]) if np.std(lct[j]) > 0 else np.nan}
    for k in sorted(set(gs)):
        sel = gs == k
        out[f"nn_correct_{k}"] = float((gt[j][sel] == k).mean())
    return out


# --------------------------------------------------------------------------------------------------------- methods
def log_cp10k(A):
    import scanpy as sc
    B = A.copy(); sc.pp.normalize_total(B, target_sum=1e4); sc.pp.log1p(B)
    return B


def run_wot(A, seed):
    import scanpy as sc
    import wot
    X = sp.csr_matrix(A.X, dtype=np.float64, copy=True)
    rng = np.random.default_rng(seed)
    tot = np.asarray(X.sum(1)).ravel()
    for i in np.flatnonzero(tot > 15000):    # the paper's downsampling to 15,000 UMI (without replacement)
        lo, hi = X.indptr[i], X.indptr[i + 1]
        X.data[lo:hi] = rng.multivariate_hypergeometric(X.data[lo:hi].astype(np.int64), 15000)
    X.eliminate_zeros()
    W = A.copy(); W.X = X
    sc.pp.normalize_total(W, target_sum=1e4); sc.pp.log1p(W)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.pp.highly_variable_genes(W, flavor="seurat", n_bins=20, min_disp=1.0)
    W = W[:, W.var.highly_variable].copy()
    W.X = sp.csr_matrix(W.X).toarray()
    m = wot.ot.OTModel(W, day_field="day", epsilon=0.05, lambda1=1, lambda2=50, growth_iters=3, local_pca=30)
    tm = m.compute_transport_map(0, 1)
    return np.asarray(tm.X), {"n_hvg": int(W.n_vars), "n_downsampled": int((tot > 15000).sum())}


def moscot_input(A, pca: bool):
    import scanpy as sc
    B = log_cp10k(A)
    if pca:
        sc.pp.pca(B, n_comps=50)
    return B


def run_moscot(A, arm):
    from moscot.problems.generic import SinkhornProblem
    from moscot.problems.time import TemporalProblem
    if arm == "moscot_temporal":
        B = moscot_input(A, False)
        p = TemporalProblem(B).prepare(time_key="day")
        p = p.solve(epsilon=1e-3, tau_a=1.0, tau_b=1.0, scale_cost="mean")
        sol = p.solutions[(0.0, 1.0)]
    else:
        B = moscot_input(A, True)
        p = SinkhornProblem(B).prepare(key="side", joint_attr="X_pca", policy="explicit", subset=[("src", "tgt")])
        p = (p.solve(epsilon=1e-3, tau_a=1.0, tau_b=1.0, scale_cost="mean") if arm == "moscot_bal"
             else p.solve(epsilon=1e-2, tau_a=0.95, tau_b=0.95))
        sol = p.solutions[("src", "tgt")]
    extra = {"converged": bool(getattr(sol, "converged", True))}
    return np.asarray(sol.transport_matrix), extra


def cinema_input(A):
    import scanpy as sc
    c = log_cp10k(A)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.pp.highly_variable_genes(c, min_mean=0.0125, max_mean=3, min_disp=0.5)
    c = c[:, c.var.highly_variable].copy()
    c.raw = c.copy()
    sc.pp.scale(c); sc.pp.pca(c, n_comps=50)
    c.X = c.raw.X.copy()
    return c


def run_cinemaot(A, n, arm):
    """Package cinemaot_unweighted; if its linear-domain Sinkhorn-Knopp returns NaN, the same objective (entropic OT,
    cost d^2 on the package's confounder components, regularisation smoothness x n_confounders) in the log domain."""
    from cinemaot import cinemaot as cm
    thres, smooth = (0.5, 1e-5) if arm == "cinemaot_tutorial" else (0.15, 1e-4)
    c = cinema_input(A)
    extra = {"thres": thres, "smoothness": smooth, "solver_substitution": False}
    cf, P, _ = cm.cinemaot_unweighted(c, obs_label="side", ref_label="src", expr_label="tgt", dim=20, thres=thres,
                                      smoothness=smooth, eps=1e-3, mode="parametric")
    P = np.asarray(P, np.float64)
    extra["n_confounders"] = int(np.asarray(cf).shape[1]) if np.ndim(cf) == 2 else 1
    if not np.isfinite(P).all() or P.sum() <= 0:
        import ot as pot
        import sklearn.metrics
        cf = np.asarray(cf).reshape(len(c), -1)
        d2 = sklearn.metrics.pairwise_distances(cf[:n], cf[n:]) ** 2
        m = len(c) - n
        # epsilon-scaling (log-stabilised, annealed epsilon): the same objective, tractable at the tiny regularisation
        a_, b_ = np.full(n, 1 / n), np.full(m, 1 / m)
        P = pot.sinkhorn(a_, b_, d2, smooth * cf.shape[1], method="sinkhorn_epsilon_scaling", numItermax=100000, stopThr=1e-7)
        P = np.asarray(P, np.float64)
        extra["solver_substitution"] = True
        extra["substitution_marginal_error"] = float(max(np.abs(P.sum(1) - a_).max() / a_[0], np.abs(P.sum(0) - b_).max() / b_[0]))
    return P, extra


def run_cinemaotw(A, n, rej, seed):
    from cinemaot import cinemaot as cm
    c = cinema_input(A)
    np.random.seed(0)
    cf, P, te, z = cm.cinemaot_weighted(c, obs_label="side", ref_label="src", expr_label="tgt", dim=20, thres=0.75,
                                        smoothness=1e-4, eps=1e-3, k=10, resolution=1)
    z = np.asarray(z, float).ravel()
    out = {"src_weight_zero_share": float((z == 0).mean()), "unmatched_auroc_source": auroc(-z, rej[:n]),
           "weight_unmatched_source": float(z[rej[:n]].mean()) if rej[:n].any() else np.nan,
           "weight_shared_source": float(z[~rej[:n]].mean()) if (~rej[:n]).any() else np.nan,
           "matching_nan": bool(not np.isfinite(np.asarray(P, float)).all())}
    return out


def run_cfm(A, n, arm, seed, steps=10000):
    import scanpy as sc
    import torch
    from torchcfm.conditional_flow_matching import (ExactOptimalTransportConditionalFlowMatcher,
                                                    SchrodingerBridgeConditionalFlowMatcher)
    from torchcfm.models.models import MLP
    from torchcfm.utils import torch_wrapper
    from torchdyn.core import NeuralODE
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    f = A.copy(); sc.pp.normalize_total(f); sc.pp.sqrt(f); sc.pp.pca(f, n_comps=5)
    Z = f.obsm["X_pca"]; Z = (Z - Z.mean(0)) / Z.std(0)
    x0_all, x1_all = torch.tensor(Z[:n], dtype=torch.float32), torch.tensor(Z[n:], dtype=torch.float32)
    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.1) if arm == "otcfm" else SchrodingerBridgeConditionalFlowMatcher(sigma=1.0, ot_method="exact")
    net = MLP(dim=5, time_varying=True, w=64)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    for _ in range(steps):
        x0 = x0_all[rng.integers(0, len(x0_all), 128)]; x1 = x1_all[rng.integers(0, len(x1_all), 128)]
        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
        loss = ((net(torch.cat([xt, t[:, None]], -1)) - ut) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        T = NeuralODE(torch_wrapper(net), solver="dopri5", sensitivity="adjoint").trajectory(x0_all, t_span=torch.linspace(0, 1, 101))[-1].numpy()
    return Z[:n], T, Z[n:], {"steps": steps, "final_loss": float(loss.detach())}


def run_cellot(A, n, work, iters, ae_iters):
    import run_learned as rl
    m = rl.cellot_train(sp.csr_matrix(A.X), np.arange(n), np.arange(n, A.n_obs), Path(os.environ.get("CELLOT_REPO", "D:/Xia_lab/CellOT_tools/cellot")),
                        iters, work, ae_iters)
    F, cond, split, idx = m["F"], m["cond"], m["split"], m["idx"]
    te = (cond == "control") & (split == "test")
    return F[te], m["push"](F[te]), F[cond != "control"], idx[te], idx[cond != "control"], m["diag"]


# ------------------------------------------------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True, choices=sorted(COUPLING | MAPS | WEIGHTS))
    p.add_argument("--source", required=True); p.add_argument("--target", required=True); p.add_argument("--truth", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--meta", default="{}")
    p.add_argument("--seed", type=int, default=20261118)
    p.add_argument("--cfm-steps", type=int, default=10000)
    p.add_argument("--cellot-iters", type=int, default=100000)
    p.add_argument("--ae-iters", type=int, default=250000)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    base = json.loads(a.meta) | {"method": a.method}
    t0 = time.time(); rows, trans = [], []
    try:
        A, n, g, rej, nc, nf = load(a)
        lc = np.log10(np.maximum(nc, 1))
        if a.method in COUPLING:
            if a.method == "wot":
                P, extra = run_wot(A, a.seed)
            elif a.method.startswith("moscot"):
                P, extra = run_moscot(A, a.method)
            else:
                P, extra = run_cinemaot(A, n, a.method)
            row, trans = coupling_metrics(P, n, g, rej, nc, nf)
            rows.append(base | {"status": "ok", "kind": "coupling"} | row | extra)
        elif a.method in WEIGHTS:
            rows.append(base | {"status": "ok", "kind": "weights"} | run_cinemaotw(A, n, rej, a.seed))
        else:
            if a.method == "cellot":
                Xs, Ts, Xt, si, ti, extra = run_cellot(A, n, (a.out / "work").resolve(), a.cellot_iters, a.ae_iters)
            else:
                Xs, Ts, Xt, extra = run_cfm(A, n, a.method, a.seed, a.cfm_steps); si, ti = np.arange(n), np.arange(n, A.n_obs)
            ms = Xs + (Xt.mean(0) - Xs.mean(0))
            for name, pred in ((a.method, Ts), ("baseline_no_transport", Xs), ("baseline_mean_shift", ms)):
                rows.append(base | {"status": "ok", "kind": "map", "solver": name, "n_eval_source": len(si)} |
                            map_metrics(Xs, pred, Xt, g[si], g[ti], lc[si], lc[ti]) | (extra if name == a.method else {}))
    except Exception:
        rows.append(base | {"status": "failed"})
        (a.out / "error.txt").write_text(traceback.format_exc())
    pd.DataFrame(rows).to_csv(a.out / "metrics.tsv", sep="\t", index=False)
    if trans:
        pd.DataFrame([base | t for t in trans]).to_csv(a.out / "transitions.tsv", sep="\t", index=False)
    (a.out / "run.json").write_text(json.dumps({"seconds": round(time.time() - t0, 1), "python": sys.executable, **base}))
    print(a.method, rows[-1].get("status"), f"{time.time() - t0:.0f}s")
    sys.exit(0 if rows and rows[-1].get("status") == "ok" else 1)


if __name__ == "__main__":
    main()
