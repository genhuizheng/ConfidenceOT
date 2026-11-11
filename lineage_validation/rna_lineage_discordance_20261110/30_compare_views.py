"""PLAN_v2.md section 3: RNA-OT (A) vs CRISPR-OT (B) vs the raw CRISPR matrix (C), per unit and RNA configuration.

    python lineage_validation/rna_lineage_discordance_20261110/30_compare_views.py --root results/rna_lineage_discordance_20261110 \
        --inputs results/crispr_matrix_benchmark_20261108/inputs --bundles results/crispr_matrix_benchmark_20261108/bundles \
        [--units ...] [--perm 99]

Per unit (bundle cells) and configuration (Splatter L5 best, logCPM + PCA):
  RNA similarity r   the configuration's corrected v3 coordinates (inputs_A/.../rna_coords): cosine for cos, -squared
                     Euclidean for sqeuc
  edit agreement s   C: 1 - D of 10_crispr_ot.UnitDistance (MATRIX_METRICS.md); Quinn cells of different LineageGroups
                     s = 0 (distinct integrations); pairs without a comparable intBC excluded
  thresholds         25th / 75th percentiles of r and s over 10^6 random same-lineage cell pairs of the unit (both sides)
  per cell (all cells of the unit, both sides; neighbours among all other cells of the unit, k = 15):
    LN = 15 largest s (same lineage), RN = 15 largest r
    div  = share of LN with r <= r_25 (lineage-close / RNA-far), conv = share of RN with s <= s_25 or another clone
    agree = |LN & RN| / 15; null means of div / conv over within-site permutations of the RNA coordinates
  per target, M4-E (A and B): retained in each; for targets retained in both, total-variation distance of the two
    couplings' source distributions, top-15 source overlap, and the mean s / r of each coupling's top-15 sources
  quadrant mass per retained target and coupling: lineage-close / RNA-far (s >= s_75, r <= r_25) and RNA-close /
    lineage-far (r >= r_75, s <= s_25 or other clone), against the independent coupling (retained source mass)
  gates: cross-table of A vs B per side, Cohen's kappa; AUROC of div / conv for rejected vs retained under A and B
Writes ROOT/compare/<unit>__<config>/{cells.tsv.gz, unit.json}. Refuses a finished output.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score

HERE = Path(__file__).resolve().parent
CONFIGS = {"l5best": "ranknm256-midrank|none|scale|cos|pca50", "logcpm_pca": "logcpm|none|noscale|sqeuc|pca30"}
K, SEED = 15, 20261110


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def rna_sim(A, B, cos):
    return A @ B.T if cos else -(np.sum(A * A, 1)[:, None] + np.sum(B * B, 1)[None, :] - 2 * A @ B.T)


def auc(y, x):
    ok = np.isfinite(x)
    return float(roc_auc_score(y[ok], x[ok])) if ok.sum() > 10 and 0 < y[ok].mean() < 1 else np.nan


def analyse(b, U, coords, rna_cc, rna_P, cr_gates, cr_P, obs, perm, rng):
    ns, nt = len(b["source_ids"]), len(b["target_ids"])
    ids = np.r_[b["source_ids"], b["target_ids"]]
    side = np.r_[np.repeat("source", ns), np.repeat("target", nt)]
    # RNA coordinates in bundle order
    X = coords["X"][pd.Index(coords["source_ids"].astype(str)).get_indexer(b["source_ids"])]
    Y = coords["Y"][pd.Index(coords["target_ids"].astype(str)).get_indexer(b["target_ids"])]
    Z = np.vstack([X, Y]).astype(np.float64); cos = bool(coords["cos"])
    if cos:
        Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    lin = np.r_[U.lin_s, U.lin_t].astype(object)
    rows = np.r_[np.arange(ns), np.arange(nt)]
    site = np.r_[np.repeat("source_site", ns), b["target_site"]].astype(object)
    if obs is not None and "site" in obs:
        o = obs.set_index("cell")
        st = o["sample_id"] if b["dataset"] == "KPTracer" else o["site"]
        site = pd.Series(ids).map(st).fillna("NA").to_numpy(dtype=object)
    # s blocks per lineage (all cells of the lineage, both sides), dense
    blocks = {}
    for g, (L, rs, rt) in U.groups.items():
        cell_idx = np.r_[rs, ns + rt]
        Ds = U.block("source", rs, "source", rs); Dt = U.block("target", rt, "target", rt); Dst = U.block("source", rs, "target", rt)
        S = np.block([[1 - Ds, 1 - Dst], [(1 - Dst).T, 1 - Dt]]).astype(np.float32)
        np.fill_diagonal(S, np.nan)
        blocks[g] = (cell_idx, S)
    pos = {}
    for g, (cell_idx, S) in blocks.items():
        for k, c in enumerate(cell_idx):
            pos[c] = (g, k)
    # thresholds over random same-lineage pairs
    rs_, ss_ = [], []
    for g, (cell_idx, S) in blocks.items():
        m = len(cell_idx); n_draw = min(10**6 * m // max(len(ids), 1) + 1000, m * m)
        a, c = rng.integers(m, size=n_draw), rng.integers(m, size=n_draw)
        keep = a != c
        a, c = a[keep], c[keep]
        rs_.append(np.einsum("ij,ij->i", Z[cell_idx[a]], Z[cell_idx[c]]) if cos else -np.sum((Z[cell_idx[a]] - Z[cell_idx[c]]) ** 2, 1))
        ss_.append(S[a, c])
    rall, sall = np.concatenate(rs_), np.concatenate(ss_)
    r25, r75 = np.percentile(rall, [25, 75]); s25, s75 = np.nanpercentile(sall, [25, 75])
    n = len(ids)
    # lineage neighbours (within lineage) and RNA neighbours (all cells), k = 15
    LN = np.full((n, K), -1, dtype=np.int64)
    for g, (cell_idx, S) in blocks.items():
        tie = (rng.random(S.shape, dtype=np.float32) * 1e-6).astype(np.float32)   # fixed-seed tie break
        Sv = np.where(np.isnan(S), np.float32(-np.inf), S + tie).astype(np.float32)
        del tie
        kk = min(K, len(cell_idx) - 1)
        top = np.argpartition(-Sv, kk, axis=1)[:, :kk] if kk > 0 else np.zeros((len(cell_idx), 0), int)
        LN[cell_idx, :kk] = cell_idx[top]
    RN = np.empty((n, K), dtype=np.int64)
    for k0 in range(0, n, 2000):
        sl = slice(k0, min(n, k0 + 2000))
        R = rna_sim(Z[sl], Z, cos); R[np.arange(R.shape[0]), np.arange(sl.start, sl.stop)] = -np.inf
        RN[sl] = np.argpartition(-R, K, axis=1)[:, :K]

    def s_of(i, j):
        """s for index pairs (arrays); 0 for another clone (Quinn), NaN not evaluable or no lineage."""
        out = np.full(len(i), np.nan)
        for t, (a, c) in enumerate(zip(i, j)):
            pa, pc = pos.get(a), pos.get(c)
            if pa is None or pc is None:
                continue
            out[t] = blocks[pa[0]][1][pa[1], pc[1]] if pa[0] == pc[0] else 0.0
        return out

    def r_of(i, j):
        return np.einsum("ij,ij->i", Z[i], Z[j]) if cos else -np.sum((Z[i] - Z[j]) ** 2, 1)

    has_ln = LN[:, 0] >= 0
    ii = np.repeat(np.arange(n), K)
    rLN = np.where(LN.ravel() >= 0, r_of(ii, np.maximum(LN.ravel(), 0)), np.nan).reshape(n, K)
    sRN = s_of(ii, RN.ravel()).reshape(n, K)
    div = np.where(has_ln, np.nanmean(rLN <= r25, 1), np.nan)
    conv = np.nanmean(np.where(np.isnan(sRN), np.nan, sRN <= s25), 1)
    agree = np.array([len(set(LN[c][LN[c] >= 0]) & set(RN[c])) / K for c in range(n)])
    # within-site permutation null of the RNA coordinates
    dnull, cnull = np.zeros(n), np.zeros(n)
    groups_site = pd.Series(np.arange(n)).groupby(site).indices
    for _ in range(perm):
        pi = np.arange(n)
        for idx in groups_site.values():
            pi[idx] = rng.permutation(idx)
        inv = np.empty(n, dtype=np.int64); inv[pi] = np.arange(n)
        rp = np.where(LN.ravel() >= 0, r_of(pi[ii], pi[np.maximum(LN.ravel(), 0)]), np.nan).reshape(n, K)
        dnull += np.where(has_ln, np.nanmean(rp <= r25, 1), np.nan)
        RNp = inv[RN[pi]]
        sp_ = s_of(ii, RNp.ravel()).reshape(n, K)
        cnull += np.nanmean(np.where(np.isnan(sp_), np.nan, sp_ <= s25), 1)
    dnull /= max(perm, 1); cnull /= max(perm, 1)
    cells = pd.DataFrame({"cell": ids, "side": side, "site": site, "lineage": lin, "div": div, "conv": conv, "agree": agree,
                          "div_null": dnull, "conv_null": cnull})
    # gates (M4-E)
    g_r = rna_cc[rna_cc.method == "M4-E"]
    for sd, sid in (("source", b["source_ids"]), ("target", b["target_ids"])):
        gg = g_r[g_r.side == sd].set_index(g_r[g_r.side == sd].observation_id.astype(str))
        cells.loc[cells.side == sd, "rna_retained"] = ~gg.rejected.reindex(sid).to_numpy(bool)
        gc = cr_gates[(cr_gates.method == "M4-E") & (cr_gates.side == sd)].set_index("observation_id")
        cells.loc[cells.side == sd, "crispr_retained"] = gc.retained.reindex(sid).to_numpy(bool)
    cells["rna_retained"] = cells.rna_retained.astype(bool); cells["crispr_retained"] = cells.crispr_retained.astype(bool)
    # couplings on retained blocks
    out = {}
    rsR, rtR = cells.rna_retained.to_numpy()[:ns], cells.rna_retained.to_numpy()[ns:]
    rsC, rtC = cells.crispr_retained.to_numpy()[:ns], cells.crispr_retained.to_numpy()[ns:]

    def qcols(P, rs, rt):
        Q = P.astype(np.float64); Q[~rs, :] = 0; Q[:, ~rt] = 0
        col = Q.sum(0); return Q / np.where(col > 0, col, np.nan)[None, :], Q.sum(1)

    qR, mR = qcols(rna_P, rsR, rtR); qC, mC = qcols(cr_P, rsC, rtC)
    both = rtR & rtC & np.isfinite(qR).all(0) & np.isfinite(qC).all(0)
    tv = np.full(nt, np.nan); ov = np.full(nt, np.nan)
    sR = np.full(nt, np.nan); sC = np.full(nt, np.nan); rR = np.full(nt, np.nan); rC = np.full(nt, np.nan)
    quad = {k: np.full(nt, np.nan) for k in ("lcrf_rna", "lcrf_cr", "lcrf_ind_rna", "rclf_rna", "rclf_cr", "rclf_ind_rna")}
    piR = mR / mR.sum() if mR.sum() > 0 else mR
    for j in np.flatnonzero(rtR | rtC):
        tj = ns + j
        s_col = s_of(np.full(ns, tj), np.arange(ns)); r_col = r_of(np.full(ns, tj), np.arange(ns))
        lcrf = (s_col >= s75) & (r_col <= r25); rclf = (r_col >= r75) & ((s_col <= s25))
        if rtR[j] and np.isfinite(qR[:, j]).all():
            quad["lcrf_rna"][j] = qR[lcrf, j].sum(); quad["rclf_rna"][j] = qR[rclf, j].sum()
            quad["lcrf_ind_rna"][j] = piR[lcrf].sum(); quad["rclf_ind_rna"][j] = piR[rclf].sum()
        if rtC[j] and np.isfinite(qC[:, j]).all():
            quad["lcrf_cr"][j] = qC[lcrf, j].sum(); quad["rclf_cr"][j] = qC[rclf, j].sum()
        if both[j]:
            tv[j] = 0.5 * np.abs(qR[:, j] - qC[:, j]).sum()
            a_, c_ = np.argsort(-qR[:, j])[:K], np.argsort(-qC[:, j])[:K]
            ov[j] = len(set(a_) & set(c_)) / K
            sR[j], sC[j] = np.nanmean(s_col[a_]), np.nanmean(s_col[c_]); rR[j], rC[j] = r_col[a_].mean(), r_col[c_].mean()
    t = cells.side == "target"
    for k, v in {"tv": tv, "top15_overlap": ov, "s_top_rnaot": sR, "s_top_crisprot": sC, "r_top_rnaot": rR, "r_top_crisprot": rC, **quad}.items():
        cells.loc[t, k] = v
    for sd in ("source", "target"):
        c = cells[cells.side == sd]
        out[f"kappa_{sd}"] = float(cohen_kappa_score(c.rna_retained, c.crispr_retained)) if c.rna_retained.nunique() > 1 or c.crispr_retained.nunique() > 1 else np.nan
        out[f"crosstab_{sd}"] = pd.crosstab(c.rna_retained, c.crispr_retained).to_dict()
        for view in ("rna", "crispr"):
            y = ~c[f"{view}_retained"].to_numpy()
            out[f"auc_div_{view}_rejected_{sd}"] = auc(y, c["div"].to_numpy()); out[f"auc_conv_{view}_rejected_{sd}"] = auc(y, c["conv"].to_numpy())
    out.update(r25=float(r25), r75=float(r75), s25=float(s25), s75=float(s75), lineages=len(blocks), cells=n)
    return cells, out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--bundles", type=Path, required=True)
    p.add_argument("--units", nargs="*")
    p.add_argument("--perm", type=int, default=99)
    a = p.parse_args()
    cot = load("cot", HERE / "10_crispr_ot.py")
    bb = cot.bundles_module(); tabs = bb.load_tables(a.inputs)
    A = a.root / "inputs_A"
    for f in sorted(glob.glob(str(a.bundles / "*.pkl")), key=lambda f: Path(f).stat().st_size):
        b = pickle.load(open(f, "rb")); u = b["unit_id"]
        if a.units and u not in a.units:
            continue
        crd = a.root / "crispr_ot" / u
        if not (crd / "audit.json").exists():
            print(f"{u}: no CRISPR-OT result yet, skipped"); continue
        key = "KPTracer" if b["dataset"] == "KPTracer" else f"Quinn2021:{b['mouse']}"
        t, prior = tabs[key]
        U = None
        obs_f = A / "rna_lineage_discordance_20261110" / "rna_coords" / u / "obs.tsv.gz"
        obs = pd.read_csv(obs_f, sep="\t", dtype={"cell": str}) if obs_f.exists() else None
        for tag, cid in CONFIGS.items():
            o = a.root / "compare" / f"{u}__{tag}"
            if (o / "unit.json").exists():
                continue
            t0 = time.perf_counter()
            if U is None:
                U = cot.UnitDistance(b, t, prior, bb.map_ids)
            kd = cid.split("|")[0] + "__none"
            coords = np.load(A / "rna_lineage_discordance_20261110" / "rna_coords" / u / f"{cid.replace('|', '__')}.npz")
            row = A / "v3_grid_20261104" / "kp" / "rows" / u / kd / cid.replace("|", "__")
            z = np.load(row / "coupling_m4e.npz")
            P = z["coupling"][np.ix_(pd.Index(z["source_ids"].astype(str)).get_indexer(b["source_ids"]),
                                     pd.Index(z["target_ids"].astype(str)).get_indexer(b["target_ids"]))]
            cc = pd.read_csv(row / "cell_confidence.csv.gz", usecols=["method", "side", "observation_id", "rejected"])
            zc = np.load(crd / "coupling_m4e.npz"); Pc = zc["coupling"]
            gates = pd.read_csv(crd / "gates.tsv.gz", sep="\t", dtype={"observation_id": str})
            rng = np.random.default_rng(SEED)
            cells, out = analyse(b, U, coords, cc, P, gates, Pc, obs, a.perm, rng)
            if obs is not None:
                oo = obs.set_index("cell")
                for col in ("total_counts_raw", "detected_genes", "cell_type"):
                    if col in oo:
                        cells[col] = cells.cell.map(oo[col])
            o.mkdir(parents=True, exist_ok=True)
            cells.to_csv(o / "cells.tsv.gz", sep="\t", index=False, compression="gzip")
            out.update(unit_id=u, config=cid, dataset=b["dataset"], mouse=b["mouse"], seconds=round(time.perf_counter() - t0, 1))
            (o / "unit.json").write_text(json.dumps(out, indent=1, default=str))
            print(f"{u} {tag}: {len(cells)} cells, kappa {out['kappa_source']:.2f}/{out['kappa_target']:.2f}, {out['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
