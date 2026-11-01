"""Shared helpers of the real-data benchmark (versioned extension; the library and the frozen runs are not changed).

Rank encoders with an explicit tie rule, beside the production one (confidenceot.preprocessing.rank_value_encode,
which breaks ties by numpy's stable argsort over each row's entries in their STORED order):
  production  the production rule exactly: the same sparse products as rank_value_encode, so ties keep the storage
              order those products leave. Measured 2026-10-31 (scipy 1.15.1): sparse.diags(...) @ csr returns every
              row in reverse column order, so for rank_no_median (no further product) ties break in REVERSE
              alphabetical gene order; for rank_value the extra product with the gene-median diagonal leaves rows
              ascending, so ties break alphabetically. Checked bit for bit against rank_value_encode;
  alphabetical ties in ascending column order (gene keys sorted as strings), another arbitrary order;
  random    ties broken uniformly at random (seeded);
  expected  the encoding averaged over random tie-breaking: every gene of a tie group receives the mean of the
            values of the slots the group occupies, and a group straddling the top-N cut shares the slots it is
            competing for, each member receiving (mean slot value) x (slots / group size). Deterministic, and does
            not depend on gene order.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

LTL_LABEL_INPUT = {True: ("ds", "pair_manifest_downsampled.csv"), False: ("rawqc", "pair_manifest_rawqc.csv")}
SEED = 20260831


def local(path: str, run_root: Path) -> Path:
    return run_root / "inputs" / str(path).replace("\\", "/").split("/inputs/")[1]


def ltl_pairs(run_root: Path, equalised: bool) -> pd.DataFrame:
    sub, name = LTL_LABEL_INPUT[equalised]
    m = pd.read_csv(run_root / "inputs" / sub / name)
    return m[m.eligible.astype(bool)].reset_index(drop=True)


def cell_types(converted: Path) -> pd.Series:
    return pd.concat([ad.read_h5ad(f, backed="r").obs["coarse_celltype"].astype(str) for f in sorted((converted / "h5ad").glob("*.h5ad"))])


def common_genes(source: ad.AnnData, target: ad.AnnData):
    """Columns as prepare_joint_representation intersects them: first occurrence, keys sorted as strings."""
    sf, tf = {}, {}
    for i, k in enumerate(source.var_names.astype(str)):
        sf.setdefault(k, i)
    for i, k in enumerate(target.var_names.astype(str)):
        tf.setdefault(k, i)
    common = sorted(set(sf) & set(tf))
    return common, [sf[k] for k in common], [tf[k] for k in common]


def scaled_for_rank(joint: sp.csr_matrix, gene_median: bool, sort: bool) -> sp.csr_matrix:
    """The products rank_value_encode takes, in the same order; sort=True puts each row in ascending column order."""
    totals = np.asarray(joint.sum(axis=1)).ravel()
    s = sp.diags(1.0 / np.maximum(totals, 1e-12)) @ sp.csr_matrix(joint)
    if gene_median:
        cols = s.tocsc(); med = np.ones(cols.shape[1])
        for g in range(cols.shape[1]):
            a, b = cols.indptr[g], cols.indptr[g + 1]
            if b > a:
                med[g] = float(np.median(cols.data[a:b]))
        med = np.where(med > 0, med, 1.0)
        s = (s @ sp.diags(1.0 / med)).tocsr()
    else:
        s = s.tocsr()
    if sort:
        s = s.copy(); s.sort_indices()
    return s


def rank_encode(joint: sp.csr_matrix, top_n: int, gene_median: bool, rule: str = "production", seed: int = 0):
    """Rank encoding with an explicit tie rule (production / alphabetical / random / expected); also per-cell tie
    statistics at the top-N cut."""
    if rule not in ("production", "alphabetical", "random", "expected"):
        raise ValueError(rule)
    s = scaled_for_rank(joint, gene_median, sort=rule == "alphabetical")
    rng = np.random.default_rng(seed)
    rows, cols, vals = [], [], []
    slot = 1.0 - np.arange(top_n, dtype=np.float64) / float(top_n)
    tied_slots = np.zeros(s.shape[0]); cut_group = np.zeros(s.shape[0]); cut_value = np.full(s.shape[0], np.nan)
    for i in range(s.shape[0]):
        a, b = s.indptr[i], s.indptr[i + 1]
        if b == a:
            continue
        d, g = s.data[a:b], s.indices[a:b]
        keep = min(top_n, d.size)
        if rule == "random":
            order = np.lexsort((rng.random(d.size), -d))
        else:
            order = np.argsort(-d, kind="stable")
        dv = d[order]
        if d.size > keep:  # tie statistics at the cut
            v = dv[keep - 1]; above = int((dv > v).sum()); grp = int((dv == v).sum())
            if grp > keep - above:
                tied_slots[i] = keep - above; cut_group[i] = grp; cut_value[i] = v
        if rule in ("production", "alphabetical", "random"):
            sel = order[:keep]
            rows.append(np.full(keep, i)); cols.append(g[sel]); vals.append(slot[:keep])
            continue
        # expected over random tie-breaking
        _, start, counts = np.unique(-dv, return_index=True, return_counts=True)
        for st, n in zip(start, counts):
            if st >= keep:
                break
            members = g[order[st:st + n]]
            taken = min(n, keep - st)
            value = slot[st:st + taken].mean() * taken / n
            rows.append(np.full(n, i)); cols.append(members); vals.append(np.full(n, value))
    enc = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=s.shape)
    stats = pd.DataFrame({"tied_slots_at_cut": tied_slots, "cut_tie_group_size": cut_group, "cut_value": cut_value,
                          "detected": np.diff(s.indptr)})
    return enc, stats


def top_variance(joint, limit: int) -> np.ndarray:
    if sp.issparse(joint):
        mean = np.asarray(joint.mean(axis=0)).ravel(); mean2 = np.asarray(joint.multiply(joint).mean(axis=0)).ravel()
        v = mean2 - mean * mean
    else:
        v = np.asarray(joint).var(axis=0)
    return np.argsort(-v, kind="stable")[:limit]


def pca_coordinates(encoded: sp.csr_matrix, n_hvg: int, n_pcs: int, scale_genes: bool, cosine: bool, seed: int):
    """The representation path after the row transform, as Preprocessing.representation runs it."""
    from sklearn.decomposition import PCA
    from confidenceot.preprocessing import unit_rows
    sel = top_variance(encoded, min(n_hvg, encoded.shape[1]))
    dense = encoded[:, sel].toarray().astype(np.float32)
    if scale_genes:
        dense -= dense.mean(axis=0); sd = dense.std(axis=0); dense /= np.where(sd > 1e-8, sd, 1.0)
    coords = PCA(n_components=min(n_pcs, dense.shape[0] - 1, dense.shape[1]), random_state=seed).fit_transform(dense)
    return (unit_rows(coords) if cosine else coords), sel


def knn_mean(C: np.ndarray, k: int) -> np.ndarray:
    k = min(k, C.shape[1])
    return np.partition(C, k - 1, axis=1)[:, :k].mean(1)


def subspace_similarity(A: np.ndarray, B: np.ndarray) -> float:
    """Mean squared cosine of the principal angles between the column spaces of two loading / score matrices."""
    qa, _ = np.linalg.qr(A); qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return float(np.mean(s ** 2))


def kappa(x, y) -> float:
    x, y = np.asarray(x, int), np.asarray(y, int)
    po = np.mean(x == y); pe = x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else np.nan
