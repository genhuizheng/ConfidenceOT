"""Shared helpers of the real-data benchmark (versioned extension; the library and the frozen runs are not changed).

Rank encoders with an explicit tie rule, beside the production one (confidenceot.preprocessing.rank_value_encode,
which breaks ties by numpy's stable argsort over each row's entries in their STORED order):
  production    the production rule exactly. The stored order is implementation-defined: Preprocessing.representation
                builds csr_matrix(counts, dtype=float64), then indexes columns into the sorted gene-key order; scipy's
                fancy column indexing leaves each row in a permutation that is neither column nor key order (measured
                2026-10-31, scipy 1.15.1: row 0 of the 16-week cells starts 15705, 8325, 24732, ...), and both
                rank_no_median and rank_value inherit it (on the equalised 16-week cells the alphabetical rule differs
                from production for both). joint_counts() rebuilds the joint exactly that way, so rank_encode(...,
                'production') equals rank_value_encode bit for bit. (An earlier note in this file and in commit 55616d0
                said "reverse alphabetical"; that came from a joint matrix built in sorted order, not production's.)
  alphabetical  ties in ascending gene-key order (keys sorted as strings): another arbitrary but documented order;
  random        ties broken uniformly at random (seeded);
  expected      the encoding averaged over random tie-breaking: every gene of a tie group receives the mean of the
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
    """Columns as prepare_joint_representation intersects them: common.gene_keys (gene_id, else gene_symbol, else
    var_names), first occurrence, keys sorted as strings. (Before 2026-10-31 15:00 this used var_names, 28 genes
    wider than production on the original-count files, whose repeated symbols carry '-1' names.)"""
    from common import gene_keys
    sf, tf = {}, {}
    for i, k in enumerate(gene_keys(source)):
        sf.setdefault(k, i)
    for i, k in enumerate(gene_keys(target)):
        tf.setdefault(k, i)
    common = sorted(set(sf) & set(tf))
    return common, [sf[k] for k in common], [tf[k] for k in common]


def joint_counts(source: ad.AnnData, target: ad.AnnData) -> sp.csr_matrix:
    """Both sides stacked on the common genes, built exactly as Preprocessing.representation builds them (float64 csr
    first, then the column index), so the stored row order, which decides production ties, is production's."""
    from common import expression_matrix
    _, si, ti = common_genes(source, target)
    sx = sp.csr_matrix(expression_matrix(source), dtype=np.float64)[:, si]
    tx = sp.csr_matrix(expression_matrix(target), dtype=np.float64)[:, ti]
    return sp.vstack([sx, tx], format="csr")


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


# ---------------------------------------------------------------------------------------------------------------
# Registered configurations (JOINT_BENCHMARK_PROTOCOL.md section 2)

DS_QUANTILES = {"P10": 0.10, "P25": 0.25, "P50": 0.50}


def ds_targets(depth: pd.Series) -> dict:
    """Pooled quantile targets of the analysed cells' pre-equalisation depth, as 27_downsample_counts.py computes P10."""
    return {k: int(np.quantile(depth.to_numpy(), q)) for k, q in DS_QUANTILES.items()}


def equalised(data: ad.AnnData, target: int, seed: int) -> ad.AnnData:
    from confidenceot import equalise_depth
    out = data.copy()
    X = sp.csr_matrix(out.X).astype(np.int64)
    out.X = sp.csr_matrix(equalise_depth(X, rng=np.random.default_rng(seed), target=target)).astype(np.float32)
    return out


def production_label(enc: str, K: int, scale: bool, ds: bool, cost: str) -> str | None:
    if enc not in ("logcpm", "ranknm"):
        return None
    stem = "logcpm" if enc == "logcpm" else f"ranknm{K}"
    return stem + ("" if scale else "_noscale") + ("_ds" if ds else "") + ("_cos" if cost == "cos" else "")


def rankhvg_encode(joint_counts: sp.csr_matrix, H: int, rule: str):
    """Rank-HVG: H genes of highest variance of log1p(1e4 x count / total) over both sides; within them each cell's
    expressed genes ranked by count (descending), value 1 - (r - 1) / H, zeros stay 0. rule 'midrank': ties share
    their average rank; 'order': ties keep column order (keys sorted as strings), to measure gene-order sensitivity."""
    totals = np.asarray(joint_counts.sum(axis=1)).ravel()
    ln = sp.csr_matrix(sp.diags(1e4 / np.maximum(totals, 1.0)) @ joint_counts); ln.data = np.log1p(ln.data)
    hvg = top_variance(ln, min(H, joint_counts.shape[1]))
    sub = sp.csr_matrix(joint_counts[:, hvg]); sub.sort_indices()
    vals = np.empty_like(sub.data, dtype=np.float64)
    for i in range(sub.shape[0]):
        a, b = sub.indptr[i], sub.indptr[i + 1]
        if b == a:
            continue
        d = sub.data[a:b]
        if rule == "midrank":
            r = pd.Series(-d).rank(method="average").to_numpy()
        elif rule == "order":
            r = np.empty(d.size); r[np.argsort(-d, kind="stable")] = np.arange(1, d.size + 1)
        else:
            raise ValueError(rule)
        vals[a:b] = 1.0 - (r - 1.0) / H
    enc = sp.csr_matrix((vals, sub.indices.copy(), sub.indptr.copy()), shape=sub.shape)
    return enc, hvg


def build_config(spec: dict, source: ad.AnnData, target: ad.AnnData, seed: int):
    """Coordinates of one registered configuration on (already depth-handled) source / target AnnData.
    Production-equivalent configurations go through the production code by their label; the others through the
    encoders above, with the production downstream (rank-K: 2,000 top-variance encoded genes; rank-HVG: all H genes;
    no gene scaling unless spec says so; PCA 30; cosine or squared Euclidean)."""
    from common import prepare_joint_representation
    from confidenceot import Preprocessing
    enc, K, H, tie, scale, cost = spec["encoding"], spec.get("K"), spec.get("H"), spec.get("tie"), spec["scale"], spec["cost"]
    ds = spec["ds"] != "none"
    label = spec.get("label") if isinstance(spec.get("label"), str) and spec.get("label") else None
    if label is None and (enc == "logcpm" or tie == "production"):
        label = production_label(enc, int(K) if K == K and K is not None else 0, scale, ds, cost)
    if label is not None:
        c = Preprocessing.from_label(label)
        X, Y, _, prov = prepare_joint_representation(
            source, target, n_hvg=c.n_hvg, n_pcs=c.n_pcs, seed=seed, representation=c.normalisation, rank_top_n=c.rank_top_n,
            cost=c.cost, equalise_depth=c.equalise_depth, regress_out=c.regress_out, regress_on_ranks=c.regress_on_ranks,
            scale_genes=c.scale_genes, label_stem=c.label_stem)
        return np.asarray(X, np.float32), np.asarray(Y, np.float32), {"path": f"production label {label}", "hvg_n": prov.get("hvg_n")}
    joint = joint_counts(source, target)
    if enc == "ranknm":
        rule, tseed = ("random", int(tie[6:])) if str(tie).startswith("random") else (tie, 0)
        e, _ = rank_encode(joint, int(K), False, rule, tseed)
        coords, sel = pca_coordinates(e, 2000, 30, scale, cost == "cos", seed)
    elif enc == "rankhvg":
        e, _ = rankhvg_encode(joint, int(H), tie)
        coords, sel = pca_coordinates(e, int(H), 30, scale, cost == "cos", seed)
    else:
        raise ValueError(enc)
    n = source.n_obs
    return coords[:n].astype(np.float32), coords[n:].astype(np.float32), {"path": f"benchmark encoder {enc} tie {tie}", "hvg_n": len(sel)}
