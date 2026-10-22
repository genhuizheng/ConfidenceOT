"""Effective coupling, matching weights and mass decomposition, written independently of classify_subclones.py.

Definitions (AUDIT_REPORT.md §B):
  P            the saved coupling (ConfidenceOTResult.coupling): the full transport plan of the terminal
               fixed-gate UOT solve on the filtered cost; gated rows / columns carry mass at the constant cost c
  r^S, r^T     the final gates of cell_confidence.csv (retained = 1); step-1-excluded cells get 0
  P~ = P * r^S r^T^T        effective coupling (the block ConfidenceOT asserts)
  q_ij = P~_ij / sum_k P~_kj    for a retained target with positive effective mass, else 0
  S_gj = sum_{i in g} q_ij
  M_g  = sum_{i in g} sum_j P~_ij       effective retained transport mass of group g
  pi_g = M_g / sum_g' M_g'              its share (= S_gj of the independent coupling of P~)
Cell counts, transport masses and matching weights are kept in separate columns everywhere.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def effective(P, rho, tau):
    P = P.toarray() if sparse.issparse(P) else np.asarray(P, dtype=float)
    return P * np.asarray(rho, float)[:, None] * np.asarray(tau, float)[None, :]


def matching(Pt, tau):
    """q (n x m): column-normalised effective coupling; columns of rejected or zero-mass targets are 0."""
    col = Pt.sum(axis=0)
    ok = np.asarray(tau, bool) & (col > 0)
    q = np.zeros_like(Pt)
    q[:, ok] = Pt[:, ok] / col[ok]
    return q, ok


def group_sum(values_by_row, labels, groups):
    """Sum rows by mutually exclusive labels; returns (len(groups) x columns). Every row must have one label."""
    labels = np.asarray(labels, dtype=object)
    missing = set(labels) - set(groups)
    if missing:
        raise ValueError(f"rows with a label outside the group list: {sorted(missing)[:5]}")
    index = {g: k for k, g in enumerate(groups)}
    onehot = sparse.csr_matrix((np.ones(len(labels)), ([index[x] for x in labels], np.arange(len(labels)))),
                               shape=(len(groups), len(labels)))
    return np.asarray(onehot @ values_by_row)


def check_unique(ids, what):
    ids = np.asarray(ids, dtype=object)
    dup = len(ids) - len(set(ids))
    if dup:
        raise ValueError(f"{what}: {dup} duplicated cell ids")


def rank_one_residual(block):
    """max |log B - (r_i + c_j)|: 0 for an exact rank-one positive block (the gated block of the filtered cost)."""
    if block.shape[0] < 2 or block.shape[1] < 2 or (block <= 0).any():
        return float("nan")
    L = np.log(block)
    R = L - L.mean(axis=1, keepdims=True) - L.mean(axis=0, keepdims=True) + L.mean()
    return float(np.abs(R).max())


def enrichment(S_mean, pi, pi_min=0.01, cells_min=5, retained_cells=None):
    """E = S / pi, undefined (nan) where pi < pi_min or the group has < cells_min retained source cells."""
    pi = np.asarray(pi, float)
    keep = pi >= pi_min
    if retained_cells is not None:
        keep &= np.asarray(retained_cells) >= cells_min
    out = np.full_like(np.asarray(S_mean, float), np.nan)
    out[..., keep] = np.asarray(S_mean, float)[..., keep] / pi[keep]
    return out
