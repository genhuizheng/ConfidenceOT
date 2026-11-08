"""Shared code of the matrix-based CRISPR validation (MATRIX_METRICS.md, fixed 2061c92): allele-table states, empirical
indel priors, founder characters, intBC events, the cell-pair edit agreement s_ij, the UPGMA partition and the
per-coupling scores. Every rule here is the one written in MATRIX_METRICS.md; nothing is tuned on a score."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import scipy.sparse as sp

SITES = ("r1", "r2", "r3")
FOUNDER = 0.9
MIN_CLASS, MIN_SIDE = 20, 5          # L2 kept class: >= 20 cells, >= 5 on each side
MIN_PI, MIN_SRC = 0.01, 5            # L2 candidate: pi_g >= 0.01 and >= 5 source cells


# ---------------------------------------------------------------- allele tables
def read_allele_table(path, dataset: str) -> pd.DataFrame:
    """cell, intBC, r1-r3 (full strings, uncut normalised to 'None'), lineage."""
    t = pd.read_csv(path, sep="\t", index_col=0, dtype=str, keep_default_na=False)
    lin = "MetFamily" if dataset == "KPTracer" else "LineageGroup"
    out = t[["cellBC", "intBC", *SITES, lin]].rename(columns={"cellBC": "cell", lin: "lineage"}).copy()
    for s in SITES:
        out[s] = out[s].where(~out[s].str.contains("None", regex=False), "None")
        if (out[s] == "").any():
            raise SystemExit(f"{path}: empty state at {s}")
    if out.duplicated(["cell", "intBC"]).any():
        raise SystemExit(f"{path}: duplicated cell x intBC")
    if dataset != "KPTracer":
        out["lineage"] = "LG" + out["lineage"].astype(str)
    return out


def indel_priors(t: pd.DataFrame) -> pd.Series:
    """Jones et al. 2020 (Cassiopeia compute_empirical_indel_priors, grouping by intBC): q(s) = number of intBCs in
    which indel string s occurs at any cut site / number of intBCs of the dataset."""
    long = t.melt(id_vars=["intBC"], value_vars=list(SITES), value_name="state")
    long = long[long.state != "None"].drop_duplicates(["intBC", "state"])
    return long.groupby("state").size() / t["intBC"].nunique()


# ---------------------------------------------------------------- one lineage of one unit
class Lineage:
    """The cells of one lineage in one unit (both sides), with events, observed intBCs, weights and L2 classes."""

    def __init__(self, t: pd.DataFrame, cells: np.ndarray, side: np.ndarray, prior: pd.Series):
        self.cells = np.asarray(cells, dtype=object); self.side = np.asarray(side, dtype=object)
        idx = pd.Index(self.cells)
        t = t[t.cell.isin(idx)].copy()
        t["row"] = idx.get_indexer(t.cell)
        self.intbcs = np.array(sorted(t.intBC.unique()), dtype=object)
        bi = pd.Index(self.intbcs); t["b"] = bi.get_indexer(t.intBC)
        n, nb = len(self.cells), len(self.intbcs)
        self.observed = sp.csr_matrix((np.ones(len(t), bool), (t.row, t.b)), shape=(n, nb))
        # founder characters: one string carried by >= 90% of the lineage cells observed at (intBC, site)
        founder = set()
        for (b, s), g in t.melt(id_vars=["b"], value_vars=list(SITES), var_name="site", value_name="state").groupby(["b", "site"]):
            vc = g.state.value_counts()
            if vc.index[0] != "None" and vc.iloc[0] >= FOUNDER * len(g):
                founder.add((b, vc.index[0]))
        self.founder_characters = len(founder)
        ev = t.melt(id_vars=["row", "b"], value_vars=list(SITES), value_name="state")
        ev = ev[ev.state != "None"].drop_duplicates(["row", "b", "state"])          # one event per intBC and string
        ev = ev[~pd.Series(list(zip(ev.b, ev.state)), index=ev.index).isin(founder)]
        keys = ev.b.astype(str) + "|" + ev.state
        self.events = np.array(sorted(keys.unique()), dtype=object)
        ei = pd.Index(self.events); ev["e"] = ei.get_indexer(keys)
        self.event_intbc = np.array([int(k.split("|", 1)[0]) for k in self.events], dtype=np.int64)
        st = np.array([k.split("|", 1)[1] for k in self.events], dtype=object)
        q = prior.reindex(st).to_numpy(float)
        if np.isnan(q).any():
            raise SystemExit("an event has no prior")
        self.weight = -np.log(q)
        self.incidence = sp.csr_matrix((np.ones(len(ev)), (ev.row, ev.e)), shape=(n, len(self.events)))
        self.intbc_weight = (self.incidence.multiply(self.weight[None, :])).tocsr() @ \
            sp.csr_matrix((np.ones(len(self.events)), (np.arange(len(self.events)), self.event_intbc)), shape=(len(self.events), nb))
        # L2 classes: the intBC allele (3-site tuple) of each observed cell
        t["allele"] = t[list(SITES)].agg("|".join, axis=1)
        nonf = ev.groupby(["row", "b"]).size()
        t["has_event"] = pd.Series(list(zip(t.row, t.b)), index=t.index).map(lambda k: k in nonf.index)
        self.partitions = {}
        for b, g in t.groupby("b"):
            keep = []
            for al, h in g.groupby("allele"):
                ns, nt = int((self.side[h.row] == "source").sum()), int((self.side[h.row] == "target").sum())
                if len(h) >= MIN_CLASS and ns >= MIN_SIDE and nt >= MIN_SIDE and h.has_event.any():
                    keep.append(al)
            if len(keep) >= 2:
                lab = np.full(n, -1, dtype=np.int64)
                for k, al in enumerate(sorted(keep)):
                    lab[g.row[g.allele == al].to_numpy()] = k
                self.partitions[f"intbc:{self.intbcs[b]}"] = lab

    def agreement(self, rows_a: np.ndarray, rows_b: np.ndarray, block: int = 4096):
        """s (float32) and evaluable (bool) for every pair rows_a x rows_b (MATRIX_METRICS.md 2(d))."""
        Ea = self.incidence[rows_a].multiply(self.weight[None, :]).tocsr(); Eb = self.incidence[rows_b].tocsr()
        Wa, Wb = self.intbc_weight[rows_a].toarray(), self.intbc_weight[rows_b].toarray()
        Oa, Ob = self.observed[rows_a].toarray().astype(np.float64), self.observed[rows_b].toarray().astype(np.float64)
        s = np.zeros((len(rows_a), len(rows_b)), dtype=np.float32); ok = np.zeros(s.shape, dtype=bool)
        for k in range(0, len(rows_b), block):
            sl = slice(k, k + block)
            A = (Ea @ Eb[sl].T).toarray()
            m = Wa @ Ob[sl].T + Oa @ Wb[sl].T - A
            good = m > 1e-12
            ok[:, sl] = good
            s[:, sl] = np.where(good, A / np.where(good, m, 1.0), 0.0)
        return s, ok

    def upgma_partition(self) -> np.ndarray:
        """Average linkage on 1 - s over all cells (both sides); pairs that are not evaluable get the median distance
        for the clustering only; cut at the first split from the root with >= 2 clusters of >= 20 cells and >= 5 cells
        on each side (SUBCLONE_RULES.md section 4 applied to this dendrogram). -1 = not in a kept cluster."""
        from scipy.cluster.hierarchy import linkage, to_tree
        n = len(self.cells)
        lab = np.full(n, -1, dtype=np.int64)
        if n < 2 * MIN_CLASS:
            return lab
        rows = np.arange(n)
        s, ok = self.agreement(rows, rows)
        d = (1.0 - s).astype(np.float64)
        iu = np.triu_indices(n, 1)
        dv, okv = d[iu], ok[iu]
        del s, ok, d
        med = float(np.median(dv[okv])) if okv.any() else 1.0
        dv[~okv] = med
        root = to_tree(linkage(dv, method="average"))
        del dv
        is_src = self.side == "source"

        def major(node):
            ids = node.pre_order()
            ns = int(is_src[ids].sum()); nt = len(ids) - ns
            return len(ids) >= MIN_CLASS and ns >= MIN_SIDE and nt >= MIN_SIDE

        node = root
        while not node.is_leaf():
            kids = [c for c in (node.get_left(), node.get_right()) if major(c)]
            if len(kids) == 2:
                for k, c in enumerate(kids):
                    lab[c.pre_order()] = k
                return lab
            if len(kids) == 1:
                node = kids[0]; continue
            return lab
        return lab


# ---------------------------------------------------------------- scores of one coupling
def retained_block(P: np.ndarray, rs: np.ndarray, rt: np.ndarray):
    Q = P.astype(np.float64, copy=True)
    Q[~rs, :] = 0.0; Q[:, ~rt] = 0.0
    m = Q.sum(1); col = Q.sum(0)
    return Q, m, col


def l2_partition(Q, m, col, src_lab, tgt_lab, rt):
    """Balanced accuracy - chance and BA of the size rule for one partition (classes 0..K-1, -1 unscored)."""
    K = int(max(src_lab.max(initial=-1), tgt_lab.max(initial=-1)) + 1)
    if K < 2:
        return None
    onehot = np.stack([(src_lab == k) for k in range(K)], 1).astype(np.float64)
    n_src = onehot.sum(0); pi = (m @ onehot); tot = pi.sum()
    if tot <= 0:
        return {"status": "no retained source mass"}
    pi = pi / tot
    cand = (pi >= MIN_PI) & (n_src >= MIN_SRC)
    S = (Q.T @ onehot) / np.where(col > 0, col, np.nan)[:, None]
    St = S / np.where(S.sum(1, keepdims=True) > 0, S.sum(1, keepdims=True), np.nan)
    scored = (tgt_lab >= 0) & rt & np.isfinite(St).all(1)
    true = tgt_lab[scored]
    present = np.unique(true)
    out = {"classes": K, "candidates": int(cand.sum()), "target_in_classes": int((tgt_lab >= 0).sum()), "scored": int(scored.sum()),
           "true_classes": int(len(present))}
    if cand.sum() < 2 or len(present) < 2:
        out["status"] = "not evaluable"; return out
    E = St[scored][:, cand] / pi[cand][None, :]
    pred = np.flatnonzero(cand)[np.argmax(E, 1)]
    size = np.flatnonzero(cand)[np.argmax(pi[cand])]
    rec = [float((pred[true == g] == g).mean()) for g in present]
    out.update(status="evaluable", ba=float(np.mean(rec)), chance=1.0 / cand.sum(),
               ba_size=float(np.mean([float(size == g) for g in present])))
    out["ba_minus_chance"] = out["ba"] - out["chance"]; out["ba_minus_size"] = out["ba"] - out["ba_size"]
    return out


def l3_cells(Q, m, col, s, ok, rt):
    """Per retained target (columns of s): S_OT, S_ind, S_best, dS, dS_norm, coverage c_j (MATRIX_METRICS.md L3)."""
    q = Q / np.where(col > 0, col, np.nan)[None, :]
    w_ot = np.where(ok, q, 0.0); w_ind = np.where(ok, m[:, None], 0.0)
    c = w_ot.sum(0)
    S_ot = (w_ot * s).sum(0) / np.where(c > 0, c, np.nan)
    wi = w_ind.sum(0)
    S_ind = (w_ind * s).sum(0) / np.where(wi > 0, wi, np.nan)
    S_best = np.where(ok & (m[:, None] > 0), s, -np.inf).max(0)   # E_j: evaluable sources with retained mass
    scored = rt & (c > 0) & (wi > 0)
    dS = S_ot - S_ind
    den = S_best - S_ind
    dn = np.where(scored & (den > 1e-12), dS / np.where(den > 1e-12, den, 1.0), np.nan)
    return pd.DataFrame({"scored": scored, "c": np.where(rt, c, np.nan), "S_ot": S_ot, "S_ind": S_ind, "S_best": S_best,
                         "dS": np.where(scored, dS, np.nan), "dS_norm": dn})
