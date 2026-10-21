"""Shared code of the lineage validation scoring. Every definition is VALIDATION_PROTOCOL.md
(FROZEN 2026-10-21, 508609f); section numbers in comments are that file's.

Nothing here reads lineage into an OT input: runs are read as written by 02_run_pair.py and
labels are joined to them afterwards (§2).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import zlib
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PROTOCOL = HERE / "VALIDATION_PROTOCOL.md"
FROZEN = "Status: **FROZEN 2026-10-21 508609f**"
LABEL = "ranknm256_noscale_ds_cos"
STRATEGIES = ("rna", "cnv", "rna_cnv", "cnv_then_rna", "rna_then_cnv")
SINGLE = ("rna", "cnv", "rna_cnv")
STEP1 = {"cnv_then_rna": "cnv", "rna_then_cnv": "rna"}
BLOCKS = {"kp_D_A": "KPTracer", "kp_D_B": "KPTracer", "kp_D_C3": "KPTracer", "quinn_D_B": "Quinn2021"}
DESIGN = {"kp_D_A": "D_A", "kp_D_B": "D_B", "kp_D_C3": "D_C3", "quinn_D_B": "D_B"}
METHODS = ("M4-E", "M4-R")
SEED = 20261021
SHUFFLES = 1000
TIE = 1e-12                       # §8.2
SIZE_BINS = (0.0, 0.01, 0.05, 0.20)   # §8.5, lower edges; the last bin is [0.20, 1]
CURVE = tuple(round(k / 10, 1) for k in range(1, 11))  # §8.4
BRUTE_LIMIT = 2000                # §10: tree S against brute force on blocks up to this size
UNKNOWN = ""
SENSITIVITY_LL = 5                # §5.2 Quinn sensitivity


def check_frozen() -> None:
    text = PROTOCOL.read_text(encoding="utf-8")
    if FROZEN not in text:
        sys.exit(f"{PROTOCOL.name} does not carry {FROZEN!r}; nothing is scored against an unfrozen protocol")


def commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance only
        return "unknown"


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_module():
    """validation_v2's Tree (parse_newick, contracted depth, leaf_of), as the audit used it."""
    if "d1" not in sys.modules:
        spec = importlib.util.spec_from_file_location("d1", REPO / "kp_lineage" / "validation_v2" / "04_lineage_concordance.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["d1"] = module
        spec.loader.exec_module(module)
    return sys.modules["d1"]


def script(name: str, filename: str):
    """Import a numbered script of this directory (its file name is not a module name)."""
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def rng_for(*parts: str) -> np.random.Generator:
    key = "/".join(parts)
    return np.random.default_rng([SEED, zlib.crc32(key.encode())])


def known(labels: np.ndarray, dataset: str) -> np.ndarray:
    """§5.1: a missing label, or a KP-Tracer family ending in _Unk, is unknown."""
    labels = np.asarray(labels, dtype=object)
    out = labels != UNKNOWN
    if dataset == "KPTracer":
        out &= ~np.array([str(x).endswith("_Unk") for x in labels], dtype=bool)
    return out


def label_classes(own: np.ndarray, own_known: np.ndarray, other: np.ndarray, other_known: np.ndarray) -> np.ndarray:
    """§5.2: matchable / unmatched_observed / ambiguous of each cell against the other side."""
    present = set(other[other_known])
    hit = np.array([x in present for x in own], dtype=bool)
    return np.where(~own_known, "ambiguous", np.where(hit, "matchable", "unmatched_observed")).astype(object)


def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    """Mann-Whitney AUROC, ties 1/2; +inf scores tie with each other."""
    positive = np.asarray(positive, dtype=bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(np.asarray(score, dtype=float), method="average")
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def spearman(x, y) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")
    # +inf margins (step-1 removed) are ranked as the largest; spearmanr ranks infinities correctly.
    return float(spearmanr(x, y).statistic)


def topk_hits(W: np.ndarray, gstar: np.ndarray, k: int) -> np.ndarray:
    """§8.2: hit_k(j) = min(1, max(0, (k - n_>) / n_=)) per column of W (groups x cells)."""
    if W.shape[1] == 0:
        return np.zeros(0)
    w = W[gstar, np.arange(W.shape[1])]
    greater = (W > w + TIE).sum(axis=0)
    equal = (np.abs(W - w) <= TIE).sum(axis=0)
    return np.clip((k - greater) / equal, 0.0, 1.0)


def mean_or_nan(values) -> float:
    values = np.asarray(values, dtype=float)
    return float(values.mean()) if values.size else float("nan")


# --- runs -----------------------------------------------------------------------------------

class RunMissing(Exception):
    pass


def run_directory(runs: Path, strategy: str, block: str, pair: str) -> Path:
    base = runs / strategy / "arm_A" / block / LABEL / pair / "scope_malignant"
    done = sorted(p.parent for p in base.glob("*/SUCCESS"))
    if len(done) != 1:
        raise RunMissing(f"{strategy}: {len(done)} completed budget directories under {base}")
    return done[0]


def read_gate(directory: Path, method: str) -> pd.DataFrame:
    wanted = {"method", "side", "observation_id", "retained", "signed_rejection_margin", "site"}
    cells = pd.read_csv(directory / "cell_confidence.csv", usecols=lambda c: c in wanted, low_memory=False)
    cells = cells[cells["method"].astype(str).eq(method)].copy()
    cells["observation_id"] = cells["observation_id"].astype(str)
    cells["retained"] = cells["retained"].astype(str).str.lower().eq("true")
    if "site" not in cells:
        cells["site"] = UNKNOWN
    cells["site"] = cells["site"].astype(str)
    return cells


def load_run(runs: Path, strategy: str, block: str, pair: str, method: str) -> dict:
    """§3.4, §6: S0 and T0, rho, tau, margins and P extended to S0 x T0."""
    directory = run_directory(runs, strategy, block, pair)
    gate = read_gate(directory, method)
    src, tgt = gate[gate["side"].eq("source")], gate[gate["side"].eq("target")]
    coupling = np.load(directory / f"coupling_{method.lower().replace('-', '')}.npz")["coupling"]
    if coupling.shape != (len(src), len(tgt)):
        raise ValueError(f"coupling {coupling.shape} but {len(src)} x {len(tgt)} {method} rows")
    metrics = pd.read_csv(directory / "pair_metrics.csv")
    mass = float(metrics.loc[metrics["method"].astype(str).eq(method), "transported_mass"].iloc[0])
    if abs(coupling.sum() - mass) > 1e-9 * max(1.0, abs(mass)):
        raise ValueError(f"coupling mass {coupling.sum():.12g} differs from transported_mass {mass:.12g}")
    record = {"directory": str(directory), "step1_directory": ""}
    if strategy in STEP1:
        first = run_directory(runs, STEP1[strategy], block, pair)
        base = read_gate(first, "M4-E")
        record["step1_directory"] = str(first)
        base_src, base_tgt = base[base["side"].eq("source")], base[base["side"].eq("target")]
    else:
        base_src, base_tgt = src, tgt
    s0 = base_src["observation_id"].to_numpy()
    t0 = base_tgt["observation_id"].to_numpy()
    if pd.Index(s0).has_duplicates or pd.Index(t0).has_duplicates:
        raise ValueError("duplicate cell ids on a side")
    s_pos = pd.Index(s0).get_indexer(src["observation_id"].to_numpy())
    t_pos = pd.Index(t0).get_indexer(tgt["observation_id"].to_numpy())
    if (s_pos < 0).any() or (t_pos < 0).any():
        raise ValueError("step-2 cells that step 1 did not analyse")
    if strategy in STEP1:
        kept_s = base_src["retained"].to_numpy()[s_pos]
        kept_t = base_tgt["retained"].to_numpy()[t_pos]
        if not (kept_s.all() and kept_t.all()):
            raise ValueError("step-2 cells that step 1 did not retain")
    n0, m0 = len(s0), len(t0)
    rho = np.zeros(n0, dtype=bool); rho[s_pos] = src["retained"].to_numpy()
    tau = np.zeros(m0, dtype=bool); tau[t_pos] = tgt["retained"].to_numpy()
    ms = np.full(n0, np.inf); ms[s_pos] = src["signed_rejection_margin"].to_numpy(float)
    mt = np.full(m0, np.inf); mt[t_pos] = tgt["signed_rejection_margin"].to_numpy(float)
    if strategy in STEP1:
        P = np.zeros((n0, m0))
        P[np.ix_(s_pos, t_pos)] = coupling
    else:
        P = np.asarray(coupling, dtype=float)
    record.update(source=s0, target=t0, source_site=base_src["site"].to_numpy(), target_site=base_tgt["site"].to_numpy(),
                  rho=rho, tau=tau, source_margin=ms, target_margin=mt, P=P,
                  step2_source=len(src), step2_target=len(tgt))
    return record


# --- cell-level scoring ---------------------------------------------------------------------

def onehot(groups: np.ndarray, n_groups: int, weights: np.ndarray | None = None) -> sparse.csr_matrix:
    n = len(groups)
    data = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    return sparse.csr_matrix((data, (groups, np.arange(n))), shape=(n_groups, n))


def score_cells(P, rho, tau, ms, mt, lab_s, lab_t, dataset: str, *, shuffles: int, seed_key: str,
                ks=(1,)) -> dict:
    """§8 and the shuffle null of §10 for one run and one method. Returns tables and a summary."""
    n0, m0 = P.shape
    known_s, known_t = known(lab_s, dataset), known(lab_t, dataset)
    cls_t = label_classes(lab_t, known_t, lab_s, known_s)
    cls_s = label_classes(lab_s, known_s, lab_t, known_t)
    groups = np.array(sorted(set(lab_s)), dtype=object)            # G, unknown included as a competitor
    gindex = {g: i for i, g in enumerate(groups)}
    gs = np.array([gindex[x] for x in lab_s], dtype=int)
    gt = np.array([gindex.get(x, -1) for x in lab_t], dtype=int)
    group_known = known(groups, dataset)
    sizes = np.bincount(gs, minlength=len(groups)).astype(float)
    m = float(P.sum())
    summary = {"n_source": n0, "n_target": m0, "groups": len(groups),
               "source_unknown": int((~known_s).sum()), "transported_mass": m}

    GP = onehot(gs, len(groups), rho.astype(float)) @ P              # retained-source mass per group and column
    D = GP.sum(axis=0)
    valid = tau & (D > 0)
    J = cls_t == "matchable"
    JR = J & valid
    nJ = int(J.sum())
    W = np.zeros_like(GP)
    W[:, valid] = GP[:, valid] / D[valid]
    w = np.zeros(m0)
    w[J] = W[gt[J], np.flatnonzero(J)]
    pistar = np.where(gt >= 0, sizes[np.maximum(gt, 0)] / n0, np.nan)
    a = P.sum(axis=1) / m if m > 0 else np.zeros(n0)
    b = P.sum(axis=0) / m if m > 0 else np.zeros(m0)
    ra = rho * a
    Wind = (onehot(gs, len(groups)) @ ra) / ra.sum() if ra.sum() > 0 else np.full(len(groups), np.nan)
    w_ind = np.where(J, Wind[np.maximum(gt, 0)], 0.0)

    summary.update(
        matchable=nJ, unmatched_observed=int((cls_t == "unmatched_observed").sum()),
        ambiguous=int((cls_t == "ambiguous").sum()), matchable_retained=int(JR.sum()),
        retained_without_source_mass=int((tau & ~(D > 0)).sum()),
        E2E=float(w[JR].sum() / nJ) if nJ else np.nan,
        CH=float(pistar[J].mean()) if nJ else np.nan,
        E2E_ind=float(w_ind[JR].sum() / nJ) if nJ else np.nan,
        COND=mean_or_nan(w[JR]), COND_ind=mean_or_nan(w_ind[JR]),
        cov_T=float(JR.sum() / nJ) if nJ else np.nan,
        target_retained=float(tau.mean()) if m0 else np.nan,
        cov_S=float(rho.mean()) if n0 else np.nan)
    summary["E2E_minus_CH"] = summary["E2E"] - summary["CH"]
    summary["COND_minus_ind"] = summary["COND"] - summary["COND_ind"]

    # §8.2 Top-k.
    cols = np.flatnonzero(JR)
    for k in ks:
        name = f"top{k}"
        if len(groups) >= max(2, k + 1 if k > 1 else 2):
            hits = topk_hits(W[:, cols], gt[cols], k)
            chance = topk_hits(np.repeat((sizes / n0)[:, None], nJ, axis=1), gt[J], k) if nJ else np.zeros(0)
            ind = topk_hits(np.repeat(Wind[:, None], len(cols), axis=1), gt[cols], k) if len(cols) else np.zeros(0)
            summary.update({f"{name}_E2E": float(hits.sum() / nJ) if nJ else np.nan, f"{name}_COND": mean_or_nan(hits),
                            f"{name}_chance": mean_or_nan(chance), f"{name}_ind_COND": mean_or_nan(ind)})
        else:
            summary.update({f"{name}_E2E": np.nan, f"{name}_COND": np.nan, f"{name}_chance": np.nan,
                            f"{name}_ind_COND": np.nan})

    # §8.3 asserted mass, as fractions of pi.
    unmatched = cls_t == "unmatched_observed"
    false_cols = J | (unmatched if dataset == "KPTracer" else np.zeros(m0, dtype=bool))

    def asserted(Mg: np.ndarray) -> dict:
        Mt = Mg * tau
        total = float(Mt.sum())
        colsum_known = Mt[group_known].sum(axis=0)
        correct = np.zeros(m0)
        correct[J] = Mt[gt[J], np.flatnonzero(J)]
        C = float(correct.sum())
        F = float((colsum_known - correct)[false_cols].sum())
        U = float(Mt[:, unmatched].sum()) if dataset != "KPTracer" else 0.0
        return {"A": total, "C": C, "F": F, "U": U, "ambiguous_mass": total - C - F - U,
                "precision": C / (C + F) if C + F > 0 else np.nan}

    observed = asserted(GP / m if m > 0 else GP)
    Hra = onehot(gs, len(groups)) @ ra
    product = asserted(np.outer(Hra, b))
    summary.update({f"asserted_{k}": v for k, v in observed.items()})
    summary.update({f"asserted_{k}_ind": v for k, v in product.items()})

    # §8.4 rejection test, both sides.
    rejection = []
    for side, cls, kept, margin in (("source", cls_s, rho, ms), ("target", cls_t, tau, mt)):
        both = np.isin(cls, ["matchable", "unmatched_observed"])
        row = {"side": side}
        for c in ("matchable", "unmatched_observed", "ambiguous"):
            hit = cls == c
            row[f"{c}_n"] = int(hit.sum())
            row[f"{c}_rejection_rate"] = float(1 - kept[hit].mean()) if hit.any() else np.nan
        row["auroc_unmatched_vs_matchable"] = auroc(margin[both], cls[both] == "unmatched_observed")
        rejection.append(row)
    summary["target_rejection_auroc"] = rejection[1]["auroc_unmatched_vs_matchable"]
    summary["source_rejection_auroc"] = rejection[0]["auroc_unmatched_vs_matchable"]

    # §8.4 coverage-accuracy curve, with source-only weights.
    curve = []
    if len(groups) >= 2 and nJ:
        Wt = np.zeros_like(GP)
        has = D > 0
        Wt[:, has] = GP[:, has] / D[has]
        jidx = np.flatnonzero(J)
        hit_all = np.where(has[jidx], topk_hits(Wt[:, jidx], gt[jidx], 1), 0.0)
        order = np.argsort(mt[jidx], kind="stable")
        for f in CURVE:
            first = order[: int(ceil(f * nJ))]
            curve.append({"fraction": f, "cells": len(first), "top1": float(hit_all[first].mean())})
        summary["curve_area"] = float(np.mean([r["top1"] for r in curve]))

    # §8.5 size control.
    group_rows, bal_e2e, bal_ch, bal_cond = [], [], [], []
    for g in np.unique(gt[J]):
        Jg = J & (gt == g)
        JRg = JR & (gt == g)
        share = sizes[g] / n0
        row = {"group": groups[g], "target_cells": int(Jg.sum()), "source_share": share,
               "source_retained": float(rho[gs == g].mean()), "target_retained": float(tau[Jg].mean()),
               "E2E": float(w[Jg].sum() / Jg.sum()), "COND": mean_or_nan(w[JRg]),
               "top1_COND": mean_or_nan(topk_hits(W[:, np.flatnonzero(JRg)], gt[JRg], 1)) if len(groups) >= 2 else np.nan}
        group_rows.append(row)
        bal_e2e.append(row["E2E"]); bal_ch.append(share)
        if JRg.any():
            bal_cond.append(row["COND"] - share)
    summary["E2E_bal_raw"] = mean_or_nan(bal_e2e)
    summary["CH_bal"] = mean_or_nan(bal_ch)
    summary["E2E_bal"] = summary["E2E_bal_raw"] - summary["CH_bal"]
    summary["COND_bal"] = mean_or_nan(bal_cond)
    bins = []
    edges = list(SIZE_BINS) + [np.inf]
    for lo, hi in zip(edges[:-1], edges[1:]):
        hit = J & (pistar >= lo) & (pistar < hi)
        bins.append({"bin": f"[{lo}, {hi})" if np.isfinite(hi) else f"[{lo}, 1]",
                     "target_cells": int(hit.sum()), "E2E": float(w[hit].sum() / hit.sum()) if hit.any() else np.nan,
                     "COND": mean_or_nan(w[hit & valid]), "CH": mean_or_nan(pistar[hit])})
    if dataset == "Quinn2021":
        # §5.2 sensitivity: matchable needs >= 5 analysed LL cells of the clone.
        strong = J & (np.where(gt >= 0, sizes[np.maximum(gt, 0)], 0) >= SENSITIVITY_LL)
        ns = int(strong.sum())
        summary.update(matchable_ge5=ns, E2E_ge5=float(w[strong & valid].sum() / ns) if ns else np.nan,
                       CH_ge5=mean_or_nan(pistar[strong]), COND_ge5=mean_or_nan(w[strong & valid]))

    # §10 lineage-label shuffle.
    shuffle_rows = []
    if shuffles and len(groups) >= 2 and nJ:
        fcols = np.flatnonzero(false_cols)
        X = P[:, fcols] * rho[:, None]
        Dc, validc, tauc = D[fcols], valid[fcols], tau[fcols]
        inJ = J[fcols]
        gstar = gt[fcols]
        rng = rng_for(seed_key)
        for draw in range(shuffles):
            perm_groups = gs[rng.permutation(n0)]
            GPp = onehot(perm_groups, len(groups)) @ X
            Wp = np.zeros_like(GPp)
            Wp[:, validc] = GPp[:, validc] / Dc[validc]
            sel = np.flatnonzero(inJ & validc)
            wp = Wp[gstar[sel], sel]
            row = {"draw": draw, "E2E": float(wp.sum() / nJ), "COND": mean_or_nan(wp)}
            for k in ks:
                if len(groups) >= max(2, k + 1 if k > 1 else 2):
                    row[f"top{k}_COND"] = mean_or_nan(topk_hits(Wp[:, sel], gstar[sel], k))
            Mt = GPp / m * tauc
            colsum_known = Mt[group_known].sum(axis=0)
            correct = np.zeros(len(fcols))
            jj = np.flatnonzero(inJ)
            correct[jj] = Mt[gstar[jj], jj]
            C, F = correct.sum(), (colsum_known - correct).sum()
            row["precision"] = float(C / (C + F)) if C + F > 0 else np.nan
            shuffle_rows.append(row)
        null = pd.DataFrame(shuffle_rows)
        for name, observed_value in (("E2E", summary["E2E"]), ("COND", summary["COND"]),
                                     ("top1_COND", summary.get("top1_COND")), ("precision", observed["precision"])):
            if name in null and observed_value is not None and np.isfinite(observed_value):
                values = null[name].to_numpy(float)
                values = values[np.isfinite(values)]
                summary[f"shuffle_{name}_mean"] = mean_or_nan(values)
                summary[f"shuffle_{name}_p"] = float((1 + (values >= observed_value - 1e-12).sum()) / (1 + len(values)))

    cells = pd.DataFrame({"class": cls_t, "lineage": lab_t, "retained": tau, "source_mass": D,
                          "w": np.where(J, w, np.nan), "pistar": pistar, "margin": mt})
    checks = {"group_weight_sum_ok": bool(np.all(np.isclose(W.sum(axis=0)[valid], 1.0, atol=1e-12)) and
                                          np.all(W.sum(axis=0)[~valid] == 0))}
    return {"summary": summary, "groups": pd.DataFrame(group_rows), "bins": pd.DataFrame(bins),
            "curve": pd.DataFrame(curve), "rejection": pd.DataFrame(rejection),
            "shuffle": pd.DataFrame(shuffle_rows), "cells": cells, "checks": checks,
            "source_classes": cls_s, "target_classes": cls_t}


# --- site-level scoring (§7) ------------------------------------------------------------------

def site_lineage(sites: np.ndarray, labels: np.ndarray, is_known: np.ndarray) -> dict:
    """The family of each KP-Tracer site; refuses a site whose cells carry more than one."""
    out = {}
    for site in pd.unique(sites):
        values = set(labels[(sites == site) & is_known])
        if len(values) > 1:
            raise ValueError(f"site {site} has cells of several lineages: {sorted(values)[:3]}")
        out[site] = values.pop() if values else UNKNOWN
    return out


def site_scores(run: dict, lab_s, lab_t, dataset: str, design: str, mass_rows: np.ndarray,
                target_classes: np.ndarray) -> tuple[pd.DataFrame, dict]:
    rho, tau = run["rho"], run["tau"]
    ss, ts = run["source_site"], run["target_site"]
    ks, kt = known(lab_s, dataset), known(lab_t, dataset)
    rows = []
    asserted_total = float(mass_rows.sum())
    for side, sites, kept in (("source", ss, rho), ("target", ts, tau)):
        for site in pd.unique(sites):
            hit = sites == site
            row = {"side": side, "site": site, "cells": int(hit.sum()), "retained": float(kept[hit].mean())}
            if side == "source":
                row["asserted_share"] = float(mass_rows[hit].sum() / asserted_total) if asserted_total > 0 else np.nan
            if side == "target" and dataset == "Quinn2021":
                labelled = hit & kt
                row["f_shared"] = float((target_classes[labelled] == "matchable").mean()) if labelled.any() else np.nan
            rows.append(row)
    table = pd.DataFrame(rows)
    out = {}
    if dataset == "KPTracer":
        fam_s, fam_t = site_lineage(ss, lab_s, ks), site_lineage(ts, lab_t, kt)
        table["lineage"] = [fam_s.get(s, UNKNOWN) if side == "source" else fam_t.get(s, UNKNOWN)
                            for side, s in zip(table["side"], table["site"])]
        if design == "D_A":
            target = list(fam_t)
            fam_m = fam_t[target[0]] if len(target) == 1 else UNKNOWN
            src = table[table["side"].eq("source")]
            fam_ok = fam_m != UNKNOWN and not fam_m.endswith("_Unk")
            related = src["lineage"].eq(fam_m) & fam_ok
            table.loc[src.index, "related"] = related.astype(object)
            out["site_truth"] = bool(related.any() and (~related).any())
            if out["site_truth"]:
                r = src["retained"].to_numpy(float); rel = related.to_numpy()
                out["e_M"] = float(r[rel].mean() - r[~rel].mean())
                top = r >= r.max() - TIE
                out["site_top1"] = float((top & rel).sum() / top.sum())
                out["site_auroc"] = float(np.mean([(x > y) + 0.5 * (x == y) for x in r[rel] for y in r[~rel]]))
            out["target_site_retained"] = float(table.loc[table["side"].eq("target"), "retained"].iloc[0])
        elif design == "D_B":
            sources = list(fam_s)
            fam_p = fam_s[sources[0]] if len(sources) == 1 else UNKNOWN
            tgt = table[table["side"].eq("target")]
            usable = tgt["lineage"].ne(UNKNOWN) & ~tgt["lineage"].str.endswith("_Unk")
            related = usable & tgt["lineage"].eq(fam_p)
            table.loc[tgt.index, "related"] = np.where(usable, related, np.nan)
            n_rel, n_unrel = int(related.sum()), int((usable & ~related).sum())
            out["B_case"] = ("both" if n_rel and n_unrel else "only_related" if n_rel else
                             "none_related" if n_unrel else "no_truth")
            t = tgt["retained"].to_numpy(float)
            if out["B_case"] == "both":
                rel, unrel = t[related.to_numpy()], t[(usable & ~related).to_numpy()]
                out["e_s"] = float(rel.mean() - unrel.mean())
                out["site_auroc"] = float(np.mean([(x > y) + 0.5 * (x == y) for x in rel for y in unrel]))
            if n_rel:
                out["mean_t_related"] = float(t[related.to_numpy()].mean())
    elif dataset == "Quinn2021":
        tgt = table[table["side"].eq("target")]
        out["spearman_t_f"] = spearman(tgt["retained"], tgt["f_shared"])
        out["target_sites"] = int(len(tgt))
    return table, out


# --- depth control (§10) ----------------------------------------------------------------------

def depth_control(run: dict, depth: pd.Series | None, keys_s, keys_t) -> pd.DataFrame:
    rows = []
    if depth is None:
        return pd.DataFrame(rows)
    for side, sites, margin, keys in (("source", run["source_site"], run["source_margin"], keys_s),
                                      ("target", run["target_site"], run["target_margin"], keys_t)):
        values = depth.reindex(pd.MultiIndex.from_arrays(keys)).to_numpy(float)
        for site in pd.unique(sites):
            hit = sites == site
            rows.append({"side": side, "site": site, "cells": int(hit.sum()), "with_depth": int(np.isfinite(values[hit]).sum()),
                         "spearman_margin_depth": spearman(margin[hit], values[hit])})
    return pd.DataFrame(rows)


# --- tree agreement (§9; Level 3 §3 definitions) ------------------------------------------------

class TreeIndex:
    """Preorder positions, clade sizes and the counted (child-of-branching) nodes of one tree."""

    def __init__(self, tree):
        self.tree = tree
        n = len(tree.parent)
        # parse_newick may root the tree below node 0 (a unary outer bracket), so only the
        # nodes reachable from tree.root get a preorder position.
        order = np.asarray(tree.order)
        self.pos = np.full(n, -1, dtype=int)
        self.pos[order] = np.arange(len(order))
        self.size = np.ones(n, dtype=int)
        for v in reversed(tree.order):
            for c in tree.children[v]:
                self.size[v] += self.size[c]
        branching = np.array([len(c) >= 2 for c in tree.children])
        parent = np.asarray(tree.parent)
        self.counted = np.zeros(n, dtype=bool)
        self.counted[parent >= 0] = branching[parent[parent >= 0]]
        self.counted &= self.pos >= 0
        self.depth = np.asarray(tree.depth)
        self.counted_nodes = np.flatnonzero(self.counted)

    def chain(self, leaf: int) -> list[int]:
        nodes, v = [], leaf
        while v >= 0:
            if self.counted[v]:
                nodes.append(v)
            v = self.tree.parent[v]
        nodes.reverse()
        if [int(self.depth[c]) for c in nodes] != list(range(1, len(nodes) + 1)):
            raise ValueError("contracted depth does not match the counted-ancestor chain")
        return nodes


class TreeBlock:
    """ΔS, ΔS_max, ΔS_norm, ΔR (Level 3 §3) of couplings on fixed source and target leaves."""

    def __init__(self, index: TreeIndex, source_nodes: np.ndarray, target_nodes: np.ndarray):
        self.index = index
        self.torder = np.argsort(index.pos[target_nodes], kind="stable")
        tpos = index.pos[target_nodes][self.torder]
        spos_order = np.argsort(index.pos[source_nodes], kind="stable")
        self.sorder = spos_order
        spos = index.pos[source_nodes][spos_order]
        chains = [index.chain(int(v)) for v in source_nodes]
        hmax = max((len(c) for c in chains), default=0)
        n = len(source_nodes)
        self.LO = np.zeros((n, hmax), dtype=int)
        self.HI = np.zeros((n, hmax), dtype=int)
        self.K = np.zeros((n, hmax))
        for r, chain in enumerate(chains):
            for k, c in enumerate(chain):
                self.LO[r, k] = np.searchsorted(tpos, index.pos[c])
                self.HI[r, k] = np.searchsorted(tpos, index.pos[c] + index.size[c])
                self.K[r, k] = k + 1
        cn = index.counted_nodes
        self.tlo = np.searchsorted(tpos, index.pos[cn]); self.thi = np.searchsorted(tpos, index.pos[cn] + index.size[cn])
        self.slo = np.searchsorted(spos, index.pos[cn]); self.shi = np.searchsorted(spos, index.pos[cn] + index.size[cn])
        self.hs = index.depth[source_nodes].astype(float)
        self.ht = index.depth[target_nodes][self.torder].astype(float)
        self.rows = np.arange(n)[:, None]

    def evaluate(self, pi: np.ndarray) -> dict:
        """pi: source rows in the given order, target columns in the given order; sums to 1."""
        pc = pi[:, self.torder]
        a, b = pc.sum(axis=1), pc.sum(axis=0)
        cs = np.zeros((pc.shape[0], pc.shape[1] + 1)); np.cumsum(pc, axis=1, out=cs[:, 1:])
        cb = np.concatenate([[0.0], np.cumsum(b)])
        Pi = cs[self.rows, self.HI] - cs[self.rows, self.LO]
        B = cb[self.HI] - cb[self.LO]
        mask = self.K > 0
        Pi, B = Pi * mask, B * mask
        S = float(Pi.sum())
        Sab = float((a[:, None] * B).sum())
        Pf = np.hstack([a[:, None], Pi, np.zeros((len(a), 1))])
        Bf = np.hstack([np.ones((len(a), 1)), B, np.zeros((len(a), 1))])
        F = (1 - Bf[:, :-1]) + 0.5 * (Bf[:, :-1] - Bf[:, 1:])
        R = float(((Pf[:, :-1] - Pf[:, 1:]) * F).sum())
        EL2 = float((a[:, None] * (2 * self.K - 1) * B * mask).sum())
        ca = np.concatenate([[0.0], np.cumsum(a[self.sorder])])
        a_c, b_c = ca[self.shi] - ca[self.slo], cb[self.thi] - cb[self.tlo]
        W = float(np.abs(a_c - b_c).sum())
        Smax = 0.5 * (float((a * self.hs).sum()) + float((b * self.ht).sum()) - W)
        dS, dSmax = S - Sab, Smax - Sab
        return {"S": S, "S_ab": Sab, "dS": dS, "dS_max": dSmax,
                "dS_norm": dS / dSmax if dSmax > 1e-15 else np.nan, "dR": R - 0.5,
                "var_L_ab": EL2 - Sab ** 2, "W_tree": W}

    def brute_S(self, pi: np.ndarray) -> float:
        pc = pi[:, self.torder]
        L = np.zeros(pc.shape)
        for r in range(pc.shape[0]):
            for k in range(self.K.shape[1]):
                if self.K[r, k] > 0:
                    L[r, self.LO[r, k]:self.HI[r, k]] = self.K[r, k]
        return float((pc * L).sum())

    def modal_share_uniform(self) -> float:
        n, m = self.K.shape[0], len(self.torder)
        cb = np.arange(m + 1) / m
        B = (cb[self.HI] - cb[self.LO]) * (self.K > 0)
        Bf = np.hstack([np.ones((n, 1)), B, np.zeros((n, 1))])
        mass = (Bf[:, :-1] - Bf[:, 1:]).mean(axis=0)
        return float(mass.max()) if mass.size else np.nan


def tree_block_scores(index: TreeIndex, source_nodes, target_nodes, block: np.ndarray, *, draws: int,
                      seed_key: str) -> dict:
    tb = TreeBlock(index, np.asarray(source_nodes), np.asarray(target_nodes))
    pi = block / block.sum()
    out = tb.evaluate(pi)
    out["modal_L_share_uniform"] = tb.modal_share_uniform()
    out["low_information"] = bool(out["modal_L_share_uniform"] >= 0.9)
    out["scored"] = bool(out["var_L_ab"] > 1e-15 and np.isfinite(out["dS_norm"]))
    if not out["dS"] <= out["dS_max"] + 1e-12:
        raise ValueError(f"tree block: dS {out['dS']} exceeds dS_max {out['dS_max']}")
    if pi.shape[0] <= BRUTE_LIMIT and pi.shape[1] <= BRUTE_LIMIT:
        brute = tb.brute_S(pi)
        if abs(brute - out["S"]) > 1e-12 * max(1.0, abs(brute)):
            raise ValueError(f"tree block: S {out['S']} differs from brute force {brute}")
        out["brute_checked"] = True
    else:
        out["brute_checked"] = False
    if out["scored"] and draws:
        rng = rng_for(seed_key)
        null_s, null_r = [], []
        for _ in range(draws):
            value = tb.evaluate(pi[:, rng.permutation(pi.shape[1])])
            null_s.append(value["dS_norm"]); null_r.append(value["dR"])
        null_s, null_r = np.asarray(null_s, float), np.asarray(null_r, float)
        null_s = null_s[np.isfinite(null_s)]
        out["perm_dS_norm_mean"] = mean_or_nan(null_s)
        out["perm_dS_norm_p"] = float((1 + (null_s >= out["dS_norm"] - 1e-12).sum()) / (1 + len(null_s)))
        out["perm_dR_mean"] = mean_or_nan(null_r)
        out["perm_dR_p"] = float((1 + (null_r >= out["dR"] - 1e-12).sum()) / (1 + len(null_r)))
    return out


def write_json(path: Path, payload: dict) -> None:
    def clean(value):
        if isinstance(value, (np.floating, float)):
            return None if not np.isfinite(value) else float(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        return value
    path.write_text(json.dumps(clean(payload), indent=2), encoding="utf-8")
