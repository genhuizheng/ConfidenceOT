"""Score the application OT rows (PLAN_TACC.md scoring sections; definitions fixed here, 2026-11-23, before any result of
the ot_20261123 runs is read). Both configurations, M4-E primary and M4-R reported. LARRY in vivo is scored by
larry_20261112/03_score.py (--units, --non-mature Undifferentiated Prog), not here.

    python -I 09_score_apps.py --app $SCRATCH/OT_project/app_search_20261117 --dataset kang [--gmt h.all...gmt]

For every unit x configuration x method (rows: <dataset>/ot_20261123/rows/<unit>/<key>/<config>/):
  gates.tsv       rejected share per side, overall and per label value (label column per dataset, below)
  coupling.tsv    retained x retained block, label x label: mass, P(target label | source label), enrichment
                  = mass * total / (row * column)
  same_label.tsv  share of retained mass between equal labels vs the independent expectation sum_k r_k c_k (labels on
                  both sides only)
  depth.tsv       AUROC of per-cell total counts, rejected vs retained (> 0.5: rejected cells are deeper), per side
Dataset-specific (PLAN_TACC):
  kang        label cell_type. ifn_shift.tsv: per retained source cell, the coupling-weighted mean over its retained
              targets minus its own value, of the mean log1p(CP10k) over HALLMARK_INTERFERON_ALPHA_RESPONSE and
              _GAMMA_RESPONSE genes (the authors' 767-gene list is not in their supplement files we hold); summary =
              mean per source cell type
  sciplex2    no cell-type label. dose.tsv: per drug, Spearman over the 7 doses of rejected share (source, target) vs dose,
              and target cells per dose (the paper's viability proxy); cycle.tsv: mean S and G2M score (mean
              log1p(CP10k) over the Tirosh 2016 lists) of rejected vs retained cells, per side
  schiebinger label stage (one per side, so per-age rejection only); WOT cell sets / signatures are not held: not scored
  celegans    labels cell_type and lineage. lineage.tsv: among source and target cells that both carry a lineage name,
              the share of retained coupling mass on lineage-compatible pairs (target name descends from or equals the
              source name: Sulston prefix, x = any letter, '/' alternatives compatible if any is) vs the independent
              expectation over the same annotated cells; gates per lineage depth (name length)
  pancreas    label cell_type. emerging.tsv: target rejection of types with < 1% of source cells (absent or rare the day
              before) vs the other target types
Added 2026-11-27 (PLAN_TACC.md sections 7 / 8 and post-hoc 2026-11-27; fixed before any SPLINTR / zebrafish OT result):
  splintr     no cell-type label; obs.clones holds every barcode of a cell. Sisters T_i of source cell i = target cells
              sharing >= 1 barcode with i; truth_i = |T_i| > 0.
              clone_truth.tsv: source rejected share for cells with / without sisters, AUROC of the (binary) rejection
              for 'no sister' (= 0.5 + (difference of the two shares) / 2), and Spearman of source clone size (max over
              the cell's barcodes of the source cells carrying it) vs rejection; target rejected share for cells with /
              without any barcode in the whole baseline library (GEX-matched sheet cells, --data) and the share of
              target cells without one.
              clone_mass.tsv: as LARRY 03_score on the retained x retained block Q: retained source cells with >= 1
              retained sister; m_i = sum_{j in T_i} Q_ij / sum_j Q_ij; uniform = |T_i| / n_retained_target;
              independent = sum_{j in T_i} b_j / sum_j b_j (b = column mass of Q); medians of m / baseline and shares
              of cells above each baseline.
  zebrafish   label Cell_type. activated_origin.tsv: for each activated fibroblast type in the retained targets
              (col11a1a, col12a1a, nppc, proliferating), the share of its incoming retained mass from each source group
              (epicardium, endocardium, constitutive fibroblasts, other fibroblasts, other) vs the group's share of
              all retained source mass. tree_relatedness.tsv: in each tree of the target condition (trees.pkl, --data),
              a cell's clade = its parent node; relatedness(a, b) = share of clades holding type a that also hold b,
              averaged over those trees; Spearman over source types between P(source type | target type a) and
              relatedness(a, source type), per activated type a. Consistency only, not lineage.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

CONFIGS = {"l5best": "ranknm256-midrank|none|scale|cos|pca50", "logcpm_pca": "logcpm|none|noscale|sqeuc|pca30"}
LABEL = {"kang": "cell_type", "sciplex2": None, "schiebinger": "stage", "celegans": "cell_type", "pancreas": "cell_type",
         "splintr": None, "zebrafish": "Cell_type"}
ZF_ACTIVATED = ["Fibroblasts (col11a1a)", "Fibroblasts (col12a1a)", "Fibroblasts (nppc)", "Fibroblasts (proliferating)"]


def zf_group(t):
    if t.startswith("Epicardium"):
        return "epicardium"
    if t.startswith("Endocardium"):
        return "endocardium"
    if t == "Fibroblasts (const.)":
        return "constitutive fibroblasts"
    if t.startswith("Fibroblasts") or t == "Valve fibroblasts":
        return "other fibroblasts"
    return "other"
S_GENES = ["MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2", "MCM6", "CDCA7", "DTL", "PRIM1", "UHRF1",
           "MLF1IP", "CENPU", "HELLS", "RFC2", "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76", "SLBP", "CCNE2", "UBR7", "POLD3",
           "MSH2", "ATAD2", "RAD51", "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", "BLM", "CASP8AP2", "USP1", "CLSPN",
           "POLA1", "CHAF1B", "BRIP1", "E2F8"]
G2M_GENES = ["HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80", "CKS2", "NUF2", "CKS1B", "MKI67", "TMPO",
             "CENPF", "TACC3", "FAM64A", "PIMREG", "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB", "BUB1", "KIF11", "ANP32E",
             "TUBB4B", "GTSE1", "KIF20B", "HJURP", "CDCA3", "HN1", "JPT1", "CDC20", "TTK", "CDC25C", "KIF2C", "RANGAP1",
             "NCAPD2", "DLGAP5", "CDCA2", "CDCA8", "ECT2", "KIF23", "HMMR", "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5", "CENPE",
             "CTCF", "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA"]


def auroc(score, lab):
    lab = np.asarray(lab, bool); r = rankdata(score); n1 = lab.sum(); n0 = len(lab) - n1
    return float((r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan


def program(A, genes):
    import scipy.sparse as sp
    X = A.X.tocsr() if sp.issparse(A.X) else sp.csr_matrix(A.X)
    tot = np.asarray(X.sum(1)).ravel()
    idx = [i for i, g in enumerate(A.var_names.astype(str)) if g in set(genes)]
    if not idx:
        return np.full(A.n_obs, np.nan), 0
    Y = X[:, idx].toarray() * (1e4 / np.maximum(tot, 1))[:, None]
    return np.log1p(Y).mean(1), len(idx)


def read_gmt(path, names):
    out = {}
    for line in Path(path).read_text().splitlines():
        f = line.split("\t")
        if f[0] in names:
            out[f[0]] = f[2:]
    return out


def compatible(src, tgt):
    """Target lineage name descends from or equals the source name (Sulston prefix; x = any; '/' alternatives)."""
    for s in src.split("/"):
        for t in tgt.split("/"):
            if len(t) >= len(s) and all(a == b or a == "x" or b == "x" for a, b in zip(s, t[:len(s)])):
                return True
    return False


def splintr_baseline_barcodes(data: Path) -> dict:
    """Barcodes carried by the GEX-matched sheet cells of each baseline library (TP0_1 / TP0_2)."""
    import gzip
    import io
    import tarfile
    sheet = pd.read_csv(data / "splintr_singletcode_figshare/extracted/allBarcodeSheets/SPLINTR/stepFourStarcodeShavedReads50.csv", dtype=str)
    out = {}
    with tarfile.open(data / "splintr_gse186305/GSE186305_RAW.tar") as tf:
        for base, s, lib in (("TP0_1", "chemoKRASBaseline_1", "GSM5643580_KF16_TP0_1"), ("TP0_2", "chemoKRASBaseline_2", "GSM5643581_KF16_TP0_2")):
            bc = pd.read_csv(io.BytesIO(gzip.decompress(tf.extractfile(f"{lib}_barcodes.tsv.gz").read())), header=None)[0]
            bc = set(bc.astype(str).str.replace("-1$", "", regex=True))
            g = sheet[(sheet["sample"] == s) & sheet.cellID.isin(bc)]
            out[base] = set(g.barcode)
            print("baseline", base, "GEX-matched cells", g.cellID.nunique(), "barcodes", len(out[base]), flush=True)
    return out


def score_splintr(key, A, pos, rj, P, base_bc):
    rows = []
    cs = [set(x.split(";")) for x in A["source"].obs["clones"].astype(str).to_numpy()[pos["source"]]]
    ct = [set(x.split(";")) for x in A["target"].obs["clones"].astype(str).to_numpy()[pos["target"]]]
    where = {}
    for j, bs in enumerate(ct):
        for b in bs:
            where.setdefault(b, set()).add(j)
    sis = [set().union(*(where.get(b, set()) for b in bs)) for bs in cs]
    has = np.array([len(x) > 0 for x in sis])
    size_ = {}
    for bs in cs:
        for b in bs:
            size_[b] = size_.get(b, 0) + 1
    csize = np.array([max(size_[b] for b in bs) for bs in cs])
    base = A["source"].obs["mouse"].astype(str).iloc[0]
    tin = np.array([len(bs & base_bc[base]) > 0 for bs in ct])
    rs_, rt_ = rj["source"], rj["target"]
    rec = {**key, "table": "clone_truth", "target_condition": A["target"].obs["condition"].astype(str).iloc[0],
           "source_cells": len(cs), "source_with_sister": int(has.sum()),
           "rejected_with_sister": float(rs_[has].mean()) if has.any() else np.nan,
           "rejected_without_sister": float(rs_[~has].mean()) if (~has).any() else np.nan,
           "auroc_rejection_for_no_sister": auroc(rs_.astype(float), ~has),
           "spearman_clone_size_vs_rejected": float(spearmanr(csize, rs_.astype(float)).correlation) if rs_.any() and (~rs_).any() else np.nan,
           "target_cells": len(ct), "target_share_without_baseline_barcode": float((~tin).mean()),
           "target_rejected_with_baseline_barcode": float(rt_[tin].mean()) if tin.any() else np.nan,
           "target_rejected_without_baseline_barcode": float(rt_[~tin].mean()) if (~tin).any() else np.nan}
    rows.append(rec)
    rs, rt = ~rs_, ~rt_
    Q = P[np.ix_(rs, rt)]
    if Q.sum() > 0:
        tmap = -np.ones(len(rt), int)
        tmap[np.flatnonzero(rt)] = np.arange(rt.sum())
        b = Q.sum(0); n_t = int(rt.sum()); R = []
        for qi, i in enumerate(np.flatnonzero(rs)):
            T = [tmap[j] for j in sis[i] if tmap[j] >= 0]
            row = Q[qi]
            if not T or row.sum() <= 0:
                continue
            R.append((row[T].sum() / row.sum(), len(T) / n_t, b[T].sum() / b.sum()))
        R = np.array(R).reshape(-1, 3)
        ok = len(R) > 0
        rows.append({**key, "table": "clone_mass", "target_condition": rec["target_condition"], "scored_cells": len(R),
                     "median_m": float(np.median(R[:, 0])) if ok else np.nan,
                     "median_m_over_uniform": float(np.median(R[:, 0] / R[:, 1])) if ok else np.nan,
                     "median_m_over_independent": float(np.median(R[:, 0] / np.maximum(R[:, 2], 1e-300))) if ok else np.nan,
                     "share_m_gt_uniform": float((R[:, 0] > R[:, 1]).mean()) if ok else np.nan,
                     "share_m_gt_independent": float((R[:, 0] > R[:, 2]).mean()) if ok else np.nan})
    return rows


def zebrafish_clades(data: Path):
    """Per tree: condition (time, morphine, inhib) of its leaves and the Cell_type sets of its clades (parent nodes)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("prep06", Path(__file__).resolve().parent / "06_prepare_tacc.py")
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    z = data / "zebrafish_heart_moslin_figshare"
    m = pd.read_csv(z / "Zebrafish_metadata.csv", dtype=str, keep_default_na=False,
                    usecols=["Cell", "time", "morphine", "inhib", "Cell_type"]).set_index("Cell")
    with open(z / "trees.pkl", "rb") as fh:
        trees = prep._GraphUnpickler(fh).load()
    out = []
    for name, t in (trees.items() if isinstance(trees, dict) else enumerate(trees)):
        clades = {}
        leaves = [n for n in t.nodes if t.out_degree(n) == 0 and n in m.index]
        for n in leaves:
            for par in t.predecessors(n):
                clades.setdefault(par, set()).add(m.at[n, "Cell_type"])
        cond = m.loc[leaves, ["time", "morphine", "inhib"]].drop_duplicates()
        assert len(cond) == 1, f"tree {name}: leaves from more than one condition"
        out.append({"tree": str(name), "condition": tuple(cond.iloc[0]), "clades": list(clades.values())})
    print("trees", len(out), flush=True)
    return out


def score_zebrafish(key, A, pos, rj, Q, zf):
    rows = []
    ls = A["source"].obs["Cell_type"].astype(str).to_numpy()[pos["source"]][~rj["source"]]
    lt = A["target"].obs["Cell_type"].astype(str).to_numpy()[pos["target"]][~rj["target"]]
    grp = np.array([zf_group(x) for x in ls])
    row_mass = Q.sum(1); tot = row_mass.sum()
    us = sorted(set(ls))
    cond = tuple(A["target"].obs[["time", "morphine", "inhib"]].astype(str).iloc[0])
    trees = [t for t in zf if t["condition"] == cond]
    for a_ in ZF_ACTIVATED:
        m = lt == a_
        if not m.any():
            continue
        col = Q[:, m].sum(1); cm = col.sum()
        if cm <= 0:
            continue
        for gname in ("epicardium", "endocardium", "constitutive fibroblasts", "other fibroblasts", "other"):
            gm = grp == gname
            rows.append({**key, "table": "activated_origin", "target_type": a_, "retained_target_cells": int(m.sum()),
                         "source_group": gname, "incoming_share": float(col[gm].sum() / cm),
                         "source_mass_share": float(row_mass[gm].sum() / tot)})
        p_src = {b: float(col[ls == b].sum() / cm) for b in us}
        rel = {}
        for b in us:
            vals = []
            for t in trees:
                with_a = [c for c in t["clades"] if a_ in c]
                if with_a:
                    vals.append(np.mean([b in c for c in with_a]))
            if vals:
                rel[b] = float(np.mean(vals))
        bb = sorted(rel)
        rho = float(spearmanr([p_src[b] for b in bb], [rel[b] for b in bb]).correlation) if len(bb) >= 3 else np.nan
        rows.append({**key, "table": "tree_relatedness", "target_type": a_, "target_condition": "/".join(cond), "trees": len(trees),
                     "trees_with_type": int(sum(any(a_ in c for c in t["clades"]) for t in trees)), "source_types": len(bb),
                     "spearman_coupling_vs_relatedness": rho})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--app", type=Path, required=True)
    p.add_argument("--dataset", required=True, choices=list(LABEL))
    p.add_argument("--gmt", type=Path, default=None, help="hallmark gmt (Kang IFN sets)")
    p.add_argument("--data", type=Path, default=None, help="downloaded data root (SPLINTR baseline library, zebrafish trees)")
    a = p.parse_args()
    import anndata as ad
    ds = a.dataset; root = a.app / ds / "ot_20261123"; out = root / "score"; out.mkdir(exist_ok=True)
    man = pd.read_csv(a.app / ds / "manifest.csv", keep_default_na=False)
    wl = pd.read_csv(root / "WORKLIST.tsv", sep="	", dtype=str).set_index(["unit_id", "key"]).out  # the worker's row folders
    lab = LABEL[ds]
    ifn = None
    if ds == "kang":
        sets = read_gmt(a.gmt, {"HALLMARK_INTERFERON_ALPHA_RESPONSE", "HALLMARK_INTERFERON_GAMMA_RESPONSE"})
        ifn = sorted(set(sum(sets.values(), [])))
    G, C, SL, D, EX = [], [], [], [], []
    base_bc, zf = {}, None
    if ds == "splintr":
        base_bc = splintr_baseline_barcodes(a.data)
    if ds == "zebrafish":
        zf = zebrafish_clades(a.data)
    for _, r in man.iterrows():
        u = r.pair_id
        A = {s: ad.read_h5ad(r[f"{s}_h5ad"]) for s in ("source", "target")}
        for s in A:
            A[s].obs_names = A[s].obs_names.astype(str)
        tot = {s: np.asarray(A[s].X.sum(1)).ravel() for s in A}
        prog = {}
        if ds == "kang":
            prog = {s: program(A[s], ifn)[0] for s in A}
        if ds == "sciplex2":
            prog = {s: (program(A[s], S_GENES)[0], program(A[s], G2M_GENES)[0]) for s in A}
        for tag, cid in CONFIGS.items():
            d = Path(wl.loc[(u, cid.rsplit("|", 3)[0])]) / cid.replace("|", "__")
            if not (d / "coupling_m4e.npz").exists():
                print("missing", d, flush=True); continue
            cc = pd.read_csv(d / "cell_confidence.csv.gz", usecols=["method", "side", "observation_id", "rejected"])
            cc["observation_id"] = cc.observation_id.astype(str)
            for method in ("M4-E", "M4-R"):
                z = np.load(d / f"coupling_{method.lower().replace('-', '')}.npz")
                P = z["coupling"].astype(np.float64); sid, tid = z["source_ids"].astype(str), z["target_ids"].astype(str)
                g = cc[cc.method == method]
                rj = {"source": g[g.side == "source"].set_index("observation_id").rejected.reindex(sid).to_numpy(bool),
                      "target": g[g.side == "target"].set_index("observation_id").rejected.reindex(tid).to_numpy(bool)}
                pos = {"source": A["source"].obs_names.get_indexer(sid), "target": A["target"].obs_names.get_indexer(tid)}
                assert (pos["source"] >= 0).all() and (pos["target"] >= 0).all()
                key = {"dataset": ds, "unit_id": u, "config": tag, "config_id": cid, "method": method}
                for side in ("source", "target"):
                    G.append({**key, "side": side, "label": "ALL", "cells": len(rj[side]), "rejected_share": float(rj[side].mean())})
                    D.append({**key, "side": side, "depth_auroc_rejected_deeper": auroc(tot[side][pos[side]], rj[side]),
                              "rejected": int(rj[side].sum()), "cells": len(rj[side])})
                    if lab:
                        lv = A[side].obs[lab].astype(str).to_numpy()[pos[side]]
                        for v in sorted(set(lv)):
                            m = lv == v
                            G.append({**key, "side": side, "label": v, "cells": int(m.sum()), "rejected_share": float(rj[side][m].mean())})
                rs, rt = ~rj["source"], ~rj["target"]
                Q = P[np.ix_(rs, rt)]
                if lab and Q.sum() > 0:
                    ls = A["source"].obs[lab].astype(str).to_numpy()[pos["source"]][rs]
                    lt = A["target"].obs[lab].astype(str).to_numpy()[pos["target"]][rt]
                    us, ut = sorted(set(ls)), sorted(set(lt))
                    M = np.array([[Q[np.ix_(ls == x, lt == y)].sum() for y in ut] for x in us]); T = M.sum()
                    for i, x in enumerate(us):
                        for j, y in enumerate(ut):
                            if M[i, j] > 0:
                                C.append({**key, "source_label": x, "target_label": y, "mass": M[i, j] / T,
                                          "p_target_given_source": M[i, j] / M[i].sum(),
                                          "enrichment": M[i, j] * T / (M[i].sum() * M[:, j].sum())})
                    shared = sorted(set(us) & set(ut))
                    if shared:
                        obs_ = sum(M[us.index(k), ut.index(k)] for k in shared) / T
                        exp_ = sum((M[us.index(k)].sum() / T) * (M[:, ut.index(k)].sum() / T) for k in shared)
                        SL.append({**key, "labels_shared": len(shared), "same_label_mass": obs_, "independent_expectation": exp_})
                if ds == "kang" and Q.sum() > 0:
                    w = Q / Q.sum(1, keepdims=True).clip(1e-300)
                    ps, pt = prog["source"][pos["source"]][rs], prog["target"][pos["target"]][rt]
                    shift = w @ pt - ps
                    ct_ = A["source"].obs["cell_type"].astype(str).to_numpy()[pos["source"]][rs]
                    for v in sorted(set(ct_)):
                        m = ct_ == v
                        EX.append({**key, "table": "ifn_shift", "source_cell_type": v, "cells": int(m.sum()),
                                   "mean_shift": float(shift[m].mean()), "ifn_genes_present": int(len(ifn))})
                if ds == "sciplex2":
                    for side in ("source", "target"):
                        sc, g2 = (x[pos[side]] for x in prog[side])
                        for nm, v in (("S", sc), ("G2M", g2)):
                            EX.append({**key, "table": "cycle", "side": side, "score": nm,
                                       "rejected_mean": float(np.nanmean(v[rj[side]])) if rj[side].any() else np.nan,
                                       "retained_mean": float(np.nanmean(v[~rj[side]])) if (~rj[side]).any() else np.nan,
                                       "genes_present": int(program(A[side], S_GENES if nm == "S" else G2M_GENES)[1])})
                    EX.append({**key, "table": "dose", "drug": A["target"].obs["drug"].astype(str).iloc[0],
                               "dose": float(A["target"].obs["dose"].iloc[0]), "target_cells": int(len(tid)),
                               "source_rejected": float(rj["source"].mean()), "target_rejected": float(rj["target"].mean())})
                if ds == "celegans" and Q.sum() > 0:
                    ln_s = A["source"].obs["lineage"].astype(str).to_numpy()[pos["source"]][rs]
                    ln_t = A["target"].obs["lineage"].astype(str).to_numpy()[pos["target"]][rt]
                    ok_s = ~np.isin(ln_s, ["nan", "", "None"]); ok_t = ~np.isin(ln_t, ["nan", "", "None"])
                    if ok_s.any() and ok_t.any():
                        S_ = Q[np.ix_(ok_s, ok_t)]; T = S_.sum()
                        us_, ui = np.unique(ln_s[ok_s], return_inverse=True); ut_, uj = np.unique(ln_t[ok_t], return_inverse=True)
                        comp = np.array([[compatible(x, y) for y in ut_] for x in us_])
                        Cm = comp[np.ix_(ui, uj)]
                        rsum, csum = S_.sum(1) / T, S_.sum(0) / T
                        EX.append({**key, "table": "lineage", "annotated_source": int(ok_s.sum()), "annotated_target": int(ok_t.sum()),
                                   "mass_annotated_block": float(T / Q.sum()), "compatible_mass_share": float(S_[Cm].sum() / T),
                                   "independent_expectation": float((rsum[:, None] * csum[None, :])[Cm].sum())})
                    for side in ("source", "target"):
                        ln = A[side].obs["lineage"].astype(str).to_numpy()[pos[side]]
                        depth_ = np.array([len(x.split("/")[0]) if x not in ("nan", "", "None") else -1 for x in ln])
                        for dv in sorted(set(depth_)):
                            m = depth_ == dv
                            EX.append({**key, "table": "lineage_depth", "side": side, "name_length": int(dv), "cells": int(m.sum()),
                                       "rejected_share": float(rj[side][m].mean())})
                if ds == "splintr":
                    EX.extend(score_splintr(key, A, pos, rj, P, base_bc))
                if ds == "zebrafish" and Q.sum() > 0:
                    EX.extend(score_zebrafish(key, A, pos, rj, Q, zf))
                if ds == "pancreas":
                    st_ = A["source"].obs["cell_type"].astype(str).value_counts(normalize=True)
                    lt_ = A["target"].obs["cell_type"].astype(str).to_numpy()[pos["target"]]
                    rare = np.array([st_.get(v, 0.0) < 0.01 for v in lt_])
                    EX.append({**key, "table": "emerging", "rare_types": ";".join(sorted(set(lt_[rare]))), "rare_cells": int(rare.sum()),
                               "rare_rejected": float(rj["target"][rare].mean()) if rare.any() else np.nan,
                               "other_rejected": float(rj["target"][~rare].mean())})
        print("scored", u, flush=True)
    for name, rows in (("gates", G), ("coupling", C), ("same_label", SL), ("depth", D)):
        pd.DataFrame(rows).to_csv(out / f"{name}.tsv", sep="\t", index=False)
    ex = pd.DataFrame(EX)
    for t in (ex.table.unique() if len(ex) else []):
        ex[ex.table == t].dropna(axis=1, how="all").to_csv(out / f"{t}.tsv", sep="\t", index=False)
    if ds == "sciplex2" and len(ex):
        dd = ex[ex.table == "dose"]; rows = []
        for (cfg, mth, drug), g in dd.groupby(["config", "method", "drug"]):
            for side in ("source", "target"):
                rho = spearmanr(g.dose, g[f"{side}_rejected"]).correlation if len(g) > 2 else np.nan
                rows.append({"config": cfg, "method": mth, "drug": drug, "side": side, "doses": len(g), "spearman_rejected_vs_dose": rho})
            rows.append({"config": cfg, "method": mth, "drug": drug, "side": "target_cells", "doses": len(g),
                         "spearman_rejected_vs_dose": spearmanr(g.dose, g.target_cells).correlation if len(g) > 2 else np.nan})
        pd.DataFrame(rows).to_csv(out / "dose_spearman.tsv", sep="\t", index=False)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
