"""RETrace2 PLAN.md (post-hoc coverage control) and the two OTs on the same sample pairs.

    python -I lineage_validation/retrace2_20261112/02_coverage_and_ot.py --out results/retrace2_20261112

Reads OUT/prep (01_). Coverage: D_L regressed on log(shared loci) (one linear fit over all pairs); residual shifted
to be positive = D_L,res; average-linkage tree, the authors' 10-20-cell clade rule, association with 01_'s nulls.
OT per pair and cost (D_L, D_L,res, D_M): ConfidenceOT M4-E as 10_crispr_ot.py (cost = D / median cross D, within-side
split null 5 + 5, grid 5, acceptance 0.99, eps 0.1, lambda 1, tol 1e-4). Compared per pair: gate kappa, common-support
TV and top-5 overlap (chance k / common sources), clade-level Shared, same-clade mass (raw-D_L clades).
Writes OUT/analysis/{association_residual.tsv, cells_residual_clades.tsv} and OUT/ot/{fits.tsv, compare.tsv, gates.tsv,
couplings/<source>__<target>__<cost>.npz}.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
from confidenceot import ConfidenceOT  # noqa: E402
from confidenceot.calibration import calibrate_confidence_cost  # noqa: E402

SEED, N_CAL, N_VAL, NPERM = 20261112, 5, 5, 999
FIT = dict(backbone="uot", epsilon=0.1, lambda_a=1.0, lambda_b=1.0, source_rejection_budget=0.95, target_rejection_budget=0.95,
           tolerance=1e-4, device="cpu")
PAIRS = [("Brain_LeftAnteriorCortex", "Brain_LeftPosteriorCortex"), ("Liver_LeftLobe", "Liver_MedianLobe"), ("Liver_LeftLobe", "Liver_RightLobe"),
         ("Liver_MedianLobe", "Liver_RightLobe"), ("Kidney_Left-Cortex-1", "Kidney_Right-Cortex-1"),
         ("Brain_LeftAnteriorCortex", "Liver_LeftLobe"), ("Liver_LeftLobe", "Kidney_Right-Cortex-1"), ("Brain_LeftPosteriorCortex", "Kidney_Right-Cortex-1")]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def clades_from_linkage(Z, names, lo=10, hi=20):
    root = to_tree(Z); out, used = [], set()

    def leaves(nd):
        return [names[i] for i in nd.pre_order()]

    def rec(nd):
        lv = leaves(nd)
        if lo <= len(lv) <= hi and not set(lv) & used:
            out.append(lv); used.update(lv); return
        if len(lv) > hi and not nd.is_leaf():
            rec(nd.get_left()); rec(nd.get_right())
    rec(root)
    return sorted(out, key=len, reverse=True)


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else np.nan


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    prep, ana, oto = a.out / "prep", a.out / "analysis", a.out / "ot"; oto.mkdir(exist_ok=True)
    m01 = load("m01", HERE / "01_state_lineage.py")
    C = pd.read_csv(prep / "cells.tsv", sep="\t"); zl = np.load(prep / "D_lineage.npz"); zm = np.load(prep / "D_meth.npz")
    DL, NT, DM = zl["D"], zl["shared_targets"], zm["D"]; n = len(DL); iu = np.triu_indices(n, 1)
    b = np.polyfit(np.log(NT[iu]), DL[iu], 1)
    res = DL[iu] - np.polyval(b, np.log(NT[iu])); res = res - res.min() + 1e-3
    DR = np.zeros((n, n)); DR[iu] = res; DR = DR + DR.T
    Z = linkage(squareform(DR, checks=False), "average")
    cl = clades_from_linkage(Z, list(C.cell))
    rc = {c: f"rclade{k + 1}" for k, lv in enumerate(cl) for c in lv}
    C["clade_res"] = C.cell.map(rc).fillna("unclustered")
    C.to_csv(ana / "cells_residual_clades.tsv", sep="\t", index=False)
    print(f"coverage fit: D = {b[1]:.4f} + {b[0]:.4f} log(shared); residual clades {len(cl)}, unclustered {(C.clade_res == 'unclustered').sum()}")
    rng = np.random.default_rng(SEED); rows = []
    for cladecol in ("clade", "clade_res"):
        for name, lab_col, within, subset in (("organ", "organ", None, None), ("tissue", "tissue", "organ", None),
                                              ("brain_celltype", "cell_type_assignment", None, "Brain")):
            D = C[C[cladecol] != "unclustered"]
            if subset:
                D = D[D.organ == subset]
            ci = pd.Index(sorted(D[cladecol].unique())).get_indexer(D[cladecol]); lv = pd.Index(sorted(D[lab_col].unique())); li = lv.get_indexer(D[lab_col])
            if ci.max() < 1 or len(lv) < 2:
                continue
            grp = D[within].to_numpy() if within else np.zeros(len(D))

            def tab(l):
                T = np.zeros((ci.max() + 1, len(lv))); np.add.at(T, (ci, l), 1); return T
            V = m01.cramer(tab(li)); Vn = np.array([m01.cramer(tab(m01.permute_within(li, grp, rng))) for _ in range(NPERM)])
            rows.append({"clades_from": "raw D_L (authors' tree)" if cladecol == "clade" else "coverage-residual D_L", "label": name,
                         "null": f"within {within}" if within else "global", "cells": len(D), "clades": ci.max() + 1, "cramer_v": V,
                         "null_mean": Vn.mean(), "excess": V - Vn.mean(), "p": float((1 + (Vn >= V).sum()) / (NPERM + 1))})
            if cladecol == "clade_res":
                pd.DataFrame(tab(li), index=sorted(D[cladecol].unique()), columns=lv).to_csv(ana / f"table_rclade_x_{name}.tsv", sep="\t")
    A = pd.DataFrame(rows); A.to_csv(ana / "association_residual.tsv", sep="\t", index=False); print(A.round(3).to_string(index=False))
    # ---------- OT ----------
    tis = C.tissue.to_numpy(); fits, comps, gl = [], [], []
    costs = {"lineage": DL, "lineage_res": DR, "methylation": DM}
    for pi_, (s, t) in enumerate(PAIRS):
        si, ti = np.flatnonzero(tis == s), np.flatnonzero(tis == t)
        res_ = {}
        for cname, D in costs.items():
            X = D[np.ix_(si, ti)]; scale = float(np.median(X[X > 0]))
            cost = X / scale; rng = np.random.default_rng(SEED + pi_)
            nulls = {}
            limit = min(len(si), len(ti))          # as 10_crispr_ot.py: both sides subsampled to one size
            for side, idx_all in (("source", si), ("target", ti)):
                idx = np.sort(rng.choice(idx_all, limit, replace=False))
                W = D[np.ix_(idx, idx)] / scale; reps = []
                for r in range(N_CAL + N_VAL):
                    o = np.random.default_rng(SEED * 31 + pi_ * 101 + r).permutation(len(idx)); h = len(idx) // 2
                    reps.append(W[np.ix_(np.sort(o[:h]), np.sort(o[h:2 * h]))])
                nulls[side] = reps
            try:
                cal = calibrate_confidence_cost(nulls["source"][:N_CAL] + nulls["target"][:N_CAL], nulls["source"][N_CAL:] + nulls["target"][N_CAL:],
                                                backbone="uot", epsilon=0.1, lambda_a=1.0, lambda_b=1.0, null_semantics="within_side_split",
                                                within_side_acceptance_minimum=0.99, source_rejection_budget=0.95, target_rejection_budget=0.95,
                                                tolerance=1e-4, grid_size=5, device="cpu", workers=1, fallback_to_cpu=False)
                fit = ConfidenceOT(variant="exact", rejection_cost=float(cal.rejection_cost), **FIT).fit(cost)
            except Exception as e:  # noqa: BLE001
                fits.append({"pair": f"{s} -> {t}", "cost": cname, "error": str(e)[:200]}); continue
            res_[cname] = fit
            (oto / "couplings").mkdir(exist_ok=True)
            np.savez_compressed(oto / "couplings" / f"{s}__{t}__{cname}.npz", coupling=fit.coupling.astype(np.float32),
                                source=C.cell.to_numpy()[si].astype(str), target=C.cell.to_numpy()[ti].astype(str),
                                source_gate=np.asarray(fit.source_gate, bool), target_gate=np.asarray(fit.target_gate, bool))
            fits.append({"pair": f"{s} -> {t}", "cost": cname, "source_cells": len(si), "target_cells": len(ti), "rejection_cost": float(cal.rejection_cost),
                         "m4e_inference_valid": bool(cal.m4e_inference_valid), "feasible_cost_found": bool(cal.feasible_cost_found),
                         "retained_source": float(np.mean(fit.source_gate)), "retained_target": float(np.mean(fit.target_gate)),
                         "outer_converged": bool(fit.outer_converged)})
            for side, ids, gate in (("source", C.cell.to_numpy()[si], fit.source_gate), ("target", C.cell.to_numpy()[ti], fit.target_gate)):
                gl.append(pd.DataFrame({"pair": f"{s} -> {t}", "cost": cname, "side": side, "cell": ids, "retained": np.asarray(gate, bool)}))
        for A_, B_ in (("lineage", "methylation"), ("lineage_res", "methylation"), ("lineage", "lineage_res")):
            if A_ not in res_ or B_ not in res_:
                continue
            fa, fb = res_[A_], res_[B_]
            rec = {"pair": f"{s} -> {t}", "a": A_, "b": B_, "kappa_source": kappa(fa.source_gate, fb.source_gate), "kappa_target": kappa(fa.target_gate, fb.target_gate)}
            S = np.flatnonzero(np.asarray(fa.source_gate, bool) & np.asarray(fb.source_gate, bool)); T = np.flatnonzero(np.asarray(fa.target_gate, bool) & np.asarray(fb.target_gate, bool))
            if len(S) >= 2 and len(T) >= 1:
                Pa, Pb = fa.coupling[np.ix_(S, T)], fb.coupling[np.ix_(S, T)]
                ca, cb = Pa / np.where(Pa.sum(0) > 0, Pa.sum(0), 1), Pb / np.where(Pb.sum(0) > 0, Pb.sum(0), 1)
                k = min(5, len(S)); ta = np.argsort(-ca, 0)[:k]; tb = np.argsort(-cb, 0)[:k]
                rec.update(common_sources=len(S), common_targets=len(T), tv=float(np.median(0.5 * np.abs(ca - cb).sum(0))),
                           top5_overlap=float(np.mean([len(set(ta[:, j]) & set(tb[:, j])) / k for j in range(len(T))])), chance=k / len(S))
                cs = C.clade.to_numpy()[si][S]; ct = C.clade.to_numpy()[ti][T]
                lv = sorted(set(cs) | set(ct)); ix = {c: i for i, c in enumerate(lv)}
                Fa = np.eye(len(lv))[[ix[c] for c in cs]].T @ (Pa / Pa.sum()) @ np.eye(len(lv))[[ix[c] for c in ct]]
                Fb = np.eye(len(lv))[[ix[c] for c in cs]].T @ (Pb / Pb.sum()) @ np.eye(len(lv))[[ix[c] for c in ct]]
                same = [i for i, c in enumerate(lv) if c != "unclustered"]
                rec.update(clade_shared=float(np.minimum(Fa, Fb).sum() / np.maximum(Fa, Fb).sum()),
                           same_clade_mass_a=float(sum(Fa[i, i] for i in same)), same_clade_mass_b=float(sum(Fb[i, i] for i in same)))
            comps.append(rec)
        print(f"{s} -> {t}: done", flush=True)
    pd.DataFrame(fits).to_csv(oto / "fits.tsv", sep="\t", index=False); pd.DataFrame(comps).to_csv(oto / "compare.tsv", sep="\t", index=False)
    if gl:
        pd.concat(gl).to_csv(oto / "gates.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(fits).round(3).to_string(index=False)); print(pd.DataFrame(comps).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
