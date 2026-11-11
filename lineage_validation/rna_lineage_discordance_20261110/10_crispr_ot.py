"""View B of PLAN_v2.md: ConfidenceOT on the CRISPR distance D = 1 - s, with the existing solver and calibration.

    python lineage_validation/rna_lineage_discordance_20261110/10_crispr_ot.py \
        --inputs results/crispr_matrix_benchmark_20261108/inputs --bundles results/crispr_matrix_benchmark_20261108/bundles \
        --out results/rna_lineage_discordance_20261110/crispr_ot [--units UNIT ...] [--workers 4]

Per unit (the bundle's cells, in its order):
  D_ij  = 1 - s_ij, s from crispr_matrix_benchmark_20261108/matrix_core.Lineage (MATRIX_METRICS.md). KP: one Lineage
          over all the unit's cells (one mouse shares its intBCs). Quinn: one Lineage per LineageGroup; cells of
          different groups get D = 1 (distinct integrations). Other pairs without a comparable intBC: the unit's
          median evaluable cross-side D (share reported).
  cost  = D / median of D over 1e6 random cross-side pairs, zeros excluded (as median_pair_scale).
  calibration: within-side split nulls from the same D (two halves of one side, <= 2,000 cells per side, seeds as
          02_run_pair.py), 5 + 5 replicates, grid 5, acceptance minimum 0.99, no MNN; calibrate_confidence_cost with
          the 02_run_pair.py arguments.
  fit   : ConfidenceOT(backbone="uot") M4-E and M4-R, epsilon 0.1, lambda 1, tolerance 1e-4, budgets 0.95 (not
          enforced), CPU.
Writes OUT/<unit>/{gates.tsv.gz, coupling_m4e.npz, coupling_m4r.npz, audit.json} and OUT/AUDIT.tsv (one row per unit).
Refuses to overwrite a finished unit.
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MX = REPO / "lineage_validation" / "crispr_matrix_benchmark_20261108"
sys.path[:0] = [str(MX), str(REPO / "src")]
import matrix_core as mc  # noqa: E402
from confidenceot import ConfidenceOT  # noqa: E402
from confidenceot.calibration import calibrate_confidence_cost  # noqa: E402

SEED, MAX_CAL, N_CAL, N_VAL = 20260831, 2000, 5, 5
FIT = dict(backbone="uot", epsilon=0.1, lambda_a=1.0, lambda_b=1.0, source_rejection_budget=0.95,
           target_rejection_budget=0.95, tolerance=1e-4, device="cpu")


def bundles_module():
    spec = importlib.util.spec_from_file_location("bb", MX / "01_build_bundles.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


class UnitDistance:
    """D on demand for (source rows x target rows) and within one side, NaN where not evaluable."""

    def __init__(self, b: dict, t: pd.DataFrame, prior: pd.Series, map_ids):
        self.kp = b["dataset"] == "KPTracer"
        ids = pd.Index(t.drop_duplicates("cell").cell)
        self.ms = map_ids(pd.Series(b["source_ids"]), ids).to_numpy(dtype=object)
        self.mt = map_ids(pd.Series(b["target_ids"]), ids).to_numpy(dtype=object)
        lin_s = np.where(pd.isna(self.ms), None, b["source_lineage"]); lin_t = np.where(pd.isna(self.mt), None, b["target_lineage"])
        if self.kp:
            lin_s = np.where(pd.isna(self.ms), None, "unit"); lin_t = np.where(pd.isna(self.mt), None, "unit")
        self.lin_s, self.lin_t = lin_s, lin_t
        self.groups = {}
        for g in sorted({x for x in np.r_[lin_s, lin_t] if x is not None}):
            rs = np.flatnonzero(lin_s == g); rt = np.flatnonzero(lin_t == g)
            sub = t if self.kp else t[t.lineage == g]
            cells = np.r_[self.ms[rs], self.mt[rt]]
            L = mc.Lineage(sub[sub.cell.isin(set(cells))], cells, np.r_[np.repeat("source", len(rs)), np.repeat("target", len(rt))], prior)
            self.groups[g] = (L, rs, rt)

    def block(self, side_a: str, rows_a: np.ndarray, side_b: str, rows_b: np.ndarray) -> np.ndarray:
        D = np.ones((len(rows_a), len(rows_b)), dtype=np.float32)   # different group (Quinn) or absent cell: 1
        la = (self.lin_s if side_a == "source" else self.lin_t)[rows_a]
        lb = (self.lin_s if side_b == "source" else self.lin_t)[rows_b]
        for g, (L, rs, rt) in self.groups.items():
            ia = np.flatnonzero(la == g); ib = np.flatnonzero(lb == g)
            if len(ia) == 0 or len(ib) == 0:
                continue
            off_a = 0 if side_a == "source" else len(rs); off_b = 0 if side_b == "source" else len(rs)
            pos_a = pd.Index(rs if side_a == "source" else rt).get_indexer(rows_a[ia]) + off_a
            pos_b = pd.Index(rs if side_b == "source" else rt).get_indexer(rows_b[ib]) + off_b
            s, ok = L.agreement(pos_a, pos_b)
            D[np.ix_(ia, ib)] = np.where(ok, 1.0 - s, np.nan)
        return D


def run_unit(b: dict, uindex: int, tabs: dict, bb, out: Path, workers: int) -> dict:
    key = "KPTracer" if b["dataset"] == "KPTracer" else f"Quinn2021:{b['mouse']}"
    t, prior = tabs[key]
    t0 = time.perf_counter()
    U = UnitDistance(b, t, prior, bb.map_ids)
    ns, nt = len(b["source_ids"]), len(b["target_ids"])
    D = U.block("source", np.arange(ns), "target", np.arange(nt))
    nan = np.isnan(D)
    med = float(np.nanmedian(D)) if (~nan).any() else 1.0
    D[nan] = med
    rng = np.random.default_rng(SEED + uindex * 104729)
    k = min(1_000_000, ns * nt)
    dv = D[rng.integers(ns, size=k), rng.integers(nt, size=k)]
    dv = dv[dv > 0]
    scale = float(np.median(dv)) if dv.size else 1.0
    cost = D.astype(np.float64) / scale
    # within-side split nulls from the same distance
    limit = min(MAX_CAL, ns, nt)
    si = np.sort(rng.choice(ns, limit, replace=False)); ti = np.sort(rng.choice(nt, limit, replace=False))
    nulls = {}
    for side, idx, seed in (("source", si, SEED + uindex), ("target", ti, SEED + uindex + 7)):
        W = U.block(side, idx, side, idx); W[np.isnan(W)] = med
        reps = []
        for r in range(N_CAL + N_VAL):
            order = np.random.default_rng(seed * 1009 + r * 9173 + 41).permutation(limit)
            half = limit // 2
            left, right = np.sort(order[:half]), np.sort(order[half:2 * half])
            reps.append(W[np.ix_(left, right)].astype(np.float64) / scale)
        nulls[side] = reps
    cal = calibrate_confidence_cost(nulls["source"][:N_CAL] + nulls["target"][:N_CAL], nulls["source"][N_CAL:] + nulls["target"][N_CAL:],
                                    backbone="uot", epsilon=0.1, lambda_a=1.0, lambda_b=1.0, null_semantics="within_side_split",
                                    within_side_acceptance_minimum=0.99, source_rejection_budget=0.95, target_rejection_budget=0.95,
                                    tolerance=1e-4, grid_size=5, device="cpu", workers=workers, fallback_to_cpu=False)
    rc = float(cal.rejection_cost)
    gates, audit = [], {"unit_id": b["unit_id"], "dataset": b["dataset"], "mouse": b["mouse"], "source_cells": ns, "target_cells": nt,
                        "groups": len(U.groups), "cross_pairs_not_evaluable": float(nan.mean()), "fill_median_D": med,
                        "cross_pairs_D1_other_group": float((D == 1.0).mean()) if not U.kp else 0.0,
                        "cross_pairs_D0": float((D == 0).mean()), "scale": scale, "rejection_cost": rc,
                        "selection_status": cal.selection_status, "m4e_inference_valid": bool(cal.m4e_inference_valid),
                        "feasible_cost_found": bool(cal.feasible_cost_found), "m4r_validation_clean": bool(cal.m4r_validation_clean),
                        "calibration_warnings": list(cal.warning_messages)}
    for method, variant in (("M4-E", "exact"), ("M4-R", "reversible")):
        res = ConfidenceOT(variant=variant, rejection_cost=rc, **FIT).fit(cost)
        tag = method.lower().replace("-", "")
        np.savez_compressed(out / f"coupling_{tag}.npz", coupling=res.coupling.astype(np.float32),
                            source_ids=np.asarray(b["source_ids"], dtype=str), target_ids=np.asarray(b["target_ids"], dtype=str))
        for side, ids, gate, conf in (("source", b["source_ids"], res.source_gate, res.source_confidence),
                                      ("target", b["target_ids"], res.target_gate, res.target_confidence)):
            gates.append(pd.DataFrame({"method": method, "side": side, "observation_id": ids, "retained": np.asarray(gate, bool),
                                       "signed_rejection_margin": conf.signed_rejection_margin}))
        audit.update({f"{tag}_retained_source": float(np.mean(res.source_gate)), f"{tag}_retained_target": float(np.mean(res.target_gate)),
                      f"{tag}_outer_converged": bool(res.outer_converged), f"{tag}_inner_converged": bool(res.inner_converged),
                      f"{tag}_transported_mass": float(res.coupling.sum())})
    pd.concat(gates).to_csv(out / "gates.tsv.gz", sep="\t", index=False, compression="gzip")
    audit["seconds"] = round(time.perf_counter() - t0, 1)
    (out / "audit.json").write_text(json.dumps(audit, indent=1, default=float))
    return audit


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--bundles", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--units", nargs="*")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    bb = bundles_module()
    tabs = bb.load_tables(a.inputs)
    names = sorted(glob.glob(str(a.bundles / "*.pkl")), key=lambda f: Path(f).stat().st_size)   # smallest first
    for k, f in enumerate(names):
        b = pickle.load(open(f, "rb"))
        if a.units and b["unit_id"] not in a.units:
            continue
        o = a.out / b["unit_id"]
        if (o / "audit.json").exists():
            print(f"{b['unit_id']}: done, kept"); continue
        o.mkdir(parents=True, exist_ok=True)
        uindex = sorted(Path(x).stem for x in names).index(b["unit_id"])
        try:
            r = run_unit(b, uindex, tabs, bb, o, a.workers)
            print(f"{b['unit_id']}: cost {r['rejection_cost']:.3f} valid {r['m4e_inference_valid']} retained "
                  f"{r['m4e_retained_source']:.2f}/{r['m4e_retained_target']:.2f}, not evaluable {r['cross_pairs_not_evaluable']:.3f}, {r['seconds']} s", flush=True)
        except Exception as e:  # recorded, never skipped silently
            (o / "FAILED.txt").write_text(repr(e)); print(f"{b['unit_id']}: FAILED {e!r}", flush=True)
    rows = [json.loads(Path(x).read_text()) for x in glob.glob(str(a.out / "*" / "audit.json"))]
    pd.DataFrame(rows).drop(columns=["calibration_warnings"], errors="ignore").to_csv(a.out / "AUDIT.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
