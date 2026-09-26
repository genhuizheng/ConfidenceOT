"""Score one finished ConfidenceOT run against the construction's ground truth.

Two layers, because each is defined where the other is blind.

**The gate.** How many cells were rejected that should not have been, and
whether rejection follows the measurement rather than the biology. The
correlations are Spearman between the continuous decision cost and each
readout; they are defined even when nothing is rejected, which the rate is not.

**The coupling.** The fraction of transported mass that lands on a cell of the
same population, which has exact ground truth because the labels came out of
the simulator, and the correlation between a cell's readout and the readout at
the centre of mass it is transported to. The first is the only measure here of
whether OT got the right answer rather than a defensible one; the second is
the only one that tests the transport rather than the threshold. A method can
have a clean gate and still be moving mass along the measurement.

The cost scale and the calibrated rejection cost are carried through as
diagnostics, not metrics. If the scale moves with the level while the metrics
stay flat, the cost normalisation is doing its job; if the metrics move while
the scale stays put, the relative geometry was distorted, and that is the
failure this benchmark exists to catch.

**nCount and nFeature are read from the h5ads, not from the run.** The gate
table does not carry them, and the benchmark wrote those files, so taking the
readouts from its own inputs removes a dependency on what the solver happened
to record. They are aligned on ``observation_id``, which the gate carries and
which is the h5ad's own index.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True,
                        help="Output root that 02_run_pair.py wrote into")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preprocessing", required=True)
    parser.add_argument("--scope", default="all")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def rank_auc(values: np.ndarray, positive: np.ndarray) -> float:
    """Mann-Whitney AUC with the retained cells as the positive class."""
    finite = np.isfinite(values)
    values, positive = values[finite], positive[finite]
    n_positive = int(positive.sum())
    n_negative = int(values.size - n_positive)
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    ranks = rankdata(values)
    return float(
        (ranks[positive].sum() - n_positive * (n_positive + 1) / 2.0)
        / (n_positive * n_negative))


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    ok = np.isfinite(left) & np.isfinite(right)
    if ok.sum() < 3 or np.ptp(left[ok]) == 0 or np.ptp(right[ok]) == 0:
        return float("nan")
    return float(spearmanr(left[ok], right[ok]).statistic)


def readouts(path: str) -> pd.DataFrame:
    """nCount and nFeature per cell, indexed the way the gate names cells."""
    import anndata as ad
    from scipy import sparse

    data = ad.read_h5ad(path)
    matrix = sparse.csr_matrix(data.X)
    return pd.DataFrame(
        {"total_counts": np.asarray(matrix.sum(axis=1)).ravel(),
         "detected_genes": np.asarray((matrix > 0).sum(axis=1)).ravel()},
        index=pd.Index(data.obs_names.astype(str), name="observation_id"))


def coupling_metrics(plan: np.ndarray, source_group: np.ndarray,
                     target_group: np.ndarray, source_value: np.ndarray,
                     target_value: np.ndarray) -> dict:
    mass = plan.sum()
    if mass <= 0 or plan.shape != (source_group.size, target_group.size):
        return {"same_population_mass": float("nan"),
                "coupling_nuisance_rho": float("nan")}
    same = np.zeros_like(plan, dtype=bool)
    for label in np.unique(source_group):
        same |= np.outer(source_group == label, target_group == label)
    row = plan.sum(axis=1)
    centre = np.where(row > 0, plan @ target_value / np.where(row > 0, row, 1.0),
                      np.nan)
    return {"same_population_mass": float(plan[same].sum() / mass),
            "coupling_nuisance_rho": correlation(source_value, centre)}


def diagnostics(run: Path) -> dict:
    """Looked up rather than assumed: the keys live across two files."""
    found: dict = {}
    for name in ("run.json", "calibration.json"):
        path = run / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            found.update(payload)
    return {key: found.get(key, float("nan")) for key in
            ("rejection_cost", "cost_scale", "cost_median", "cost_max")} | {
        "calibration_null": found.get("calibration_null"),
        "calibration_valid_for_m4r": found.get("calibration_valid_for_m4r"),
        "preprocessing_label": found.get("preprocessing_label")}


def score_pair(run: Path, truth: pd.DataFrame, row: pd.Series) -> list[dict]:
    gate = pd.read_csv(run / "cell_confidence.csv")
    summary = diagnostics(run)
    measured = {side: readouts(row[f"{side}_h5ad"])
                for side in ("source", "target")}
    rows = []
    for method, block in gate.groupby("method"):
        record: dict = {"method": str(method)}
        group_of: dict[str, np.ndarray] = {}
        value_of: dict[str, np.ndarray] = {}
        for side in ("source", "target"):
            cells = block[block["side"] == side]
            answer = truth[truth["side"] == side]
            if cells.empty:
                continue
            if len(answer) != len(cells):
                raise ValueError(
                    f"{side}: {len(cells)} gated cells against "
                    f"{len(answer)} in the truth table")
            # The truth table is in the h5ad's order, and so is the gate, but
            # a join on the name is the only version of that which cannot go
            # quietly wrong.
            joined = cells.set_index(
                cells["observation_id"].astype(str)).join(
                measured[side], how="left")
            if joined[["total_counts", "detected_genes"]].isna().any().any():
                raise ValueError(f"{side}: a gated cell is not in the h5ad")

            retained = joined["retained"].to_numpy(dtype=bool)
            should_reject = answer["should_reject"].to_numpy(dtype=bool)
            cost = joined["decision_cost"].to_numpy(dtype=float)
            total = joined["total_counts"].to_numpy(dtype=float)
            detected = joined["detected_genes"].to_numpy(dtype=float)
            group_of[side] = answer["group"].to_numpy()
            value_of[side] = total

            matched = ~should_reject
            record[f"{side}_false_rejection_rate"] = float(
                (~retained[matched]).mean()) if matched.any() else float("nan")
            rejected = ~retained
            if should_reject.any():
                record[f"{side}_recall"] = float(rejected[should_reject].mean())
                record[f"{side}_precision"] = (
                    float(should_reject[rejected].mean())
                    if rejected.any() else float("nan"))
            else:
                # Nothing to find on this side, so recall and precision are
                # undefined rather than perfect; the false rejection rate is
                # what carries the answer here.
                record[f"{side}_recall"] = float("nan")
                record[f"{side}_precision"] = float("nan")
            record[f"{side}_auc_total_counts"] = rank_auc(total, retained)
            record[f"{side}_auc_detected_genes"] = rank_auc(detected, retained)
            record[f"{side}_rho_cost_total_counts"] = correlation(cost, total)
            record[f"{side}_rho_cost_detected_genes"] = correlation(cost, detected)
            record[f"{side}_rejected_n"] = int(rejected.sum())
            record[f"{side}_cells_n"] = int(retained.size)

        plan_path = run / f"coupling_{str(method).lower().replace('-', '')}.npz"
        if plan_path.exists() and {"source", "target"} <= set(group_of):
            with np.load(plan_path) as handle:
                plan = np.asarray(handle["coupling"], dtype=np.float64)
            record.update(coupling_metrics(
                plan, group_of["source"], group_of["target"],
                value_of["source"], value_of["target"]))
        else:
            record.update({"same_population_mass": float("nan"),
                           "coupling_nuisance_rho": float("nan")})
        record.update(summary)
        rows.append(record)
    return rows


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    collected = []
    missing = []
    for _, row in manifest.iterrows():
        # The solver writes scope_<scope>/budget_<value>, and the budget is
        # not knowable from here: it is whatever the run used.
        found = sorted((args.run_root / str(row["pair_id"])
                        / f"scope_{args.scope}").glob("budget_*"))
        ran = [path for path in found if (path / "cell_confidence.csv").exists()]
        if not ran:
            missing.append(str(row["pair_id"]))
            continue
        truth = pd.read_csv(row["truth_csv"])
        for run in ran:
            for record in score_pair(run, truth, row):
                record.update({
                    "pair_id": row["pair_id"],
                    "preprocessing": args.preprocessing,
                    "budget_dir": run.name,
                    "n_cells": row["n_cells"], "replicate": row["replicate"],
                    "technical_level": row["technical_level"],
                    "biological_case": row["biological_case"],
                    "R_nCount": row["R_nCount"], "R_nFeature": row["R_nFeature"],
                    "rank_cut_valid": row["rank_cut_valid"],
                })
                collected.append(record)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(collected)
    table.to_csv(args.out, index=False)
    print(f"scored {len(table)} rows into {args.out}")
    if missing:
        print(f"{len(missing)} pairs had no gate output: {missing[:8]}"
              + (" ..." if len(missing) > 8 else ""))


if __name__ == "__main__":
    main()
