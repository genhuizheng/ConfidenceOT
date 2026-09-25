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

The median pair scale and the calibrated rejection cost are carried through as
diagnostics, not metrics. If the scale moves with the level while the metrics
stay flat, the cost normalisation is doing its job; if the metrics move while
the scale stays put, the relative geometry was distorted, and that is the
failure this benchmark exists to catch.
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


def coupling_metrics(plan: np.ndarray, source_group: np.ndarray,
                     target_group: np.ndarray, source_value: np.ndarray,
                     target_value: np.ndarray) -> dict:
    mass = plan.sum()
    if mass <= 0:
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
    """The two numbers that separate a scale change from a geometry change.

    Looked up rather than assumed: run.json carries the calibrated rejection
    cost and the preprocessing record, and the cost scale lives in whichever
    of the two files the version in use happens to write it to.
    """
    found: dict = {}
    for name in ("run.json", "calibration.json"):
        path = run / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            found.update(payload)
    scale = next((found[key] for key in
                  ("cost_scale", "scale", "median_pair_scale")
                  if isinstance(found.get(key), (int, float))), float("nan"))
    return {"rejection_cost": found.get("rejection_cost", float("nan")),
            "median_pair_scale": scale,
            "calibration_null": found.get("calibration_null"),
            "preprocessing_label": found.get("preprocessing_label")}


def score_pair(run: Path, truth: pd.DataFrame) -> list[dict]:
    gate = pd.read_csv(run / "cell_confidence.csv")
    summary = diagnostics(run)
    rows = []
    for method, block in gate.groupby("method"):
        record: dict = {"method": str(method)}
        for side in ("source", "target"):
            cells = block[block["side"] == side]
            if cells.empty:
                continue
            answer = truth[truth["side"] == side]
            if len(answer) != len(cells):
                raise ValueError(
                    f"{side}: {len(cells)} gated cells against "
                    f"{len(answer)} in the truth table")
            retained = cells["retained"].to_numpy(dtype=bool)
            should_reject = answer["should_reject"].to_numpy(dtype=bool)
            cost = cells["decision_cost"].to_numpy(dtype=float)
            total = cells["total_counts"].to_numpy(dtype=float)
            detected = cells["n_genes_by_counts"].to_numpy(dtype=float)

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
            record[f"{side}_rejected_n"] = int((~retained).sum())
            record[f"{side}_cells_n"] = int(retained.size)
        plan_path = run / f"coupling_{str(method).lower().replace('-', '')}.npz"
        if plan_path.exists():
            with np.load(plan_path) as handle:
                plan = np.asarray(handle["coupling"], dtype=np.float64)
            source_cells = block[block["side"] == "source"]
            target_cells = block[block["side"] == "target"]
            record.update(coupling_metrics(
                plan,
                truth.loc[truth["side"] == "source", "group"].to_numpy(),
                truth.loc[truth["side"] == "target", "group"].to_numpy(),
                source_cells["total_counts"].to_numpy(dtype=float),
                target_cells["total_counts"].to_numpy(dtype=float)))
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
        run = args.run_root / str(row["pair_id"]) / f"scope_{args.scope}"
        if not (run / "cell_confidence.csv").exists():
            missing.append(str(row["pair_id"]))
            continue
        truth = pd.read_csv(row["truth_csv"])
        for record in score_pair(run, truth):
            record.update({
                "pair_id": row["pair_id"], "preprocessing": args.preprocessing,
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
