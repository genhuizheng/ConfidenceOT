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
from functools import lru_cache
from scipy.stats import rankdata, spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True, action="append",
                        help="Output root that 02_run_pair.py wrote into. "
                             "Repeat it to score several configurations of "
                             "one condition in a single pass, which is what "
                             "makes the readout cache worth having: nCount "
                             "and nFeature are properties of the input h5ad "
                             "and do not depend on the preprocessing, so "
                             "reading them per configuration is the same "
                             "work twelve times over.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preprocessing", action="append", default=None,
                        help="Label for each --run-root, in the same order. "
                             "Omitted, each root is named after its own "
                             "directory, which is how the runs were laid out.")
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


@lru_cache(maxsize=None)
def readouts(path: str) -> pd.DataFrame:
    """nCount and nFeature per cell, indexed the way the gate names cells.

    Cached on the path. Every configuration scored against one condition
    reads the same files: the counts are the benchmark's input, not the
    run's output, so nothing about them changes with the preprocessing.
    """
    import anndata as ad
    from scipy import sparse

    data = ad.read_h5ad(path)
    matrix = sparse.csr_matrix(data.X)
    return pd.DataFrame(
        {"total_counts": np.asarray(matrix.sum(axis=1)).ravel(),
         "detected_genes": np.asarray((matrix > 0).sum(axis=1)).ravel()},
        index=pd.Index(data.obs_names.astype(str), name="observation_id"))


COUPLING_COLUMNS = ("same_population_mass", "coupling_nuisance_rho",
                    "transported_mass", "asserted_mass", "false_asserted_mass",
                    "asserted_same_population_fraction")


def coupling_metrics(plan: np.ndarray, source_group: np.ndarray,
                     target_group: np.ndarray, source_value: np.ndarray,
                     target_value: np.ndarray, source_kept: np.ndarray,
                     target_kept: np.ndarray) -> dict:
    """What the transport did, and separately what the gate stands behind.

    ``same_population_mass`` is over the whole plan, and it describes the
    backbone: where the transport put mass, gate or no gate. It is not what
    ConfidenceOT asserts. The transport runs over every cell -- a rejected
    one pays the rejection cost instead of its own -- so a rejected cell still
    carries mass, and on the disjoint case most of the plan is exactly that:
    rejected sources sending mass into retained targets. Scoring the whole
    plan charges the method for mass it explicitly disowned.

    What it does assert is the block between retained cells on both sides.
    ``false_asserted_mass`` is the cross-population mass inside that block,
    and its ideal is 0 in every case: nothing wrong asserted when everything
    is shared, the unmatched population not asserted when one is missing, and
    nothing at all when nothing is shared. That makes it the one metric that
    reads the disjoint case correctly. Rejecting every source cell empties the
    block whatever the target gate says, so a one-sided reject-all is a
    complete answer here rather than half of one, and a rate-based reading
    that demands both sides reach 1.0 marks it wrong.
    """
    mass = plan.sum()
    if mass <= 0 or plan.shape != (source_group.size, target_group.size):
        return {name: float("nan") for name in COUPLING_COLUMNS}
    # Mass carried from each source population to each target population,
    # as one population-by-population table; the same-population mass is its
    # diagonal. This replaced a boolean mask the size of the plan, built from
    # one outer product per label and then used to index the plan -- at
    # N=10000 that was 10^8 booleans allocated six times per coupling, and it
    # was most of why scoring took three hours. Labels are the union of both
    # sides, so a population present on one side only contributes a zero
    # column or row and therefore nothing to the diagonal, which is what
    # populations_disjoint needs: no mass can be same-population there.
    labels = np.union1d(source_group, target_group)
    source_onehot = (source_group[:, None] == labels[None, :]).astype(np.float64)
    target_onehot = (target_group[:, None] == labels[None, :]).astype(np.float64)
    by_population = source_onehot.T @ plan @ target_onehot
    # The same table restricted to retained cells on both sides: zeroing a
    # rejected cell's one-hot row removes its mass from every entry.
    kept_by_population = (
        (source_onehot * source_kept[:, None]).T
        @ plan
        @ (target_onehot * target_kept[:, None]))
    asserted = float(kept_by_population.sum())
    asserted_same = float(np.trace(kept_by_population))
    row = plan.sum(axis=1)
    centre = np.where(row > 0, plan @ target_value / np.where(row > 0, row, 1.0),
                      np.nan)
    return {"same_population_mass": float(np.trace(by_population) / mass),
            "coupling_nuisance_rho": correlation(source_value, centre),
            "transported_mass": float(mass),
            "asserted_mass": asserted,
            "false_asserted_mass": asserted - asserted_same,
            "asserted_same_population_fraction": (
                asserted_same / asserted if asserted > 0 else float("nan"))}


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
        # calibration.json's own statement of what the cost had to accept.
        "acceptance_requirement": found.get("acceptance_requirement"),
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
        kept_of: dict[str, np.ndarray] = {}
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
            kept_of[side] = retained.astype(np.float64)

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
                value_of["source"], value_of["target"],
                kept_of["source"], kept_of["target"]))
        else:
            record.update({name: float("nan") for name in COUPLING_COLUMNS})
        record.update(summary)
        rows.append(record)
    return rows


def score_root(run_root, label: str, manifest, truths: dict, scope: str,
               missing: list) -> list[dict]:
    """Every pair of one condition under one configuration."""
    collected = []
    for _, row in manifest.iterrows():
        # The solver writes scope_<scope>/budget_<value>, and the budget is
        # not knowable from here: it is whatever the run used.
        found = sorted((run_root / str(row["pair_id"])
                        / f"scope_{scope}").glob("budget_*"))
        ran = [path for path in found if (path / "cell_confidence.csv").exists()]
        if not ran:
            missing.append(f"{label}/{row['pair_id']}")
            continue
        truth = truths[str(row["pair_id"])]
        for run in ran:
            for record in score_pair(run, truth, row):
                record.update({
                    "pair_id": row["pair_id"],
                    "preprocessing": label,
                    "budget_dir": run.name,
                    "n_cells": row["n_cells"], "replicate": row["replicate"],
                    "technical_level": row["technical_level"],
                    "biological_case": row["biological_case"],
                    "R_nCount": row["R_nCount"], "R_nFeature": row["R_nFeature"],
                    "rank_cut_valid": row["rank_cut_valid"],
                })
                collected.append(record)
    return collected


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    labels = args.preprocessing or [root.name for root in args.run_root]
    if len(labels) != len(args.run_root):
        raise SystemExit(
            f"{len(args.run_root)} run roots against {len(labels)} labels")
    # The truth table belongs to the condition, not to the configuration, so
    # it is read once rather than once per root.
    truths = {str(row["pair_id"]): pd.read_csv(row["truth_csv"])
              for _, row in manifest.iterrows()}
    collected = []
    missing = []
    for run_root, label in zip(args.run_root, labels):
        collected.extend(score_root(run_root, label, manifest, truths,
                                    args.scope, missing))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(collected)
    table.to_csv(args.out, index=False)
    print(f"scored {len(table)} rows into {args.out}")
    if missing:
        print(f"{len(missing)} pairs had no gate output: {missing[:8]}"
              + (" ..." if len(missing) > 8 else ""))


if __name__ == "__main__":
    main()
