"""Standalone specificity check for the ConfidenceOT rejection decision.

This script is deliberately independent of the Splatter scaling benchmark.  It
builds its own counts so that the ground truth is a construction rather than a
simulator parameter, then runs the exact production path used by the cancer
workflow: ``raw counts -> library size 1e4 -> log1p -> joint HVG -> gene
scaling -> joint PCA -> squared Euclidean cost -> median scaling -> rotation
null calibration -> M4-E``.

Two separate questions are measured, because they have different consequences.

1. Is the rejection *rate* informative?  The ``homogeneous`` arms contain one
   population with no incompatible cells at all, so a specific method should
   reject almost none of them.  The rotation null calibration, however,
   selects the largest cost whose null raw acceptance stays at or below 10%,
   and for a homogeneous cloud the rotated null closely resembles the observed
   data.  If the homogeneous arms still reject at the cap, the rate is fixed by
   the calibration target rather than by compatibility, and no rejection rate
   reported anywhere in the project carries biological information.
2. Is the rejection *identity* driven by depth?  The homogeneous arms differ
   only in how widely per-cell sequencing depth is spread.  Every cell's
   underlying relative expression profile is drawn from the same distribution
   regardless of its depth, so any dependence of the gate on depth is a
   specificity failure.  ``auc_total_counts`` and
   ``spearman_decision_cost_total_counts`` use the same definitions as
   ``cancer_metastasis/25_diagnose_gate_covariates.py`` so simulated and real
   values can be read side by side.

The ``perturbed`` arm is a positive control at uniform depth: a genuine
subpopulation is present only on the source side, so a method with any power
should reject those cells and few others.  Comparing it against the
homogeneous arms separates sensitivity from specificity.

Passing ``--fixed-rejection-cost`` bypasses null calibration, which separates
"the calibration target forces this rejection rate" from "the cost geometry
forces it".  Running both modes is the point of the script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cancer_metastasis.common import prepare_joint_representation  # noqa: E402
from confidenceot import (  # noqa: E402
    ConfidenceOT,
    calibrate_confidence_cost,
    rotation_null_costs,
)


ARMS = {
    # Depth spread only; no biological difference of any kind.
    "homogeneous_depth_cv0": {"depth_sigma": 0.0, "perturbed_fraction": 0.0},
    "homogeneous_depth_cv_low": {"depth_sigma": 0.3, "perturbed_fraction": 0.0},
    "homogeneous_depth_cv_mid": {"depth_sigma": 0.6, "perturbed_fraction": 0.0},
    "homogeneous_depth_cv_high": {"depth_sigma": 0.9, "perturbed_fraction": 0.0},
    # Positive control: a real source-only subpopulation at uniform depth.
    "perturbed_depth_cv0": {"depth_sigma": 0.0, "perturbed_fraction": 0.2},
}


def squared_euclidean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    value = (
        np.sum(left * left, axis=1)[:, None]
        + np.sum(right * right, axis=1)[None, :]
        - 2.0 * left @ right.T
    )
    return np.maximum(value, 0.0)


def rank_auc(values: np.ndarray, positive: np.ndarray) -> float:
    """Mann--Whitney AUC with ``positive`` as the positive class."""
    numeric = np.asarray(values, dtype=np.float64)
    label = np.asarray(positive, dtype=bool)
    finite = np.isfinite(numeric)
    numeric, label = numeric[finite], label[finite]
    n_positive = int(label.sum())
    n_negative = int(numeric.size - n_positive)
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    ranks = rankdata(numeric)
    return float(
        (ranks[label].sum() - n_positive * (n_positive + 1) / 2.0)
        / (n_positive * n_negative)
    )


def rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 10 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return float(spearmanr(x, y).statistic)


def simulate_counts(
    rng: np.random.Generator,
    *,
    n_cells: int,
    n_genes: int,
    median_depth: float,
    depth_sigma: float,
    dispersion: float,
    perturbed_fraction: float,
    perturbation_log2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return integer counts, realised depth, and the perturbed-cell mask.

    Every cell's relative expression profile is drawn from one shared gene mean
    vector with gamma overdispersion, so cells differ biologically only through
    the optional perturbed subpopulation.  Depth is applied afterwards by
    multinomial sampling, which is why depth carries no biological signal by
    construction.
    """
    gene_mean = rng.lognormal(mean=0.0, sigma=1.6, size=n_genes)
    gene_mean /= gene_mean.sum()
    rates = rng.gamma(
        shape=dispersion, scale=gene_mean / dispersion, size=(n_cells, n_genes)
    )
    perturbed = np.zeros(n_cells, dtype=bool)
    if perturbed_fraction > 0.0:
        count = int(round(perturbed_fraction * n_cells))
        perturbed[rng.choice(n_cells, count, replace=False)] = True
        affected = rng.choice(n_genes, max(1, n_genes // 10), replace=False)
        rates[np.ix_(perturbed, affected)] *= 2.0 ** perturbation_log2
    rates /= rates.sum(axis=1, keepdims=True)
    if depth_sigma <= 0.0:
        depth = np.full(n_cells, float(median_depth))
    else:
        depth = median_depth * rng.lognormal(0.0, depth_sigma, size=n_cells)
    depth = np.maximum(depth.round(), 100.0)
    counts = np.empty((n_cells, n_genes), dtype=np.int64)
    for index in range(n_cells):
        counts[index] = rng.multinomial(int(depth[index]), rates[index])
    return counts, np.asarray(counts.sum(axis=1), dtype=np.float64), perturbed


def as_anndata(counts: np.ndarray, prefix: str):
    import anndata as ad

    return ad.AnnData(
        X=counts.astype(np.float32),
        obs=pd.DataFrame(index=[f"{prefix}{i:05d}" for i in range(counts.shape[0])]),
        var=pd.DataFrame(index=[f"gene{j:05d}" for j in range(counts.shape[1])]),
    )


def run_replicate(
    arm: str, settings: dict, replicate: int, args: argparse.Namespace
) -> dict[str, object]:
    seed = args.seed + 7919 * replicate + abs(hash(arm)) % 10_000
    rng = np.random.default_rng(seed)
    shared = dict(
        n_genes=args.n_genes, median_depth=args.median_depth,
        dispersion=args.dispersion,
        perturbation_log2=args.perturbation_log2,
    )
    source_counts, source_depth, perturbed = simulate_counts(
        rng, n_cells=args.n_cells, depth_sigma=settings["depth_sigma"],
        perturbed_fraction=settings["perturbed_fraction"], **shared,
    )
    # The target side never carries the perturbed subpopulation, so perturbed
    # source cells are the only genuinely incompatible cells in any arm.
    target_counts, _, _ = simulate_counts(
        rng, n_cells=args.n_cells, depth_sigma=settings["depth_sigma"],
        perturbed_fraction=0.0, **shared,
    )
    source = as_anndata(source_counts, "s")
    target = as_anndata(target_counts, "t")
    source_pca, target_pca, hvg, _ = prepare_joint_representation(
        source, target, n_hvg=args.n_hvg, n_pcs=args.n_pcs, seed=seed
    )
    pairs = min(1_000_000, len(source_pca) * len(target_pca))
    sampled = np.sum(
        (
            source_pca[rng.integers(len(source_pca), size=pairs)]
            - target_pca[rng.integers(len(target_pca), size=pairs)]
        ) ** 2,
        axis=1,
    )
    positive = sampled[sampled > 0]
    scale = float(np.median(positive)) if positive.size else 1.0
    cost = squared_euclidean(source_pca, target_pca) / scale

    calibration_status = "fixed_user_supplied"
    calibration_valid = False
    if args.fixed_rejection_cost is not None:
        rejection_cost = float(args.fixed_rejection_cost)
    else:
        limit = min(args.calibration_max_cells, len(source_pca), len(target_pca))
        source_index = np.sort(rng.choice(len(source_pca), limit, replace=False))
        target_index = np.sort(rng.choice(len(target_pca), limit, replace=False))
        total = args.null_calibration_replicates + args.null_validation_replicates
        source_nulls, target_nulls = rotation_null_costs(
            source_pca[source_index], target_pca[target_index],
            observed_scale=scale, seed=seed, n_replicates=total,
        )
        split = args.null_calibration_replicates
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            calibration = calibrate_confidence_cost(
                source_nulls[:split] + target_nulls[:split],
                source_nulls[split:] + target_nulls[split:],
                backbone="uot", epsilon=args.epsilon,
                lambda_a=args.lambda_a, lambda_b=args.lambda_b,
                source_rejection_budget=args.source_rejection_budget,
                target_rejection_budget=args.target_rejection_budget,
                tolerance=args.tolerance, grid_size=args.calibration_grid_size,
                device="cpu", emit_warnings=False,
            )
        rejection_cost = float(calibration.rejection_cost)
        calibration_status = str(calibration.selection_status)
        calibration_valid = bool(calibration.calibration_valid)

    model = ConfidenceOT(
        backbone="uot", variant="exact", rejection_cost=rejection_cost,
        epsilon=args.epsilon, lambda_a=args.lambda_a, lambda_b=args.lambda_b,
        source_rejection_budget=args.source_rejection_budget,
        target_rejection_budget=args.target_rejection_budget,
        tolerance=args.tolerance, device="cpu", warn_on_terminal=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = model.fit(cost)
    retained = np.asarray(result.source_gate, dtype=bool)
    decision = np.asarray(result.source_confidence.decision_cost, dtype=np.float64)
    sign_rule = decision < rejection_cost

    record: dict[str, object] = {
        "arm": arm,
        "replicate": replicate,
        "depth_sigma": settings["depth_sigma"],
        "perturbed_fraction": settings["perturbed_fraction"],
        "rejection_cost_mode": (
            "fixed" if args.fixed_rejection_cost is not None else "null_calibrated"
        ),
        "rejection_cost": rejection_cost,
        "calibration_selection_status": calibration_status,
        "calibration_valid": calibration_valid,
        "hvg_n": len(hvg),
        "source_n": int(retained.size),
        # Question 1: is the rate informative?
        "source_rejection_rate": float(np.mean(~retained)),
        "source_rejection_budget_cap": args.source_rejection_budget,
        "sign_rule_retained_fraction": float(np.mean(sign_rule)),
        "median_decision_cost": float(np.median(decision)),
        # Question 2: is the identity depth-driven?
        "auc_total_counts": rank_auc(source_depth, retained),
        "spearman_decision_cost_total_counts": rank_correlation(
            decision, source_depth
        ),
        "median_total_counts_retained": float(np.median(source_depth[retained]))
        if retained.any() else float("nan"),
        "median_total_counts_rejected": float(np.median(source_depth[~retained]))
        if (~retained).any() else float("nan"),
        "observed_depth_cv": float(source_depth.std() / source_depth.mean()),
    }
    if settings["perturbed_fraction"] > 0.0:
        # Sensitivity against the only ground-truth incompatible cells.
        true_positive = int(np.sum(perturbed & ~retained))
        predicted = int(np.sum(~retained))
        actual = int(np.sum(perturbed))
        precision = true_positive / predicted if predicted else float("nan")
        recall = true_positive / actual if actual else float("nan")
        record["perturbed_recall"] = recall
        record["perturbed_precision"] = precision
        record["perturbed_f1"] = (
            2 * precision * recall / (precision + recall)
            if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0
            else float("nan")
        )
        record["auc_perturbed_rejected"] = rank_auc(decision, perturbed)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--n-cells", type=int, default=1500,
                        help="Cells per side; 1500 matches the real per-pair median")
    parser.add_argument("--n-genes", type=int, default=4000)
    parser.add_argument("--median-depth", type=float, default=10000.0)
    parser.add_argument("--dispersion", type=float, default=2.0)
    parser.add_argument("--perturbation-log2", type=float, default=1.0)
    parser.add_argument("--n-hvg", type=int, default=2000)
    parser.add_argument("--n-pcs", type=int, default=30)
    parser.add_argument("--source-rejection-budget", type=float, default=0.85)
    parser.add_argument("--target-rejection-budget", type=float, default=0.00)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--lambda-a", type=float, default=1.0)
    parser.add_argument("--lambda-b", type=float, default=1.0)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--calibration-max-cells", type=int, default=2000)
    parser.add_argument("--calibration-grid-size", type=int, default=5)
    parser.add_argument("--null-calibration-replicates", type=int, default=5)
    parser.add_argument("--null-validation-replicates", type=int, default=5)
    parser.add_argument(
        "--fixed-rejection-cost", type=float, default=None,
        help="Bypass null calibration to isolate the calibration target's effect",
    )
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--arm", action="append", choices=sorted(ARMS),
                        help="Restrict to these arms; default runs all")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    selected = args.arm or sorted(ARMS)
    records = []
    for arm in selected:
        for replicate in range(args.replicates):
            record = run_replicate(arm, ARMS[arm], replicate, args)
            records.append(record)
            print(
                f"{arm} rep={replicate} "
                f"rejection={record['source_rejection_rate']:.3f} "
                f"auc_depth={record['auc_total_counts']:.3f} "
                f"rho_cost_depth={record['spearman_decision_cost_total_counts']:.3f}",
                flush=True,
            )

    replicate_table = pd.DataFrame(records)
    numeric = [
        column for column in replicate_table.columns
        if column not in {"arm", "rejection_cost_mode", "calibration_selection_status"}
        and pd.api.types.is_numeric_dtype(replicate_table[column])
    ]
    summary = (
        replicate_table.groupby("arm", sort=True)[numeric]
        .median()
        .reset_index()
    )
    replicate_table.to_csv(args.output_root / "depth_null_replicates.csv", index=False)
    summary.to_csv(args.output_root / "depth_null_arm_summary.csv", index=False)
    report = {
        "rejection_cost_mode": (
            "fixed" if args.fixed_rejection_cost is not None else "null_calibrated"
        ),
        "source_rejection_budget": args.source_rejection_budget,
        "target_rejection_budget": args.target_rejection_budget,
        "replicates_per_arm": args.replicates,
        "cells_per_side": args.n_cells,
        "arms": ARMS,
        "expected_outcomes": {
            "homogeneous arms": (
                "No incompatible cell exists, so a specific method rejects "
                "almost none. A rate near the cap means the rate is set by the "
                "calibration target rather than by compatibility."
            ),
            "auc_total_counts": (
                "0.5 under specificity. Compare against the real data, where "
                "GSE180661 gives 0.67 and GSE225857 gives 0.79."
            ),
            "perturbed_depth_cv0": (
                "Positive control: recall should be high and precision should "
                "greatly exceed the perturbed fraction, otherwise the method "
                "has no power in this regime either."
            ),
        },
    }
    (args.output_root / "depth_null_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, default=str), flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
