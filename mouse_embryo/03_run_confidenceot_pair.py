"""Run baseline OT and ConfidenceOT on one prepared MOSTA section pair."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from confidenceot import ConfidenceOT, calibrate_confidence_cost, rotation_null_costs
from traditional_ot import balanced_ot, partial_wasserstein_uniform, unbalanced_ot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared_pair", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--rejection-mode", choices=("fixed", "calibrated", "both"), default="fixed")
    parser.add_argument("--fixed-rejection-cost", type=float, default=0.5)
    parser.add_argument("--confidence-backbone", choices=("balanced", "uot"), default="uot")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--lambda-a", type=float, default=1.0)
    parser.add_argument("--lambda-b", type=float, default=1.0)
    parser.add_argument("--rejection-budget", type=float, default=0.15)
    parser.add_argument("--partial-transported-mass", type=float, default=0.85)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--max-iterations", type=int, default=20_000)
    parser.add_argument("--max-outer-iterations", type=int, default=30)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--null-calibration-replicates", type=int, default=2)
    parser.add_argument("--null-validation-replicates", type=int, default=2)
    parser.add_argument("--calibration-grid-size", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


def squared_euclidean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    values = (
        np.sum(left * left, axis=1)[:, None]
        + np.sum(right * right, axis=1)[None, :]
        - 2.0 * left @ right.T
    )
    return np.maximum(values, 0.0).astype(np.float64)


def timed(function):
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    result = function()
    return result, time.perf_counter() - wall_start, time.process_time() - cpu_start


def marginal_deficit(mass: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.maximum(1.0 - np.divide(mass, reference, out=np.zeros_like(mass), where=reference > 0), 0.0)


def population_rejection(method: str, side: str, labels: np.ndarray, signal: np.ndarray) -> list[dict]:
    rows = []
    for label in np.unique(labels):
        selected = labels == label
        rows.append({
            "method": method, "side": side, "annotation": str(label),
            "n": int(selected.sum()), "mean_rejection_signal": float(np.mean(signal[selected])),
        })
    return rows


def population_transitions(method: str, coupling: np.ndarray, source: np.ndarray, target: np.ndarray) -> list[dict]:
    rows = []
    source_categories = np.unique(source)
    target_categories = np.unique(target)
    for source_label in source_categories:
        source_mask = source == source_label
        total = float(coupling[source_mask].sum())
        for target_label in target_categories:
            mass = float(coupling[np.ix_(source_mask, target == target_label)].sum())
            rows.append({
                "method": method, "source_annotation": str(source_label),
                "target_annotation": str(target_label), "transported_mass": mass,
                "source_conditional_probability": mass / total if total > 0 else 0.0,
            })
    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    method_root = args.output_dir / "methods"
    method_root.mkdir(exist_ok=True)
    prepared = np.load(args.prepared_pair, allow_pickle=False)
    source = prepared["source_pca"].astype(np.float64)
    target = prepared["target_pca"].astype(np.float64)
    source_labels = prepared["source_labels"].astype(str)
    target_labels = prepared["target_labels"].astype(str)
    raw_cost = squared_euclidean(source, target)
    positive = raw_cost[raw_cost > 0]
    if positive.size == 0:
        raise ValueError("The observed PCA distance matrix is identically zero.")
    cost_scale = float(np.median(positive))
    cost = raw_cost / cost_scale
    metrics: list[dict] = []
    rejection_rows: list[dict] = []
    transition_rows: list[dict] = []
    cell_rows: list[dict] = []

    def record(method: str, coupling: np.ndarray, source_signal: np.ndarray, target_signal: np.ndarray,
               wall: float, cpu: float, extra: dict, *, signal_definition: str,
               explicit_source_rejected: np.ndarray | None = None,
               explicit_target_rejected: np.ndarray | None = None) -> None:
        analysis_coupling = coupling
        if explicit_source_rejected is not None and explicit_target_rejected is not None:
            retained_mask = (~explicit_source_rejected)[:, None] & (~explicit_target_rejected)[None, :]
            analysis_coupling = np.where(retained_mask, coupling, 0.0)
        solver_mass = float(coupling.sum())
        retained_mass = float(analysis_coupling.sum())
        concordant = float(analysis_coupling[source_labels[:, None] == target_labels[None, :]].sum())
        weighted_cost = (
            float(np.sum(analysis_coupling * cost) / retained_mass) if retained_mass > 0 else np.nan
        )
        row_sum = analysis_coupling.sum(axis=1, keepdims=True)
        conditional = np.divide(
            analysis_coupling, row_sum, out=np.zeros_like(analysis_coupling), where=row_sum > 0
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            entropy_terms = np.where(conditional > 0, -conditional * np.log(conditional), 0.0)
        row = {
            "method": method, "wall_seconds": wall, "cpu_seconds": cpu,
            "solver_transport_mass": solver_mass,
            "retained_analysis_mass": retained_mass,
            "retained_same_annotation_transport_fraction": (
                concordant / retained_mass if retained_mass > 0 else np.nan
            ),
            "retained_mean_transport_cost": weighted_cost,
            "retained_mean_source_transport_entropy": float(
                entropy_terms.sum(axis=1)[row_sum.ravel() > 0].mean()
            ) if np.any(row_sum > 0) else np.nan,
            "source_mean_rejection_signal": float(np.mean(source_signal)),
            "target_mean_rejection_signal": float(np.mean(target_signal)),
            "rejection_signal_definition": signal_definition,
            **extra,
        }
        metrics.append(row)
        rejection_rows.extend(population_rejection(method, "source", source_labels, source_signal))
        rejection_rows.extend(population_rejection(method, "target", target_labels, target_signal))
        transition_rows.extend(
            population_transitions(method, analysis_coupling, source_labels, target_labels)
        )
        for side, labels, coordinates, ids, signal, explicit in (
            ("source", source_labels, prepared["source_spatial"], prepared.get("source_ids", np.arange(len(source_labels))),
             source_signal, explicit_source_rejected),
            ("target", target_labels, prepared["target_spatial"], prepared.get("target_ids", np.arange(len(target_labels))),
             target_signal, explicit_target_rejected),
        ):
            for index in range(len(labels)):
                cell_rows.append({
                    "method": method, "side": side, "sample_index": index,
                    "observation_id": str(ids[index]), "annotation": str(labels[index]),
                    "spatial_x": float(coordinates[index, 0]), "spatial_y": float(coordinates[index, 1]),
                    "rejection_signal": float(signal[index]),
                    "explicit_rejected": np.nan if explicit is None else bool(explicit[index]),
                    "signal_definition": signal_definition,
                })
        safe = method.lower().replace(" ", "_").replace("/", "_").replace("|", "_")
        np.savez_compressed(
            method_root / f"{safe}.npz", coupling=coupling.astype(np.float32),
            retained_analysis_coupling=analysis_coupling.astype(np.float32),
            source_rejection_signal=source_signal.astype(np.float32),
            target_rejection_signal=target_signal.astype(np.float32),
        )

    balanced, wall, cpu = timed(lambda: balanced_ot(
        cost, epsilon=args.epsilon, threshold=args.tolerance, max_iterations=args.max_iterations
    ))
    record("Balanced OT", balanced.coupling, np.zeros(source.shape[0]), np.zeros(target.shape[0]),
           wall, cpu, {"converged": balanced.converged, "iterations": balanced.n_iterations},
           signal_definition="no_rejection_gate",
           explicit_source_rejected=np.zeros(source.shape[0], dtype=bool),
           explicit_target_rejected=np.zeros(target.shape[0], dtype=bool))

    vanilla, wall, cpu = timed(lambda: unbalanced_ot(
        cost, epsilon=args.epsilon, lambda_a=args.lambda_a, lambda_b=args.lambda_b,
        threshold=args.tolerance, max_iterations=args.max_iterations,
    ))
    record(
        "Vanilla UOT", vanilla.coupling,
        marginal_deficit(vanilla.source_mass, vanilla.source_marginal),
        marginal_deficit(vanilla.target_mass, vanilla.target_marginal), wall, cpu,
        {"converged": vanilla.converged, "iterations": vanilla.n_iterations},
        signal_definition="positive_marginal_mass_deficit",
    )

    if source.shape[0] == target.shape[0]:
        partial, wall, cpu = timed(lambda: partial_wasserstein_uniform(
            cost, transported_mass=args.partial_transported_mass
        ))
        record("Partial OT", partial.coupling, (~partial.source_gate).astype(float),
               (~partial.target_gate).astype(float), wall, cpu,
               {"converged": partial.success, "iterations": np.nan},
               signal_definition="binary_unmatched_support",
               explicit_source_rejected=~partial.source_gate,
               explicit_target_rejected=~partial.target_gate)

    rejection_costs: list[tuple[str, float]] = []
    if args.rejection_mode in ("fixed", "both"):
        rejection_costs.append(("Fixed", args.fixed_rejection_cost))
    calibration_payload = None
    if args.rejection_mode in ("calibrated", "both"):
        total_nulls = args.null_calibration_replicates + args.null_validation_replicates
        source_nulls, target_nulls = rotation_null_costs(
            source, target, observed_scale=cost_scale, seed=args.seed, n_replicates=total_nulls
        )
        split = args.null_calibration_replicates
        calibration_nulls = source_nulls[:split] + target_nulls[:split]
        validation_nulls = source_nulls[split:] + target_nulls[split:]
        calibration, calibration_wall, calibration_cpu = timed(lambda: calibrate_confidence_cost(
            calibration_nulls, validation_nulls, backbone=args.confidence_backbone,
            epsilon=args.epsilon, lambda_a=args.lambda_a, lambda_b=args.lambda_b,
            source_rejection_budget=args.rejection_budget,
            target_rejection_budget=args.rejection_budget, tolerance=args.tolerance,
            max_iterations=args.max_iterations, max_outer_iterations=args.max_outer_iterations,
            grid_size=args.calibration_grid_size, device=args.device,
            fallback_to_cpu=False, workers=args.workers,
        ))
        rejection_costs.append(("Calibrated", calibration.rejection_cost))
        calibration_payload = {
            "rejection_cost": calibration.rejection_cost,
            "selection_status": calibration.selection_status,
            "calibration_valid": calibration.calibration_valid,
            "source_monotone": calibration.source_monotone,
            "target_monotone": calibration.target_monotone,
            "refinement_method": calibration.refinement_method,
            "warning_messages": list(calibration.warning_messages),
            "validation_source_raw_acceptance": calibration.validation_source_raw_acceptance,
            "validation_target_raw_acceptance": calibration.validation_target_raw_acceptance,
            "validation_aggregate_valid": calibration.validation_aggregate_valid,
            "validation_records": [
                {
                    "null_index": record.null_index,
                    "source_raw_acceptance": record.source_raw_acceptance,
                    "target_raw_acceptance": record.target_raw_acceptance,
                    "inner_converged": record.inner_converged,
                    "outer_converged": record.outer_converged,
                    "cycle_detected": record.cycle_detected,
                }
                for record in calibration.validation
            ],
            "curve_costs": calibration.curve_costs.tolist(),
            "source_raw_acceptance_curve": calibration.source_raw_acceptance_curve.tolist(),
            "target_raw_acceptance_curve": calibration.target_raw_acceptance_curve.tolist(),
            "wall_seconds": calibration_wall, "cpu_seconds": calibration_cpu,
        }
        (args.output_dir / "calibration.json").write_text(
            json.dumps(calibration_payload, indent=2), encoding="utf-8"
        )

    for mode, rejection_cost in rejection_costs:
        for variant, label in (("exact", "M4-E"), ("reversible", "M4-R")):
            model = ConfidenceOT(
                backbone=args.confidence_backbone, variant=variant,
                rejection_cost=rejection_cost, epsilon=args.epsilon,
                lambda_a=args.lambda_a, lambda_b=args.lambda_b,
                source_rejection_budget=args.rejection_budget,
                target_rejection_budget=args.rejection_budget,
                tolerance=args.tolerance, max_iterations=args.max_iterations,
                max_outer_iterations=args.max_outer_iterations, device=args.device,
            )
            fitted, wall, cpu = timed(lambda model=model: model.fit(cost))
            method = f"{mode} | {label} / {args.confidence_backbone.upper()}"
            record(method, fitted.coupling, (~fitted.source_gate).astype(float),
                   (~fitted.target_gate).astype(float), wall, cpu, {
                       "converged": fitted.inner_converged and fitted.outer_converged,
                       "iterations": fitted.total_inner_iterations,
                       "outer_iterations": fitted.n_outer_iterations,
                       "cycle_detected": fitted.cycle_detected,
                       "rejection_cost": fitted.rejection_cost,
                       "device": fitted.device, "backend": fitted.backend,
                   }, signal_definition="binary_confidence_gate",
                   explicit_source_rejected=~fitted.source_gate,
                   explicit_target_rejected=~fitted.target_gate)

    pd.DataFrame(metrics).to_csv(args.output_dir / "method_metrics.csv", index=False)
    pd.DataFrame(rejection_rows).to_csv(args.output_dir / "population_rejection.csv", index=False)
    pd.DataFrame(transition_rows).to_csv(args.output_dir / "population_transitions.csv", index=False)
    pd.DataFrame(cell_rows).to_csv(args.output_dir / "cell_rejection.csv", index=False)
    preanalysis_rows = []
    for side in ("source", "target"):
        coordinates = prepared[f"{side}_background_spatial"]
        labels = prepared.get(f"{side}_background_labels", np.repeat("Cavity", len(coordinates)))
        ids = prepared.get(f"{side}_background_ids", np.arange(len(coordinates)))
        for index in range(len(coordinates)):
            preanalysis_rows.append({
                "side": side, "observation_id": str(ids[index]), "annotation": str(labels[index]),
                "spatial_x": float(coordinates[index, 0]), "spatial_y": float(coordinates[index, 1]),
                "exclusion_reason": "anatomical_background_excluded_before_ot",
            })
    pd.DataFrame(preanalysis_rows).to_csv(args.output_dir / "preanalysis_exclusions.csv", index=False)
    run = {
        "prepared_pair": str(args.prepared_pair.resolve()), "cost_scale": cost_scale,
        "cost_shape": list(cost.shape), "epsilon": args.epsilon,
        "lambda_a": args.lambda_a, "lambda_b": args.lambda_b,
        "rejection_budget": args.rejection_budget, "rejection_mode": args.rejection_mode,
        "confidence_backbone": args.confidence_backbone, "device": args.device,
        "pid": os.getpid(), "calibration": calibration_payload,
    }
    (args.output_dir / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(pd.DataFrame(metrics).to_string(index=False))
    print(f"Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
