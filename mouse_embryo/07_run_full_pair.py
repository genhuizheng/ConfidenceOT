"""Run one all-bin calibrated ConfidenceOT developmental pair on a GH node.

Dense cost matrices live in node-local storage.  Dense couplings are reduced
to biological tables in memory and are never persisted to shared storage.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from confidenceot import ConfidenceOT, calibrate_confidence_cost


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared_pair", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("local_work_dir", type=Path)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--lambda-a", type=float, default=1.0)
    parser.add_argument("--lambda-b", type=float, default=1.0)
    parser.add_argument("--rejection-budget", type=float, default=0.15)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--null-calibration-replicates", type=int, default=5)
    parser.add_argument("--null-validation-replicates", type=int, default=5)
    parser.add_argument("--calibration-grid-size", type=int, default=5)
    parser.add_argument("--cost-block-rows", type=int, default=256)
    parser.add_argument("--scale-sample-pairs", type=int, default=1_000_000)
    parser.add_argument(
        "--calibration-max-bins", type=int, default=2000,
        help="Label-blind deterministic null-calibration subsample per side; 0 uses every bin.",
    )
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def sampled_cost_scale(left: np.ndarray, right: np.ndarray, *, seed: int, n_pairs: int) -> float:
    rng = np.random.default_rng(seed)
    count = min(int(n_pairs), max(left.shape[0] * right.shape[0], 1))
    left_index = rng.integers(0, left.shape[0], size=count)
    right_index = rng.integers(0, right.shape[0], size=count)
    distances = np.sum((left[left_index] - right[right_index]) ** 2, axis=1, dtype=np.float64)
    positive = distances[distances > 0]
    if positive.size == 0:
        raise ValueError("The sampled observed expression distances are all zero.")
    return float(np.median(positive))


def write_cost_memmap(
    left: np.ndarray,
    right: np.ndarray,
    destination: Path,
    *,
    scale: float,
    block_rows: int,
) -> np.memmap:
    destination.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.memmap(destination, mode="w+", dtype=np.float64, shape=(left.shape[0], right.shape[0]))
    right_norm = np.sum(right * right, axis=1, dtype=np.float64)[None, :]
    for start in range(0, left.shape[0], block_rows):
        stop = min(start + block_rows, left.shape[0])
        block = left[start:stop]
        distances = (
            np.sum(block * block, axis=1, dtype=np.float64)[:, None]
            + right_norm - 2.0 * block @ right.T
        )
        matrix[start:stop] = np.maximum(distances, 0.0) / scale
    matrix.flush()
    del matrix
    return np.memmap(destination, mode="r", dtype=np.float64, shape=(left.shape[0], right.shape[0]))


def write_rotation_nulls(
    source: np.ndarray,
    target: np.ndarray,
    root: Path,
    *,
    observed_scale: float,
    seed: int,
    n_replicates: int,
    block_rows: int,
) -> tuple[list[np.memmap], list[np.memmap], list[Path]]:
    source_nulls, target_nulls, paths = [], [], []
    dimension = source.shape[1]
    for replicate in range(n_replicates):
        rng = np.random.default_rng(seed * 1009 + replicate * 9173 + 41)
        source_rotation, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
        target_rotation, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
        rotated_target = (target - target.mean(axis=0)) @ target_rotation + target.mean(axis=0)
        source_path = root / f"source_null_{replicate:02d}.f64"
        source_nulls.append(write_cost_memmap(
            source, rotated_target, source_path, scale=observed_scale, block_rows=block_rows
        ))
        paths.append(source_path)
        del rotated_target
        rotated_source = (source - source.mean(axis=0)) @ source_rotation + source.mean(axis=0)
        target_path = root / f"target_null_{replicate:02d}.f64"
        target_nulls.append(write_cost_memmap(
            rotated_source, target, target_path, scale=observed_scale, block_rows=block_rows
        ))
        paths.append(target_path)
        del rotated_source
    return source_nulls, target_nulls, paths


def one_hot(labels: np.ndarray, categories: list[str], gate: np.ndarray | None = None) -> np.ndarray:
    values = np.column_stack([labels == category for category in categories]).astype(np.float64)
    if gate is not None:
        values *= gate[:, None]
    return values


def grouped_mass(
    coupling: np.ndarray,
    source_hot: np.ndarray,
    target_hot: np.ndarray,
) -> np.ndarray:
    return (source_hot.T @ coupling) @ target_hot


def transition_rows(method: str, matrix: np.ndarray, source_categories: list[str], target_categories: list[str]):
    totals = matrix.sum(axis=1, keepdims=True)
    conditional = np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0)
    return [
        {
            "method": method, "source_annotation": source_label,
            "target_annotation": target_label, "transported_mass": matrix[i, j],
            "source_conditional_probability": conditional[i, j],
        }
        for i, source_label in enumerate(source_categories)
        for j, target_label in enumerate(target_categories)
    ]


def coupling_metrics(
    coupling: np.ndarray,
    cost: np.ndarray,
    source_gate: np.ndarray,
    target_gate: np.ndarray,
    *,
    block_rows: int,
) -> dict[str, float]:
    mass = 0.0
    weighted_cost = 0.0
    entropies = []
    for start in range(0, coupling.shape[0], block_rows):
        stop = min(start + block_rows, coupling.shape[0])
        active_source = source_gate[start:stop]
        if not np.any(active_source):
            continue
        block = coupling[start:stop][active_source][:, target_gate]
        cost_block = cost[start:stop][active_source][:, target_gate]
        row_mass = block.sum(axis=1)
        mass += float(row_mass.sum())
        weighted_cost += float(np.sum(block * cost_block))
        probabilities = np.divide(block, row_mass[:, None], out=np.zeros_like(block), where=row_mass[:, None] > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            entropy = np.where(probabilities > 0, -probabilities * np.log(probabilities), 0.0).sum(axis=1)
        entropies.extend(entropy[row_mass > 0].tolist())
    return {
        "retained_analysis_mass": mass,
        "retained_mean_transport_cost": weighted_cost / mass if mass > 0 else float("nan"),
        "retained_mean_source_transport_entropy": float(np.mean(entropies)) if entropies else float("nan"),
    }


def population_rejection_rows(method: str, side: str, labels: np.ndarray, gate: np.ndarray):
    return [
        {
            "method": method, "side": side, "annotation": category,
            "n": int(np.sum(labels == category)),
            "rejected_n": int(np.sum((labels == category) & ~gate)),
            "rejected_fraction": float(np.mean(~gate[labels == category])),
        }
        for category in sorted(np.unique(labels))
    ]


def same_annotation_fraction(matrix: np.ndarray, source_categories: list[str], target_categories: list[str]) -> float:
    total = float(matrix.sum())
    if total <= 0:
        return float("nan")
    target_lookup = {label: j for j, label in enumerate(target_categories)}
    same = sum(matrix[i, target_lookup[label]] for i, label in enumerate(source_categories) if label in target_lookup)
    return float(same / total)


def main() -> None:
    pipeline_started = time.perf_counter()
    args = parse_args()
    if args.calibration_max_bins < 0:
        raise ValueError("calibration_max_bins must be nonnegative.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.local_work_dir.mkdir(parents=True, exist_ok=True)
    prepared = np.load(args.prepared_pair, allow_pickle=False)
    source = prepared["source_pca"].astype(np.float64)
    target = prepared["target_pca"].astype(np.float64)
    source_labels = prepared["source_labels"].astype(str)
    target_labels = prepared["target_labels"].astype(str)
    source_ids = prepared["source_ids"].astype(str) if "source_ids" in prepared else np.asarray(
        [f"source_{index}" for index in range(len(source))], dtype=str
    )
    target_ids = prepared["target_ids"].astype(str) if "target_ids" in prepared else np.asarray(
        [f"target_{index}" for index in range(len(target))], dtype=str
    )
    source_categories = sorted(np.unique(source_labels))
    target_categories = sorted(np.unique(target_labels))
    calibration_rng = np.random.default_rng(args.seed + 29009)
    source_calibration_n = len(source) if args.calibration_max_bins == 0 else min(
        len(source), args.calibration_max_bins
    )
    target_calibration_n = len(target) if args.calibration_max_bins == 0 else min(
        len(target), args.calibration_max_bins
    )
    source_calibration_index = np.sort(calibration_rng.choice(
        len(source), size=source_calibration_n, replace=False
    ))
    target_calibration_index = np.sort(calibration_rng.choice(
        len(target), size=target_calibration_n, replace=False
    ))
    calibration_source = source[source_calibration_index]
    calibration_target = target[target_calibration_index]
    pd.concat([
        pd.DataFrame({
            "side": "source", "row_index": source_calibration_index,
            "observation_id": source_ids[source_calibration_index],
        }),
        pd.DataFrame({
            "side": "target", "row_index": target_calibration_index,
            "observation_id": target_ids[target_calibration_index],
        }),
    ], ignore_index=True).to_csv(args.output_dir / "calibration_subsample.csv", index=False)
    scale = sampled_cost_scale(source, target, seed=args.seed, n_pairs=args.scale_sample_pairs)
    observed_path = args.local_work_dir / "observed_cost.f64"
    null_paths: list[Path] = []
    observed = write_cost_memmap(
        source, target, observed_path, scale=scale, block_rows=args.cost_block_rows
    )
    started = time.perf_counter()
    try:
        total_nulls = args.null_calibration_replicates + args.null_validation_replicates
        source_nulls, target_nulls, null_paths = write_rotation_nulls(
            calibration_source, calibration_target, args.local_work_dir,
            observed_scale=scale, seed=args.seed,
            n_replicates=total_nulls, block_rows=args.cost_block_rows,
        )
        split = args.null_calibration_replicates
        calibration = calibrate_confidence_cost(
            source_nulls[:split] + target_nulls[:split],
            source_nulls[split:] + target_nulls[split:],
            backbone="uot", epsilon=args.epsilon, lambda_a=args.lambda_a, lambda_b=args.lambda_b,
            source_rejection_budget=args.rejection_budget,
            target_rejection_budget=args.rejection_budget, tolerance=args.tolerance,
            grid_size=args.calibration_grid_size, device=args.device,
            fallback_to_cpu=False, workers=1,
        )
        del source_nulls, target_nulls
        gc.collect()
        confidence_model = ConfidenceOT(
            backbone="uot", variant="reversible", rejection_cost=calibration.rejection_cost,
            epsilon=args.epsilon, lambda_a=args.lambda_a, lambda_b=args.lambda_b,
            source_rejection_budget=args.rejection_budget,
            target_rejection_budget=args.rejection_budget, tolerance=args.tolerance,
            device=args.device,
        )
        confidence = confidence_model.fit(observed)
        source_hot_retained = one_hot(source_labels, source_categories, confidence.source_gate)
        target_hot_retained = one_hot(target_labels, target_categories, confidence.target_gate)
        confidence_grouped = grouped_mass(confidence.coupling, source_hot_retained, target_hot_retained)
        confidence_metrics = coupling_metrics(
            confidence.coupling, observed, confidence.source_gate, confidence.target_gate,
            block_rows=args.cost_block_rows,
        )
        confidence_metrics.update({
            "method": "Calibrated | M4-R / UOT", "solver_transport_mass": float(confidence.coupling.sum()),
            "source_rejection_rate": float(np.mean(~confidence.source_gate)),
            "target_rejection_rate": float(np.mean(~confidence.target_gate)),
            "fit_seconds": confidence.fit_seconds, "inner_converged": confidence.inner_converged,
            "outer_converged": confidence.outer_converged, "cycle_detected": confidence.cycle_detected,
            "same_annotation_transport_fraction": same_annotation_fraction(
                confidence_grouped, source_categories, target_categories
            ),
        })
        source_gate, target_gate = confidence.source_gate.copy(), confidence.target_gate.copy()
        source_score, target_score = confidence.source_score.copy(), confidence.target_score.copy()
        source_raw_gate = confidence.source_raw_gate.copy()
        target_raw_gate = confidence.target_raw_gate.copy()
        del confidence
        gc.collect()

        traditional_model = ConfidenceOT(
            backbone="balanced", variant="exact", rejection_cost=calibration.rejection_cost,
            epsilon=args.epsilon, source_rejection_budget=0.0, target_rejection_budget=0.0,
            tolerance=args.tolerance, device=args.device,
        )
        traditional = traditional_model.fit(observed)
        source_hot = one_hot(source_labels, source_categories)
        target_hot = one_hot(target_labels, target_categories)
        traditional_grouped = grouped_mass(traditional.coupling, source_hot, target_hot)
        traditional_metrics = coupling_metrics(
            traditional.coupling, observed, np.ones(len(source), dtype=bool),
            np.ones(len(target), dtype=bool), block_rows=args.cost_block_rows,
        )
        traditional_metrics.update({
            "method": "Traditional Balanced OT", "solver_transport_mass": float(traditional.coupling.sum()),
            "source_rejection_rate": 0.0, "target_rejection_rate": 0.0,
            "fit_seconds": traditional.fit_seconds, "inner_converged": traditional.inner_converged,
            "outer_converged": traditional.outer_converged, "cycle_detected": traditional.cycle_detected,
            "same_annotation_transport_fraction": same_annotation_fraction(
                traditional_grouped, source_categories, target_categories
            ),
        })
        forced_source = grouped_mass(
            traditional.coupling, one_hot(source_labels, source_categories, ~source_gate), target_hot
        )
        forced_target = grouped_mass(
            traditional.coupling, source_hot, one_hot(target_labels, target_categories, ~target_gate)
        ).T
        transition = transition_rows(
            "Traditional Balanced OT", traditional_grouped, source_categories, target_categories
        ) + transition_rows(
            "Calibrated | M4-R / UOT", confidence_grouped, source_categories, target_categories
        )
        pd.DataFrame(transition).to_csv(args.output_dir / "population_transitions.csv", index=False)
        forced_rows = []
        for i, source_label in enumerate(source_categories):
            total = forced_source[i].sum()
            for j, target_label in enumerate(target_categories):
                forced_rows.append({
                    "rejection_side": "source", "rejected_annotation": source_label,
                    "traditional_forced_partner_annotation": target_label,
                    "transported_mass": forced_source[i, j],
                    "conditional_probability": forced_source[i, j] / total if total > 0 else 0.0,
                })
        for i, target_label in enumerate(target_categories):
            total = forced_target[i].sum()
            for j, source_label in enumerate(source_categories):
                forced_rows.append({
                    "rejection_side": "target", "rejected_annotation": target_label,
                    "traditional_forced_partner_annotation": source_label,
                    "transported_mass": forced_target[i, j],
                    "conditional_probability": forced_target[i, j] / total if total > 0 else 0.0,
                })
        pd.DataFrame(forced_rows).to_csv(args.output_dir / "traditional_forced_matches.csv", index=False)
        rejection = population_rejection_rows(
            "Calibrated | M4-R / UOT", "source", source_labels, source_gate
        ) + population_rejection_rows(
            "Calibrated | M4-R / UOT", "target", target_labels, target_gate
        )
        pd.DataFrame(rejection).to_csv(args.output_dir / "population_rejection.csv", index=False)
        cell_rows = pd.concat([
            pd.DataFrame({
                "side": "source", "observation_id": source_ids,
                "annotation": source_labels, "spatial_x": prepared["source_spatial"][:, 0],
                "spatial_y": prepared["source_spatial"][:, 1], "retained": source_gate,
                "rejected": ~source_gate, "raw_accepted": source_raw_gate,
                "confidence_coefficient": source_score,
            }),
            pd.DataFrame({
                "side": "target", "observation_id": target_ids,
                "annotation": target_labels, "spatial_x": prepared["target_spatial"][:, 0],
                "spatial_y": prepared["target_spatial"][:, 1], "retained": target_gate,
                "rejected": ~target_gate, "raw_accepted": target_raw_gate,
                "confidence_coefficient": target_score,
            }),
        ], ignore_index=True)
        cell_rows.to_csv(args.output_dir / "cell_confidence.csv", index=False)
        source_background_spatial = prepared["source_background_spatial"] if "source_background_spatial" in prepared else np.empty((0, 2))
        target_background_spatial = prepared["target_background_spatial"] if "target_background_spatial" in prepared else np.empty((0, 2))
        source_background_ids = prepared["source_background_ids"].astype(str) if "source_background_ids" in prepared else np.asarray(
            [f"source_background_{index}" for index in range(len(source_background_spatial))], dtype=str
        )
        target_background_ids = prepared["target_background_ids"].astype(str) if "target_background_ids" in prepared else np.asarray(
            [f"target_background_{index}" for index in range(len(target_background_spatial))], dtype=str
        )
        source_background_labels = prepared["source_background_labels"].astype(str) if "source_background_labels" in prepared else np.repeat(
            "Cavity", len(source_background_spatial)
        )
        target_background_labels = prepared["target_background_labels"].astype(str) if "target_background_labels" in prepared else np.repeat(
            "Cavity", len(target_background_spatial)
        )
        background_rows = pd.concat([
            pd.DataFrame({
                "side": np.repeat("source", len(source_background_ids)), "observation_id": source_background_ids,
                "annotation": source_background_labels,
                "spatial_x": source_background_spatial[:, 0],
                "spatial_y": source_background_spatial[:, 1],
                "exclusion_reason": np.repeat("preanalysis_anatomical_background", len(source_background_ids)),
            }),
            pd.DataFrame({
                "side": np.repeat("target", len(target_background_ids)), "observation_id": target_background_ids,
                "annotation": target_background_labels,
                "spatial_x": target_background_spatial[:, 0],
                "spatial_y": target_background_spatial[:, 1],
                "exclusion_reason": np.repeat("preanalysis_anatomical_background", len(target_background_ids)),
            }),
        ], ignore_index=True)
        background_rows.to_csv(args.output_dir / "preanalysis_exclusions.csv", index=False)
        pd.DataFrame([traditional_metrics, confidence_metrics]).to_csv(
            args.output_dir / "method_metrics.csv", index=False
        )
        certificate = {
            "rejection_cost": calibration.rejection_cost,
            "calibration_valid": calibration.calibration_valid,
            "selection_status": calibration.selection_status,
            "source_monotone": calibration.source_monotone,
            "target_monotone": calibration.target_monotone,
            "refinement_method": calibration.refinement_method,
            "validation_source_raw_acceptance": calibration.validation_source_raw_acceptance,
            "validation_target_raw_acceptance": calibration.validation_target_raw_acceptance,
            "validation_aggregate_valid": calibration.validation_aggregate_valid,
            "warning_messages": list(calibration.warning_messages),
            "validation_records": [record.__dict__ for record in calibration.validation],
            "curve_costs": calibration.curve_costs.tolist(),
            "source_raw_acceptance_curve": calibration.source_raw_acceptance_curve.tolist(),
            "target_raw_acceptance_curve": calibration.target_raw_acceptance_curve.tolist(),
            "source_projected_acceptance_curve": calibration.source_projected_acceptance_curve.tolist(),
            "target_projected_acceptance_curve": calibration.target_projected_acceptance_curve.tolist(),
            "cost_scale": scale, "cost_scale_estimator": f"median_of_{args.scale_sample_pairs}_sampled_pairs",
            "cost_shape": list(observed.shape), "device": args.device,
            "observed_inference_uses_all_prepared_bins": True,
            "calibration_shape": [source_calibration_n, target_calibration_n],
            "calibration_sampling": "label_blind_deterministic_without_replacement",
            "wall_seconds": time.perf_counter() - started,
            "pipeline_wall_seconds_including_cost_construction": time.perf_counter() - pipeline_started,
        }
        (args.output_dir / "calibration.json").write_text(json.dumps(certificate, indent=2), encoding="utf-8")
        (args.output_dir / "SUCCESS").write_text("complete\n", encoding="utf-8")
        print(json.dumps(certificate, indent=2))
    finally:
        del observed
        gc.collect()
        for path in [observed_path, *null_paths]:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    main()
