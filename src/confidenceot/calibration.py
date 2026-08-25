"""Portable, label-free null calibration for ConfidenceOT."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from confidenceot.api import ConfidenceOT


def _squared_euclidean(left: NDArray[np.float64], right: NDArray[np.float64]) -> NDArray[np.float64]:
    """Pairwise squared Euclidean distance using only NumPy."""
    distances = (
        np.sum(left * left, axis=1)[:, None]
        + np.sum(right * right, axis=1)[None, :]
        - 2.0 * left @ right.T
    )
    return np.maximum(distances, 0.0)


@dataclass(frozen=True)
class NullValidationRecord:
    null_index: int
    source_raw_acceptance: float
    target_raw_acceptance: float
    inner_converged: bool
    outer_converged: bool
    cycle_detected: bool


@dataclass(frozen=True)
class NullCalibrationResult:
    backbone: Literal["balanced", "uot"]
    rejection_cost: float
    curve_costs: NDArray[np.float64]
    source_raw_acceptance_curve: NDArray[np.float64]
    target_raw_acceptance_curve: NDArray[np.float64]
    source_projected_acceptance_curve: NDArray[np.float64]
    target_projected_acceptance_curve: NDArray[np.float64]
    selection_status: str
    calibration_valid: bool
    source_monotone: bool
    target_monotone: bool
    refinement_method: str
    warning_messages: tuple[str, ...]
    validation: tuple[NullValidationRecord, ...]


def rotation_null_costs(
    source: ArrayLike,
    target: ArrayLike,
    *,
    observed_scale: float,
    seed: int,
    n_replicates: int,
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """Create directional random-rotation null costs without scikit-learn."""
    source_values = np.asarray(source, dtype=np.float64)
    target_values = np.asarray(target, dtype=np.float64)
    if source_values.ndim != 2 or target_values.ndim != 2:
        raise ValueError("source and target must be two-dimensional coordinate arrays.")
    if source_values.shape[1] != target_values.shape[1]:
        raise ValueError("source and target must use the same feature dimension.")
    if not np.isfinite(observed_scale) or observed_scale <= 0:
        raise ValueError("observed_scale must be positive and finite.")
    if n_replicates <= 0:
        raise ValueError("n_replicates must be positive.")
    source_nulls: list[NDArray[np.float64]] = []
    target_nulls: list[NDArray[np.float64]] = []
    dimension = source_values.shape[1]
    for replicate in range(n_replicates):
        rng = np.random.default_rng(seed * 1009 + replicate * 9173 + 41)
        source_rotation, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
        target_rotation, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
        target_center = target_values.mean(axis=0)
        rotated_target = (target_values - target_center) @ target_rotation + target_center
        source_nulls.append(_squared_euclidean(source_values, rotated_target) / observed_scale)
        source_center = source_values.mean(axis=0)
        rotated_source = (source_values - source_center) @ source_rotation + source_center
        target_nulls.append(_squared_euclidean(rotated_source, target_values) / observed_scale)
    return source_nulls, target_nulls


def _initial_grid(
    nulls: Sequence[NDArray[np.float64]],
    *,
    backbone: str,
    epsilon: float,
    lambda_a: float,
    lambda_b: float,
    tolerance: float,
    grid_size: int,
) -> NDArray[np.float64]:
    from traditional_ot.balanced import balanced_ot
    from traditional_ot.unbalanced import unbalanced_ot

    losses: list[NDArray[np.float64]] = []
    for cost in nulls:
        if backbone == "balanced":
            fit = balanced_ot(cost, epsilon=epsilon, threshold=tolerance, max_iterations=20_000)
            source = np.divide((fit.coupling * cost).sum(axis=1), fit.source_marginal)
            target = np.divide((fit.coupling * cost).sum(axis=0), fit.target_marginal)
        else:
            fit = unbalanced_ot(
                cost, epsilon=epsilon, lambda_a=lambda_a, lambda_b=lambda_b,
                threshold=tolerance, max_iterations=20_000,
            )
            source_logits = np.log(fit.target_marginal)[None, :] + fit.log_target_scaling[None, :] - cost / epsilon
            source_logits -= np.max(source_logits, axis=1, keepdims=True)
            source_probability = np.exp(source_logits)
            source_probability /= source_probability.sum(axis=1, keepdims=True)
            target_logits = np.log(fit.source_marginal)[:, None] + fit.log_source_scaling[:, None] - cost / epsilon
            target_logits -= np.max(target_logits, axis=0, keepdims=True)
            target_probability = np.exp(target_logits)
            target_probability /= target_probability.sum(axis=0, keepdims=True)
            source = np.sum(source_probability * cost, axis=1)
            target = np.sum(target_probability * cost, axis=0)
        losses.extend((source, target))
    positive = np.concatenate(losses)
    positive = positive[np.isfinite(positive) & (positive > 0)]
    if positive.size == 0:
        raise RuntimeError("Null calibration produced no positive matching losses.")
    return np.geomspace(max(float(positive.min()) * 0.1, np.finfo(float).tiny), float(positive.max()) * 1.1, grid_size)


def calibrate_confidence_cost(
    calibration_nulls: Sequence[ArrayLike],
    validation_nulls: Sequence[ArrayLike] = (),
    *,
    backbone: Literal["balanced", "uot"] = "uot",
    epsilon: float = 0.1,
    lambda_a: float = 1.0,
    lambda_b: float = 1.0,
    source_raw_acceptance_target: float = 0.10,
    target_raw_acceptance_target: float = 0.10,
    source_rejection_budget: float = 0.15,
    target_rejection_budget: float = 0.15,
    tolerance: float = 1e-3,
    max_iterations: int = 20_000,
    max_outer_iterations: int = 30,
    grid_size: int = 7,
    refinement_relative_tolerance: float = 1e-3,
    device: Literal["auto", "cpu", "cuda"] = "auto",
    cuda_dtype: Literal["float32", "float64"] = "float32",
    fallback_to_cpu: bool = False,
    workers: int = 1,
    emit_warnings: bool = True,
) -> NullCalibrationResult:
    """Estimate a rejection cost with M4-E and validate it with M4-R."""
    nulls = tuple(np.asarray(value, dtype=np.float64) for value in calibration_nulls)
    validation_values = tuple(np.asarray(value, dtype=np.float64) for value in validation_nulls)
    if not nulls:
        raise ValueError("At least one calibration null is required.")
    if grid_size < 3:
        raise ValueError("grid_size must be at least 3.")
    if workers <= 0:
        raise ValueError("workers must be a positive integer.")
    shape = nulls[0].shape
    if any(value.shape != shape for value in (*nulls, *validation_values)):
        raise ValueError("All calibration and validation nulls must share one shape.")
    curve_costs = _initial_grid(
        nulls, backbone=backbone, epsilon=epsilon, lambda_a=lambda_a,
        lambda_b=lambda_b, tolerance=tolerance, grid_size=grid_size,
    )
    cache: dict[float, tuple[float, float, float, float]] = {}
    warning_set: set[str] = set()

    def evaluate(c_value: float) -> tuple[float, float, float, float]:
        key = float(c_value)
        if key in cache:
            return cache[key]
        model = ConfidenceOT(
            backbone=backbone, variant="exact", rejection_cost=key,
            epsilon=epsilon, lambda_a=lambda_a, lambda_b=lambda_b,
            source_rejection_budget=source_rejection_budget,
            target_rejection_budget=target_rejection_budget,
            tolerance=tolerance, max_iterations=max_iterations,
            max_outer_iterations=max_outer_iterations, device=device,
            cuda_dtype=cuda_dtype, fallback_to_cpu=fallback_to_cpu,
            warn_on_terminal=False,
        )
        fitted = model.fit_many(nulls, workers=workers)
        for fit in fitted:
            if not fit.inner_converged or not fit.outer_converged or fit.cycle_detected:
                warning_set.add("At least one M4-E calibration fit ended with a terminal warning.")
        cache[key] = (
            float(np.mean([fit.source_raw_acceptance for fit in fitted])),
            float(np.mean([np.mean(fit.source_gate) for fit in fitted])),
            float(np.mean([fit.target_raw_acceptance for fit in fitted])),
            float(np.mean([np.mean(fit.target_gate) for fit in fitted])),
        )
        return cache[key]

    curve = np.asarray([evaluate(float(value)) for value in curve_costs])
    source_raw, source_projected = curve[:, 0], curve[:, 1]
    target_raw, target_projected = curve[:, 2], curve[:, 3]
    source_monotone = bool(np.all(np.diff(source_raw) >= -1e-12))
    target_monotone = bool(np.all(np.diff(target_raw) >= -1e-12))
    feasible = (source_raw <= source_raw_acceptance_target) & (target_raw <= target_raw_acceptance_target)
    if np.any(feasible):
        index = int(np.flatnonzero(feasible)[-1])
        c_star = float(curve_costs[index])
        selection = "largest_jointly_feasible"
    else:
        violation = np.maximum(
            np.maximum(source_raw - source_raw_acceptance_target, 0),
            np.maximum(target_raw - target_raw_acceptance_target, 0),
        )
        index = int(np.flatnonzero(np.isclose(violation, violation.min()))[-1])
        c_star = float(curve_costs[index])
        selection = "minimum_joint_violation_fallback"
        warning_set.add("No jointly feasible rejection cost was found; the least-violating value was retained.")
    refinement = "grid"
    if np.any(feasible) and source_monotone and target_monotone and index + 1 < len(curve_costs):
        low, high = c_star, float(curve_costs[index + 1])
        while (high - low) / max(low, np.finfo(float).tiny) > refinement_relative_tolerance:
            middle = math.sqrt(low * high)
            sx, _, sy, _ = evaluate(middle)
            if sx <= source_raw_acceptance_target and sy <= target_raw_acceptance_target:
                low = middle
            else:
                high = middle
        c_star = low
        refinement = "joint_bisection"
    elif not source_monotone or not target_monotone:
        warning_set.add("A raw-acceptance curve was nonmonotone; continuous refinement was skipped.")

    validation_records: list[NullValidationRecord] = []
    validation_model = ConfidenceOT(
            backbone=backbone, variant="reversible", rejection_cost=c_star,
            epsilon=epsilon, lambda_a=lambda_a, lambda_b=lambda_b,
            source_rejection_budget=source_rejection_budget,
            target_rejection_budget=target_rejection_budget,
            tolerance=tolerance, max_iterations=max_iterations,
            max_outer_iterations=max_outer_iterations, device=device,
            cuda_dtype=cuda_dtype, fallback_to_cpu=fallback_to_cpu,
            warn_on_terminal=False,
        )
    validation_fits = validation_model.fit_many(validation_values, workers=workers)
    for index, fit in enumerate(validation_fits):
        validation_records.append(NullValidationRecord(
            null_index=index,
            source_raw_acceptance=fit.source_raw_acceptance,
            target_raw_acceptance=fit.target_raw_acceptance,
            inner_converged=fit.inner_converged,
            outer_converged=fit.outer_converged,
            cycle_detected=fit.cycle_detected,
        ))
        if not fit.inner_converged or not fit.outer_converged or fit.cycle_detected:
            warning_set.add("At least one held-out M4-R validation fit ended with a terminal warning.")
        if (
            fit.source_raw_acceptance > source_raw_acceptance_target
            or fit.target_raw_acceptance > target_raw_acceptance_target
        ):
            warning_set.add("The frozen rejection cost exceeded a held-out M4-R raw-acceptance target.")
    messages = tuple(sorted(warning_set))
    if emit_warnings:
        for message in messages:
            warnings.warn(message, RuntimeWarning, stacklevel=2)
    return NullCalibrationResult(
        backbone=backbone,
        rejection_cost=c_star,
        curve_costs=np.asarray(sorted(cache), dtype=np.float64),
        source_raw_acceptance_curve=np.asarray([cache[c][0] for c in sorted(cache)]),
        target_raw_acceptance_curve=np.asarray([cache[c][2] for c in sorted(cache)]),
        source_projected_acceptance_curve=np.asarray([cache[c][1] for c in sorted(cache)]),
        target_projected_acceptance_curve=np.asarray([cache[c][3] for c in sorted(cache)]),
        selection_status=selection,
        calibration_valid=bool(np.any(feasible) and not messages),
        source_monotone=source_monotone,
        target_monotone=target_monotone,
        refinement_method=refinement,
        warning_messages=messages,
        validation=tuple(validation_records),
    )
