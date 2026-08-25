"""Annotation-free cross-snapshot outlier filtering followed by balanced OT."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from traditional_ot.balanced import BalancedOTResult, balanced_ot
from traditional_ot.unbalanced import UOTResult, squared_euclidean_cost, unbalanced_ot


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class OutlierTrimmedOTResult:
    transition_probability: FloatArray
    coupling: FloatArray
    source_outlier_score: FloatArray
    target_outlier_score: FloatArray
    source_outlier: BoolArray
    target_outlier: BoolArray
    source_inlier: BoolArray
    target_inlier: BoolArray
    inlier_result: BalancedOTResult | UOTResult
    k_neighbors: int
    solver: str


def _matrix(values: ArrayLike, *, name: str) -> FloatArray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"`{name}` must be a numeric 2D matrix.") from error
    if array.ndim != 2 or min(array.shape) == 0:
        raise ValueError(f"`{name}` must be a non-empty 2D matrix.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"`{name}` must contain only finite values.")
    return np.ascontiguousarray(array)


def _fraction(value: object, *, name: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"`{name}` must be a finite fraction in [0, 1).") from error
    if not np.isfinite(converted) or not 0.0 <= converted < 1.0:
        raise ValueError(f"`{name}` must be a finite fraction in [0, 1).")
    return converted


def _kth_distance(cost: FloatArray, k: int) -> FloatArray:
    index = min(k - 1, cost.shape[1] - 1)
    return np.sqrt(np.partition(cost, index, axis=1)[:, index])


def cross_snapshot_outlier_scores(
    source: ArrayLike,
    target: ArrayLike,
    *,
    k_neighbors: int = 10,
) -> tuple[FloatArray, FloatArray]:
    """Return cross-snapshot support scores on a snapshot-wide local scale.

    For each cell the score is its cross-snapshot kNN radius divided by the
    median within-snapshot kNN radius. Using a snapshot-wide denominator lets
    the score detect both coherent unmatched clusters and diffuse technical
    outliers; a cell-specific denominator would hide diffuse outliers because
    they are sparse in both the within- and cross-snapshot neighborhoods.
    """
    source_values = _matrix(source, name="source")
    target_values = _matrix(target, name="target")
    if source_values.shape[1] != target_values.shape[1]:
        raise ValueError("`source` and `target` must share their feature dimension.")
    if isinstance(k_neighbors, (bool, np.bool_)) or not isinstance(k_neighbors, (int, np.integer)) or k_neighbors <= 0:
        raise ValueError("`k_neighbors` must be a positive integer.")
    if min(len(source_values), len(target_values)) <= k_neighbors:
        raise ValueError("Each snapshot must contain more cells than `k_neighbors`.")
    cross = squared_euclidean_cost(source_values, target_values)
    source_within = squared_euclidean_cost(source_values, source_values)
    target_within = squared_euclidean_cost(target_values, target_values)
    np.fill_diagonal(source_within, np.inf)
    np.fill_diagonal(target_within, np.inf)
    source_cross_radius = _kth_distance(cross, int(k_neighbors))
    target_cross_radius = _kth_distance(cross.T, int(k_neighbors))
    source_local_scale = float(np.median(_kth_distance(source_within, int(k_neighbors))))
    target_local_scale = float(np.median(_kth_distance(target_within, int(k_neighbors))))
    tiny = np.finfo(np.float64).eps
    return (
        np.log((source_cross_radius + tiny) / (source_local_scale + tiny)),
        np.log((target_cross_radius + tiny) / (target_local_scale + tiny)),
    )


def _top_fraction(scores: FloatArray, fraction: float) -> BoolArray:
    count = int(round(len(scores) * fraction))
    selected = np.zeros(len(scores), dtype=bool)
    if count:
        selected[np.argsort(-scores, kind="mergesort")[:count]] = True
    return selected


def outlier_trimmed_ot(
    source: ArrayLike,
    target: ArrayLike,
    *,
    source_outlier_fraction: float,
    target_outlier_fraction: float,
    k_neighbors: int = 10,
    epsilon: float = 0.08,
    threshold: float = 1e-7,
    max_iterations: int = 5_000,
) -> OutlierTrimmedOTResult:
    """Detect unsupported cells without annotations and run OT on inliers."""
    source_values = _matrix(source, name="source")
    target_values = _matrix(target, name="target")
    source_fraction = _fraction(source_outlier_fraction, name="source_outlier_fraction")
    target_fraction = _fraction(target_outlier_fraction, name="target_outlier_fraction")
    source_score, target_score = cross_snapshot_outlier_scores(
        source_values, target_values, k_neighbors=k_neighbors
    )
    source_outlier = _top_fraction(source_score, source_fraction)
    target_outlier = _top_fraction(target_score, target_fraction)
    source_inlier, target_inlier = ~source_outlier, ~target_outlier
    if not source_inlier.any() or not target_inlier.any():
        raise ValueError("Outlier fractions leave no inlier cells for OT.")
    cost = squared_euclidean_cost(source_values[source_inlier], target_values[target_inlier])
    mean_cost = float(cost.mean())
    if not np.isfinite(mean_cost) or mean_cost <= 0.0:
        raise ValueError("The inlier cost matrix is degenerate.")
    result = balanced_ot(
        cost / mean_cost,
        epsilon=epsilon,
        threshold=threshold,
        max_iterations=max_iterations,
    )
    transition = np.zeros((len(source_values), len(target_values)))
    coupling = np.zeros_like(transition)
    transition[np.ix_(source_inlier, target_inlier)] = result.transition_probability
    coupling[np.ix_(source_inlier, target_inlier)] = result.coupling
    return OutlierTrimmedOTResult(
        transition_probability=transition,
        coupling=coupling,
        source_outlier_score=source_score,
        target_outlier_score=target_score,
        source_outlier=source_outlier,
        target_outlier=target_outlier,
        source_inlier=source_inlier,
        target_inlier=target_inlier,
        inlier_result=result,
        k_neighbors=int(k_neighbors),
        solver="balanced",
    )


def outlier_trimmed_uot(
    source: ArrayLike,
    target: ArrayLike,
    *,
    source_outlier_fraction: float,
    target_outlier_fraction: float,
    k_neighbors: int = 10,
    epsilon: float = 0.08,
    lambda_a: float = 0.5,
    lambda_b: float = 10.0,
    threshold: float = 1e-7,
    max_iterations: int = 5_000,
) -> OutlierTrimmedOTResult:
    """Detect unsupported cells and run KL-unbalanced OT on the inliers."""
    source_values = _matrix(source, name="source")
    target_values = _matrix(target, name="target")
    source_fraction = _fraction(source_outlier_fraction, name="source_outlier_fraction")
    target_fraction = _fraction(target_outlier_fraction, name="target_outlier_fraction")
    source_score, target_score = cross_snapshot_outlier_scores(
        source_values, target_values, k_neighbors=k_neighbors
    )
    source_outlier = _top_fraction(source_score, source_fraction)
    target_outlier = _top_fraction(target_score, target_fraction)
    source_inlier, target_inlier = ~source_outlier, ~target_outlier
    if not source_inlier.any() or not target_inlier.any():
        raise ValueError("Outlier fractions leave no inlier cells for UOT.")
    cost = squared_euclidean_cost(source_values[source_inlier], target_values[target_inlier])
    mean_cost = float(cost.mean())
    if not np.isfinite(mean_cost) or mean_cost <= 0.0:
        raise ValueError("The inlier cost matrix is degenerate.")
    result = unbalanced_ot(
        cost / mean_cost,
        epsilon=epsilon,
        lambda_a=lambda_a,
        lambda_b=lambda_b,
        threshold=threshold,
        max_iterations=max_iterations,
    )
    transition = np.zeros((len(source_values), len(target_values)))
    coupling = np.zeros_like(transition)
    transition[np.ix_(source_inlier, target_inlier)] = result.transition_probability
    coupling[np.ix_(source_inlier, target_inlier)] = result.coupling
    return OutlierTrimmedOTResult(
        transition_probability=transition,
        coupling=coupling,
        source_outlier_score=source_score,
        target_outlier_score=target_score,
        source_outlier=source_outlier,
        target_outlier=target_outlier,
        source_inlier=source_inlier,
        target_inlier=target_inlier,
        inlier_result=result,
        k_neighbors=int(k_neighbors),
        solver="unbalanced",
    )
