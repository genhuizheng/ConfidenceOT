"""Balanced entropic Sinkhorn OT matching moscot's full-rank linear baseline.

The implementation follows the numerical path used by
``moscot.problems.generic.SinkhornProblem`` for a point-cloud geometry with
uniform marginals, squared-Euclidean cost, mean cost scaling, and the
log-sum-exp (LSE) Sinkhorn solver.

Moscot stores cells as rows.  The public function below defaults to cells as
columns because that is the convention required by this project, and
transposes the input before constructing the point clouds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
ScaleCost = Literal["mean", "max_cost"] | float


@dataclass(frozen=True)
class TraditionalOTResult:
    """Result of the traditional balanced Sinkhorn OT baseline.

    Attributes
    ----------
    coupling
        Joint transport mass with shape ``(n_source_cells, n_target_cells)``.
        Its row and column sums equal ``source_marginal`` and
        ``target_marginal``, respectively, up to the requested tolerance.
    transition_probability
        Forward conditional transition probabilities.  Entry ``[i, j]`` is
        ``P(target column j | source column i)`` and every positive-mass source
        row sums to one.
    cost_matrix
        Unscaled pairwise squared-Euclidean cost.
    scaled_cost_matrix
        Cost used by Sinkhorn after moscot-compatible scaling.
    source_marginal, target_marginal
        Normalized input masses.  They are uniform when weights are omitted.
    converged
        Whether the target-marginal L1 error crossed ``threshold``.
    n_iterations
        Number of Sinkhorn updates performed.
    marginal_error
        Final target-marginal L1 error, matching the balanced OTT stopping
        criterion.
    transport_cost
        Unregularized transport component ``sum(coupling * scaled_cost)``.
    """

    coupling: FloatArray
    transition_probability: FloatArray
    cost_matrix: FloatArray
    scaled_cost_matrix: FloatArray
    source_marginal: FloatArray
    target_marginal: FloatArray
    converged: bool
    n_iterations: int
    marginal_error: float
    transport_cost: float


def _as_cell_rows(matrix: ArrayLike, *, cells_axis: int, name: str) -> FloatArray:
    try:
        values = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError(f"`{name}` must be a numeric 2D matrix.") from e
    if values.ndim != 2:
        raise ValueError(f"`{name}` must be a 2D matrix, found shape {values.shape}.")
    if isinstance(cells_axis, (bool, np.bool_)) or cells_axis not in (0, 1):
        raise ValueError(f"`cells_axis` must be 0 or 1, found {cells_axis}.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"`{name}` contains NaN or infinite values.")
    cells = values if cells_axis == 0 else values.T
    if cells.shape[0] == 0:
        raise ValueError(f"`{name}` contains no cells along axis {cells_axis}.")
    if cells.shape[1] == 0:
        raise ValueError(f"`{name}` contains no features.")
    return np.ascontiguousarray(cells)


def _normalize_marginal(
    weights: ArrayLike | None, *, n: int, name: str
) -> FloatArray:
    if weights is None:
        return np.full(n, 1.0 / n, dtype=np.float64)
    try:
        marginal = np.asarray(weights, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError(f"`{name}` must be a numeric 1D array.") from e
    if marginal.ndim != 1 or marginal.shape[0] != n:
        raise ValueError(f"`{name}` must have shape ({n},), found {marginal.shape}.")
    if not np.all(np.isfinite(marginal)) or np.any(marginal < 0.0):
        raise ValueError(f"`{name}` must contain finite, non-negative values.")
    total = float(np.sum(marginal))
    if total <= 0.0:
        raise ValueError(f"`{name}` must have positive total mass.")
    return marginal / total


def _squared_euclidean(source: FloatArray, target: FloatArray) -> FloatArray:
    source_norm = np.sum(source * source, axis=1, keepdims=True)
    target_norm = np.sum(target * target, axis=1, keepdims=True).T
    cost = source_norm + target_norm - 2.0 * (source @ target.T)
    # OTT's algebraically equivalent computation can produce tiny negatives
    # from floating-point roundoff; squared distances are non-negative.
    return np.maximum(cost, 0.0)


def _scale_cost(cost: FloatArray, scale_cost: ScaleCost) -> FloatArray:
    if isinstance(scale_cost, str):
        if scale_cost == "mean":
            scale = float(np.mean(cost))
        elif scale_cost == "max_cost":
            scale = float(np.max(cost))
        else:
            raise ValueError(
                "`scale_cost` must be 'mean', 'max_cost', or a positive float."
            )
    else:
        try:
            scale = float(scale_cost)
        except (TypeError, ValueError) as e:
            raise ValueError(
                "`scale_cost` must be 'mean', 'max_cost', or a positive float."
            ) from e
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            "Cost scaling is zero or non-finite. The two point clouds may be "
            "degenerate; provide a positive numeric `scale_cost`."
        )
    return cost / scale


def _safe_log(values: FloatArray) -> FloatArray:
    result = np.full_like(values, -np.inf)
    positive = values > 0.0
    result[positive] = np.log(values[positive])
    return result


def _logsumexp(values: FloatArray, *, axis: int) -> FloatArray:
    maximum = np.max(values, axis=axis, keepdims=True)
    finite = np.isfinite(maximum)
    with np.errstate(invalid="ignore", divide="ignore"):
        shifted = np.where(finite, values - maximum, -np.inf)
        summed = np.sum(np.exp(shifted), axis=axis, keepdims=True)
        output = np.where(finite, maximum + np.log(summed), -np.inf)
    return np.squeeze(output, axis=axis)


def _coupling_from_potentials(
    f: FloatArray, g: FloatArray, cost: FloatArray, epsilon: float
) -> FloatArray:
    log_coupling = (f[:, None] + g[None, :] - cost) / epsilon
    return np.exp(log_coupling)


def _solve_traditional_ot(
    source: ArrayLike,
    target: ArrayLike,
    *,
    cells_axis: int = 1,
    epsilon: float = 1e-3,
    scale_cost: ScaleCost = "mean",
    source_weights: ArrayLike | None = None,
    target_weights: ArrayLike | None = None,
    threshold: float = 1e-3,
    inner_iterations: int = 10,
    max_iterations: int = 2000,
) -> TraditionalOTResult:
    """Compute column-to-column transition probabilities with traditional OT.

    This reproduces one moscot method: the full-rank, balanced, entropy-
    regularized linear OT problem solved by LSE Sinkhorn.  The defaults mirror
    ``moscot.problems.generic.SinkhornProblem.solve``.

    Parameters
    ----------
    source, target
        Two matrices sharing the same feature dimension. By default, columns
        are cells and rows are features.
    cells_axis
        Axis containing cells in both matrices. The project convention and
        default is ``1``; use ``0`` for moscot-style cells-as-rows matrices.
    epsilon
        Entropic regularization after cost scaling.
    scale_cost
        Moscot-compatible cost scaling. The default divides squared-Euclidean
        costs by their mean.
    source_weights, target_weights
        Optional non-negative column masses. Omitted weights are uniform.
    threshold
        Target-marginal L1 convergence threshold.
    inner_iterations
        Number of Sinkhorn updates between convergence checks.
    max_iterations
        Maximum number of Sinkhorn updates.

    Returns
    -------
    TraditionalOTResult
        Both the joint coupling and the source-conditioned transition matrix.
    """
    try:
        epsilon = float(epsilon)
    except (TypeError, ValueError) as e:
        raise ValueError("`epsilon` must be a positive finite float.") from e
    try:
        threshold = float(threshold)
    except (TypeError, ValueError) as e:
        raise ValueError("`threshold` must be a positive finite float.") from e
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError(f"`epsilon` must be positive and finite, found {epsilon}.")
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError(f"`threshold` must be positive and finite, found {threshold}.")
    if (
        isinstance(inner_iterations, (bool, np.bool_))
        or not isinstance(inner_iterations, (int, np.integer))
        or inner_iterations <= 0
    ):
        raise ValueError("`inner_iterations` must be a positive integer.")
    if (
        isinstance(max_iterations, (bool, np.bool_))
        or not isinstance(max_iterations, (int, np.integer))
        or max_iterations <= 0
    ):
        raise ValueError("`max_iterations` must be a positive integer.")

    source_cells = _as_cell_rows(source, cells_axis=cells_axis, name="source")
    target_cells = _as_cell_rows(target, cells_axis=cells_axis, name="target")
    if source_cells.shape[1] != target_cells.shape[1]:
        raise ValueError(
            "Source and target must share the feature dimension, found "
            f"{source_cells.shape[1]} and {target_cells.shape[1]}."
        )

    a = _normalize_marginal(source_weights, n=source_cells.shape[0], name="source_weights")
    b = _normalize_marginal(target_weights, n=target_cells.shape[0], name="target_weights")
    raw_cost = _squared_euclidean(source_cells, target_cells)
    cost = _scale_cost(raw_cost, scale_cost)

    # OTT's DefaultInitializer starts both LSE potentials at zero. Its
    # Gauss-Seidel update order is target potential g, then source potential f.
    f = np.zeros_like(a)
    g = np.zeros_like(b)
    log_a = _safe_log(a)
    log_b = _safe_log(b)

    coupling = np.empty_like(cost)
    error = np.inf
    converged = False
    n_iterations = 0

    for iteration in range(max_iterations):
        g = epsilon * (
            log_b - _logsumexp((f[:, None] - cost) / epsilon, axis=0)
        )
        f = epsilon * (
            log_a - _logsumexp((g[None, :] - cost) / epsilon, axis=1)
        )
        n_iterations = iteration + 1

        should_check = (
            n_iterations % inner_iterations == 0 or n_iterations == max_iterations
        )
        if should_check:
            coupling = _coupling_from_potentials(f, g, cost, epsilon)
            error = float(np.sum(np.abs(np.sum(coupling, axis=0) - b)))
            if np.isfinite(error) and error < threshold:
                converged = True
                break
            if not np.isfinite(error):
                break

    # Ensure the returned matrix is reconstructed from the final potentials
    # even when max_iterations is not a multiple of inner_iterations.
    coupling = _coupling_from_potentials(f, g, cost, epsilon)
    error = float(np.sum(np.abs(np.sum(coupling, axis=0) - b)))
    converged = bool(np.isfinite(error) and error < threshold)

    row_mass = np.sum(coupling, axis=1, keepdims=True)
    transition = np.divide(
        coupling,
        row_mass,
        out=np.zeros_like(coupling),
        where=row_mass > 0.0,
    )

    return TraditionalOTResult(
        coupling=coupling,
        transition_probability=transition,
        cost_matrix=raw_cost,
        scaled_cost_matrix=cost,
        source_marginal=a,
        target_marginal=b,
        converged=converged,
        n_iterations=n_iterations,
        marginal_error=error,
        transport_cost=float(np.sum(coupling * cost)),
    )


def traditional_method(
    source_matrix: ArrayLike,
    target_matrix: ArrayLike,
    *,
    cells_axis: int = 1,
    epsilon: float = 1e-3,
    scale_cost: ScaleCost = "mean",
    source_weights: ArrayLike | None = None,
    target_weights: ArrayLike | None = None,
    threshold: float = 1e-3,
    inner_iterations: int = 10,
    max_iterations: int = 2000,
) -> FloatArray:
    """Return source-column to target-column transition probabilities.

    This is the single public entry point for the traditional baseline. By
    default, ``source_matrix`` and ``target_matrix`` have shape
    ``(n_features, n_cells)``. The returned matrix has shape
    ``(n_source_cells, n_target_cells)`` and every row sums to one.

    Invalid input or parameters raise :class:`ValueError`. A solver that does
    not meet the requested convergence threshold raises :class:`RuntimeError`
    instead of silently returning an unreliable transition matrix.
    """
    result = _solve_traditional_ot(
        source_matrix,
        target_matrix,
        cells_axis=cells_axis,
        epsilon=epsilon,
        scale_cost=scale_cost,
        source_weights=source_weights,
        target_weights=target_weights,
        threshold=threshold,
        inner_iterations=inner_iterations,
        max_iterations=max_iterations,
    )
    if not result.converged:
        raise RuntimeError(
            "Traditional Sinkhorn OT did not converge: "
            f"target-marginal L1 error={result.marginal_error:.6g}, "
            f"threshold={threshold:.6g}, iterations={result.n_iterations}."
        )
    return result.transition_probability
