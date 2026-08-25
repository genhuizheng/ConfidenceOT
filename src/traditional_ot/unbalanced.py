"""KL-unbalanced OT and confidence-filtered UOT.

This module implements the objective and algorithms in the accompanying
CF-UOT manuscript.  Costs are accepted directly and are therefore assumed to
have already been embedded and scaled consistently across observed and null
pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
CFVariant = Literal["exact", "reversible"]


@dataclass(frozen=True)
class UOTResult:
    """Solution of entropic KL-unbalanced optimal transport."""

    coupling: FloatArray
    transition_probability: FloatArray
    source_mass: FloatArray
    target_mass: FloatArray
    conditional_loss: FloatArray
    cost_matrix: FloatArray
    source_marginal: FloatArray
    target_marginal: FloatArray
    log_source_scaling: FloatArray
    log_target_scaling: FloatArray
    epsilon: float
    lambda_a: float
    lambda_b: float
    alpha: float
    beta: float
    objective: float
    converged: bool
    n_iterations: int
    fixed_point_error: float


@dataclass(frozen=True)
class CFUOTResult:
    """Solution of exact or reversible confidence-filtered UOT."""

    coupling: FloatArray
    transition_probability: FloatArray
    source_mass: FloatArray
    target_mass: FloatArray
    conditional_loss: FloatArray
    gate_score: FloatArray
    gate: BoolArray
    log_source_scaling: FloatArray
    log_target_scaling: FloatArray
    objective: float
    objective_history: tuple[float, ...]
    gate_history: tuple[BoolArray, ...]
    variant: CFVariant
    rejection_cost: float
    tau: float
    boundary_count: int
    inner_converged: bool
    outer_converged: bool
    n_outer_solves: int
    total_inner_iterations: int


@dataclass(frozen=True)
class CalibrationResult:
    """Null-calibrated rejection cost and its required diagnostics."""

    rejection_cost: float
    initial_estimate: float
    null_quantiles: FloatArray
    observed_losses: FloatArray
    null_losses: tuple[FloatArray, ...]
    observed_acceptance: float
    null_acceptance: float
    refinement_method: str
    acceptance_curve_costs: FloatArray
    acceptance_curve: FloatArray
    monotonic_on_grid: bool


def _positive_finite(value: object, *, name: str, allow_zero: bool = False) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"`{name}` must be a {qualifier} finite float.") from error
    invalid = converted < 0.0 if allow_zero else converted <= 0.0
    if not np.isfinite(converted) or invalid:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"`{name}` must be {qualifier} and finite.")
    return converted


def _positive_integer(value: object, *, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value <= 0
    ):
        raise ValueError(f"`{name}` must be a positive integer.")
    return int(value)


def _cost_matrix(cost: ArrayLike, *, name: str = "cost_matrix") -> FloatArray:
    try:
        values = np.asarray(cost, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"`{name}` must be a numeric 2D matrix.") from error
    if values.ndim != 2 or min(values.shape) == 0:
        raise ValueError(f"`{name}` must be a non-empty 2D matrix.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"`{name}` contains NaN or infinite values.")
    if np.any(values < 0.0):
        raise ValueError(f"`{name}` must be non-negative.")
    return np.ascontiguousarray(values)


def _marginal(
    weights: ArrayLike | None, *, n: int, name: str, normalize: bool = True
) -> FloatArray:
    if weights is None:
        if not normalize:
            raise ValueError(f"`{name}` is required when `normalize_weights=False`.")
        return np.full(n, 1.0 / n, dtype=np.float64)
    try:
        values = np.asarray(weights, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"`{name}` must be a numeric 1D array.") from error
    if values.ndim != 1 or values.shape != (n,):
        raise ValueError(f"`{name}` must have shape ({n},), found {values.shape}.")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(f"`{name}` must contain strictly positive finite values.")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError(f"`{name}` must have positive total mass.")
    return values / total if normalize else values.copy()


def _gate(gate: ArrayLike, *, n_source: int) -> BoolArray:
    values = np.asarray(gate)
    if values.ndim != 1 or values.shape != (n_source,):
        raise ValueError(f"`gate` must have shape ({n_source},), found {values.shape}.")
    if not np.all(np.isin(values, (0, 1, False, True))):
        raise ValueError("`gate` must contain only 0/1 or boolean values.")
    return values.astype(bool, copy=True)


def _logsumexp(values: FloatArray, *, axis: int) -> FloatArray:
    maximum = np.max(values, axis=axis, keepdims=True)
    if not np.all(np.isfinite(maximum)):
        raise FloatingPointError(
            "A Sinkhorn reduction has no finite support. Increase `epsilon` or "
            "rescale the cost matrix."
        )
    shifted = values - maximum
    result = maximum + np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis)


def _generalized_kl(values: FloatArray, reference: FloatArray) -> float:
    positive = values > 0.0
    terms = reference - values
    terms = terms.copy()
    terms[positive] += values[positive] * (
        np.log(values[positive]) - np.log(reference[positive])
    )
    return float(np.sum(terms))


def _conditional_from_log_coupling(
    log_coupling: FloatArray, scoring_cost: FloatArray
) -> tuple[FloatArray, FloatArray]:
    log_mass = _logsumexp(log_coupling, axis=1)
    conditional = np.exp(log_coupling - log_mass[:, None])
    loss = np.sum(conditional * scoring_cost, axis=1)
    return conditional, loss


def _objective(
    coupling: FloatArray,
    optimization_cost: FloatArray,
    a: FloatArray,
    b: FloatArray,
    epsilon: float,
    lambda_a: float,
    lambda_b: float,
) -> float:
    reference = a[:, None] * b[None, :]
    source_mass = coupling.sum(axis=1)
    target_mass = coupling.sum(axis=0)
    return float(
        np.sum(coupling * optimization_cost)
        + epsilon * _generalized_kl(coupling, reference)
        + lambda_a * _generalized_kl(source_mass, a)
        + lambda_b * _generalized_kl(target_mass, b)
    )


def _solve_uot(
    optimization_cost: FloatArray,
    *,
    scoring_cost: FloatArray,
    epsilon: float,
    lambda_a: float,
    lambda_b: float,
    source_weights: ArrayLike | None,
    target_weights: ArrayLike | None,
    threshold: float,
    max_iterations: int,
    warm_start: tuple[ArrayLike, ArrayLike] | None = None,
    normalize_weights: bool = True,
) -> UOTResult:
    n_source, n_target = optimization_cost.shape
    if not isinstance(normalize_weights, (bool, np.bool_)):
        raise ValueError("`normalize_weights` must be boolean.")
    a = _marginal(
        source_weights, n=n_source, name="source_weights", normalize=normalize_weights
    )
    b = _marginal(
        target_weights, n=n_target, name="target_weights", normalize=normalize_weights
    )
    log_a = np.log(a)
    log_b = np.log(b)
    alpha = lambda_a / (lambda_a + epsilon)
    beta = lambda_b / (lambda_b + epsilon)
    with np.errstate(over="ignore", invalid="ignore"):
        scaled_cost = optimization_cost / epsilon
    if not np.all(np.isfinite(scaled_cost)):
        raise ValueError(
            "`cost_matrix / epsilon` is non-finite; rescale costs or increase epsilon."
        )
    log_kernel = log_a[:, None] + log_b[None, :] - scaled_cost

    if warm_start is None:
        log_u = np.zeros(n_source, dtype=np.float64)
        log_v = np.zeros(n_target, dtype=np.float64)
    else:
        if len(warm_start) != 2:
            raise ValueError("`warm_start` must be a pair `(log_u, log_v)`.")
        log_u = np.asarray(warm_start[0], dtype=np.float64).copy()
        log_v = np.asarray(warm_start[1], dtype=np.float64).copy()
        if log_u.shape != (n_source,) or log_v.shape != (n_target,):
            raise ValueError("`warm_start` arrays have incompatible shapes.")
        if not np.all(np.isfinite(log_u)) or not np.all(np.isfinite(log_v)):
            raise ValueError("`warm_start` must contain finite log-scalings.")

    error = np.inf
    converged = False
    n_iterations = 0
    for iteration in range(max_iterations):
        old_log_u = log_u
        old_log_v = log_v
        if alpha == 0.0:
            log_u = np.zeros_like(log_u)
        else:
            log_u = alpha * (
                log_a - _logsumexp(log_kernel + log_v[None, :], axis=1)
            )
        if beta == 0.0:
            log_v = np.zeros_like(log_v)
        else:
            log_v = beta * (
                log_b - _logsumexp(log_kernel + log_u[:, None], axis=0)
            )
        n_iterations = iteration + 1
        error = float(
            max(
                np.max(np.abs(log_u - old_log_u)),
                np.max(np.abs(log_v - old_log_v)),
            )
        )
        if not np.isfinite(error):
            break
        if error < threshold:
            converged = True
            break

    log_coupling = log_u[:, None] + log_kernel + log_v[None, :]
    if float(np.max(log_coupling)) > np.log(np.finfo(np.float64).max):
        raise FloatingPointError("The UOT coupling overflowed; rescale the problem.")
    coupling = np.exp(log_coupling)
    transition, conditional_loss = _conditional_from_log_coupling(
        log_coupling, scoring_cost
    )
    source_mass = coupling.sum(axis=1)
    target_mass = coupling.sum(axis=0)
    return UOTResult(
        coupling=coupling,
        transition_probability=transition,
        source_mass=source_mass,
        target_mass=target_mass,
        conditional_loss=conditional_loss,
        cost_matrix=optimization_cost.copy(),
        source_marginal=a,
        target_marginal=b,
        log_source_scaling=log_u,
        log_target_scaling=log_v,
        epsilon=epsilon,
        lambda_a=lambda_a,
        lambda_b=lambda_b,
        alpha=alpha,
        beta=beta,
        objective=_objective(
            coupling, optimization_cost, a, b, epsilon, lambda_a, lambda_b
        ),
        converged=converged,
        n_iterations=n_iterations,
        fixed_point_error=error,
    )


def unbalanced_ot(
    cost_matrix: ArrayLike,
    *,
    epsilon: float,
    lambda_a: float,
    lambda_b: float,
    source_weights: ArrayLike | None = None,
    target_weights: ArrayLike | None = None,
    threshold: float = 1e-9,
    max_iterations: int = 10_000,
    normalize_weights: bool = True,
) -> UOTResult:
    """Solve standard entropic KL-UOT from a precomputed non-negative cost."""
    cost = _cost_matrix(cost_matrix)
    epsilon = _positive_finite(epsilon, name="epsilon")
    lambda_a = _positive_finite(lambda_a, name="lambda_a", allow_zero=True)
    lambda_b = _positive_finite(lambda_b, name="lambda_b", allow_zero=True)
    threshold = _positive_finite(threshold, name="threshold")
    max_iterations = _positive_integer(max_iterations, name="max_iterations")
    return _solve_uot(
        cost,
        scoring_cost=cost,
        epsilon=epsilon,
        lambda_a=lambda_a,
        lambda_b=lambda_b,
        source_weights=source_weights,
        target_weights=target_weights,
        threshold=threshold,
        max_iterations=max_iterations,
        normalize_weights=normalize_weights,
    )


def solve_fixed_gate_uot(
    cost_matrix: ArrayLike,
    gate: ArrayLike,
    *,
    rejection_cost: float,
    epsilon: float,
    lambda_a: float,
    lambda_b: float,
    source_weights: ArrayLike | None = None,
    target_weights: ArrayLike | None = None,
    threshold: float = 1e-9,
    max_iterations: int = 10_000,
    warm_start: tuple[ArrayLike, ArrayLike] | None = None,
    normalize_weights: bool = True,
) -> UOTResult:
    """Solve CF-UOT for a fixed binary source gate."""
    cost = _cost_matrix(cost_matrix)
    accepted = _gate(gate, n_source=cost.shape[0])
    rejection_cost = _positive_finite(rejection_cost, name="rejection_cost")
    epsilon = _positive_finite(epsilon, name="epsilon")
    lambda_a = _positive_finite(lambda_a, name="lambda_a", allow_zero=True)
    lambda_b = _positive_finite(lambda_b, name="lambda_b", allow_zero=True)
    threshold = _positive_finite(threshold, name="threshold")
    max_iterations = _positive_integer(max_iterations, name="max_iterations")
    filtered = np.where(accepted[:, None], cost, rejection_cost)
    return _solve_uot(
        filtered,
        scoring_cost=cost,
        epsilon=epsilon,
        lambda_a=lambda_a,
        lambda_b=lambda_b,
        source_weights=source_weights,
        target_weights=target_weights,
        threshold=threshold,
        max_iterations=max_iterations,
        warm_start=warm_start,
        normalize_weights=normalize_weights,
    )


def _counterfactual_loss(result: UOTResult, original_cost: FloatArray) -> FloatArray:
    log_weights = (
        np.log(result.target_marginal)[None, :]
        + result.log_target_scaling[None, :]
        - original_cost / result.epsilon
    )
    normalizer = _logsumexp(log_weights, axis=1)
    conditional = np.exp(log_weights - normalizer[:, None])
    return np.sum(conditional * original_cost, axis=1)


def _updated_gate(
    score: FloatArray,
    current_gate: BoolArray,
    rejection_cost: float,
    tau: float,
) -> BoolArray:
    updated = current_gate.copy()
    updated[score < rejection_cost - tau] = True
    updated[score > rejection_cost + tau] = False
    return updated


def confidence_filtered_uot(
    cost_matrix: ArrayLike,
    *,
    rejection_cost: float,
    epsilon: float,
    lambda_a: float,
    lambda_b: float,
    variant: CFVariant = "reversible",
    source_weights: ArrayLike | None = None,
    target_weights: ArrayLike | None = None,
    initial_gate: ArrayLike | None = None,
    tau: float = 0.0,
    threshold: float = 1e-9,
    max_iterations: int = 10_000,
    max_outer_iterations: int = 100,
    warm_start: bool = True,
) -> CFUOTResult:
    """Run exact or reversible confidence-filtered KL-UOT.

    The returned coupling is always solved for the returned gate, including
    when the reversible outer loop exhausts its iteration budget.
    """
    cost = _cost_matrix(cost_matrix)
    rejection_cost = _positive_finite(rejection_cost, name="rejection_cost")
    epsilon = _positive_finite(epsilon, name="epsilon")
    lambda_a = _positive_finite(lambda_a, name="lambda_a", allow_zero=True)
    lambda_b = _positive_finite(lambda_b, name="lambda_b", allow_zero=True)
    tau = _positive_finite(tau, name="tau", allow_zero=True)
    threshold = _positive_finite(threshold, name="threshold")
    max_iterations = _positive_integer(max_iterations, name="max_iterations")
    max_outer_iterations = _positive_integer(
        max_outer_iterations, name="max_outer_iterations"
    )
    if variant not in ("exact", "reversible"):
        raise ValueError("`variant` must be 'exact' or 'reversible'.")
    if not isinstance(warm_start, (bool, np.bool_)):
        raise ValueError("`warm_start` must be boolean.")
    gate = (
        np.ones(cost.shape[0], dtype=bool)
        if initial_gate is None
        else _gate(initial_gate, n_source=cost.shape[0])
    )

    objective_history: list[float] = []
    gate_history: list[BoolArray] = [gate.copy()]
    log_warm: tuple[FloatArray, FloatArray] | None = None
    total_inner = 0
    all_inner_converged = True
    outer_converged = False
    result: UOTResult | None = None
    score: FloatArray | None = None
    solved_gate: BoolArray | None = None

    for _ in range(max_outer_iterations):
        solved_gate = gate.copy()
        result = solve_fixed_gate_uot(
            cost,
            gate,
            rejection_cost=rejection_cost,
            epsilon=epsilon,
            lambda_a=lambda_a,
            lambda_b=lambda_b,
            source_weights=source_weights,
            target_weights=target_weights,
            threshold=threshold,
            max_iterations=max_iterations,
            warm_start=log_warm if warm_start else None,
        )
        total_inner += result.n_iterations
        all_inner_converged &= result.converged
        objective_history.append(result.objective)
        score = (
            result.conditional_loss
            if variant == "exact"
            else _counterfactual_loss(result, cost)
        )
        new_gate = _updated_gate(score, gate, rejection_cost, tau)
        if np.array_equal(new_gate, gate):
            outer_converged = True
            break
        gate = new_gate
        gate_history.append(gate.copy())
        if warm_start:
            log_warm = (
                result.log_source_scaling.copy(),
                result.log_target_scaling.copy(),
            )

    assert result is not None and score is not None and solved_gate is not None
    # If the final update changed the gate at the budget boundary, re-solve so
    # that the returned coupling and returned gate are internally consistent.
    if not np.array_equal(solved_gate, gate):
        result = solve_fixed_gate_uot(
            cost,
            gate,
            rejection_cost=rejection_cost,
            epsilon=epsilon,
            lambda_a=lambda_a,
            lambda_b=lambda_b,
            source_weights=source_weights,
            target_weights=target_weights,
            threshold=threshold,
            max_iterations=max_iterations,
            warm_start=(
                result.log_source_scaling, result.log_target_scaling
            ) if warm_start else None,
        )
        total_inner += result.n_iterations
        all_inner_converged &= result.converged
        objective_history.append(result.objective)
        score = (
            result.conditional_loss
            if variant == "exact"
            else _counterfactual_loss(result, cost)
        )

    boundary_count = int(np.count_nonzero(np.abs(score - rejection_cost) <= tau))
    return CFUOTResult(
        coupling=result.coupling,
        transition_probability=result.transition_probability,
        source_mass=result.source_mass,
        target_mass=result.target_mass,
        conditional_loss=result.conditional_loss,
        gate_score=score,
        gate=gate,
        log_source_scaling=result.log_source_scaling,
        log_target_scaling=result.log_target_scaling,
        objective=result.objective,
        objective_history=tuple(objective_history),
        gate_history=tuple(gate_history),
        variant=variant,
        rejection_cost=rejection_cost,
        tau=tau,
        boundary_count=boundary_count,
        inner_converged=all_inner_converged,
        outer_converged=outer_converged,
        n_outer_solves=len(objective_history),
        total_inner_iterations=total_inner,
    )


def calibrate_rejection_cost(
    observed_cost: ArrayLike,
    null_costs: Sequence[ArrayLike],
    *,
    epsilon: float,
    lambda_a: float,
    lambda_b: float,
    acceptance_target: float = 0.10,
    acceptance_tolerance: float = 0.02,
    variant: CFVariant = "reversible",
    threshold: float = 1e-9,
    max_iterations: int = 10_000,
    max_outer_iterations: int = 100,
    grid_size: int = 15,
    refinement_relative_tolerance: float = 0.01,
) -> CalibrationResult:
    """Calibrate ``c`` from one or more precomputed geometric-null costs."""
    observed = _cost_matrix(observed_cost, name="observed_cost")
    if not isinstance(null_costs, Sequence) or len(null_costs) == 0:
        raise ValueError("`null_costs` must contain at least one cost matrix.")
    nulls = tuple(
        _cost_matrix(cost, name=f"null_costs[{index}]")
        for index, cost in enumerate(null_costs)
    )
    epsilon = _positive_finite(epsilon, name="epsilon")
    lambda_a = _positive_finite(lambda_a, name="lambda_a", allow_zero=True)
    lambda_b = _positive_finite(lambda_b, name="lambda_b", allow_zero=True)
    acceptance_target = _positive_finite(
        acceptance_target, name="acceptance_target"
    )
    if acceptance_target >= 1.0:
        raise ValueError("`acceptance_target` must be smaller than 1.")
    acceptance_tolerance = _positive_finite(
        acceptance_tolerance, name="acceptance_tolerance", allow_zero=True
    )
    grid_size = _positive_integer(grid_size, name="grid_size")
    if grid_size < 3:
        raise ValueError("`grid_size` must be at least 3.")
    refinement_relative_tolerance = _positive_finite(
        refinement_relative_tolerance, name="refinement_relative_tolerance"
    )

    null_vanilla = tuple(
        unbalanced_ot(
            cost,
            epsilon=epsilon,
            lambda_a=lambda_a,
            lambda_b=lambda_b,
            threshold=threshold,
            max_iterations=max_iterations,
        )
        for cost in nulls
    )
    if not all(result.converged for result in null_vanilla):
        raise RuntimeError("At least one vanilla UOT null solve did not converge.")
    null_losses = tuple(result.conditional_loss for result in null_vanilla)
    null_quantiles = np.array([
        np.quantile(losses, acceptance_target) for losses in null_losses
    ])
    estimate = float(np.median(null_quantiles))
    if not np.isfinite(estimate) or estimate <= 0.0:
        raise RuntimeError("Null losses produced a non-positive rejection-cost estimate.")

    observed_vanilla = unbalanced_ot(
        observed,
        epsilon=epsilon,
        lambda_a=lambda_a,
        lambda_b=lambda_b,
        threshold=threshold,
        max_iterations=max_iterations,
    )
    if not observed_vanilla.converged:
        raise RuntimeError("The vanilla UOT observed solve did not converge.")

    cache: dict[float, float] = {}

    def null_acceptance(candidate: float) -> float:
        key = float(candidate)
        if key not in cache:
            fractions = []
            for null in nulls:
                fitted = confidence_filtered_uot(
                    null,
                    rejection_cost=key,
                    epsilon=epsilon,
                    lambda_a=lambda_a,
                    lambda_b=lambda_b,
                    variant=variant,
                    threshold=threshold,
                    max_iterations=max_iterations,
                    max_outer_iterations=max_outer_iterations,
                )
                if not fitted.inner_converged:
                    raise RuntimeError("A CF-UOT null solve did not converge.")
                fractions.append(float(fitted.gate.mean()))
            cache[key] = float(np.mean(fractions))
        return cache[key]

    estimate_acceptance = null_acceptance(estimate)
    grid = np.geomspace(estimate * 1e-3, estimate * 2.0, grid_size)
    curve = np.array([null_acceptance(float(candidate)) for candidate in grid])
    monotonic = bool(np.all(np.diff(curve) >= -1e-12))
    refinement_method = "quantile"
    calibrated = estimate

    if abs(estimate_acceptance - acceptance_target) > acceptance_tolerance:
        feasible = np.flatnonzero(curve <= acceptance_target)
        if not len(feasible):
            raise RuntimeError(
                "No rejection cost on the calibration grid attains the requested "
                "null acceptance. The null or cost scale is uninformative."
            )
        if monotonic:
            lower_index = int(feasible[-1])
            if lower_index == len(grid) - 1:
                upper = float(grid[-1])
                for _ in range(20):
                    upper *= 2.0
                    if null_acceptance(upper) > acceptance_target:
                        break
                else:
                    raise RuntimeError(
                        "Could not bracket the largest feasible rejection cost."
                    )
                lower = float(grid[-1])
            else:
                lower = float(grid[lower_index])
                upper = float(grid[lower_index + 1])
            tolerance = refinement_relative_tolerance * estimate
            for _ in range(80):
                if upper - lower <= tolerance:
                    break
                middle = 0.5 * (lower + upper)
                if null_acceptance(middle) <= acceptance_target:
                    lower = middle
                else:
                    upper = middle
            calibrated = lower
            refinement_method = "bisection"
        else:
            calibrated = float(grid[int(feasible[-1])])
            refinement_method = "grid_nonmonotone"

    final_null_acceptance = null_acceptance(calibrated)
    observed_fit = confidence_filtered_uot(
        observed,
        rejection_cost=calibrated,
        epsilon=epsilon,
        lambda_a=lambda_a,
        lambda_b=lambda_b,
        variant=variant,
        threshold=threshold,
        max_iterations=max_iterations,
        max_outer_iterations=max_outer_iterations,
    )
    if not observed_fit.inner_converged:
        raise RuntimeError("The final observed CF-UOT solve did not converge.")
    return CalibrationResult(
        rejection_cost=calibrated,
        initial_estimate=estimate,
        null_quantiles=null_quantiles,
        observed_losses=observed_vanilla.conditional_loss,
        null_losses=null_losses,
        observed_acceptance=float(observed_fit.gate.mean()),
        null_acceptance=final_null_acceptance,
        refinement_method=refinement_method,
        acceptance_curve_costs=grid,
        acceptance_curve=curve,
        monotonic_on_grid=monotonic,
    )


def random_rotation_null(
    source: ArrayLike, target: ArrayLike, *, rng: np.random.Generator
) -> FloatArray:
    """Return an N2 squared-Euclidean null cost for cells stored as rows."""
    source_values = np.asarray(source, dtype=np.float64)
    target_values = np.asarray(target, dtype=np.float64)
    if source_values.ndim != 2 or target_values.ndim != 2:
        raise ValueError("`source` and `target` must be 2D matrices with cells as rows.")
    if source_values.shape[1] != target_values.shape[1]:
        raise ValueError("`source` and `target` must share their feature dimension.")
    if not np.all(np.isfinite(source_values)) or not np.all(np.isfinite(target_values)):
        raise ValueError("`source` and `target` must contain only finite values.")
    gaussian = rng.normal(size=(source_values.shape[1], source_values.shape[1]))
    q, r = np.linalg.qr(gaussian)
    q *= np.sign(np.diag(r))[None, :]
    center = source_values.mean(axis=0)
    rotated = center + (source_values - center) @ q
    return squared_euclidean_cost(rotated, target_values)


def feature_permutation_null(
    source: ArrayLike, target: ArrayLike, *, rng: np.random.Generator
) -> FloatArray:
    """Return an N3 feature-wise-permutation null cost for row-wise cells."""
    source_values = np.asarray(source, dtype=np.float64)
    target_values = np.asarray(target, dtype=np.float64)
    if source_values.ndim != 2 or target_values.ndim != 2:
        raise ValueError("`source` and `target` must be 2D matrices with cells as rows.")
    if source_values.shape[1] != target_values.shape[1]:
        raise ValueError("`source` and `target` must share their feature dimension.")
    if not np.all(np.isfinite(source_values)) or not np.all(np.isfinite(target_values)):
        raise ValueError("`source` and `target` must contain only finite values.")
    permuted = source_values.copy()
    for feature in range(permuted.shape[1]):
        permuted[:, feature] = permuted[rng.permutation(len(permuted)), feature]
    return squared_euclidean_cost(permuted, target_values)


def squared_euclidean_cost(source: ArrayLike, target: ArrayLike) -> FloatArray:
    """Compute a non-negative squared-Euclidean cost for row-wise cells."""
    source_values = np.asarray(source, dtype=np.float64)
    target_values = np.asarray(target, dtype=np.float64)
    if source_values.ndim != 2 or target_values.ndim != 2:
        raise ValueError("`source` and `target` must be 2D matrices with cells as rows.")
    if source_values.shape[1] != target_values.shape[1]:
        raise ValueError("`source` and `target` must share their feature dimension.")
    if min(source_values.shape[0], target_values.shape[0]) == 0:
        raise ValueError("`source` and `target` must each contain at least one cell.")
    if not np.all(np.isfinite(source_values)) or not np.all(np.isfinite(target_values)):
        raise ValueError("`source` and `target` must contain only finite values.")
    source_norm = np.sum(source_values**2, axis=1, keepdims=True)
    target_norm = np.sum(target_values**2, axis=1, keepdims=True).T
    return np.maximum(source_norm + target_norm - 2 * source_values @ target_values.T, 0.0)
