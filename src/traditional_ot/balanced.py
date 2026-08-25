"""Balanced entropic OT and confidence-filtered balanced OT.

The confidence-filtered formulation keeps both marginals exact.  Rejecting a
source row therefore removes only its target preference; it never removes its
transported mass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
CFVariant = Literal["exact", "reversible"]


@dataclass(frozen=True)
class BalancedOTResult:
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
    objective: float
    converged: bool
    n_iterations: int
    marginal_error: float


@dataclass(frozen=True)
class CFBOTResult:
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


def _positive_finite(value: object, *, name: str, allow_zero: bool = False) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"`{name}` must be a finite float.") from error
    invalid = converted < 0.0 if allow_zero else converted <= 0.0
    if not np.isfinite(converted) or invalid:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"`{name}` must be {qualifier} and finite.")
    return converted


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"`{name}` must be a positive integer.")
    return int(value)


def _cost_matrix(cost: ArrayLike) -> FloatArray:
    try:
        values = np.asarray(cost, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("`cost_matrix` must be a numeric 2D matrix.") from error
    if values.ndim != 2 or min(values.shape) == 0:
        raise ValueError("`cost_matrix` must be a non-empty 2D matrix.")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("`cost_matrix` must be finite and non-negative.")
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
    if values.shape != (n,) or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(f"`{name}` must have shape ({n},) and be strictly positive and finite.")
    return values / values.sum() if normalize else values.copy()


def _gate(gate: ArrayLike, *, n_source: int) -> BoolArray:
    values = np.asarray(gate)
    if values.shape != (n_source,) or not np.all(np.isin(values, (0, 1, False, True))):
        raise ValueError(f"`gate` must have shape ({n_source},) and contain only booleans.")
    return values.astype(bool, copy=True)


def _logsumexp(values: FloatArray, *, axis: int) -> FloatArray:
    maximum = np.max(values, axis=axis, keepdims=True)
    if not np.all(np.isfinite(maximum)):
        raise FloatingPointError("A Sinkhorn reduction has no finite support.")
    return np.squeeze(maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True)), axis=axis)


def _generalized_kl(values: FloatArray, reference: FloatArray) -> float:
    positive = values > 0.0
    terms = reference - values
    terms = terms.copy()
    terms[positive] += values[positive] * (np.log(values[positive]) - np.log(reference[positive]))
    return float(terms.sum())


def _solve_balanced(
    optimization_cost: FloatArray,
    *,
    scoring_cost: FloatArray,
    epsilon: float,
    source_weights: ArrayLike | None,
    target_weights: ArrayLike | None,
    threshold: float,
    max_iterations: int,
    warm_start: tuple[ArrayLike, ArrayLike] | None = None,
    normalize_weights: bool = True,
) -> BalancedOTResult:
    n_source, n_target = optimization_cost.shape
    if not isinstance(normalize_weights, (bool, np.bool_)):
        raise ValueError("`normalize_weights` must be boolean.")
    a = _marginal(
        source_weights, n=n_source, name="source_weights", normalize=normalize_weights
    )
    b = _marginal(
        target_weights, n=n_target, name="target_weights", normalize=normalize_weights
    )
    if not np.isclose(a.sum(), b.sum(), rtol=1e-10, atol=1e-12):
        raise ValueError(
            "Balanced OT requires source and target measures with equal total mass."
        )
    log_a, log_b = np.log(a), np.log(b)
    scaled = optimization_cost / epsilon
    if not np.all(np.isfinite(scaled)):
        raise ValueError("`cost_matrix / epsilon` is non-finite; rescale costs or increase epsilon.")
    log_kernel = log_a[:, None] + log_b[None, :] - scaled
    if warm_start is None:
        log_u = np.zeros(n_source)
        log_v = np.zeros(n_target)
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
    for iteration in range(max_iterations):
        log_u = log_a - _logsumexp(log_kernel + log_v[None, :], axis=1)
        log_v = log_b - _logsumexp(log_kernel + log_u[:, None], axis=0)
        log_coupling = log_u[:, None] + log_kernel + log_v[None, :]
        coupling = np.exp(log_coupling)
        error = float(max(np.abs(coupling.sum(axis=1) - a).sum(), np.abs(coupling.sum(axis=0) - b).sum()))
        if error < threshold:
            converged = True
            break
    n_iterations = iteration + 1
    log_coupling = log_u[:, None] + log_kernel + log_v[None, :]
    coupling = np.exp(log_coupling)
    source_mass = coupling.sum(axis=1)
    target_mass = coupling.sum(axis=0)
    transition = coupling / source_mass[:, None]
    conditional_loss = np.sum(transition * scoring_cost, axis=1)
    reference = a[:, None] * b[None, :]
    objective = float(np.sum(coupling * optimization_cost) + epsilon * _generalized_kl(coupling, reference))
    return BalancedOTResult(
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
        objective=objective,
        converged=converged,
        n_iterations=n_iterations,
        marginal_error=error,
    )


def balanced_ot(
    cost_matrix: ArrayLike,
    *,
    epsilon: float,
    source_weights: ArrayLike | None = None,
    target_weights: ArrayLike | None = None,
    threshold: float = 1e-9,
    max_iterations: int = 10_000,
    normalize_weights: bool = True,
) -> BalancedOTResult:
    """Solve balanced entropic OT from a precomputed non-negative cost."""
    cost = _cost_matrix(cost_matrix)
    epsilon = _positive_finite(epsilon, name="epsilon")
    threshold = _positive_finite(threshold, name="threshold")
    max_iterations = _positive_integer(max_iterations, name="max_iterations")
    return _solve_balanced(
        cost,
        scoring_cost=cost,
        epsilon=epsilon,
        source_weights=source_weights,
        target_weights=target_weights,
        threshold=threshold,
        max_iterations=max_iterations,
        normalize_weights=normalize_weights,
    )


def solve_fixed_gate_balanced_ot(
    cost_matrix: ArrayLike,
    gate: ArrayLike,
    *,
    rejection_cost: float,
    epsilon: float,
    source_weights: ArrayLike | None = None,
    target_weights: ArrayLike | None = None,
    threshold: float = 1e-9,
    max_iterations: int = 10_000,
    warm_start: tuple[ArrayLike, ArrayLike] | None = None,
    normalize_weights: bool = True,
) -> BalancedOTResult:
    """Solve confidence-filtered balanced OT for a fixed source gate."""
    cost = _cost_matrix(cost_matrix)
    accepted = _gate(gate, n_source=cost.shape[0])
    rejection_cost = _positive_finite(rejection_cost, name="rejection_cost")
    epsilon = _positive_finite(epsilon, name="epsilon")
    threshold = _positive_finite(threshold, name="threshold")
    max_iterations = _positive_integer(max_iterations, name="max_iterations")
    filtered = np.where(accepted[:, None], cost, rejection_cost)
    return _solve_balanced(
        filtered,
        scoring_cost=cost,
        epsilon=epsilon,
        source_weights=source_weights,
        target_weights=target_weights,
        threshold=threshold,
        max_iterations=max_iterations,
        warm_start=warm_start,
        normalize_weights=normalize_weights,
    )


def _counterfactual_loss(result: BalancedOTResult, original_cost: FloatArray) -> FloatArray:
    log_weights = np.log(result.target_marginal)[None, :] + result.log_target_scaling[None, :] - original_cost / result.epsilon
    normalizer = _logsumexp(log_weights, axis=1)
    conditional = np.exp(log_weights - normalizer[:, None])
    return np.sum(conditional * original_cost, axis=1)


def _updated_gate(score: FloatArray, current: BoolArray, rejection_cost: float, tau: float) -> BoolArray:
    updated = current.copy()
    updated[score < rejection_cost - tau] = True
    updated[score > rejection_cost + tau] = False
    return updated


def confidence_filtered_balanced_ot(
    cost_matrix: ArrayLike,
    *,
    rejection_cost: float,
    epsilon: float,
    variant: CFVariant = "reversible",
    source_weights: ArrayLike | None = None,
    target_weights: ArrayLike | None = None,
    initial_gate: ArrayLike | None = None,
    tau: float = 0.0,
    threshold: float = 1e-9,
    max_iterations: int = 10_000,
    max_outer_iterations: int = 100,
    warm_start: bool = True,
) -> CFBOTResult:
    """Run exact or reversible confidence-filtered balanced OT."""
    cost = _cost_matrix(cost_matrix)
    rejection_cost = _positive_finite(rejection_cost, name="rejection_cost")
    epsilon = _positive_finite(epsilon, name="epsilon")
    tau = _positive_finite(tau, name="tau", allow_zero=True)
    threshold = _positive_finite(threshold, name="threshold")
    max_iterations = _positive_integer(max_iterations, name="max_iterations")
    max_outer_iterations = _positive_integer(max_outer_iterations, name="max_outer_iterations")
    if variant not in ("exact", "reversible"):
        raise ValueError("`variant` must be 'exact' or 'reversible'.")
    if not isinstance(warm_start, (bool, np.bool_)):
        raise ValueError("`warm_start` must be boolean.")
    gate = np.ones(cost.shape[0], dtype=bool) if initial_gate is None else _gate(initial_gate, n_source=cost.shape[0])
    history: list[float] = []
    gate_history: list[BoolArray] = [gate.copy()]
    log_warm = None
    total_inner = 0
    all_inner = True
    outer_converged = False
    result = None
    score = None
    solved_gate = None
    for _ in range(max_outer_iterations):
        solved_gate = gate.copy()
        result = solve_fixed_gate_balanced_ot(
            cost, gate, rejection_cost=rejection_cost, epsilon=epsilon,
            source_weights=source_weights, target_weights=target_weights,
            threshold=threshold, max_iterations=max_iterations,
            warm_start=log_warm if warm_start else None,
        )
        total_inner += result.n_iterations
        all_inner &= result.converged
        history.append(result.objective)
        score = result.conditional_loss if variant == "exact" else _counterfactual_loss(result, cost)
        new_gate = _updated_gate(score, gate, rejection_cost, tau)
        if np.array_equal(new_gate, gate):
            outer_converged = True
            break
        gate = new_gate
        gate_history.append(gate.copy())
        if warm_start:
            log_warm = (result.log_source_scaling.copy(), result.log_target_scaling.copy())
    assert result is not None and score is not None and solved_gate is not None
    if not np.array_equal(solved_gate, gate):
        result = solve_fixed_gate_balanced_ot(
            cost, gate, rejection_cost=rejection_cost, epsilon=epsilon,
            source_weights=source_weights, target_weights=target_weights,
            threshold=threshold, max_iterations=max_iterations,
            warm_start=(result.log_source_scaling, result.log_target_scaling) if warm_start else None,
        )
        total_inner += result.n_iterations
        all_inner &= result.converged
        history.append(result.objective)
        score = result.conditional_loss if variant == "exact" else _counterfactual_loss(result, cost)
    return CFBOTResult(
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
        objective_history=tuple(history),
        gate_history=tuple(gate_history),
        variant=variant,
        rejection_cost=rejection_cost,
        tau=tau,
        boundary_count=int(np.count_nonzero(np.abs(score - rejection_cost) <= tau)),
        inner_converged=all_inner,
        outer_converged=outer_converged,
        n_outer_solves=len(history),
        total_inner_iterations=total_inner,
    )
